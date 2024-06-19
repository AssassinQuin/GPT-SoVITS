import os
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
import librosa
from feature_extractor import cnhubert
from module.models import SynthesizerTrn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from time import time as ttime
import LangSegment
import soundfile as sf
from tools.asr.funasr_asr import only_asr
from inference_help import (
    clean_text_inf,
    DictToAttrRecursive,
    dict_language,
    format_text,
    splits,
    get_first,
    get_texts,
    get_spepc,
    load_model_config,
    get_project_path,
    has_omission,
    logger,
    i18n,
)

project_path = get_project_path()

cnhubert_base_path = os.environ.get(
    "cnhubert_base_path",
    f"{project_path}/GPT_SoVITS/pretrained_models/chinese-hubert-base",
)
bert_path = os.environ.get(
    "bert_path",
    f"{project_path}/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
)

if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
is_half = eval(os.environ.get("is_half", "False")) and torch.cuda.is_available()

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


cnhubert.cnhubert_base_path = cnhubert_base_path

tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)

if is_half is True:
    bert_model = bert_model.half().to(device)
else:
    bert_model = bert_model.to(device)

dtype = torch.float16 if is_half is True else torch.float32


ssl_model = cnhubert.get_model()
if is_half is True:
    ssl_model = ssl_model.half().to(device)
else:
    ssl_model = ssl_model.to(device)

# 加载模型
models = load_model_config()
curr_model_name = ""

logger.info(f"===== cnhubert_base_path: {cnhubert_base_path}")
logger.info(f"===== bert_path: {bert_path}")
logger.info(f"===== is_half: {is_half}")
logger.info(f"===== device: {device}")
logger.info(f"===== dtype: {dtype}")
logger.info(f"===== models: {models}")


#
def get_bert_inf(phones, word2ph, norm_text, language):
    language = language.replace("all_", "")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)  # .to(dtype)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half is True else torch.float32,
        ).to(device)

    return bert


#
def get_phones_and_bert(text, language):
    if language in {"en", "all_zh", "all_ja"}:
        language = language.replace("all_", "")
        if language == "en":
            LangSegment.setfilters(["en"])
            formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
        else:
            # 因无法区别中日文汉字,以用户输入为准
            formattext = text
        while "  " in formattext:
            formattext = formattext.replace("  ", " ")
        phones, word2ph, norm_text = clean_text_inf(formattext, language)
        if language == "zh":
            bert = get_bert_feature(norm_text, word2ph).to(device)
        else:
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float16 if is_half is True else torch.float32,
            ).to(device)
    elif language in {"zh", "ja", "auto"}:
        textlist = []
        langlist = []
        LangSegment.setfilters(["zh", "ja", "en", "ko"])
        if language == "auto":
            for tmp in LangSegment.getTexts(text):
                if tmp["lang"] == "ko":
                    langlist.append("zh")
                    textlist.append(tmp["text"])
                else:
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
        else:
            for tmp in LangSegment.getTexts(text):
                if tmp["lang"] == "en":
                    langlist.append(tmp["lang"])
                else:
                    # 因无法区别中日文汉字,以用户输入为准
                    langlist.append(language)
                textlist.append(tmp["text"])
        logger.info(textlist)
        logger.info(langlist)
        phones_list = []
        bert_list = []
        norm_text_list = []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = clean_text_inf(textlist[i], lang)
            bert = get_bert_inf(phones, word2ph, norm_text, lang)
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            bert_list.append(bert)
        bert = torch.cat(bert_list, dim=1)
        phones = sum(phones_list, [])
        norm_text = "".join(norm_text_list)

    return phones, bert.to(dtype), norm_text


#
def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T


#
def change_sovits_weights(sovits_path):
    global vq_model, hps
    dict_s2 = torch.load(sovits_path, map_location="cpu")
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    )
    if "pretrained" not in sovits_path:
        del vq_model.enc_q
    if is_half is True:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    logger.info(vq_model.load_state_dict(dict_s2["weight"], strict=False))
    with open("./sweight.txt", "w", encoding="utf-8") as f:
        f.write(sovits_path)


#
def change_gpt_weights(gpt_path):
    global hz, max_sec, t2s_model, config
    hz = 50
    dict_s1 = torch.load(gpt_path, map_location="cpu")
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half is True:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()
    total = sum([param.nelement() for param in t2s_model.parameters()])
    logger.info("Number of parameter: %.2fM" % (total / 1e6))
    with open("./gweight.txt", "w", encoding="utf-8") as f:
        f.write(gpt_path)


def inference(
    text,
    text_language,
    model_name="旁白",
    top_k=30,
    top_p=1,
    temperature=1,
    ref_free=False,
):
    global curr_model_name

    if curr_model_name != model_name:
        if models[model_name] is not None:
            curr_model_name = model_name
            change_sovits_weights(models[curr_model_name]["sovits_path"])
            change_gpt_weights(models[curr_model_name]["gpt_path"])
        else:
            raise Exception("模型不存在")

    prompt_wav_path = models[model_name]["prompt_wav_path"]
    prompt_text = models[model_name]["prompt_text"]
    prompt_language = models[model_name]["prompt_language"]

    if prompt_text is None or len(prompt_text) == 0:
        ref_free = True
    prompt_language = dict_language[prompt_language]
    text_language = dict_language[text_language]

    prompt_text = format_text(prompt_text, prompt_language)
    text = format_text(text, text_language)

    if not ref_free:
        prompt_text = prompt_text.strip("\n")
        if prompt_text[-1] not in splits:
            prompt_text += "。" if prompt_language != "en" else "."
        logger.info(i18n("实际输入的参考文本:"), prompt_text)

    if text[0] not in splits and len(get_first(text)) < 4:
        text = "。" + text if text_language != "en" else "." + text

    logger.info(i18n("实际输入的目标文本:"), text)

    texts = get_texts(text, text_language)

    return do_inference(
        prompt_wav_path,
        prompt_text,
        prompt_language,
        texts,
        text_language,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        ref_free=ref_free,
    )


# 执行推理，合成音频
"""
top_k：

作用：限制每次生成时只考虑概率最高的 k 个单词。
控制方式：通过限制选择范围，使生成过程更集中于最可能的选项，减少生成内容的随机性。
效果：较低的 top_k 值会使生成内容更确定、更保守，可能更符合训练数据的模式，但也可能显得单调。
较高的 top_k 值会增加生成内容的多样性，但可能会引入一些不符合上下文的单词。
top_p (核采样, nucleus sampling)：

作用：限制生成时只考虑概率总和达到 p 的单词集合。
控制方式：根据概率累积值选择单词，而不是固定数量的单词。这种方法允许根据概率分布的形状动态调整候选单词的数量。
效果：较低的 top_p 值会使生成内容更确定，只考虑最可能的选项。较高的 top_p 值会增加生成内容的多样性，但可能会引入更多不确定性。
temperature：

作用：通过控制生成过程中的随机性来调整模型输出的多样性。
控制方式：通过对预测概率分布进行缩放，使得模型对不同选项的区分度更大或更小。
具体公式为： p_i' = p_i ^ (1 / temperature)，其中 p_i 是某个选项的原始概率。
效果：较低的 temperature 值（接近于0）会使模型输出更确定，更倾向于选择最高概率的选项。
较高的 temperature 值会使生成内容更加随机和多样化。
"""


def do_inference(
    prompt_wav_path,
    prompt_text,
    prompt_language,
    texts,
    text_language,
    top_k=1,
    top_p=1,
    temperature=0.8,
    ref_free=False,
):
    t0 = ttime()
    zero_wav = np.zeros(
        int(hps.data.sampling_rate * 0.3),
        dtype=np.float16 if is_half else np.float32,
    )

    try:
        wav16k, sr = librosa.load(prompt_wav_path, sr=16000)
        if not (48000 <= wav16k.shape[0] <= 160000):
            raise ValueError("参考音频在3~10秒范围外，请更换！")

        wav16k = torch.from_numpy(wav16k).to(device)
        zero_wav_torch = torch.from_numpy(zero_wav).to(device)
        if is_half:
            wav16k = wav16k.half()
            zero_wav_torch = zero_wav_torch.half()

        wav16k = torch.cat([wav16k, zero_wav_torch])
        ssl_content = ssl_model.model(wav16k.unsqueeze(0))[
            "last_hidden_state"
        ].transpose(1, 2)
        codes = vq_model.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
    except Exception as e:
        logger.error(f"Error processing prompt audio: {e}")
        return

    t1 = ttime()
    audio_opt = []

    if not ref_free:
        phones1, bert1, norm_text1 = get_phones_and_bert(prompt_text, prompt_language)

    index = 0
    try_again = 0
    count = 0
    step = 0
    reduce_count = 0.03
    reduce_step = 3

    max_try_again = int((1 - 0.2) / reduce_count) * int((30 - 1) / reduce_step)

    pinyin_similarity_map = {}

    while index < len(texts):
        text = texts[index].strip()
        if len(text) == 0:
            index += 1
            continue
        if text[-1] not in splits:
            text += "。" if text_language != "en" else "."

        phones2, bert2, norm_text2 = get_phones_and_bert(text, text_language)

        if not ref_free:
            bert = torch.cat([bert1, bert2], 1)
            all_phoneme_ids = (
                torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
            )
        else:
            bert = bert2
            all_phoneme_ids = torch.LongTensor(phones2).to(device).unsqueeze(0)

        bert = bert.to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
        prompt = prompt_semantic.unsqueeze(0).to(device)
        t2 = ttime()

        try:
            with torch.no_grad():
                pred_semantic, idx = t2s_model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_len,
                    None if ref_free else prompt,
                    bert,
                    top_k=top_k - step,
                    top_p=top_p - count,
                    temperature=temperature - count,
                    early_stop_num=hz * max_sec,
                )
            t3 = ttime()
            pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
            refer = get_spepc(hps, prompt_wav_path)
            refer = refer.half().to(device) if is_half else refer.to(device)

            audio = (
                vq_model.decode(
                    pred_semantic,
                    torch.LongTensor(phones2).to(device).unsqueeze(0),
                    refer,
                )
                .detach()
                .cpu()
                .numpy()[0, 0]
            )

            max_audio = np.abs(audio).max()  # Prevent 16-bit overflow
            if max_audio > 1:
                audio /= max_audio

            temp_audio_path = os.path.join(
                project_path, "tmp", "gen", "audio", f"temp_audio_{t2}.wav"
            )
            os.makedirs(os.path.dirname(temp_audio_path), exist_ok=True)
            sf.write(temp_audio_path, audio, hps.data.sampling_rate)

            asr_result = only_asr(temp_audio_path)
            _phones, _bert, _norm_text = get_phones_and_bert(asr_result, text_language)

            is_continu, pinyin_similarity, gen_text_clean, text_clean = has_omission(
                _norm_text, norm_text2
            )

            logger.info(f"""
=========
【输入目标】: {text}
【处理之后】: {norm_text2}
-----------------------------
【try_again】  : {try_again}
【top_k】      : {top_k + step}
【top_p】      : {top_p + count}
【temperature】: {temperature + count}
-----------------------------
生成文本：{gen_text_clean}
输入文本：{text_clean}
相似度：{pinyin_similarity}
=========
""")

            pinyin_similarity_map[pinyin_similarity] = audio
            if is_continu:
                try_again += 1
                count -= reduce_count
                if try_again % int((1 - 0.2) / reduce_count) == 0:
                    step -= reduce_count
                    count = 0
                if try_again >= max_try_again:
                    index += 1
                    try_again = 0
                    count = 0
                    step = 0
                    best_similarity = max(pinyin_similarity_map.keys())
                    best_audio = pinyin_similarity_map[best_similarity]
                    audio_opt.append(best_audio)
                    audio_opt.append(zero_wav)
                    t4 = ttime()
                    os.remove(temp_audio_path)
                    pinyin_similarity_map = {}
                else:
                    os.remove(temp_audio_path)
                    continue
            else:
                os.remove(temp_audio_path)
                audio_opt.append(audio)
                audio_opt.append(zero_wav)
                t4 = ttime()
                index += 1
                try_again = 0
                count = 0
                step = 0
                pinyin_similarity_map = {}

        except Exception as e:
            logger.error(f"Error during inference: {e}")
            index += 1
            try_again = 0
            continue

    logger.info("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))
    yield (
        hps.data.sampling_rate,
        (np.concatenate(audio_opt, 0) * 32768).astype(np.int16),
    )
