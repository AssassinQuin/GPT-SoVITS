import os
import logging
from venv import logger
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
import librosa
from feature_extractor import cnhubert
from module.models import SynthesizerTrn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from time import time as ttime
from tools.i18n.i18n import I18nAuto
import LangSegment

from inference_help import (
    clean_text_inf,
    DictToAttrRecursive,
    dict_language,
    format_text,
    splits,
    get_first,
    cut1,
    cut2,
    cut3,
    cut5,
    cut4,
    process_text,
    merge_short_text_in_array,
    get_spepc,
)


# 初始化日志信息
logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)

# 多语言
i18n = I18nAuto()

cnhubert_base_path = os.environ.get(
    "cnhubert_base_path", "GPT_SoVITS/pretrained_models/chinese-hubert-base"
)
bert_path = os.environ.get(
    "bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
)

if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()

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

logger.info(f"===== cnhubert_base_path: {cnhubert_base_path}")
logger.info(f"===== bert_path: {bert_path}")
logger.info(f"===== is_half: {is_half}")
logger.info(f"===== device: {device}")
logger.info(f"===== dtype: {dtype}")


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
        print(textlist)
        print(langlist)
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
    print(vq_model.load_state_dict(dict_s2["weight"], strict=False))
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
    print("Number of parameter: %.2fM" % (total / 1e6))
    with open("./gweight.txt", "w", encoding="utf-8") as f:
        f.write(gpt_path)


def inference(
    model_name,
    prompt_wav_path,
    prompt_text,
    prompt_language,
    text,
    text_language,
    how_to_cut=i18n("按中文句号。切"),
    top_k=100,
    top_p=0.7,
    temperature=0.7,
    ref_free=False,
):
    if prompt_text is None or len(prompt_text) == 0:
        ref_free = True
    prompt_language = dict_language[prompt_language]
    text_language = dict_language[text_language]

    prompt_text = format_text(prompt_text, prompt_language)
    text = format_text(text, text_language)

    prompt_text = clean_text_inf(prompt_text, prompt_language)

    if not ref_free:
        prompt_text = prompt_text.strip("\n")
        if prompt_text[-1] not in splits:
            prompt_text += "。" if prompt_language != "en" else "."
        print(i18n("实际输入的参考文本:"), prompt_text)

    # text = replace_consecutive_punctuation(text)
    if text[0] not in splits and len(get_first(text)) < 4:
        text = "。" + text if text_language != "en" else "." + text

    print(i18n("实际输入的目标文本:"), text)

    if how_to_cut == i18n("凑四句一切"):
        text = cut1(text)
    elif how_to_cut == i18n("凑50字一切"):
        text = cut2(text)
    elif how_to_cut == i18n("按中文句号。切"):
        text = cut3(text)
    elif how_to_cut == i18n("按英文句号.切"):
        text = cut4(text)
    elif how_to_cut == i18n("按标点符号切"):
        text = cut5(text)
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    print(i18n("实际输入的目标文本(切句后):"), text)
    texts = text.split("\n")
    texts = process_text(texts)
    texts = merge_short_text_in_array(texts, 5)

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
def do_inference(
    prompt_wav_path,
    prompt_text,
    prompt_language,
    texts,
    text_language,
    top_k=100,
    top_p=0.7,
    temperature=0.7,
    ref_free=False,
):
    t0 = ttime()
    zero_wav = np.zeros(
        int(hps.data.sampling_rate * 0.3),
        dtype=np.float16 if is_half is True else np.float32,
    )
    with torch.no_grad():
        wav16k, sr = librosa.load(prompt_wav_path, sr=16000)
        if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
            raise OSError(i18n("参考音频在3~10秒范围外，请更换！"))
        wav16k = torch.from_numpy(wav16k)
        zero_wav_torch = torch.from_numpy(zero_wav)
        if is_half is True:
            wav16k = wav16k.half().to(device)
            zero_wav_torch = zero_wav_torch.half().to(device)
        else:
            wav16k = wav16k.to(device)
            zero_wav_torch = zero_wav_torch.to(device)
        wav16k = torch.cat([wav16k, zero_wav_torch])
        ssl_content = ssl_model.model(wav16k.unsqueeze(0))[
            "last_hidden_state"
        ].transpose(1, 2)
        codes = vq_model.extract_latent(ssl_content)

        prompt_semantic = codes[0, 0]
    t1 = ttime()
    audio_opt = []
    if not ref_free:
        phones1, bert1, norm_text1 = get_phones_and_bert(prompt_text, prompt_language)

    for text in texts:
        # 解决输入目标文本的空行导致报错的问题
        if len(text.strip()) == 0:
            continue
        if text[-1] not in splits:
            text += "。" if text_language != "en" else "."
        print(i18n("实际输入的目标文本(每句):"), text)
        phones2, bert2, norm_text2 = get_phones_and_bert(text, text_language)
        print(i18n("前端处理后的文本(每句):"), norm_text2)
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
        with torch.no_grad():
            pred_semantic, idx = t2s_model.model.infer_panel(
                all_phoneme_ids,
                all_phoneme_len,
                None if ref_free else prompt,
                bert,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                early_stop_num=hz * max_sec,
            )
        t3 = ttime()
        pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
        refer = get_spepc(hps, prompt_wav_path)
        if is_half is True:
            refer = refer.half().to(device)
        else:
            refer = refer.to(device)
        audio = (
            vq_model.decode(
                pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refer
            )
            .detach()
            .cpu()
            .numpy()[0, 0]
        )
        max_audio = np.abs(audio).max()  # 简单防止16bit爆音
        if max_audio > 1:
            audio /= max_audio
        audio_opt.append(audio)
        audio_opt.append(zero_wav)
        t4 = ttime()
    print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))
    yield (
        hps.data.sampling_rate,
        (np.concatenate(audio_opt, 0) * 32768).astype(np.int16),
    )
