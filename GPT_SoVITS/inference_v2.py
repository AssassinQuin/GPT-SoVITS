import os
import re
import traceback
import numpy as np
import torch
from module.models import SynthesizerTrn
from transformers import AutoModelForMaskedLM, AutoTokenizer
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
import librosa
from feature_extractor import cnhubert
import LangSegment
from text import chinese
from text.cleaner import clean_text
from text import cleaned_text_to_sequence
from tools.my_utils import load_audio
from module.mel_processing import spectrogram_torch
import soundfile as sf
from GPT_SoVITS.role import gen_role
from loguru import logger
from GPT_SoVITS.inference_help import (
    i18n,
    dict_language,
    splits,
    DictToAttrRecursive,
    cut5,
    process_text,
    merge_short_text_in_array,
)


class TTSGenerator:
    def __init__(self, role_json_path="/root/code/GPT-SoVITS/model/role.json"):
        """
        初始化 TTSGenerator 类，加载所有必要的模型和配置。
        """
        # 加载角色映射
        self.role_map = gen_role(role_json_path)
        self.cache_role_info = ""
        self.cache = {}

        # 设备配置
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # 数据类型配置
        self.is_half = (
            eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
        )
        self.dtype = torch.float16 if self.is_half else torch.float32

        # BERT 模型加载
        bert_path = os.environ.get(
            "bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
        self.bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
        self.bert_model = (
            self.bert_model.half().to(self.device)
            if self.is_half
            else self.bert_model.to(self.device)
        )
        self.bert_model.eval()
        logger.info("BERT 模型加载完成")

        # CNHubert 模型加载
        cnhubert_base_path = os.environ.get(
            "cnhubert_base_path", "GPT_SoVITS/pretrained_models/chinese-hubert-base"
        )
        cnhubert.cnhubert_base_path = cnhubert_base_path

        self.ssl_model = cnhubert.get_model()
        self.ssl_model = (
            self.ssl_model.half().to(self.device)
            if self.is_half
            else self.ssl_model.to(self.device)
        )
        self.ssl_model.eval()
        logger.info("CNHubert 模型加载完成")

        # 初始化其他模型变量
        self.vq_model = None
        self.hps = None
        self.version = None
        self.t2s_model = None
        self.config = None
        self.hz = 50
        self.max_sec = 0  # 将在 change_gpt_weights 中设置

    def get_spectrogram(self, hps, filename):
        """
        读取音频文件并转换为频谱图。

        Args:
            hps (Any): 配置参数。
            filename (str): 音频文件路径。

        Returns:
            torch.Tensor: 频谱图张量。
        """
        audio = load_audio(filename, int(hps.data.sampling_rate))
        audio_tensor = torch.FloatTensor(audio)
        max_val = audio_tensor.abs().max()
        if max_val > 1:
            audio_tensor /= min(2, max_val)
        audio_norm = audio_tensor.unsqueeze(0)
        spectrogram = spectrogram_torch(
            audio_norm,
            hps.data.filter_length,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            center=False,
        )
        return spectrogram

    def get_bert_feature(self, text, word2ph):
        """
        获取 BERT 特征并根据每个字的音素数量重复特征向量。

        Args:
            text (str): 输入文本。
            word2ph (List[int]): 每个字对应的音素数量。

        Returns:
            torch.Tensor: 按音素重复后的 BERT 特征张量。
        """
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(self.device)
            res = self.bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        return phone_level_feature.T

    def clean_text_inf(self, text, language, version):
        phones, word2ph, norm_text = clean_text(text, language, version)
        phones = cleaned_text_to_sequence(phones, version)
        return phones, word2ph, norm_text

    def get_bert_inf(self, phones, word2ph, norm_text, language):
        language = language.replace("all_", "")
        if language == "zh":
            bert = self.get_bert_feature(norm_text, word2ph).to(self.device)
        else:
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=self.dtype,
            ).to(self.device)
        return bert

    def get_phones_and_bert(self, text, language, version, final=False):
        if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
            language = language.replace("all_", "")
            if language == "en":
                LangSegment.setfilters(["en"])
                formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
            else:
                # 因无法区别中日韩文汉字,以用户输入为准
                formattext = text
            while "  " in formattext:
                formattext = formattext.replace("  ", " ")
            if language == "zh":
                if re.search(r"[A-Za-z]", formattext):
                    formattext = re.sub(
                        r"[a-z]", lambda x: x.group(0).upper(), formattext
                    )
                    formattext = chinese.mix_text_normalize(formattext)
                    return self.get_phones_and_bert(formattext, "zh", version)
                else:
                    phones, word2ph, norm_text = self.clean_text_inf(
                        formattext, language, version
                    )
                    bert = self.get_bert_feature(norm_text, word2ph).to(self.device)
            elif language == "yue" and re.search(r"[A-Za-z]", formattext):
                formattext = re.sub(r"[a-z]", lambda x: x.group(0).upper(), formattext)
                formattext = chinese.mix_text_normalize(formattext)
                return self.get_phones_and_bert(formattext, "yue", version)
            else:
                phones, word2ph, norm_text = self.clean_text_inf(
                    formattext, language, version
                )
                bert = torch.zeros(
                    (1024, len(phones)),
                    dtype=self.dtype,
                ).to(self.device)
        elif language in {"zh", "ja", "ko", "yue", "auto", "auto_yue"}:
            textlist = []
            langlist = []
            LangSegment.setfilters(["zh", "ja", "en", "ko"])
            if language == "auto":
                for tmp in LangSegment.getTexts(text):
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
            elif language == "auto_yue":
                for tmp in LangSegment.getTexts(text):
                    if tmp["lang"] == "zh":
                        tmp["lang"] = "yue"
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
            else:
                for tmp in LangSegment.getTexts(text):
                    if tmp["lang"] == "en":
                        langlist.append(tmp["lang"])
                    else:
                        # 因无法区别中日韩文汉字,以用户输入为准
                        langlist.append(language)
                    textlist.append(tmp["text"])

            phones_list = []
            bert_list = []
            norm_text_list = []
            for i in range(len(textlist)):
                lang = langlist[i]
                phones, word2ph, norm_text = self.clean_text_inf(
                    textlist[i], lang, version
                )
                bert = self.get_bert_inf(phones, word2ph, norm_text, lang)
                phones_list.append(phones)
                norm_text_list.append(norm_text)
                bert_list.append(bert)
            bert = torch.cat(bert_list, dim=1)
            phones = sum(phones_list, [])
            norm_text = "".join(norm_text_list)
        if not final and len(phones) < 6:
            return self.get_phones_and_bert("." + text, language, version, final=True)

        return phones, bert.to(self.dtype), norm_text

    def change_sovits_weights(
        self, sovits_path, prompt_language=None, text_language=None
    ):
        """
        加载并设置 Sovits 模型的权重。

        Args:
            sovits_path (str): Sovits 模型路径。
            prompt_language (str, optional): 提示语言。
            text_language (str, optional): 文本语言。
        """
        # 加载 Sovits 模型权重
        dict_s2 = torch.load(sovits_path, map_location="cpu")
        self.hps = DictToAttrRecursive(dict_s2["config"])
        self.hps.model.semantic_frame_rate = "25hz"
        if dict_s2["weight"]["enc_p.text_embedding.weight"].shape[0] == 322:
            self.hps.model.version = "v1"
        else:
            self.hps.model.version = "v2"
        self.version = self.hps.model.version
        logger.info(f"sovits版本: {self.hps.model.version}")

        # 初始化 SynthesizerTrn 模型
        self.vq_model = SynthesizerTrn(
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model,
        )
        if "pretrained" not in sovits_path:
            del self.vq_model.enc_q
        if self.is_half:
            self.vq_model = self.vq_model.half().to(self.device)
        else:
            self.vq_model = self.vq_model.to(self.device)
        self.vq_model.eval()
        self.vq_model.load_state_dict(dict_s2["weight"])

        # 更新语言设置
        if prompt_language is not None and text_language is not None:
            if prompt_language in list(dict_language.keys()):
                prompt_text_update, prompt_language_update = (
                    {"__type__": "update"},
                    {"__type__": "update", "value": prompt_language},
                )
            else:
                prompt_text_update = {"__type__": "update", "value": ""}
                prompt_language_update = {"__type__": "update", "value": i18n("中文")}
            if text_language in list(dict_language.keys()):
                text_update, text_language_update = (
                    {"__type__": "update"},
                    {"__type__": "update", "value": text_language},
                )
            else:
                text_update = {"__type__": "update", "value": ""}
                text_language_update = {"__type__": "update", "value": i18n("中文")}
            return (
                {"__type__": "update", "choices": list(dict_language.keys())},
                {"__type__": "update", "choices": list(dict_language.keys())},
                prompt_text_update,
                prompt_language_update,
                text_update,
                text_language_update,
            )

    def change_gpt_weights(self, gpt_path):
        """
        加载并设置 GPT 模型的权重。

        Args:
            gpt_path (str): GPT 模型路径。
        """
        # 加载 GPT 模型权重
        dict_s1 = torch.load(gpt_path, map_location="cpu")
        self.config = dict_s1["config"]
        self.max_sec = self.config["data"]["max_sec"]
        self.t2s_model = Text2SemanticLightningModule(
            self.config, "****", is_train=False
        )
        self.t2s_model.load_state_dict(dict_s1["weight"])
        if self.is_half:
            self.t2s_model = self.t2s_model.half()
        self.t2s_model = self.t2s_model.to(self.device)
        self.t2s_model.eval()
        total = sum([param.nelement() for param in self.t2s_model.parameters()])
        logger.info(f"Number of parameters: {total / 1e6:.2f}M")

    def inference(
        self,
        role_name,
        text,
        text_language="多语种混合",
        top_k=20,
        top_p=0.6,
        temperature=0.6,
        ref_free=False,
        speed=1,
        if_freeze=False,
        inp_refs=None,
    ):
        """
        执行 TTS 推理。

        Args:
            role_name (str): 角色名称。
            text (str): 输入文本。
            text_language (str, optional): 文本语言。默认为 "中英混合"。
            top_k (int, optional): top_k 超参数。默认为 15。
            top_p (float, optional): top_p 超参数。默认为 1。
            temperature (float, optional): 温度参数。默认为 1。
            ref_free (bool, optional): 是否不使用参考音频。默认为 False。
            speed (float, optional): 语速。默认为 1。
            if_freeze (bool, optional): 是否冻结缓存。默认为 False。
            inp_refs (Optional[Generator], optional): 输入参考音频。默认为 None。

        Yields:
            Tuple[int, np.ndarray]: 采样率和生成的音频数组。
        """
        if self.cache_role_info == "" or self.cache_role_info != role_name:
            role_info = self.role_map.get(role_name, {})
            if role_info == {}:
                logger.info(f"角色 {role_name} 不存在，使用默认角色")
                role_info = self.role_map.get("旁白1")
            self.role_info = role_info
            self.change_gpt_weights(role_info.get("gpt_model", ""))
            self.change_sovits_weights(role_info.get("sovits_model", ""))
            self.cache_role_info = role_name

        logger.info(f"""
    =====================
    spk：{self.cache_role_info}
    速度：{speed}
    文本：{text}
    语言：{text_language}
    =====================
    """)

        tts_generator = self.get_tts_wav(
            self.role_info.get("ref_audio_path", ""),
            self.role_info.get("ref_text", ""),
            "中文",
            text,
            text_language,
            top_k,
            top_p,
            temperature,
            ref_free,
            speed,
            if_freeze,
            inp_refs,
        )
        for rate, wav_arr in tts_generator:
            return rate, wav_arr

    def get_tts_wav(
        self,
        ref_wav_path,
        prompt_text,
        prompt_language,
        text,
        text_language,
        top_k=20,
        top_p=0.6,
        temperature=0.6,
        ref_free=False,
        speed=1,
        if_freeze=False,
        inp_refs=None,
    ):
        if prompt_text is None or len(prompt_text) == 0:
            ref_free = True
        prompt_language = dict_language[prompt_language]
        text_language = dict_language[text_language]

        if not ref_free:
            prompt_text = prompt_text.strip("\n")
            if prompt_text[-1] not in splits:
                prompt_text += "。" if prompt_language != "en" else "."
        text = text.strip("\n")

        zero_wav = np.zeros(
            int(self.hps.data.sampling_rate * 0.3),
            dtype=np.float16 if self.is_half else np.float32,
        )
        if not ref_free:
            with torch.no_grad():
                wav16k, sr = librosa.load(ref_wav_path, sr=16000)
                if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
                    raise OSError(i18n("参考音频在3~10秒范围外，请更换！"))
                wav16k = torch.from_numpy(wav16k)
                zero_wav_torch = torch.from_numpy(zero_wav)
                if self.is_half:
                    wav16k = wav16k.half().to(self.device)
                    zero_wav_torch = zero_wav_torch.half().to(self.device)
                else:
                    wav16k = wav16k.to(self.device)
                    zero_wav_torch = zero_wav_torch.to(self.device)
                wav16k = torch.cat([wav16k, zero_wav_torch])
                ssl_content = self.ssl_model.model(wav16k.unsqueeze(0))[
                    "last_hidden_state"
                ].transpose(1, 2)
                codes = self.vq_model.extract_latent(ssl_content)
                prompt_semantic = codes[0, 0]
                prompt = prompt_semantic.unsqueeze(0).to(self.device)

        text = cut5(text)
        while "\n\n" in text:
            text = text.replace("\n\n", "\n")
        texts = text.split("\n")
        texts = process_text(texts)
        texts = merge_short_text_in_array(texts, 5)
        audio_opt = []
        if not ref_free:
            phones1, bert1, norm_text1 = self.get_phones_and_bert(
                prompt_text, prompt_language, self.version
            )

        for i_text, text in enumerate(texts):
            if len(text.strip()) == 0:
                continue
            if text[-1] not in splits:
                text += "。" if text_language != "en" else "."
            phones2, word2ph, norm_text2 = self.get_phones_and_bert(
                text, text_language, self.version
            )
            if not ref_free:
                bert = torch.cat([bert1, word2ph], 1)
                all_phoneme_ids = (
                    torch.LongTensor(phones1 + phones2).to(self.device).unsqueeze(0)
                )
            else:
                bert = word2ph
                all_phoneme_ids = torch.LongTensor(phones2).to(self.device).unsqueeze(0)

            bert = bert.to(self.device).unsqueeze(0)
            all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(self.device)

            if i_text in self.cache and if_freeze:
                pred_semantic = self.cache[i_text]
            else:
                with torch.no_grad():
                    pred_semantic, idx = self.t2s_model.model.infer_panel(
                        all_phoneme_ids,
                        all_phoneme_len,
                        None if ref_free else prompt,
                        bert,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                        early_stop_num=self.hz * self.max_sec,
                    )
                    if idx > 1400:
                        # If idx == 1499, yield only 0.2 seconds of silence
                        silence_duration = int(0.2 * self.hps.data.sampling_rate)
                        silence_audio = np.zeros(silence_duration, dtype=np.int16)
                        yield (self.hps.data.sampling_rate, silence_audio)
                        return  # Exit the function early
                    pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
                    self.cache[i_text] = pred_semantic

            refers = []
            if inp_refs:
                for path in inp_refs:
                    try:
                        refer = (
                            self.get_spectrogram(self.hps, path.name)
                            .to(self.dtype)
                            .to(self.device)
                        )
                        refers.append(refer)
                    except:
                        traceback.print_exc()
            if len(refers) == 0:
                refers = [
                    self.get_spectrogram(self.hps, ref_wav_path)
                    .to(self.dtype)
                    .to(self.device)
                ]

            audio = (
                self.vq_model.decode(
                    pred_semantic,
                    torch.LongTensor(phones2).to(self.device).unsqueeze(0),
                    refers,
                    speed=speed,
                )
                .detach()
                .cpu()
                .numpy()[0, 0]
            )
            max_audio = np.abs(audio).max()
            if max_audio > 1:
                audio /= max_audio
            audio_opt.append(audio)
            audio_opt.append(zero_wav)

        yield (
            self.hps.data.sampling_rate,
            (np.concatenate(audio_opt, 0) * 32768).astype(np.int16),
        )


if __name__ == "__main__":
    try:
        tts_generator = TTSGenerator()
        rate, wav_array = tts_generator.inference(
            "旁白1",
            """
DEATH NOTE——洛杉矶BB连续杀人事件

HOW TO USE IT
当彼永德巴斯蒂杀害第三个人的时候，他做了一个实验。那就是尝试在不破坏内脏的情况下，能否让被害人因内出血致死。具体的做法是，把因为药物失去意识的被害人捆绑起来，让其动弹不得，然后在不损伤皮肤的情况下，持续彻底殴打对方的左腕——也就是说，他企图通过左腕的失血过量杀人。可惜，这个实验最终以失败告终了。虽然被害人的左腕因为淤血全部变成紫黑色，但是他并没有因此死亡。被害人只是出现了奇怪的痉挛反应，完全没有生命之忧。虽然他从书本上看到过，人若失去一只手臂分量的血液生命就会停止，但是这似乎并不属实。本身杀人的方法对于彼永德巴斯蒂来说并不是最重要的，这只是余兴，只是一个实验而矣，成功也好失败也罢，对他来说根本无所谓。彼永德巴斯蒂轻轻耸了耸肩，然后抽出了刀子——不。
不，不，不。
""",
        )
        sf.write("output.wav", wav_array, rate)
    except Exception as e:
        logger.error(f"音频生成失败: {e}")
        traceback.print_exc()
