import os
import json
import torch
from TTS_infer_pack.TTS import TTS, TTS_Config, NO_PROMPT_ERROR


class GPTSoVITSWrapper:
    """GPT-SoVITS 语音合成封装类（仅支持v4版本）"""

    def __init__(
        self, config_path="GPT_SoVITS/configs/tts_infer.yaml", device=None, is_half=True
    ):
        # 初始化配置
        self.tts_config = TTS_Config(config_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.is_half = is_half and torch.cuda.is_available()

        # 固定版本为v4
        self.version = "v4"
        self.model_version = "v4"

        # 初始化模型参数
        self._init_config()
        self.tts_pipeline = TTS(self.tts_config)

        # 语言选项（v4专用）
        self.dict_language = {
            "中文": "all_zh",
            "英文": "en",
            "日文": "all_ja",
            "粤语": "all_yue",
            "韩文": "all_ko",
            "中英混合": "zh",
            "日英混合": "ja",
            "粤英混合": "yue",
            "韩英混合": "ko",
            "多语种混合": "auto",
            "多语种混合(粤语)": "auto_yue",
        }

    def _init_config(self):
        """初始化TTS配置"""
        self.tts_config.device = self.device
        self.tts_config.is_half = self.is_half
        self.tts_config.version = self.version

        # 加载默认模型路径
        self._load_default_weights()

    def _load_default_weights(self):
        """加载默认模型权重"""
        with open("./weight.json", "r", encoding="utf-8") as f:
            weights = json.load(f)

        # GPT模型路径
        self.gpt_path = weights.get("GPT", {}).get(
            self.version, "GPT_SoVITS/pretrained_models/s1v3.ckpt"
        )
        self.tts_config.t2s_weights_path = self.gpt_path

        # SoVITS模型路径
        self.sovits_path = weights.get("SoVITS", {}).get(
            self.version, "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth"
        )
        self.tts_config.vits_weights_path = self.sovits_path

    def set_gpt_weights(self, gpt_path: str):
        """设置GPT模型权重路径"""
        if not os.path.exists(gpt_path):
            raise FileNotFoundError(f"GPT权重文件不存在: {gpt_path}")

        self.tts_config.t2s_weights_path = gpt_path
        self.gpt_path = gpt_path
        self.tts_pipeline.init_t2s_weights(gpt_path)

    def set_sovits_weights(self, sovits_path: str):
        """设置SoVITS模型权重路径"""
        if not os.path.exists(sovits_path):
            raise FileNotFoundError(f"SoVITS权重文件不存在: {sovits_path}")

        self.tts_config.vits_weights_path = sovits_path
        self.sovits_path = sovits_path
        self.tts_pipeline.init_vits_weights(sovits_path)

    def inference(
        self,
        text: str,
        text_lang: str = "中文",
        ref_audio_path: str = None,
        prompt_text: str = None,
        prompt_lang: str = "中文",
        top_k: int = 5,
        top_p: float = 1.0,
        temperature: float = 1.0,
        speed_factor: float = 1.0,
        sample_steps: int = 32,
        **kwargs,
    ):
        """
        执行语音合成推理

        :param text: 需要合成的文本
        :param text_lang: 文本语言（参考dict_language的键）
        :param ref_audio_path: 参考音频路径
        :param prompt_text: 提示文本
        :param prompt_lang: 提示语言
        :param top_k: 采样top_k
        :param top_p: 采样top_p
        :param temperature: 温度参数
        :param speed_factor: 语速因子（0.6-1.65）
        :param sample_steps: 采样步数（推荐32）
        :return: 生成音频数据（采样率, 音频数组）
        """
        # 验证语言参数
        if text_lang not in self.dict_language:
            raise ValueError(f"不支持的文本语言: {text_lang}")
        if prompt_lang not in self.dict_language:
            raise ValueError(f"不支持的提示语言: {prompt_lang}")

        # 构建输入参数
        inputs = {
            "text": text,
            "text_lang": self.dict_language[text_lang],
            "ref_audio_path": ref_audio_path,
            "prompt_text": prompt_text or "",
            "prompt_lang": self.dict_language[prompt_lang],
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "speed_factor": speed_factor,
            "sample_steps": sample_steps,
            **kwargs,
        }

        try:
            for result in self.tts_pipeline.run(inputs):
                if isinstance(result, tuple) and len(result) == 2:
                    return result  # (sr, audio)
                if hasattr(result, "cpu"):
                    return result.cpu().numpy()
        except NO_PROMPT_ERROR:
            raise ValueError("V4版本需要提供参考文本！")

    @staticmethod
    def tts_to_wav(audio_data, sample_rate, output_path):
        """
        将音频数据保存为WAV文件

        :param audio_data: 音频数据数组
        :param sample_rate: 采样率
        :param output_path: 输出文件路径
        """
        import soundfile as sf

        sf.write(output_path, audio_data, sample_rate)


if __name__ == "__main__":
    # 使用示例
    tts = GPTSoVITSWrapper()
    tts.set_gpt_weights("/root/code/GPT-SoVITS/GPT_weights_v4/丹恒-e15.ckpt")
    tts.set_sovits_weights(
        "/root/code/GPT-SoVITS/SoVITS_weights_v4/丹恒_e4_s6664_l32.pth"
    )
    # 合成语音
    sr, audio = tts.inference(
        text="""
“‘V’……‘C’……‘I’？不对……‘M’……这个还是‘V’……‘X’……‘D’……这里是并排三个‘I’……‘L’？……这是‘L’……嗯……果然还是太勉强了……”
果然也就只能这样看了。别说什么汉字谚语了，伤痕根本就是单纯用直线和曲线构成的排序，不管是用铅笔还是和刀子画线，看起来都会像什么东西吧。
""",
        text_lang="中英混合",
        ref_audio_path="/root/code/GPT-SoVITS/output/丹恒/archive_danheng_1.wav",
        prompt_text="重新介绍一下自己，我叫丹恒，担任列车的护卫，列车的智库也由我维护。",
        prompt_lang="中文",
    )

    # 保存结果
    tts.tts_to_wav(audio, sr, "output_tmp.wav")
