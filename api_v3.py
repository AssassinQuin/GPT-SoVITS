from loguru import logger
import os
import json
import torch
from TTS_infer_pack.TTS import TTS, TTS_Config, NO_PROMPT_ERROR


class GPTSoVITSWrapper:
    """GPT-SoVITS 语音合成封装类（仅支持v4版本）"""

    def __init__(
        self, config_path="GPT_SoVITS/configs/tts_infer.yaml", device=None, is_half=True
    ):
        self.gen_model_info()

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

        self.spk = list(self.spkMap.keys())[0]
        self.spk_info = self.spkMap[self.spk]
        self.change_spk(self.spk)  # 修改 api_v3.py 第44行为如下内容：

    def _init_config(self):
        """初始化TTS配置"""
        self.tts_config.device = self.device
        self.tts_config.is_half = self.is_half
        self.tts_config.version = self.version

        # 加载默认模型路径
        self._load_default_weights()

    def gen_model_info(self, model_dir="/root/code/GPT-SoVITS/model"):
        """
        加载模型信息并应用角色配置文件（改进版）
        :param model_dir: 模型根目录路径
        """
        self.spkMap = {}
        role_info = {}  # 初始化角色配置

        # 1. 读取角色配置文件
        role_info_path = os.path.join(model_dir, "role_info.json")
        try:
            if os.path.exists(role_info_path):
                with open(role_info_path, "r", encoding="utf-8") as f:
                    role_info = json.load(f)
                    logger.info(f"成功加载角色配置文件: {role_info_path}")
            else:
                logger.warning(f"角色配置文件不存在: {role_info_path}")
        except Exception as e:
            logger.error(f"加载角色配置文件失败: {str(e)}", exc_info=True)

        # 2. 遍历模型目录
        if not os.path.isdir(model_dir):
            logger.error(f"模型目录不存在: {model_dir}")
            return

        for speaker_dir in os.listdir(model_dir):
            speaker_path = os.path.join(model_dir, speaker_dir)

            if not os.path.isdir(speaker_path):
                continue

            # 3. 初始化配置项（包含speed_factor）
            self.spkMap[speaker_dir] = {
                "ref_audio_path": None,
                "prompt_text": None,
                "sovits_path": None,
                "gpt_path": None,
                "prompt_lang": "中文",
                "speed_factor": 1.0,  # 默认值
            }

            # 4. 应用角色配置
            if speaker_dir in role_info:
                config = role_info[speaker_dir]
                # 类型安全校验
                if isinstance(config, dict):
                    speed_factor = config.get("speed_factor")
                    if isinstance(speed_factor, (int, float)):
                        self.spkMap[speaker_dir]["speed_factor"] = float(speed_factor)
                    else:
                        logger.warning(
                            f"角色 {speaker_dir} 的speed_factor类型错误，使用默认值"
                        )
                else:
                    logger.warning(f"角色 {speaker_dir} 配置格式错误")

            # 5. 加载模型文件（原有逻辑保持不变）
            for filename in os.listdir(speaker_path):
                file_path = os.path.join(speaker_path, filename)
                base_name, ext = os.path.splitext(filename)

                if ext == ".wav":
                    self.spkMap[speaker_dir]["ref_audio_path"] = file_path
                elif ext == ".lab":
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            self.spkMap[speaker_dir]["prompt_text"] = f.read().strip()
                    except Exception as e:
                        logger.error(
                            f"读取 {speaker_dir} 的lab文件失败: {file_path}, 错误: {e}"
                        )
                elif ext == ".pth" and speaker_dir in base_name:
                    self.spkMap[speaker_dir]["sovits_path"] = file_path
                elif ext == ".ckpt" and speaker_dir in base_name:
                    self.spkMap[speaker_dir]["gpt_path"] = file_path

            # 6. 验证必要文件
            required_files = [
                "ref_audio_path",
                "prompt_text",
                "sovits_path",
                "gpt_path",
            ]
            missing = [f for f in required_files if not self.spkMap[speaker_dir][f]]
            if missing:
                logger.warning(
                    f"说话人 {speaker_dir} 缺失关键文件: {', '.join(missing)}"
                )
                del self.spkMap[speaker_dir]

        logger.info(f"成功加载 {len(self.spkMap)} 个说话人模型信息")

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

    def change_spk(self, spk: str):
        if spk not in self.spkMap:
            raise ValueError(f"不支持的spk: {spk}")
        if self.spk != spk:
            self.spk = spk
            self.spk_info = self.spkMap[spk]
            self.set_gpt_weights(self.spkMap[spk]["gpt_path"])
            self.set_sovits_weights(self.spkMap[spk]["sovits_path"])
            logger.info(f"""
##################################
切换新模型
模型: {self.spk}
speed_factor: {self.spkMap[spk]["speed_factor"]}
##################################
""")

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

    def inference_with_spk(
        self,
        spk: str,
        text: str,
        text_lang: str = "中文",
        # ref_audio_path: str = None,
        # prompt_text: str = None,
        # prompt_lang: str = "中文",
        top_k: int = 5,
        top_p: float = 1.0,
        temperature: float = 1.0,
        # speed_factor: float = 1.0,
        sample_steps: int = 32,
        **kwargs,
    ):
        if spk not in self.spkMap:
            raise ValueError(f"不支持的spk: {spk}")
        self.change_spk(spk)
        return self.inference(
            text,
            text_lang,
            self.spk_info["ref_audio_path"],
            self.spk_info["prompt_text"],
            self.spk_info["prompt_lang"],
            top_k,
            top_p,
            temperature,
            self.spk_info["speed_factor"],
            sample_steps,
            **kwargs,
        )

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
    # 若没有目录则创建
    out_putfile = "test_gen"
    if not os.path.exists(out_putfile):
        os.makedirs(out_putfile)
    # 使用示例
    tts = GPTSoVITSWrapper()
    # 合成语音
    spk_list = tts.spkMap.keys()
    idx = 0
    for spk in spk_list:
        sr, audio = tts.inference_with_spk(
            spk,
            text="""
第119章 我全都要
　　灵光飞起。
　　雷俊用息壤旗一卷，然后人便向上升。
　　鉴于上次高家和德相和尚伏击一事，雷俊现在出入深谷地形更加谨慎。
　　虽然他行事尽量不招摇，但随着他修为实力越来越强，也开始少不了有树大招风的征兆。
　　出谷前，先仔细侦查一二后，风雷符化夜风，再悄然溜出。
　　好在，这次没人来堵他。
    """,
            text_lang="中文",
        )

        # 保存结果

        tts.tts_to_wav(audio, sr, os.path.join(out_putfile, f"{spk}.wav"))
        logger.info(f"""
##################
已完成：{idx}/{len(spk_list)}
##################
""")
        idx += 1
