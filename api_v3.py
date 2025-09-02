import json
import os

import torch
from loguru import logger
from TTS_infer_pack.TTS import NO_PROMPT_ERROR, TTS, TTS_Config


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
        self.version = "v2proPlus"
        self.model_version = "v2proPlus"

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

        # 添加参考音频时长缓存
        self._ref_audio_cache = {}  # 缓存格式: {(ref_audio_path, prompt_text): (ref_duration, char_duration)}

        self.spk = list(self.spkMap.keys())[0]
        self.spk_info = self.spkMap[self.spk]
        self.change_spk(self.spk)

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
                elif ext == ".lab" or ext == ".txt":
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
        max_retries: int = 5,
        duration_tolerance: float = 2.0,
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
        :param max_retries: 最大重试次数（默认5次）
        :param duration_tolerance: 时长容忍倍数（默认2.0倍）
        :return: 生成音频数据（采样率, 音频数组）
        """
        import librosa
        import numpy as np

        # 验证语言参数
        if text_lang not in self.dict_language:
            raise ValueError(f"不支持的文本语言: {text_lang}")
        if prompt_lang not in self.dict_language:
            raise ValueError(f"不支持的提示语言: {prompt_lang}")

        # 计算参考音频的时长和字符数，用于估算每个字符的时长
        ref_duration = None
        char_duration = None
        expected_duration = None

        if ref_audio_path and prompt_text:
            # 使用缓存键
            cache_key = (ref_audio_path, prompt_text)

            # 检查缓存
            if cache_key in self._ref_audio_cache:
                ref_duration, char_duration = self._ref_audio_cache[cache_key]
                logger.info(
                    f"使用缓存的参考音频数据: 时长={ref_duration:.2f}秒, 每字符时长={char_duration:.3f}秒"
                )
            else:
                try:
                    # 加载参考音频获取时长
                    ref_audio, ref_sr = librosa.load(ref_audio_path, sr=None)
                    ref_duration = len(ref_audio) / ref_sr

                    # 计算参考文本的字符数（去除空格和标点）
                    ref_char_count = len(
                        [c for c in prompt_text if c.strip() and c.isalnum()]
                    )

                    if ref_char_count > 0:
                        char_duration = ref_duration / ref_char_count

                        # 保存到缓存
                        self._ref_audio_cache[cache_key] = (ref_duration, char_duration)

                        logger.info(
                            f"计算并缓存参考音频数据: 时长={ref_duration:.2f}秒, 参考字符数={ref_char_count}, 每字符时长={char_duration:.3f}秒"
                        )
                    else:
                        logger.warning(f"参考文本中没有有效字符，无法计算每字符时长")
                except Exception as e:
                    logger.warning(f"无法分析参考音频时长: {e}")

            # 计算目标文本的预期时长
            if char_duration is not None:
                target_char_count = len([c for c in text if c.strip() and c.isalnum()])
                expected_duration = char_duration * target_char_count
                logger.info(
                    f"目标字符数: {target_char_count}, 预期时长: {expected_duration:.2f}秒"
                )

        # 存储所有生成的音频结果
        generated_results = []

        for attempt in range(max_retries + 1):
            try:
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

                # 执行TTS推理
                for result in self.tts_pipeline.run(inputs):
                    if isinstance(result, tuple) and len(result) == 2:
                        sr, audio = result

                        # 计算生成音频的时长
                        generated_duration = len(audio) / sr

                        # 如果没有参考时长信息，直接返回结果
                        if expected_duration is None:
                            logger.info(
                                f"生成音频时长: {generated_duration:.2f}秒（无参考时长比较）"
                            )
                            return result

                        # 检查生成音频时长是否合理
                        duration_ratio = generated_duration / expected_duration

                        logger.info(
                            f"第{attempt + 1}次尝试 - 生成时长: {generated_duration:.2f}秒, 预期时长: {expected_duration:.2f}秒, 比例: {duration_ratio:.2f}"
                        )

                        # 存储结果
                        generated_results.append(
                            {
                                "result": result,
                                "duration": generated_duration,
                                "ratio": duration_ratio,
                                "attempt": attempt + 1,
                            }
                        )

                        # 如果时长在合理范围内，直接返回
                        if duration_ratio <= duration_tolerance:
                            logger.info(f"音频时长合理，返回结果")
                            return result

                        # 如果超过容忍范围，继续重试
                        if attempt < max_retries:
                            logger.warning(
                                f"音频时长超出预期 {duration_tolerance} 倍，进行第{attempt + 2}次重试"
                            )
                            # 可以调整参数来影响下次生成
                            temperature = min(temperature * 0.9, 0.1)  # 降低温度
                            top_p = max(top_p * 0.95, 0.1)  # 降低top_p
                            top_k = max(top_k - 1, 1)  # 降低top_k
                            break
                        else:
                            logger.warning(f"已达到最大重试次数 {max_retries}")
                            break

                    if hasattr(result, "cpu"):
                        return result.cpu().numpy()

            except Exception as e:
                logger.error(f"第{attempt + 1}次推理失败: {e}")
                if attempt == max_retries:
                    raise e
                continue

        # 如果所有重试都完成，选择时长最短的结果
        if generated_results:
            best_result = min(generated_results, key=lambda x: x["duration"])
            logger.info(
                f"选择时长最短的结果: 第{best_result['attempt']}次尝试, 时长: {best_result['duration']:.2f}秒"
            )
            return best_result["result"]

        # 如果没有任何成功的结果，抛出异常
        raise RuntimeError("所有推理尝试都失败了")

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
第83章 血淋漓的吃人
　　阳台外面，陈楚看似在晒太阳休息，实则早已经意识一沉来到了重甲兽那边。
　　二次进化到现在已经过去了四十来天。
　　经过不断吞食进化，重甲兽的体型又膨胀了一圈，身躯显得粗壮，厚重，宛若一头披着装甲的蜥鳄。
　　尤其是脑袋两侧的三对羽角，鲜红如血，足有半米长，上面根根红色尖刺耸立，让重甲兽看起来十分威猛霸气。
　　二十多米幽暗的水深下，重甲兽四肢放在身体两侧，粗壮的尾巴微微晃动，搜寻着适合的食物。
　　Biu！
　　这时一道快如闪电的影子在前方一闪而过，要不是重甲兽一直都在观察四周的话还不一定能发现。
　　biu！微不可查的水流划过声响起，那道黑影再次出现，不过这次是从重甲兽身后一闪而过。
　　“什么东西，好快。”重甲兽微微一惊。
　　从重甲兽入水后，它【他】吃过的普通变异鱼不下两百条，什么品种都有，但还没遇到过速度这么快的。
　　哪怕是当初在重甲兽腹部留下白痕的那条也比不上。
　　Biu！
　　就在陈楚惊疑时，那道黑影又从重甲兽左侧一闪而过，因为水深和相隔十多米远，因此看不清具体模样。
　　但很显然那条变异鱼已经盯上了重甲兽。
　　现在这条变异鱼环绕巡游的行为应该是在观察它，寻找适合的时机发起进攻。
　　既然你送上门来，我就不客气了。重甲兽尾巴摆动，厚重的身躯似没有注意到那条变异鱼一样缓缓向前游动。
　　忽然重甲兽身后一道黑影闪现，以可怕速度刹那出现在它后侧，长着锋利獠牙的大嘴狠狠咬向重甲兽尾巴。
　　因为速度太快，直到它一口咬在尾巴上时重甲兽才反应过来。
　　嘭！急速游动带来的冲击动能直接将重甲兽撞翻，然后。咔嚓，一排牙齿混合着血液在水中缓缓沉浮。
　　袭杀重甲兽不成反被崩断满嘴利齿，那条变异鱼吃痛下连忙松口想要逃窜，但想来就来，想走就走吗。
　　轰！
　　就在它松开尾巴的瞬间，水花爆炸，重甲兽全身肌肉爆发下猛然反身，一爪拍向那条变异鱼。
　　移动速度不快，但不代表它近距离攻击速度也慢。
　　在百倍力量的肌肉爆发下，重甲兽的左爪轰的一声落在那条变异鱼后背，恐怖力量宛若水中重炮，直接将变异鱼的身体都拍成了两截。
　　可怕的力量甚至去势不减，在水下形成一股冲击波掀起狂暴水流，直到冲出十多米后才缓缓平息。
　　随着体型成长，拥有两大天赋的重甲兽实力越发可怕，同级防御无敌，百倍爆发的力量更是恐怖，完全和体型不相匹配。
　　而直到这时重甲兽才看清这条变异鱼什么模样。
　　血水浑浊的河水中两截两米多长，形状就像一把扁长尖刀，全身被灰色鳞片覆盖的变异鱼尸体缓缓沉浮。
　　在这条变异鱼的背上，一排鱼鳍就像帆船的桅杆展开，脑袋更是凶猛，类似于海狼的嘴巴一样狭长，长满利齿。
　　不过已经被重甲兽的尾巴崩断。
　　难怪速度这么快，只从外形重甲兽就看出这条鱼的变异方向是速度。
　　可惜单独速度快没用，防御太弱的情况下只要被抓到机会一巴掌就被拍死了。
　　尾巴游动，重甲兽上前一爪抓住一截变异鱼身体，长满獠牙的大嘴狠狠一咬就撕下大块血肉吞食了起来。
　　只是扁长的变异鱼看起来很长，但加起来血肉还没几百公斤，一会就被重甲兽吃光了。
　　不够的它又在入海口肆虐了一番，在感觉全身都传来饱腹感后，才游向岸边准备回老巢休息消化。
　　十来米深的江面下，当重甲兽接近河岸时，扑通一声有什么东西落入水中，缓缓沉到它面前。
　　看着眼前巴掌大小，背鳍被鱼钩勾住不断游动的小鱼，重甲兽眼中露出意外。
　　什么情况，有人在这里钓鱼？
　　忽然重甲兽狰狞的嘴角上扬，露出雪白锋利的牙齿，锋利的右爪微微探出，抓住了那条小鱼一拉。
　　顿时河面上浮飘着的浮漂一沉，岸边拿着抄网的中年人兴奋大喊：“来了，快，快拉。”
　　“我看见了。”另一个拿着海竿的中年人很沉重，用力一拉。
　　吱吱吱！！巨大力量下九米长的海竿弯曲到了极限，足以承受数百公斤的鱼线发出吱吱声。
　　“我靠，好沉，这条鱼至少有几百斤重。”
　　“真的假的？”
　　“真的，不骗你，要知道我可是修炼者，虽然只有二重天，但双臂一晃也有几百公斤的力量。”
　　“这种情况下我居然不能把那条鱼一下拉起来，你说是不是很大。”
　　“也是，快，我用手机准备给你拍下来。”
　　“看我的，等我遛一遛把那条鱼的体力耗尽，就可以强行拉上来了，今天就是我张某人的高光时刻。”
　　岸边上，两个出来放松的中年人开始拼命的左拉右扯，却无法撼动水下的那条大鱼。
　　一时间反而更加兴奋。
　　水面下，重甲兽懒洋洋的趴在河底，右爪抓着鱼钩任由上面拉扯，偶尔心情好时会微微抬抓让他们有点动力。
　　差不多半个小时后，觉得玩的差不多的重甲兽才猛然探出左爪，将一条从身旁游过长二十多公分的小鱼按住。
　　岸上，张峰忽然感觉手上一轻，兴奋大喊：“来了！”说着手上用力往上面一甩
　　嘭！河面水花炸开，一条二十多公分长的小鱼冲出水面，啪的一声落在岸上。
　　看着草地上不断挣扎的小鱼。张峰两人脸上都露出了茫然，怎么回事，我辣么大的鱼呢？
　　。
　　接下来两天，大家都在努力修炼。
　　在经过连翻的生死战斗，又获得大量贡献点兑换资源的情况下，这个时候修炼的效果最好。
　　包括陈楚也一样。
　　突破三重的他手握蓝晶修炼下吸收能量的效率暴涨，原本逸散的三成能量也被他全部炼化，没有一点浪费。
　　这两天除了修炼外，三玖小群偶尔也会聊天，交流一下遇到的敌人和战斗经验。
　　在陈楚侧面询问下，果然林雪姐妹，伊睿，洛妃家里人都给她们准备了类似一些修炼资源。
　　这段时间连续战斗，近距离接近死亡的情况下潜力激发，不止是陈楚，林雪等人实力也在勇猛精进。
　　第三天早晨，陈楚穿上战甲来到酒店门外集合。
　　在几次血与火的战斗后，所有新生脸上的那抹稚气已经消失，变得成熟刚毅了许多。
　　只是画风更歪了。
　　两天过去，其它同学身上的战甲也变了个样，有的学刘风涂成了黄金色，有的人学白幕加了翅膀。
　　不过翅膀不是天使羽翼，而是类似天空套装那种包裹半个身体，并且可以折叠收拢。
　　除此之外有的人还给厚重的战甲配了件黑色披风，看起来也很帅气。
　　就连李昊这个浓眉大眼的家伙，他的战甲不但全身加厚，并且还把那头变异黑牛的四根牛角焊接在双肩和肩背后面。
　　四根一米多长的弯曲尖角，再配上他背着的那根三米长柱子，整个人就像一个远古战士一样威猛迫人。
　　这种情况下，反而是什么都没改装的陈楚显得比较普通，和周围的同学格格不入。
　　话说，我要不要也去改装一下？就在陈楚思考这个问题时，庞龙三个老师走了出来。
　　庞龙沉声道：“目前莱斯特鲁市的邪神教主力已经被剿灭，因此我还有大部分军方高手和军队将调往其它地方支援。”
　　“什么，老师你们要调走。”
　　“那我们呢？”顿时人群一阵骚动，所有人都有些意外这个消息。
　　庞龙沉声道：“别慌，刘菲絮老师会留下来坐镇，同时你们还有重要任务。”
　　“按照审讯信息，莱斯特鲁潜伏了一支十多人数的普通血神教徒，最强的负责人只有三重天，目的是散播血种。”
　　“血种，顾名思义是一种邪恶的力量种子，普通人在吸收后可以一跃成为真正的修炼者。”
　　“吸收了血种的人被称为血徒，可以通过猎杀普通人，炼化他们的血肉精气快速提升实力。”
　　刘风惊呼：“炼化人的血肉精气，这不是吃人吗！？”
　　庞龙沉声道：“不错，这就是血神教的可怕之处，前面联盟强者镇杀的许多邪神教高手中就有这类血使。”
　　“血神教对可罗雅的暗中渗透已经很久，通过杀人炼化血肉精气，在这个贫瘠地方催生出了一批高级血使和大批灌顶教徒。”
　　“这也是联邦对邪神教赶尽杀绝的原因，这些人早已经疯了。”
　　庞龙的话让所有人脸色都十分难看，沉重。
　　“不过你们不用担心，在莱斯特鲁散播的只是普通血种，数量不多，并且最高成长潜力只有三重天。”
　　“要不是那些血种平时和普通人一样，隐藏在人群不好区分前面就将它们一网打尽了。”
　　“接下来你们可以单人行动，也可以几人组队，注意发生命案的地方，基本能发现一些蛛丝马迹。”
　　“除此之外周边还有一些零散的叛军残存，需要你们将其清理干净，避免他们与其他城市邪神教的人勾连死灰复燃。”
　　接下来，庞龙又对陈楚等人说了一些注意细节，然后才宣布散会。
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
