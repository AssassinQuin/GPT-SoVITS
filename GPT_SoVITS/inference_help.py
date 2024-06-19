import os
import re
import string
import torch
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from module.mel_processing import spectrogram_torch
from my_utils import load_audio
from tools.i18n.i18n import I18nAuto
import json
import inspect
from pypinyin import pinyin, Style
from difflib import SequenceMatcher
import logging
from venv import logger


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

punctuation = set(["!", "?", "…", ",", ".", "-", " "])


# 定义语言对应的符号
language_punctuation = {
    "zh": {
        "comma": "，",
        "period": "。",
        "question_mark": "？",
        "exclamation_mark": "！",
        "ellipsis": "…",
        "colon": "：",
        "newline": "。",
    },
    "en": {
        "comma": ",",
        "period": ".",
        "question_mark": "?",
        "exclamation_mark": "!",
        "ellipsis": "...",
        "colon": ":",
        "newline": ".",
    },
}

dict_language = {
    i18n("中文"): "all_zh",  # 全部按中文识别
    i18n("英文"): "en",  # 全部按英文识别#######不变
    i18n("日文"): "all_ja",  # 全部按日文识别
    i18n("中英混合"): "zh",  # 按中英混合识别####不变
    i18n("日英混合"): "ja",  # 按日英混合识别####不变
    i18n("多语种混合"): "auto",  # 多语种启动切分识别语种
}

splits = {
    "，",
    "。",
    "？",
    "！",
    ",",
    ".",
    "?",
    "!",
    "~",
    ":",
    "：",
    "—",
    "…",
}


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


#
def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    return spec


def format_text(text, language="zh"):
    text = text.strip("\n")
    # 根据语言获取对应的符号
    punct = (
        language_punctuation["zh"] if "zh" in language else language_punctuation["en"]
    )

    # 替换规则
    text = re.sub(r" {2,}", punct["period"], text)  # 多个空格替换为句号
    text = re.sub(r"\n|\r", punct["newline"], text)  # 回车，换行符替换为句号
    text = re.sub(r" ", punct["comma"], text)  # 一个空格替换为逗号
    text = re.sub(r"[\"\'‘’“”\[\]【】〖〗]", "", text)  # 删除特殊符号
    text = re.sub(r"[:：……—]", punct["period"], text)  # 替换为句号

    # 替换所有非当前语言的符号为对应语言的符号
    if language == "en":
        text = re.sub(
            r"[，。？！…～：]",
            lambda match: (
                punct["comma"]
                if match.group(0) == "，"
                else punct["period"]
                if match.group(0) == "。"
                else punct["question_mark"]
                if match.group(0) == "？"
                else punct["exclamation_mark"]
                if match.group(0) == "！"
                else punct["ellipsis"]
                if match.group(0) == "…"
                else punct["period"]
            ),
            text,
        )
    elif language == "zh":
        text = re.sub(
            r"[,.\?!~:]+",
            lambda match: (
                punct["comma"]
                if match.group(0) == ","
                else punct["period"]
                if match.group(0) == "."
                else punct["question_mark"]
                if match.group(0) == "?"
                else punct["exclamation_mark"]
                if match.group(0) == "!"
                else punct["ellipsis"]
                if match.group(0) == "..."
                else punct["period"]
            ),
            text,
        )

    # 确保文本开头没有标点符号
    text = re.sub(r"^[，。？！…～：]|^[,.?!~:]", "", text)

    # 保留多个连续标点符号中的第一个
    def remove_consecutive_punctuation(text):
        result = []
        i = 0
        while i < len(text):
            if text[i] in punct.values():
                result.append(text[i])
                while i + 1 < len(text) and text[i + 1] in punct.values():
                    i += 1
            else:
                result.append(text[i])
            i += 1
        return "".join(result)

    text = remove_consecutive_punctuation(text)

    return text


#
def clean_text_inf(text, language):
    phones, word2ph, norm_text = clean_text(text, language)
    phones = cleaned_text_to_sequence(phones)
    return phones, word2ph, norm_text


#
def get_first(text):
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text


def merge_short_text_in_array(texts, threshold):
    """
    合并短文本，直到文本长度达到指定的阈值。

    参数:
    texts (list): 包含短文本的列表。
    threshold (int): 合并文本的阈值长度。

    返回:
    list: 合并后的文本列表。
    """
    # 如果文本列表长度小于2，直接返回原列表
    if len(texts) < 2:
        return texts

    result = []
    current_text = ""

    for text in texts:
        # 将当前文本添加到合并文本中
        current_text += text
        # 如果合并文本长度达到阈值，将其添加到结果列表中，并重置合并文本
        if len(current_text) >= threshold:
            result.append(current_text)
            current_text = ""

    # 处理剩余的文本
    if current_text:
        # 如果结果列表为空，直接添加剩余文本
        if not result:
            result.append(current_text)
        else:
            # 否则，将剩余文本添加到结果列表的最后一个元素中
            result[-1] += current_text

    return result


def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts


def cut1(inp):
    """
    将输入文本按每4个元素一组进行切割，并去除包含标点符号的组。

    参数:
    inp (str): 输入文本。

    返回:
    str: 切割后的文本，每组之间用换行符分隔。
    """
    inp = inp.strip("\n")
    inps = inp.split()
    split_idx = list(range(0, len(inps), 4))
    split_idx[-1] = None
    opts = [" ".join(inps[i:j]) for i, j in zip([None] + split_idx, split_idx + [None])]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut2(inp):
    """
    将输入文本按长度切割，每段不超过50个字符，并去除包含标点符号的段。

    参数:
    inp (str): 输入文本。

    返回:
    str: 切割后的文本，每段之间用换行符分隔。
    """
    inp = inp.strip("\n")
    inps = inp.split()
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += " " + inps[i] if tmp_str else inps[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str.strip())
            tmp_str = ""
    if tmp_str:
        opts.append(tmp_str.strip())
    if len(opts) > 1 and len(opts[-1]) < 50:
        opts[-2] += " " + opts[-1]
        opts = opts[:-1]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


# 定义标点符号集合
punctuation = set(string.punctuation + "，。？！…～：")


def split_text_by_punctuation(text, punctuations):
    """
    将输入文本按指定的标点符号进行切割，并保留句末标点符号。

    参数:
    text (str): 输入文本。
    punctuations (set): 用于切割文本的标点符号集合。

    返回:
    str: 切割后的文本，每句之间用换行符分隔。
    """
    segments = []
    start = 0
    for i, char in enumerate(text):
        if char in punctuations:
            segments.append(text[start : i + 1].strip())
            start = i + 1
    if start < len(text):
        segments.append(text[start:].strip())
    return "\n".join(segments)


def cut3(inp):
    """
    将输入文本按句号、问号和感叹号进行切割，并去除包含标点符号的句。

    参数:
    inp (str): 输入文本。

    返回:
    str: 切割后的文本，每句之间用换行符分隔。
    """
    return split_text_by_punctuation(inp, {"。", "？", "！", "～"})


def cut4(inp):
    """
    将输入文本按英文句号进行切割，并去除包含标点符号的句。

    参数:
    inp (str): 输入文本。

    返回:
    str: 切割后的文本，每句之间用换行符分隔。
    """
    return split_text_by_punctuation(inp, {".", "?", "!", "~"})


def cut5(inp):
    """
    将输入文本按标点符号进行切割，并去除包含标点符号的段。

    参数:
    inp (str): 输入文本。

    返回:
    str: 切割后的文本，每段之间用换行符分隔。
    """
    inp = inp.strip("\n")
    punds = r"[,.;?!、，。？！;：…]"
    items = re.split(f"({punds})", inp)
    mergeitems = ["".join(group) for group in zip(items[::2], items[1::2])]
    if len(items) % 2 == 1:
        mergeitems.append(items[-1])
    opt = [item for item in mergeitems if not set(item).issubset(punctuation)]
    return "\n".join(opt)


def cut(ips, language="zh"):
    if language == "en":
        return cut4(ips)
    else:
        return cut3(ips)


def process_text(texts):
    """
    处理输入的文本列表，移除无效的文本条目。

    参数:
    texts (list): 包含文本条目的列表。

    返回:
    list: 处理后的有效文本列表。

    异常:
    ValueError: 当输入的文本列表中全是无效条目时抛出。
    """
    # 判断列表中是否全是无效的文本条目
    if all(text in [None, " ", "\n", ""] for text in texts):
        raise ValueError("请输入有效文本")

    # 过滤并返回有效的文本条目
    return [text for text in texts if text not in [None, " ", "", "\n"]]


def get_project_path():
    """
    获取项目路径。

    Returns:
        str: 项目路径。
    """
    # 获取当前脚本所在的路径
    current_path = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    )
    # 获取当前项目根路径（去除最后一级目录）
    project_path = os.path.dirname(current_path)
    return project_path


def load_model_config():
    """
    加载模型配置文件，并返回包含模型路径和文件信息的字典。

    Returns:
        dict: 包含模型路径和文件信息的字典。
    """
    project_path = get_project_path()
    config_path = os.path.join(project_path, "GPT_SoVITS", "model_config.json")

    # 读取并解析 model_config.json 文件
    with open(config_path, "r", encoding="utf-8") as file:
        model_config = json.load(file)

    models = {}
    for model_name in model_config:
        models[model_name] = {}

        # 设置模型路径
        absolute_model_path = os.path.join(
            project_path, "model", model_config[model_name]["model_path"]
        )
        models[model_name]["model_path"] = absolute_model_path

        # 查找 GPT_weights 模型路径
        gpt_path = next(
            (
                os.path.join(root, file)
                for root, dirs, files in os.walk(absolute_model_path)
                for file in files
                if file.endswith(".ckpt")
            ),
            None,
        )
        if gpt_path:
            models[model_name]["gpt_path"] = gpt_path

        # 查找 SoVITS_weights 模型路径
        sovits_path = next(
            (
                os.path.join(root, file)
                for root, dirs, files in os.walk(absolute_model_path)
                for file in files
                if file.endswith(".pth")
            ),
            None,
        )
        if sovits_path:
            models[model_name]["sovits_path"] = sovits_path

        # 查找参考音频路径
        prompt_wav_path = next(
            (
                os.path.join(root, file)
                for root, dirs, files in os.walk(absolute_model_path)
                for file in files
                if file.endswith(".wav")
            ),
            None,
        )
        if prompt_wav_path:
            models[model_name]["prompt_wav_path"] = prompt_wav_path

        # 查找参考文本路径
        prompt_text = next(
            (
                file.replace(".wav", "")
                for root, dirs, files in os.walk(absolute_model_path)
                for file in files
                if file.endswith(".wav")
            ),
            None,
        )
        if prompt_text:
            models[model_name]["prompt_text"] = prompt_text

        models[model_name]["prompt_language"] = model_config[model_name][
            "prompt_language"
        ]

    return models


def has_omission(gen_text, text):
    """
    检查生成的文本是否有遗漏原文的内容，通过拼音比较和相似度判断。
    :param gen_text: 生成的文本
    :param text: 原始文本
    :return: 若生成的文本拼音相似度超过98%且没有增加重复字，则返回 False（没有遗漏），否则返回 True（有遗漏）
    """

    def remove_punctuation(text):
        """
        移除文本中的标点符号。
        """
        return re.sub(r"[^\w\s]", "", text)

    def get_pinyin(text):
        """
        获取文本的拼音表示。
        """
        return " ".join(["".join(p) for p in pinyin(text, style=Style.NORMAL)])

    def get_pinyin_duo(text):
        """
        获取文本的拼音表示，支持多音字。
        """
        return pinyin(text, style=Style.NORMAL, heteronym=True)

    def calculate_similarity(pinyin1, pinyin2):
        """
        计算两个拼音字符串的相似度。
        """
        return SequenceMatcher(None, pinyin1, pinyin2).ratio()

    # 去除标点符号
    gen_text_clean = remove_punctuation(gen_text)
    text_clean = remove_punctuation(text)

    # 获取拼音
    gen_text_pinyin = get_pinyin(gen_text_clean)
    text_pinyin = get_pinyin(text_clean)

    gen_text_ping_duo = get_pinyin_duo(gen_text_clean)
    text_ping_duo = get_pinyin_duo(text_clean)

    # 计算拼音相似度
    sim_ratio = calculate_similarity(gen_text_pinyin, text_pinyin) * 100

    res = True
    # 判断是否有遗漏
    if len(gen_text_clean) != len(text_clean):
        # 如果字数不等，根据拼音相似度判断，每个字的差异减少5%的相似度
        length_difference = abs(len(gen_text_clean) - len(text_clean))
        res = True
        sim_ratio = sim_ratio - length_difference * 5
    else:
        # 对比 gen_text_ping_duo 与 text_ping_duo
        # 判断每个字符是否存在多音字，若存在，则对比两个字符多音字是否有相同，若有则满足字符相等，若不存在则减 5
        for gen_word, text_word in zip(gen_text_ping_duo, text_ping_duo):
            if not any(gen in text_word for gen in gen_word):
                sim_ratio -= 5
        res = sim_ratio < 98

    return res, sim_ratio, gen_text_clean, text_clean


def test_has_omission():
    gen_text = "而在长缩不定的雾气中他仿佛已经听到海浪声传浪声传入耳边"
    org_text = "而在涨缩不定的雾气中他仿佛已经听到海浪声传入耳边"
    logger.info(has_omission(gen_text, org_text))  # true

    gen_text = "外观古典精美地黑色燧发手枪是的就连自身都要打个问号"
    org_text = "外观古典精美的黑色燧发手枪是的就连自身都要打个问号"
    logger.info(has_omission(gen_text, org_text))  # false 100% 多音字匹配

    gen_text = "而是消耗消耗最后一片残存的幻影正如雾般从空气中消散干净"
    org_text = "而失乡号最后一片残存的幻影正如雾般从空气中消散干净"
    logger.info(has_omission(gen_text, org_text))  # true

    gen_text = "那位虔诚的牧师正趴在齐刀台旁喘着粗气"
    org_text = "那位虔诚的牧师正趴在祈祷台旁大口喘着粗气"
    logger.info(has_omission(gen_text, org_text))  # true

    gen_text = "那位虔诚地牧师正趴在起到台旁大口喘着粗气"
    org_text = "那位虔诚的牧师正趴在祈祷台旁大口喘着粗气"
    logger.info(has_omission(gen_text, org_text))  # false 100% 多音字匹配

    gen_text = "略显凌乱的单身公寓内周铭伏案桌前"
    org_text = "却显凌乱的单身公寓内周明福案桌前"
    logger.info(has_omission(gen_text, org_text))  # true


def cut_text(texts, num=30, language="zh"):
    """
    将文本列表按指定长度切割，尽量在标点符号处进行切割，确保每段长度大致相等。

    参数:
    texts (list): 包含文本段落的列表。
    num (int): 每段的最大字符数。
    language (str): 文本语言（用于选择标点符号）。

    返回:
    list: 切割后的文本段落列表。
    """
    result = []
    for t in texts:
        while len(t) > num:
            punctuation_positions = [
                t.rfind(p, 0, num) for p in language_punctuation[language].values()
            ]
            punctuation_positions = [pos for pos in punctuation_positions if pos != -1]

            if punctuation_positions:
                cut_index = max(punctuation_positions)
            else:
                # 找不到标点符号时，在最接近 num 的地方切割
                cut_index = num - 1
                for offset in range(1, num):
                    if num - offset > 0 and re.match(r"\W", t[num - offset]):
                        cut_index = num - offset
                        break
                    if num + offset < len(t) and re.match(r"\W", t[num + offset]):
                        cut_index = num + offset
                        break

            result.append(t[: cut_index + 1].strip())
            t = t[cut_index + 1 :].strip()

        if t:
            result.append(t)
    return result


def get_texts(text, language="zh"):
    text = cut(text, language)
    texts = text.split("\n")
    texts = process_text(texts)
    texts = merge_short_text_in_array(texts, 15)

    texts = cut_text(texts, 40)
    return texts


def test_format_text():
    text = """
第1章 那天，起了很大的雾
无边无际的浓雾在窗外翻滚，浓郁的仿佛整个世界都已经消失在雾的彼端，唯有混沌未明的天光穿透雾气照进屋来，让这安静的房间里维持着一种半昏半明的光线。
略显凌乱的单身公寓内，周铭伏案桌前，桌上的杂物被粗暴地推到了一旁，而形容憔悴的他正在奋笔疾书：
〖第七天，情况没有任何改变，浓雾笼罩着窗外的一切，窗户被不知名的力量封锁……整个房间仿佛被什么东西给整个“浇铸”进了某种异常的空间里……
没办法与外界联系，也没有水电，但电灯一直亮着，电脑也能打开——尽管我已经拔掉了它的电源线……〗
仿佛有轻微的风声突然从窗户方向传来，正埋头在日记本上书写的周铭猛然间抬起了头，憔悴的双眼中微微亮起光来，然而下一秒他便发现那只是自己的幻觉，那扇窗外仍旧只有盘踞不散的苍白浓雾，一个死寂的世界冷漠地笼罩着他这小小的蜗居之所。
他的目光扫过窗台，看到了被胡乱丢弃的扳手与铁锤——那是他过去几天里尝试离开房间的痕迹，然而现在这些坚硬粗苯的工具只是静静地躺在那里，仿佛在嘲讽着他的窘迫局面。
几秒钟后，周铭的表情重新变得平静下来——带着这种异常的平静，他再次低下头，回到自己的书写中：
〖我被困住了，完全没有头绪的困局，过去几天里，我甚至尝试过拆掉屋顶、墙壁和地板，但用尽全身力气也没能在墙面上留下一丁点痕迹，这房间变得像是……像是一个和空间“浇铸”在一起的盒子，没有任何出路……
除了那扇门。
但那扇门外的情况……更不对劲。〗
周铭再一次停了下来，他慢慢审视着自己刚刚留下的字迹，又有些漫不经心地翻动日记本，看着自己在过去几天里留下的东西——压抑的言语，无意义的胡思乱想，烦躁的涂鸦，以及强行放松精神时写下的冷笑话。
他不知道自己写下这些有什么意义，不知道这些胡言乱语的东西将来能给谁看，事实上他甚至都不是一个习惯写日记的人——作为一个闲暇时间相当有限的中学教师，他可没多少精力花在这上面。
但现在，不管愿不愿意，他有了大把的闲暇时间。
在一觉醒来之后，他被困在了自己的房间。
窗外是不会消散的浓雾，雾气浓郁到甚至根本看不见除了雾之外的任何东西，整个世界仿佛失去了昼夜交替，二十四小时恒定的、昏昏沉沉的光线充斥着房间，窗户锁死，水电中断，手机没有信号，在房间里搞出再大的动静也引不来外界的救援。
仿佛一个荒诞的噩梦，梦中的一切都在违背自然规律地运转，但周铭已经用尽了所有的办法来确定一件事：这里没有幻觉，也没有梦境，有的只是不再正常的世界，以及一个暂时还算正常的自己。
他深深吸了口气，目光最后落在房间尽头那唯一的一扇门上。
普普通通的廉价白色木门，上面还钉着自己从去年就忘记换下来而一直留到今天的日历，门把手被磨得铮亮，门口脚垫放得有些歪。
那扇门可以打开。
如果说这封闭异化的房间如同一个囚笼，那么这囚笼最恶毒之处莫过于它其实保留了一扇随时可以推开的大门，在时时刻刻引诱着笼中的囚徒推门离开——可那大门对面却不是周铭想要的“外面”。
那里没有陈旧却亲切的楼道走廊，没有阳光明媚的街道与充满活力的人群，没有自己所熟悉的一切。
那里只有一个陌生而令人心生不安的异域他乡，而且“那边”同样是个无法逃脱的困境。
但周铭知道，留给自己犹豫的时间已经不多了，所谓的“选择”更是从一开始就不存在。
他的食物储备是有限的，几桶矿泉水也只剩下最后四分之一，他已经在这封闭的房间中尝试过了所有脱困、求救的手段，如今摆在他面前的路只有一个，那就是做好准备，去“门”的对面求得一线生机。
或许，还能有机会调查清楚到底是什么原因造就了如今这诡异窘迫的超自然局面。
周铭轻轻吸了口气，低下头在日记本上留下最后几段：
    〖……但不管怎样，现在唯一的选择都只剩下了前往门的对面，至少在那艘诡异的船上还能找到些吃的东西，而我过去几天在那边的探索和准备应该也足以让自己在那艘船上生存下来……尽管我在那边能做的准备其实也实在有限。
最后的最后，致后来者，如果我没能回来，而未来的某一天真的有什么救援人员之类的人打开了这间房间，看到了这本日记，请不要把我所写下的这一切当成是个荒诞的故事——它真的发生了，尽管这令人毛骨悚然，但真的有一个名叫周铭的人，被困在了疯狂诡异的时空异象里面。
我尽己所能地在这本日记中描述了自己所见到的种种异常现象，也记录下了自己为脱困而做出的所有努力，如果真的有什么“后来者”的话，请至少记住我的名字，至少记住这一切曾经发生过。〗
周铭合上了日记本，把笔扔进旁边的笔筒，慢慢从桌后站起身来。
是离开的时候了，在彻底陷入被动与绝境之前。
但在短暂的思考之后，他却没有直接走向那唯一可以通向“外界”的大门，而是径直走向了自己的床铺。
他必须以万全的姿态来面对门对面的“异乡”——而他现在的状态，尤其是精神状态还不够好。
周铭不知道自己能不能睡着，但哪怕是强迫自己躺在床上放空大脑，也好过在精神过于疲惫的状态下前往“对面”。
八小时后，周铭睁开了眼睛。
窗外仍然是一片混沌雾霭，昼夜不明的天光带着令人压抑的晦暗。
周铭直接无视了窗外的情况，他从所剩不多的储备中拿出食物，吃到八分饱，随后来到房间角落的穿衣镜前。
镜子中的男人仍然头发杂乱，显得颇为狼狈，也没有什么气质可言，但周铭仍然死死地盯着镜子中的自己，就仿佛是为了把这副模样永久地印在脑海中一般。
他就这样盯着镜子看了好几分钟，然后低声自言自语着，仿佛是要说给镜子里的那个人般开口：“你叫周铭，至少在‘这边’，你叫周铭，要时刻牢记这一点。”
这之后，他才转身离开。
来到那扇再熟悉不过的房门前，周铭深深吸了口气，将手放在把手上面。
除了一身衣服，他没有携带任何额外的东西，既没有带食物，也没有带防身的装备，这是之前几次“探索”留下的经验——除了自身之外，他没办法把任何东西带过这扇门。
事实上，他甚至觉得连这“自身”都要打个问号，因为……
周铭转动把手，一把推开了房门，一团涨缩蠕动的灰黑色雾气如某种帷幕般出现在他眼前，而在涨缩不定的雾气中，他仿佛已经听到海浪声传入耳边。
迈步跨过那层雾气，略显腥咸的海风迎面而来，耳边虚幻的海浪声变得真切，脚下也传来了微微的摇晃感，周铭在短暂的眩晕后睁开眼睛，入目之处是一片宽阔空旷的木质甲板，伫立在黑暗阴云下的高耸桅杆，以及船舷外根本看不到边际的、正在微微起伏的海面。
周铭低下头，看到的是比自己记忆中要更加强壮一些的身体，一身看起来做工精致造价不菲但风格完全陌生的船长制服，一双骨节粗大的手掌，以及正握在自己手中的、外观古典精美的黑色燧发手枪。
是的，就连“自身”都要打个问号。
"""
    text = format_text(text)
    texts = get_texts(text)
    for i in range(len(texts)):
        print(f"{i+1}. {texts[i]}")


# 测试例子
if __name__ == "__main__":
    test_format_text()
