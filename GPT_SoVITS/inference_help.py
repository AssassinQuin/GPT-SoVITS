import os
import re
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


def cut3(inp):
    """
    将输入文本按句号进行切割，并去除包含标点符号的句。

    参数:
    inp (str): 输入文本。

    返回:
    str: 切割后的文本，每句之间用换行符分隔。
    """
    inp = inp.strip("\n")
    opts = [
        item
        for item in inp.strip("。").split("。")
        if not set(item).issubset(punctuation)
    ]
    return "\n".join(opts)


def cut4(inp):
    """
    将输入文本按英文句号进行切割，并去除包含标点符号的句。

    参数:
    inp (str): 输入文本。

    返回:
    str: 切割后的文本，每句之间用换行符分隔。
    """
    inp = inp.strip("\n")
    opts = [
        item
        for item in inp.strip(".").split(".")
        if not set(item).issubset(punctuation)
    ]
    return "\n".join(opts)


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
        adjusted_sim_ratio = sim_ratio - length_difference * 5
        res = True
        sim_ratio = adjusted_sim_ratio
    else:
        # 对比 gen_text_ping_duo 与 text_ping_duo
        # 判断每个字符是否存在多音字，若存在，则对比两个字符多音字是否有相同，若有则满足字符相等，若不存在则减 5
        sim_ratio = 100
        for gen_word, text_word in zip(gen_text_ping_duo, text_ping_duo):
            if not any(gen in text_word for gen in gen_word):
                sim_ratio -= 5
        res = sim_ratio < 98

    print(f"""
=========
生成文本：{gen_text_clean}
生成文本长度：{len(gen_text_clean)}
输入文本：{text_clean}
输入文本长度：{len(text_clean)}
相似度：{sim_ratio}
=========
""")
    return res, sim_ratio


# 测试例子
if __name__ == "__main__":
    gen_text = "而在长缩不定的雾气中他仿佛已经听到海浪声传浪声传入耳边"
    org_text = "而在涨缩不定的雾气中他仿佛已经听到海浪声传入耳边"
    print(has_omission(gen_text, org_text))  # true

    gen_text = "外观古典精美地黑色燧发手枪是的就连自身都要打个问号"
    org_text = "外观古典精美的黑色燧发手枪是的就连自身都要打个问号"
    print(has_omission(gen_text, org_text))  # false 100% 多音字匹配

    gen_text = "而是消耗消耗最后一片残存的幻影正如雾般从空气中消散干净"
    org_text = "而失乡号最后一片残存的幻影正如雾般从空气中消散干净"
    print(has_omission(gen_text, org_text))  # true

    gen_text = "那位虔诚的牧师正趴在齐刀台旁喘着粗气"
    org_text = "那位虔诚的牧师正趴在祈祷台旁大口喘着粗气"
    print(has_omission(gen_text, org_text))  # true

    gen_text = "那位虔诚地牧师正趴在起到台旁大口喘着粗气"
    org_text = "那位虔诚的牧师正趴在祈祷台旁大口喘着粗气"
    print(has_omission(gen_text, org_text))  # false 100% 多音字匹配

    gen_text = "略显凌乱的单身公寓内周铭伏案桌前"
    org_text = "却显凌乱的单身公寓内周明福案桌前"
    print(has_omission(gen_text, org_text))  # true
