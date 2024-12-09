import sys
import os
import re
from typing import List, Any, Tuple


from pypinyin import pinyin, Style
from difflib import SequenceMatcher
from tools.zh_normalization.text_normlization import TextNormalizer

# 将项目根目录添加到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 预编译正则表达式
CHINESE_PUNCTUATION = "。！？；：，、……"
QUOTE_PATTERN = re.compile(r"“(.*?)”")
REPEAT_PUNCTUATION_PATTERN = re.compile(r"[，！？。～、]+")
NON_WORD_PATTERN = re.compile(r"[^\w\s，。！？]")


def replace_invalid_quotes(match: re.Match) -> str:
    """
    替换无效的中文双引号。

    Args:
        match (re.Match): 正则匹配对象。

    Returns:
        str: 替换后的字符串。
    """
    quoted_text = match.group(1)
    if quoted_text and quoted_text[-1] in CHINESE_PUNCTUATION:
        return f"“{quoted_text}”"  # 保留双引号
    return quoted_text  # 移除双引号


def remove_invalid_quotes(text: str) -> str:
    """
    移除或保留无效的中文双引号内的内容。

    Args:
        text (str): 输入文本。

    Returns:
        str: 处理后的文本。
    """
    return QUOTE_PATTERN.sub(replace_invalid_quotes, text)


def split_text(text: str, max_length: int = 50) -> List[Tuple[str, str]]:
    """
    根据中文终止符号和最大长度将文本分割成多个块。
    1. 双引号内的内容单独拆分，不管里面有多少内容。
    2. 英文内容单独拆分，不管里面有多少内容，除非位于标点符号分隔的块内。
    3. 拆分出来的每个块分类为 "中文"、"英文" 或 "中英混合"。
    4. 单个字母不作为独立的英文块，只拆分完整的单词或英文短语。
    5. 强制拆分时，必须在最近的标点符号处分割，不能在无标点符号的情况下进行拆分。

    Args:
        text (str): 输入的文本。
        max_length (int, optional): 每块的最大长度。默认为50。

    Returns:
        List[Tuple[str, str]]: 分割后的文本块及其分类。
    """
    chunks = []
    length = len(text)

    # 定义正则表达式模式
    quote_pattern = re.compile(r"“[^”]*”")  # 匹配双引号内的内容
    # 英文匹配调整为匹配两个或以上字母的单词或短语
    english_pattern = re.compile(
        r"\b[A-Za-z]{2,}(?:\s+[A-Za-z]{2,})*\b"
    )  # 至少两个字母的英文单词或短语
    # 强制分割的中文终止符号
    strong_punctuation = "。！？"
    # 其他中文标点符号
    other_punctuation = "，；：、"

    # 使用finditer查找所有匹配
    matches = list(quote_pattern.finditer(text)) + list(english_pattern.finditer(text))
    # 按照出现位置排序
    matches.sort(key=lambda m: m.start())

    last_idx = 0  # 上一个匹配的结束位置

    for match in matches:
        start, end = match.span()
        # 处理匹配前的中文内容
        if start > last_idx:
            non_matched_text = text[last_idx:start]
            # 对中文内容进行拆分
            chinese_chunks = split_non_quote_text(
                non_matched_text, max_length, strong_punctuation, other_punctuation
            )
            for chunk in chinese_chunks:
                label = classify_text(chunk)
                chunks.append((chunk, label))

        matched_text = match.group()
        if quote_pattern.fullmatch(matched_text):
            # 双引号内的内容作为一个整体，分类为中英混合或中文
            label = classify_text(matched_text)
            chunks.append((matched_text, label))
        elif english_pattern.fullmatch(matched_text):
            # 英文内容作为一个整体，分类为英文
            label = classify_text(matched_text)
            chunks.append((matched_text.lower(), label))

        last_idx = end  # 更新上一个匹配位置

    # 处理文本末尾的中文内容
    if last_idx < length:
        remaining_text = text[last_idx:]
        # 对中文内容进行拆分
        chinese_chunks = split_non_quote_text(
            remaining_text, max_length, strong_punctuation, other_punctuation
        )
        for chunk in chinese_chunks:
            label = classify_text(chunk)
            chunks.append((chunk, label))

    return chunks


def split_non_quote_text(
    text: str, max_length: int, strong_punc: str, other_punc: str
) -> List[str]:
    """
    对不在双引号内的中文文本，根据中文终止符号或最大长度进行拆分，
    遇到强制分割符号（。！？）时，在最近的标点符号处分割。
    如果在达到最大长度附近找不到标点符号，则不进行拆分。

    Args:
        text (str): 要拆分的中文文本。
        max_length (int): 每块的最大长度。
        strong_punc (str): 强制分割的中文终止符号。
        other_punc (str): 其他中文标点符号集合。

    Returns:
        List[str]: 拆分后的文本块。
    """
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        # 如果剩余文本的长度小于等于 max_length，且包含标点符号，则全部作为一个块
        if length - start <= max_length:
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # 取从 start 开始的 max_length 字符
        end = start + max_length
        if end >= length:
            end = length

        # 在 [start, end] 范围内寻找最后一个标点符号
        last_punc_idx = -1
        for i in range(end - 1, start - 1, -1):
            if text[i] in strong_punc + other_punc:
                last_punc_idx = i
                break

        if last_punc_idx != -1:
            # 在找到的标点符号处分割
            split_pos = last_punc_idx + 1  # 包含标点符号
            chunk = text[start:split_pos].strip()
            if chunk:
                chunks.append(chunk)
            start = split_pos
        else:
            # 尝试在 [start, end + max_length] 范围内寻找标点符号
            search_end = min(end + max_length, length)
            found = False
            for i in range(end, search_end):
                if text[i] in strong_punc + other_punc:
                    split_pos = i + 1
                    chunk = text[start:split_pos].strip()
                    if chunk:
                        chunks.append(chunk)
                    start = split_pos
                    found = True
                    break
            if not found:
                # 没有找到标点符号，无法拆分，保留整个剩余文本为一个块
                remaining = text[start:].strip()
                if remaining:
                    chunks.append(remaining)
                break

    return chunks


def is_chinese(char: str) -> bool:
    """判断字符是否为中文字符"""
    return "\u4e00" <= char <= "\u9fff"


def is_english(char: str) -> bool:
    """判断字符是否为英文字符"""
    return "a" <= char <= "z" or "A" <= char <= "Z"


def classify_text(text: str) -> str:
    """
    根据文本内容分类。
    仅中文：返回 "中文"
    仅英文：返回 "英文"
    中英混合：返回 "中英混合"
    含有非中英文字符：返回 "多语种混合"

    Args:
        text (str): 要分类的文本。

    Returns:
        str: 分类结果。
    """
    # 清除标点符号
    cleaned_text = re.sub(r"[^\w\s]", "", text)

    has_chinese = False
    has_english = False
    has_other_language = False

    for char in cleaned_text:
        if is_chinese(char):
            has_chinese = True
        elif is_english(char):
            has_english = True
        elif char.isspace():
            continue
        else:
            has_other_language = True

        # 提前终止判断
        if has_chinese and has_english:
            return "中英混合"
        if has_other_language:
            return "多语种混合"

    if has_chinese:
        return "中文"
    elif has_english:
        return "英文"
    else:
        return "中文"


def get_texts(text: str, ignore_punctuation: bool = False) -> List[str]:
    """
    清洗和分割文本为多个句子块。

    Args:
        text (str): 输入文本。
        ignore_punctuation (bool, optional): 是否忽略标点符号。默认为False。

    Returns:
        List[str]: 分割后的文本块。
    """
    if ignore_punctuation:
        text = text.replace("“", "").replace("”", "")
    text = (
        text.replace("……", "，")
        .replace("，，", "，")
        .replace("、", "，")
        .replace("=", "")
        .replace("—", "")
    )
    text = REPEAT_PUNCTUATION_PATTERN.sub(lambda m: m.group(0)[-1], text)
    text = remove_invalid_quotes(text)
    texts = [line.strip() for line in text.split("\n") if line.strip()]

    lines = []
    text_normalizer = TextNormalizer()

    for t in texts:
        tmp_chunks = split_text(t)
        for chunk in tmp_chunks:
            (line, language) = chunk
            if language == "中英混合":
                normalized_sentences = text_normalizer.normalize(line)
                normalized_line = "".join(normalized_sentences)
                if normalized_line:
                    lines.append((normalized_line, language))
            else:
                lines.append(chunk)

    return lines


def transcribe_and_clean(data_in: Any, rate: int) -> str:
    """
    转录音频并清洗文本。

    Args:
        data_in (Any): 输入音频数据。
        rate (int): 采样率。

    Returns:
        str: 清洗后的文本。
    """
    from wav2text import only_asr

    raw_text = only_asr(data_in, rate)
    cleaned_text = re.sub(r"<\|.*?\|>", "", raw_text)
    cleaned_text = re.sub(r"\s+", "", cleaned_text)
    cleaned_text = NON_WORD_PATTERN.sub("", cleaned_text)
    return cleaned_text


def has_omission(
    gen_data: Any, text: str, rate: int, lang
) -> Tuple[bool, float, str, str]:
    """
    检查生成的音频是否遗漏原文内容。

    Args:
        gen_data (Any): 生成的音频数据。
        text (str): 原始文本。
        rate (int): 采样率。

    Returns:
        Tuple[bool, float, str, str]: 是否有遗漏，相似度，生成的清洗文本，原始清洗文本。
    """
    gen_text = transcribe_and_clean(gen_data, rate)
    if lang == "中英混合" or lang == "多语种混合":
        text_normalizer = TextNormalizer()
        gen_text = "".join(text_normalizer.normalize(gen_text))
    return _has_omission(gen_text, text)


def _has_omission(gen_text: str, text: str) -> Tuple[bool, float, str, str]:
    """
    检查生成的文本是否有遗漏原文的内容，通过拼音比较和相似度判断。

    Args:
        gen_text (str): 生成的文本。
        text (str): 原始文本。

    Returns:
        Tuple[bool, float, str, str]: 有遗漏与否，相似度，生成的清洗文本，原始清洗文本。
    """
    if gen_text == "":
        return True, 0, "", text

    def clean_text(text: str) -> str:
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", "", text)
        return text.lower()

    def get_pinyin_str(text: str) -> str:
        return " ".join(word[0] for word in pinyin(text, style=Style.TONE2))

    def has_pinyin_intersection(str1, str2):
        # 将字符串转换为拼音，处理多音字
        pinyin_list1 = [
            item[0]
            for sublist in pinyin(hans=str1, heteronym=True, style=Style.TONE2)
            for item in sublist
        ]
        pinyin_list2 = [
            item[0]
            for sublist in pinyin(hans=str2, heteronym=True, style=Style.TONE2)
            for item in sublist
        ]

        # 将拼音列表转换为集合
        pinyin_set1 = set(pinyin_list1)
        pinyin_set2 = set(pinyin_list2)

        # 检查拼音集合是否有交集
        return bool(pinyin_set1 & pinyin_set2)

    def calculate_similarity(pinyin1: str, pinyin2: str) -> float:
        return SequenceMatcher(None, pinyin1, pinyin2).ratio()

    gen_text_clean = clean_text(gen_text)
    text_clean = clean_text(text)

    if gen_text_clean == text_clean:
        return False, 100.0, gen_text_clean, text_clean

    gen_pinyin_str = get_pinyin_str(gen_text_clean)
    text_pinyin_str = get_pinyin_str(text_clean)

    weight = 100 / len(text_clean) / 2

    sim_ratio = calculate_similarity(gen_pinyin_str, text_pinyin_str) * 100

    needs_repeat = True
    if len(gen_text_clean) != len(text_clean):
        length_diff = abs(len(gen_text_clean) - len(text_clean))
        sim_ratio -= length_diff * weight
        needs_repeat = sim_ratio < max(1 - weight * 2, 95)
    else:
        mismatch = False
        sim_ratio = 100.0
        for idx in range(len(text_clean)):
            if text_clean[idx] == gen_text_clean[idx]:
                continue
            else:
                # 若 text_clean[idx] 与 gen_text_clean[idx] 没有交集
                if not has_pinyin_intersection(text_clean[idx], gen_text_clean[idx]):
                    sim_ratio -= weight
                    mismatch = True

        if not mismatch:
            sim_ratio = 100
        print(max(1 - weight * 2, 95))
        needs_repeat = sim_ratio < max(1 - weight * 2, 95)

    return needs_repeat, sim_ratio, gen_text_clean, text_clean


def clear_text(text: str, ignore_punctuation: bool = False) -> str:
    """
    清理文本，移除标点符号和空白字符，并进行规范化。

    Args:
        text (str): 输入文本。
        ignore_punctuation (bool, optional): 是否忽略标点符号。默认为False。

    Returns:
        str: 清理后的文本。
    """
    if ignore_punctuation:
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", "", text)
        text = text.lower()
    return "".join(TextNormalizer().normalize(text))


if __name__ == "__main__":
    text = """
跟他搭话的是，同为英文老师的高冢阳二老师。因为体型太胖，被酒井副校长命令减肥，却一直不见他瘦下去。另外学生给他起的外号叫重金属（日文是ヘビメタ，和Heavy Metal的日语简称一样。），本人似乎是以为学生记得他之前讲过自己以前玩摇滚的事，听着非常受用，其实只是高代谢（日文是ヘビー？メタボリック，Heavy Metabolic，简称和上文的Heavy Metal的一样。）的略称而已。
“啊，我班上的学生接二连三地出问题呢。”
莲实开口说道。

        """
    texts = get_texts(text)

    for text_line in texts:
        print(text_line)
# gen_text = "小猿你没事吧"
# target_text = "小媛你没事吧"
# b, f, g, t = _has_omission(gen_text, target_text)
# print(b, f, g, t)
