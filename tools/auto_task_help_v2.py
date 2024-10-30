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


def split_text(text: str, max_length: int = 30) -> List[str]:
    """
    根据中文标点符号和最大长度将文本分割成多个块。
    保留双引号内的内容作为独立的拆分块，连续的双引号内容也分别作为独立的拆分块。
    对双引号外的内容根据标点符号或 max_length 进行拆分，优先保留语义完整性。
    保留每个拆分块末尾的最后一个标点符号。

    Args:
        text (str): 输入文本。
        max_length (int, optional): 每块的最大长度。默认为30。

    Returns:
        List[str]: 分割后的文本块。
    """
    chunks = []
    text = text.replace(" ", "，")
    length = len(text)
    idx = 0
    buffer = []
    inside_quote = False

    while idx < length:
        ch = text[idx]

        if ch == "“":
            # 如果当前缓冲区有未处理的非引号内容，先拆分它
            if not inside_quote and buffer:
                non_quote_chunk = "".join(buffer).strip()
                if non_quote_chunk:
                    # 进一步拆分非引号内容
                    non_quote_chunks = split_non_quote_text(non_quote_chunk, max_length)
                    chunks.extend(non_quote_chunks)
                buffer = []
            # 开始收集引号内的内容
            inside_quote = True
            buffer.append(ch)
            idx += 1
            continue

        elif ch == "”" and inside_quote:
            buffer.append(ch)
            # 完成一个引号内容的收集
            chunks.append("".join(buffer).strip())
            buffer = []
            inside_quote = False
            idx += 1
            continue

        buffer.append(ch)
        idx += 1

    # 处理剩余的非引号内容
    if buffer:
        remaining_text = "".join(buffer).strip()
        if remaining_text:
            non_quote_chunks = split_non_quote_text(remaining_text, max_length)
            chunks.extend(non_quote_chunks)

    return chunks


def split_non_quote_text(text: str, max_length: int = 30) -> List[str]:
    """
    将非引号内的文本根据标点符号和最大长度进行拆分。
    保留每个拆分块末尾的最后一个标点符号。

    Args:
        text (str): 非引号内的文本。
        max_length (int, optional): 每块的最大长度。默认为30。

    Returns:
        List[str]: 拆分后的文本块。
    """
    result = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + max_length, length)

        # 如果剩余长度小于等于 max_length，直接添加
        if end == length:
            chunk = text[start:end].strip()
            if chunk:
                result.append(chunk)
            break

        # 优先在 end 前向窗口查找标点
        split_pos = -1
        window = 30  # 查找窗口大小，可根据需要调整

        for i in range(end - 1, max(start, end - window) - 1, -1):
            if text[i] in "。？！～，;；，":
                split_pos = i + 1
                break

        # 如果未找到，尝试在 end 处向后查找标点
        if split_pos == -1:
            for i in range(end, min(end + window, length)):
                if text[i] in "。？！～，;；，":
                    split_pos = i + 1
                    break

        # 如果仍未找到标点，则强制按 max_length 拆分
        if split_pos == -1 or split_pos > length:
            split_pos = end

        chunk = text[start:split_pos].strip()
        if chunk:
            result.append(chunk)
        start = split_pos

    return result


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
    text = text.replace("……", "。").replace("。。", "。").replace("、", "，")
    text = REPEAT_PUNCTUATION_PATTERN.sub(lambda m: m.group(0)[-1], text)
    text = remove_invalid_quotes(text)
    texts = [line.strip() for line in text.split("\n") if line.strip()]

    lines = []
    text_normalizer = TextNormalizer()
    for t in texts:
        tmp_chunks = split_text(t)
        for chunk in tmp_chunks:
            normalized_sentences = text_normalizer.normalize(chunk)
            normalized_line = "".join(normalized_sentences)
            if normalized_line:
                lines.append(normalized_line)

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


def has_omission(gen_data: Any, text: str, rate: int) -> Tuple[bool, float, str, str]:
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
我之所以想离开家，是因为我哥一直都没有音讯，所以我想出去找到他，可是想要离开家的话，至少要到二十岁，所以我必须通过特殊手段才能得到提前离开的机会，所以我靠近了他！
        """
    texts = get_texts(text)

    for text_line in texts:
        print(text_line)
# gen_text = "小猿你没事吧"
# target_text = "小媛你没事吧"
# b, f, g, t = _has_omission(gen_text, target_text)
# print(b, f, g, t)
