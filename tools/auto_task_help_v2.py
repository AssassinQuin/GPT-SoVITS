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
        window = 5  # 查找窗口大小，可根据需要调整

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


# def split_text(text: str, max_length: int = 30) -> List[str]:
#     """
#     根据中文标点符号和最大长度将文本分割成多个块。

#     Args:
#         text (str): 输入文本。
#         max_length (int, optional): 每块的最大长度。默认为30。

#     Returns:
#         List[str]: 分割后的文本块。
#     """
#     chunks = []
#     curr_text = []
#     last_punct_idx = -1

#     for idx, ch in enumerate(text):
#         if ch == "“":
#             if curr_text:
#                 tmp_text = "".join(curr_text)
#                 tmp_text = tmp_text.lstrip("。？！，")  # 移除开头的标点
#                 if tmp_text:
#                     chunks.append(tmp_text)
#                 curr_text = []
#             tmp_quote = [ch]
#             while idx < len(text) and text[idx] != "”":
#                 tmp_quote.append(text[idx])
#                 idx += 1
#             if idx < len(text):
#                 tmp_quote.append("”")
#                 chunks.append("".join(tmp_quote))
#             continue

#         if ch in "。？！～，":
#             last_punct_idx = len(curr_text)
#         curr_text.append(ch)

#         if len(curr_text) > max_length:
#             if last_punct_idx != -1:
#                 cut_point = last_punct_idx + 1
#                 tmp_text = "".join(curr_text[:cut_point]).lstrip("。？！～，")
#                 if tmp_text:
#                     chunks.append(tmp_text)
#                 curr_text = curr_text[cut_point:]
#                 last_punct_idx = -1
#             else:
#                 # 强制切割到下一个标点符号
#                 for future_idx in range(idx + 1, len(text)):
#                     if text[future_idx] in "。？！～，":
#                         tmp_text = "".join(curr_text) + text[idx + 1 : future_idx + 1]
#                         tmp_text = tmp_text.lstrip("。？！～，")
#                         if tmp_text:
#                             chunks.append(tmp_text)
#                         curr_text = []
#                         idx = future_idx
#                         break
#                 else:
#                     # 如果未找到标点，强制切割
#                     tmp_text = "".join(curr_text[:max_length]).lstrip("。？！～，")
#                     if tmp_text:
#                         chunks.append(tmp_text)
#                     curr_text = curr_text[max_length:]

#     if curr_text:
#         tmp_text = "".join(curr_text).lstrip("。？！～，")
#         if tmp_text:
#             chunks.append(tmp_text)

#     return chunks


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

    def get_pinyin_list(text: str) -> List[str]:
        pinyins = []
        for ch in text:
            pinyin_list = pinyin(ch, heteronym=True, style=Style.FIRST_LETTER)
            if pinyin_list:
                pinyins.append(pinyin_list[0])
        return [item for sublist in pinyins for item in sublist]

    def calculate_similarity(pinyin1: str, pinyin2: str) -> float:
        return SequenceMatcher(None, pinyin1, pinyin2).ratio()

    gen_text_clean = clean_text(gen_text)
    text_clean = clean_text(text)

    if gen_text_clean == text_clean:
        return False, 100.0, gen_text_clean, text_clean

    gen_pinyin_str = get_pinyin_str(gen_text_clean)
    text_pinyin_str = get_pinyin_str(text_clean)

    gen_pinyin_list = get_pinyin_list(gen_text_clean)
    text_pinyin_list = get_pinyin_list(text_clean)

    sim_ratio = calculate_similarity(gen_pinyin_str, text_pinyin_str) * 100

    if "儿" in text:
        sim_ratio += 5

    has_omission_flag = True
    if len(gen_text_clean) != len(text_clean):
        length_diff = abs(len(gen_text_clean) - len(text_clean))
        sim_ratio -= length_diff * 5
        has_omission_flag = sim_ratio < 98
    else:
        mismatch = False
        for gen_p, text_p in zip(gen_pinyin_list, text_pinyin_list):
            if gen_p not in text_p:
                sim_ratio -= 5
                mismatch = True
        if not mismatch:
            sim_ratio = 100
        has_omission_flag = sim_ratio < 98

    return has_omission_flag, sim_ratio, gen_text_clean, text_clean


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
第1章 唯一的号码        
黎簇从沙漠里出来，身体一直没有完全恢复，还在接受持续治疗。他的神志完全清醒，已是他在北京医院醒来的第三天，他第一次完全想起了所有的事情。
背后的伤口奇迹般的成功结痂了，轻微的瘙痒让他很不舒服，这种感觉让一切细节开始回到他的脑子里。他想起了那只手机。那个黑瞎子，在给了他食物和水之后，和他说过，他必须活下去，他需要拨打一个电话，来告诉电话另一头的人所有事情的经过。
黎簇不敢说他是真正的刚刚想起来。经历了太阳下的暴晒，他所有的精力都用在了走路上。他有无数次回忆时就要想起这些细节，但是脑海中那刺目的毒日让他的记忆一想到沙漠就自动停止了。
即便他现在想起来，也没有马上拨打这个电话。他忽然想到，自己已经走出了这件事情，如果他不去回忆，这一切都会过去。唯独他背后的伤疤在时刻提醒他这些已经发生的事情。当时吴邪说过，带他去沙漠就是因为他背后的伤疤。
如果他拨打了这个电话，电话另一头的人决定去沙漠中救吴邪和黑瞎子的话，他们是不是也会来找他？ 如果他背后的伤疤真像吴邪认为的那么重要的话，电话另一头的人，肯定也会来找他。那么，事情还会再重复发生一遍。
不，他无法在经历一次了躺在床上，他身上所有的肌肉都麻木了。这棉质被子的质感，空调吹出的风所散发出的臭味和适宜的温度，还有四周人说话的声音，让他忽然意识到了文明的美好。
不能就这么简单的打这个电话。
"""
    texts = get_texts(text)

    for text_line in texts:
        print(text_line)
