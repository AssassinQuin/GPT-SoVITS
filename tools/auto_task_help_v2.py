import sys
import os
import re
from pypinyin import pinyin, Style
from difflib import SequenceMatcher
from tools.zh_normalization.text_normlization import TextNormalizer

# 将项目根目录添加到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def remove_invalid_quotes(text):
    # 定义常见的中文标点符号
    chinese_punctuation = "。！？；：，、……"

    def replace_func(match):
        # 获取匹配到的文本
        quoted_text = match.group(1)
        # 检查最后一个字符是否为中文标点符号
        if quoted_text and quoted_text[-1] in chinese_punctuation:
            return f"“{quoted_text}”"  # 保留双引号
        else:
            return quoted_text  # 移除双引号

    # 使用正则表达式匹配中文双引号内的内容
    pattern = r"“(.*?)”"
    result = re.sub(pattern, replace_func, text)

    return result


def split_text(text, max_length=30):
    chunks = []
    curr_idx = 0
    curr_text = ""
    last_end_idx = -1

    while curr_idx < len(text):
        ch = text[curr_idx]
        if ch == "“":
            # 保存之前所有文本到chunks
            if curr_text:
                tmp_text = curr_text
                if tmp_text[0] in "。？！，":
                    tmp_text = tmp_text[1:]
                if tmp_text:
                    chunks.append(tmp_text)
            curr_text = ""
            tmp_text = ""
            while curr_idx < len(text):
                temp_ch = text[curr_idx]
                tmp_text += temp_ch
                if temp_ch == "”":
                    chunks.append(tmp_text)
                    last_end_idx = -1  # 清空 last_end_idx
                    break
                curr_idx += 1
            curr_idx += 1
            continue
        if ch in "。？！～，":
            last_end_idx = len(curr_text)
        curr_text += ch

        if len(curr_text) > max_length:
            if last_end_idx != -1:
                cut_point = last_end_idx + 1
                tmp_text = curr_text[:cut_point]
                if tmp_text[0] in "。？！～，":
                    tmp_text = tmp_text[1:]
                chunks.append(tmp_text)
                curr_text = curr_text[cut_point:]
                last_end_idx = -1  # 清空 last_end_idx 每次切分后
            else:
                # 找到超过阈值后的第一个标点符号
                found = False
                for i in range(curr_idx + 1, len(text)):
                    if text[i] in "。？！～，":
                        cut_point = i + 1
                        tmp_text = curr_text + text[curr_idx + 1 : cut_point]
                        if tmp_text[0] in "。？！～，":
                            tmp_text = tmp_text[1:]
                        chunks.append(tmp_text)
                        curr_text = ""
                        curr_idx = cut_point - 1  # 更新 curr_idx
                        found = True
                        break

                # 如果未找到标点符号，强制切割
                if not found:
                    tmp_text = curr_text[:max_length]
                    if tmp_text[0] in "。？！～，":
                        tmp_text = tmp_text[1:]
                    chunks.append(tmp_text)
                    curr_text = curr_text[max_length:]

        curr_idx += 1

    if curr_text:
        chunks.append(curr_text)

    return chunks


def get_texts(text, is_ignore_double_quotes=False):
    if is_ignore_double_quotes:
        text = text.replace("“", "")
        text = text.replace("”", "")
    text = text.replace("……", "。")
    text = text.replace("。。", "。")
    text = text.replace("、", "，")
    # 移除连续重复符号，只保留最后一个
    pattern = r"[，！？。～、]+"
    text = re.sub(pattern, lambda m: m.group(0)[-1], text)

    text = remove_invalid_quotes(text)
    texts = text.split("\n")
    # 清除空白符号
    texts = [text.strip() for text in texts if text.strip()]

    lines = []
    for t in texts:
        tmp = split_text(t)
        for line in tmp:
            # 文本格式化
            tx = TextNormalizer()
            sentences = tx.normalize(line)
            line = "".join(sentences)
            lines.append(line)

    return lines


def transcribe_and_clean(data_in, rate):
    from wav2text import only_asr

    # 调用模型的inference方法
    res = only_asr(data_in, rate)

    # 从结果中提取文本
    raw_text = res

    cleaned_text = re.sub(r"<\|.*?\|>", "", raw_text)

    # 去除空白字符
    cleaned_text = re.sub(r"\s+", "", cleaned_text)

    # 去除多余标点符号（只保留常用标点）
    cleaned_text = re.sub(r"[^\w\s，。！？]", "", cleaned_text)

    return cleaned_text


def has_omission(gen_data, text, rate):
    gen_text = transcribe_and_clean(gen_data, rate)
    tx = TextNormalizer()
    sentences = tx.normalize(gen_text)
    gen_text = "".join(sentences)
    return _has_omission(gen_text, text)


def _has_omission(gen_text, text):
    """
    检查生成的文本是否有遗漏原文的内容，通过拼音比较和相似度判断。
    :param gen_text: 生成的文本
    :param text: 原始文本
    :return: 若生成的文本拼音相似度超过98%且没有增加重复字，则返回 False（没有遗漏），否则返回 True（有遗漏）
    """

    def clean_text(text):
        """
        移除文本中的标点符号、空白符，并将英文全部转为小写。
        """
        # 移除标点符号
        text = re.sub(r"[^\w\s]", "", text)
        # 移除空白符
        text = re.sub(r"\s+", "", text)
        # 将英文全部转为小写
        text = text.lower()
        return text

    def get_pinyin(text):
        """
        获取文本的拼音表示。
        """
        return " ".join(["".join(p) for p in pinyin(text, style=Style.TONE2)])

    def get_pinyin_duo(text):
        """
        获取文本的拼音表示，支持多音字。
        """
        res = []
        for ch in text:
            res.extend(pinyin(ch, heteronym=True, style=Style.FIRST_LETTER))
        return res

    def calculate_similarity(pinyin1, pinyin2):
        """
        计算两个拼音字符串的相似度。
        """
        return SequenceMatcher(None, pinyin1, pinyin2).ratio()

    # 去除标点符号
    gen_text_clean = clean_text(gen_text)
    text_clean = clean_text(text)

    if gen_text_clean == text_clean:
        return False, 100, gen_text_clean, text_clean

    # 获取拼音
    gen_text_pinyin = get_pinyin(gen_text_clean)
    text_pinyin = get_pinyin(text_clean)

    gen_text_ping_duo = get_pinyin_duo(gen_text_clean)
    text_ping_duo = get_pinyin_duo(text_clean)

    # 计算拼音相似度
    sim_ratio = calculate_similarity(gen_text_pinyin, text_pinyin) * 100
    # 如果有“儿”字，则增加5分
    if "儿" in text:
        sim_ratio += 5

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
        is_multi_word = True
        for gen_word, text_word in zip(gen_text_ping_duo, text_ping_duo):
            if not any(g in text_word for g in gen_word):
                sim_ratio -= 5
                is_multi_word = False
        if is_multi_word:
            sim_ratio = 100
        res = sim_ratio < 98

    return res, sim_ratio, gen_text_clean, text_clean


def clear_text(text, ignore_punctuation=False):
    if ignore_punctuation:
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", "", text)
        text = text.lower()
    return "".join(TextNormalizer().normalize(text))
