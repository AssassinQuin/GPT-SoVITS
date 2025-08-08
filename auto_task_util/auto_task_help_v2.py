#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import re
from typing import List, Any, Tuple
from pypinyin import pinyin, Style
from difflib import SequenceMatcher

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auto_task_util.zh_normalization.text_normlization import TextNormalizer

CHINESE_PUNCTUATION = "。！？；：，、……"
QUOTE_PATTERN = re.compile(r"“(.*?)”")
REPEAT_PUNCTUATION_PATTERN = re.compile(r"[，！？。～、]+")
NON_WORD_PATTERN = re.compile(r"[^\w\s，。！？]")


def replace_invalid_quotes(match: re.Match) -> str:
    quoted = match.group(1)
    return f"“{quoted}”" if quoted and quoted[-1] in CHINESE_PUNCTUATION else quoted


def remove_invalid_quotes(text: str) -> str:
    return QUOTE_PATTERN.sub(replace_invalid_quotes, text)


def split_non_quote_text(
    text: str, max_length: int, strong_punc: str, other_punc: str
) -> List[str]:
    chunks = []
    start, length = 0, len(text)
    while start < length:
        if length - start <= max_length:
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break
        end = min(start + max_length, length)
        last_punc_idx = -1
        for i in range(end - 1, start - 1, -1):
            if text[i] in strong_punc + other_punc:
                last_punc_idx = i
                break
        if last_punc_idx != -1:
            split_pos = last_punc_idx + 1
            chunk = text[start:split_pos].strip()
            if chunk:
                chunks.append(chunk)
            start = split_pos
        else:
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
                remaining = text[start:].strip()
                if remaining:
                    chunks.append(remaining)
                break
    return chunks


def is_english(char: str) -> bool:
    return "A" <= char <= "Z" or "a" <= char <= "z"


def is_chinese(char: str) -> bool:
    # 扩展判断：除汉字外，若字符为中文全角括号也视为中文
    return ("\u4e00" <= char <= "\u9fff") or char in "（）"


def classify_text(text: str) -> str:
    """
    根据内容判断文本类型。
      - 仅含中文：返回 "中文"
      - 仅含英文：返回 "英文"
      - 同时含有中文和英文：返回 "中英混合"
      - 含其他字符：返回 "多语种混合"
    """
    # 去除所有标点和空白字符
    cleaned_text = re.sub(r"[^\w\s]", "", text)
    has_chinese = has_english = has_other = False
    for char in cleaned_text:
        if char.isdigit() or char.isspace():
            continue
        if "\u4e00" <= char <= "\u9fff":
            has_chinese = True
        elif "a" <= char <= "z" or "A" <= char <= "Z":
            has_english = True
        else:
            has_other = True
        if has_other:
            return "多语种混合"
        if has_chinese and has_english:
            return "中英混合"
    if has_chinese:
        return "中文"
    if has_english:
        return "英文"
    return "中文"


def split_text(text: str, max_length: int = 100) -> List[Tuple[str, str]]:
    """
    采用双引号和英文短语匹配拆分文本，但如果匹配到的英文短语位于双引号范围内则不单独拆分，
    保证同一部分内容仅返回一个分块。
    """
    chunks = []
    length = len(text)
    quote_pat = re.compile(r"“[^”]*”")
    eng_pat = re.compile(r"\b[A-Za-z]{2,}(?:\s+[A-Za-z]{2,})*\b")
    strong_punc, other_punc = "。！？", "，；：、"

    quote_matches = list(quote_pat.finditer(text))
    eng_matches = []
    for m in eng_pat.finditer(text):
        if any(m.start() >= q.start() and m.end() <= q.end() for q in quote_matches):
            continue
        eng_matches.append(m)
    all_matches = quote_matches + eng_matches
    all_matches.sort(key=lambda m: m.start())
    last_idx = 0
    for match in all_matches:
        start, end = match.span()
        if start > last_idx:
            non_match = text[last_idx:start]
            for seg in split_non_quote_text(
                non_match, max_length, strong_punc, other_punc
            ):
                chunks.append((seg, classify_text(seg)))
        mtext = match.group()
        if quote_pat.fullmatch(mtext):
            chunks.append((mtext, classify_text(mtext)))
        elif eng_pat.fullmatch(mtext):
            chunks.append((mtext, classify_text(mtext)))
        last_idx = end
    if last_idx < length:
        remaining = text[last_idx:]
        for seg in split_non_quote_text(remaining, max_length, strong_punc, other_punc):
            chunks.append((seg, classify_text(seg)))
    return chunks


def get_texts(text: str, ignore_punctuation: bool = False) -> List[Tuple[str, str]]:
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
    lines_raw = [line.strip() for line in text.split("\n") if line.strip()]
    result = []
    normalizer = TextNormalizer()
    for t in lines_raw:
        for chunk, lang in split_text(t):
            if lang == "中文":
                normalized = "".join(normalizer.normalize(chunk))
                if normalized:
                    result.append((normalized, lang))
            else:
                result.append((chunk, lang))
    return result


def transcribe_and_clean(data_in: Any, rate: int) -> str:
    from wav2text import only_asr

    raw = only_asr(data_in, rate)
    cleaned = re.sub(r"<\|.*?\|>", "", raw)
    cleaned = re.sub(r"\s+", "", cleaned)
    return NON_WORD_PATTERN.sub("", cleaned)


def _has_omission(gen_text: str, text: str) -> Tuple[bool, float, str, str]:
    if not gen_text:
        return True, 0.0, "", text

    def clean(txt: str) -> str:
        txt = re.sub(r"[^\w\s]", "", txt)
        return re.sub(r"\s+", "", txt).lower()

    def get_pinyin(txt: str) -> str:
        return " ".join(item[0] for item in pinyin(txt, style=Style.TONE2))

    def has_pinyin_intersection(a: str, b: str) -> bool:
        set_a = {
            item[0]
            for sub in pinyin(hans=a, heteronym=True, style=Style.TONE2)
            for item in sub
        }
        set_b = {
            item[0]
            for sub in pinyin(hans=b, heteronym=True, style=Style.TONE2)
            for item in sub
        }
        return bool(set_a & set_b)

    def calc_sim(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    gen_clean = clean(gen_text)
    text_clean = clean(text)
    if gen_clean == text_clean:
        return False, 100.0, gen_clean, text_clean

    gen_pinyin = get_pinyin(gen_clean)
    text_pinyin = get_pinyin(text_clean)
    weight = 100 / len(text_clean) / 2
    sim_ratio = calc_sim(gen_pinyin, text_pinyin) * 100

    if len(gen_clean) != len(text_clean):
        diff = abs(len(gen_clean) - len(text_clean))
        sim_ratio -= diff * weight
        need_repeat = sim_ratio < max(95, 1 - weight * 2)
    else:
        mismatch = False
        for i in range(len(text_clean)):
            if text_clean[i] != gen_clean[i] and not has_pinyin_intersection(
                text_clean[i], gen_clean[i]
            ):
                sim_ratio -= weight
                mismatch = True
        if not mismatch:
            sim_ratio = 100.0
        need_repeat = sim_ratio < max(95, 1 - weight * 2)
    return need_repeat, sim_ratio, gen_clean, text_clean


def has_omission(
    gen_data: Any, text: str, rate: int, lang: str
) -> Tuple[bool, float, str, str]:
    gen_text = transcribe_and_clean(gen_data, rate)
    if lang in ("中英混合", "多语种混合"):
        gen_text = "".join(TextNormalizer().normalize(gen_text))
    return _has_omission(gen_text, text)


def clear_text(text: str, ignore_punctuation: bool = False) -> str:
    if ignore_punctuation:
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", "", text).lower()
    return "".join(TextNormalizer().normalize(text))


if __name__ == "__main__":
    sample_text = """
    跟他搭话的是，同为英文老师的高冢阳二老师。因为体型太胖，被酒井副校长命令减肥，却一直不见他瘦下去。另外学生给他起的外号叫重金属（日文是ヘビメタ，和Heavy Metal的日语简称一样。），本人似乎是以为学生记得他之前讲过自己以前玩摇滚的事，听着非常受用，其实只是高代谢（日文是ヘビー？メタボリック，Heavy Metabolic，简称和上文的Heavy Metal的一样。）的略称而已。
“啊，我班上的学生接二连三地出问题呢。”
莲实开口说道。

迈络思于1999年在以色列设立，在美国纳斯达克证券交易所上市，主要从事网络互联产品的研发、生产和销售。英伟达于1998年在美国设立，在美国纳斯达克证券交易所上市，主要从事图形处理器的研发、生产和销售。

北京时间12月9日晚间，英伟达盘前下跌2.20%，股价报139.31美元/股。
    机器人马上有反应了，“Kinesiskapaketetnedladdning……10%…50%……70%……中文数据包加载完成，”声音平和但又机械。
    说着，机器人脑袋的屏幕上直接弹出一个颜文字来。（^ω^）
    """
    texts = get_texts(sample_text)
    for line, lang in texts:
        print(f"{line}  —— [{lang}]")
