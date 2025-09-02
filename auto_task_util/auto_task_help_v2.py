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


def split_text(text: str, max_length: int = 100, min_length: int = 20) -> List[Tuple[str, str]]:
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
第72章 基地参观
　　闪光城，酒馆后院。
　　呼喝声此起彼伏，一片吵杂。
　　“手脚麻利点，把东西都卸下来。”
　　“都小心点，摔坏东西我扣你薪水！”
　　“把标记好的东西单独挑出来，等会我亲自要送往城外。”
　　穆卓大声指挥著下属，时不时喝骂一句。
　　原本以他的身份，根本不需要亲自跟随车队送货过来。
　　不过为了表示对闪光领的重视，他每一回都会亲自带队。
　　想起当初在酒馆和苏南初次见面的情形，穆卓就忍不住一阵唏嘘。
　　当初虽然看出苏南不是池中之物，未来前途肯定远大，却没想到对方的表现会这么惊人和亮眼。
　　身为荆花公国有数的大商人，他自有自己的情报渠道，深知苏南在这次夺回闪光领的战争中发挥著多么巨大的作用。
　　可以说，要是没有苏南，黑石城别说逆袭了，早就在那次刺杀事件中就分崩离析了！苏南几乎是以一己之力逆转了双方的强弱对比！
　　穆卓虽然听说过不少关于巫师学徒的隐秘，却还是第一次了解到原来巫师学徒这么厉害。
　　对这种前途远大的强者，他自然要好好结交。
　　更别说苏南还和商会有合作，那些药剂可是给商会带来了十分可观的利润。
　　卸好货物后，穆卓将其他事宜交给副手，自己则是亲自带著人和货物赶往城外的基地。
　　奥塔寸步不离的跟在穆卓身边，马车也是同坐一辆。
　　上了马车，放下门帘，穆卓随口问道：“染霜行省那边还没消息吗？”
　　奥塔脸色沉重的摇了摇头，说道：“没有，我寄了好几封信，可那边的族人一直都没回复。”
　　“奇了怪了，染霜行省的兽化人生活得好好的，怎么会去主动散播兽化症？”穆卓摩挲著下巴沉吟道。
　　奥塔闷声道：“应该不是他们，只有原始兽化人才能传染兽化症，我们这些后裔做不到，我觉得里面应该另有隐情。”
　　穆卓不予置否，想了想说道：“这次送完货，你去染霜行省走一趟，调查一下是怎么回事，要是情况太危急的话，商会就得考虑收缩在染霜行省的产业了。”
　　“我知道了。”奥塔肃然点头。
　　两人随后不再交谈，车厢里很快陷入静谧。
　　就在这时，外面突然传来一阵喧哗。
　　穆卓好奇的掀开门帘向外看去，发现三头浑身漆黑的豹子从大街上奔跑而过，引得路人纷纷注目。
　　黑岩豹！穆卓立刻认出了黑豹的来历。
　　自从黑石城一战后，黑岩骑兵团的威名就传遍了整个金岩行省。
　　许多人都知道这个骑兵团拥有一种名为黑岩豹的强大坐骑，每一头都有骑士级战力，且悍不畏死，比最优秀的战马都要出色。
　　黑岩骑兵团的威名，大半都来自于黑岩豹。
　　不少贵族领主对此十分眼热。
　　穆卓曾经试探过克伊的口风，想要购买黑岩豹，可惜被干脆利落的拒绝了。
　　不过这会让穆卓惊讶的不是这个，而是骑在黑岩豹背上的三个小孩子。
　　两男一女，都是十三四岁的年龄。
　　“这三人是什么来路？”穆卓暗暗疑惑。
　　这三人的年龄显然不可能是黑岩骑兵团的人，却能用黑岩豹当坐骑，在闪光城的地位绝对不低。
　　蓦地，穆卓忽然想起一件事来。
　　听说苏南年初收了三个学徒，不会就是这三个小家伙吧？
　　呼！劲风在耳边呼啸不停，两侧的景色飞快后掠。
　　谢曼满脸兴奋的骑在黑岩豹背上，体会著人生第一次的策豹狂奔，只觉浑身血液都在沸腾。
　　另一边的科雷同样神色难掩兴奋，双眼发亮。
　　唯独阿蒂尔小脸有些发白，不太适应这么快的速度。没多久，三人就抵达基地所在之处。
　　“这里就是老师所说的基地？”
　　谢曼好奇的打量面前高耸的石墙和紧闭的大门。
　　仔细观察的话，可以发现石墙表面有浅浅的奇异纹路。
　　他依稀在书上看到过这种纹路的图案，貌似是一种强化物体硬度的附魔符纹。
　　轰隆隆！紧闭的大门忽然打开，紧跟著一个熟悉的身影从里面冲了出来，瞬移般落在三人面前。
　　正是艾米。
　　“喵，你们终于来了，我等伱们很久了。”
　　艾米朝三人招了招手。
　　“快进来吧，苏南正在忙著，让我带你们先在基地里参观一下，熟悉熟悉。”
　　在艾米的带领下，谢曼三人有些拘谨的踏进基地。
　　走进大门后，他们才发现门后站著两头巨大的岩石生物。
　　“石魔像！”阿蒂尔低呼一声。
　　她在书上看到过石魔像的图案，但还是第一次亲眼见到实物。
　　谢曼和科雷也瞪大双眼打量面前的石魔像，眼中满是新奇。
　　“这是基地的守卫喵。”艾米解释了一句。
　　正说著，又有一队黏土魔像从远处走来，似乎要出基地。
　　艾米向后看了一眼，说道：“有商队送东西过来了，它们要去搬运，走吧，我们继续。”
　　说著往前蹦蹦跳跳的走去，谢曼三人连忙跟上。
　　艾米十分尽职，每到一处就向三人讲解这个地方的用处。
　　“这里是冥想区，你们以后可以在这里冥想，随便挑个空的冥想室就行。”
　　“这里是法术修炼区，看到那些人型标靶没有，等你们学习了法术，就可以到这里练习。”
　　“这里是实验区，做实验的地方。”
　　“这里是居住区，你们以后的住处。”
　　这里是制药区.
　　基地的面积大得夸张。
　　一路走来，各种各样的设施看得三个小家伙眼花缭乱。
　　不知不觉间，一行人来到傀儡工厂。
　　谢曼第一时间注意到站在角落的钢铁庞然大物，忍不住瞪圆了双眼。
　　“那是什么？”
　　艾米看了一眼，说道：“那是钢铁魔像。”
　　“原来这就是钢铁魔像。”
　　谢曼三人看过的书籍上只描述了低阶魔像，对钢铁魔像只提了一笔，没有详细说明，这会看到实物，只觉得满心震撼。
　　这也太大了！“艾米大人，钢铁魔像比石魔像厉害很多吗？”科雷好奇问道。
　　“那是当然，钢铁魔像的战斗力可是传奇骑士级别的。”
　　听到艾米的回答，三个小家伙齐齐倒吸口气，露出震惊无比的目光。
　　这个大块头居然和传奇骑士一样强大！三人忍不住咽了口唾沫，看向钢铁魔像的目光顿时充满了敬畏。
　　连带著对苏南的敬仰又加深了几分。
　　能制造出钢铁魔像的老师，又该有多么厉害？
    """
    texts = get_texts(sample_text)
    for line, lang in texts:
        print(f"{line}  —— [{lang}]")
