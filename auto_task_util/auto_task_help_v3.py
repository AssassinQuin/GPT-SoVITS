import os
import re

from chardet import detect

# from pkg_resources import split_sections  # removed unused import to avoid confusion

# Robust import for TextNormalizer whether running as a module or a script
try:
    from auto_task_util.zh_normalization.text_normlization import TextNormalizer
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from auto_task_util.zh_normalization.text_normlization import TextNormalizer


def format_text(text):
    # 1. 清除所有特殊字符、保留中文、英文、日文、数字、半角标点
    # 2. 将全角/中文标点符号替换为半角标点符号
    # 先将全角/中文标点统一为半角，避免在清理阶段被删除
    text = replace_punctuation(text)
    # 2. 清除所有特殊字符
    text = clean_text(text)
    # 3. 合并过短行：若一行字符串数据长度小于 10，则与上一行或下一行中"数据长度更小"的一行合并
    text = merge_short_lines(text, min_len=20)
    return text


# 章节标题识别函数，供多处复用
def is_chapter_title(s: str) -> bool:
    """
    识别常见的章节标题行（中英文）。
    规则覆盖：
    - 中文：第X章；常见标题词：序章/序言/楔子/前言/引子/后记/尾声/终章/上篇/下篇/中篇
    - 英文：Chapter/Chap./Ch. + 数字或罗马数字；Prologue/Epilogue
    """
    ss = s.strip()
    if not ss:
        return False
    # 中文常见章节：仅识别"第X章"，且需在行首，避免误识别
    if re.match(r"^第[零〇两一二三四五六七八九十百千0-9０-９]{1,8}章", ss):
        return True
    # 常见中文标题词：序章/序言/楔子/前言/引子/后记/尾声/终章/上篇/下篇/中篇
    if re.match(r"^(序章|序言|楔子|前言|引子|后记|尾声|终章|上篇|下篇|中篇)", ss):
        return True
    # 英文章节：Chapter/Chap./Ch. + 数字或罗马数字；Prologue/Epilogue
    if re.match(r"^(chapter|chap\.|ch\.)\s*(\d+|[ivxlcdm]+)\b", ss, flags=re.I):
        return True
    if re.match(r"^(prologue|epilogue)\b", ss, flags=re.I):
        return True
    return False


def clean_text(text: str) -> str:
    """
    清除所有特殊字符，仅保留：
    - 中文（含扩展区）
    - 日文（平假名、片假名、片假名扩展、半角片假名）
    - 英文（A-Z a-z）
    - 数字（0-9）
    - 半角标点（ASCII）
    - 空白符（空格、制表、换行等）
    - 若某行存在不成对的引号（" 或 '），删除那个没有成对的引号
    - 若某行仅由标点（及空白）构成，则删除该行的全部符号（该行变为空行）
    - 每行首尾的空白符将被清理
    - 删除空白行
    - 若行末无标点符号，则在末尾补一个句号

    Args:
        text: 输入文本

    Returns:
        清理后的文本
    """
    # 允许字符：中文/日文/英文/数字/空白 + ASCII 半角标点
    ascii_punct = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    ascii_punct_escaped = re.escape(ascii_punct)
    pattern = rf"[^\u4e00-\u9fff\u3400-\u4dbf\u3040-\u309f\u30a0-\u30ff\u31f0-\u31ff\uff65-\uff9fA-Za-z0-9\s{ascii_punct_escaped}]"
    cleaned = re.sub(pattern, "", text)

    # 行级处理辅助：移除不成对的引号
    def remove_unpaired_quotes(s: str) -> str:
        if not s:
            return s
        to_del = set()
        # 处理双引号 "
        in_double = False
        stack_double = []
        for i, ch in enumerate(s):
            if ch == '"':
                if not in_double:
                    in_double = True
                    stack_double.append(i)
                else:
                    in_double = False
                    stack_double.pop()
        if in_double and stack_double:
            # 删除最后一个未配对的开引号
            to_del.add(stack_double.pop())

        # 处理单引号 '（避免删英文缩写中的撇号，如 don't）
        positions = []
        for i, ch in enumerate(s):
            if ch == "'":
                prev_c = s[i - 1] if i > 0 else ""
                next_c = s[i + 1] if i + 1 < len(s) else ""
                # 仅当两侧都是 ASCII 字母或数字时，视为英文撇号（如 don't、O'Neill、90's）
                prev_ascii = True if re.match(r"[A-Za-z0-9]", prev_c) else False
                next_ascii = True if re.match(r"[A-Za-z0-9]", next_c) else False
                if prev_ascii and next_ascii:
                    continue
                positions.append(i)
        in_single = False
        stack_single = []
        for idx in positions:
            if not in_single:
                in_single = True
                stack_single.append(idx)
            else:
                in_single = False
                stack_single.pop()
        if in_single and stack_single:
            to_del.add(stack_single.pop())

        if not to_del:
            return s
        return "".join(ch for i, ch in enumerate(s) if i not in to_del)

    # 删除仅由标点符号（及空白）组成的整行的符号（保留该行为空行）
    lines = cleaned.splitlines()
    filtered_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue  # 删除空白行
        # 移除不成对的引号
        stripped = remove_unpaired_quotes(stripped)
        if stripped and all((ch in ascii_punct) for ch in stripped):
            continue  # 删除仅由标点组成的行
        if stripped and stripped[-1] not in ascii_punct:
            stripped += "."  # 若末尾无标点，补句号
        filtered_lines.append(stripped)
    cleaned = "\n".join(filtered_lines)
    return cleaned


# 从文本中提取被识别为章节标题的段落
def extract_chapter_paragraphs(text: str, return_indices: bool = True):
    """
    提取文本中被识别为章节标题的行（段落）。

    参数：
        text: 输入的原始或已清洗文本（按行组织）
        return_indices: 是否返回行索引（基于去空行后的 0-based 索引）
    返回：
        - 若 return_indices 为 True: List[Tuple[int, str]] = [(idx, line), ...]
        - 否则: List[str] = [line, ...]
    """
    if not text:
        return []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    results = []
    for i, ln in enumerate(lines):
        if is_chapter_title(ln):
            results.append((i, ln) if return_indices else ln)
    return results


def replace_punctuation(text: str) -> str:
    """
    将全角/中文标点统一替换为半角 ASCII 标点，并规范连续标点：
    - ， 。 ！ ？ ； ： -> , . ! ? ; :
    - （）［］【】｛｝《》 -> ()[][]<>{}
    - 全角连接符等：— – － ～ -> - - - ~
    - 斜杠、竖线、等号等全角形式 -> 对应半角
    - 中文引号 “ ” ‘ ’ 以及日文引号 「 」 『 』 -> 对应半角 " 和 '
    - 省略号（…、……、...）-> "."
    - 连续引号只保留第一个字符，其他连续标点不合并
    """
    if not text:
        return text

    # 先将中文省略号替换为句号
    text = re.sub(r"…+", ".", text)
    text = re.sub(r"\.{3,}", ".", text)

    # 全角/中文 -> 半角 映射（1:1 字符）
    cn_to_ascii = {
        "，": ",",
        "。": ".",
        "．": ".",
        "！": "!",
        "？": "?",
        "；": ";",
        "：": ":",
        "（": "(",
        "）": ")",
        "［": "[",
        "］": "]",
        "【": "[",
        "】": "]",
        "｛": "{",
        "｝": "}",
        "《": "<",
        "》": ">",
        "、": ",",
        "—": "-",
        "–": "-",
        "－": "-",
        "～": "~",
        "·": ".",
        "／": "/",
        "＼": "\\",
        "｜": "|",
        "＠": "@",
        "＃": "#",
        "＄": "$",
        "％": "%",
        "＾": "^",
        "＆": "&",
        "＋": "+",
        "＝": "=",
        "＊": "*",
        "＿": "_",
        "＜": "<",
        "＞": ">",
        # 引号相关
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "「": '"',
        "」": '"',
        "『": "'",
        "』": "'",
    }

    text = text.translate(str.maketrans(cn_to_ascii))

    # 仅合并连续引号（" 或 ' 的连续串），保留第一个引号；其他连续标点不处理
    text = re.sub(r'(["\'])(?:["\']+)', r"\1", text)

    return text


def merge_short_lines(text: str, min_len: int = 10) -> str:
    """
    合并过短的行：
    - 定义“字符串数据长度”为去除空白后的字符数
    - 遍历所有行，当某行数据长度 < min_len 时，与“上一行或下一行中数据长度更小的那一行”合并为一行
      • 若同时存在上一行与下一行且上一行数据长度 <= 下一行，则合并到上一行（保持顺序：上一行 + 空格 + 当前行）
      • 否则合并到下一行（当前行 + 空格 + 下一行），并跳过下一行
      • 边界情况：若只有一侧存在，则与存在的一侧合并；若两侧都不存在则保留
    - 合并是迭代进行的，直到不再发生合并为止
    - 跳过“章节标题”行（如“第X章/第X节/Chapter X/序章/尾声”等），这些行不与相邻行合并，且其他行也不会与其合并

    注意：
    - 输入通常已由 clean_text 处理过：无空白行、每行已去首尾空白、末尾有标点
    - 合并时仅以空格连接，不额外更改标点
    """
    if not text:
        return text

    # 拆分为行（保持已有换行）
    lines = text.splitlines()
    # 保险：去掉纯空白行
    lines = [ln.strip() for ln in lines if ln.strip()]

    # 章节标题识别：遇到章节标题行不参与合并（自身不与其他行合并，他人也不与其合并）
    # 使用全局 is_chapter_title()

    def data_len(s: str) -> int:
        return len(re.sub(r"\s+", "", s))

    changed = True
    while changed:
        changed = False
        new_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]

            # 章节标题直接保留，跳过合并
            if is_chapter_title(line):
                new_lines.append(line)
                i += 1
                continue

            if data_len(line) >= min_len or len(lines) == 1:
                new_lines.append(line)
                i += 1
                continue

            # 需要合并的过短行
            prev_exists = len(new_lines) > 0
            next_exists = (i + 1) < len(lines)
            prev_len = data_len(new_lines[-1]) if prev_exists else None
            next_len = data_len(lines[i + 1]) if next_exists else None
            prev_is_title = is_chapter_title(new_lines[-1]) if prev_exists else False
            next_is_title = is_chapter_title(lines[i + 1]) if next_exists else False

            if (
                prev_exists
                and (not prev_is_title)
                and (
                    (not next_exists)
                    or next_is_title
                    or (
                        prev_len is not None
                        and next_len is not None
                        and prev_len <= next_len
                    )
                )
            ):
                # 合并到上一行（上一行不能是章节标题）
                new_lines[-1] = f"{new_lines[-1]} {line}".strip()
                changed = True
                i += 1
            elif next_exists and (not next_is_title):
                # 与下一行合并（下一行不能是章节标题）
                combined = f"{line} {lines[i + 1]}".strip()
                new_lines.append(combined)
                changed = True
                i += 2
            else:
                # 无可合并对象，或两侧为章节标题，保留
                new_lines.append(line)
                i += 1

        lines = new_lines

    return "\n".join(lines)


def convert_txt_file_to_utf8(
    input_file: str, output_dir: str = "tmp/tmp_txt", overwrite: bool = True
) -> str:
    """
    将单个 .txt 文件按原编码读取并以 UTF-8 写入到 output_dir，返回写出的文件路径。
    规则与 convert_txt_dir_to_utf8 保持一致。

    参数：
        input_file: 输入 .txt 文件路径
        output_dir: 输出目录，默认 tmp/tmp_txt
        overwrite: 是否覆盖已存在文件
    返回：
        写出的 UTF-8 文件完整路径
    """
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"输入文件不存在或不是文件: {input_file}")
    if not input_file.lower().endswith(".txt"):
        raise ValueError(f"仅支持 .txt 文件: {input_file}")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, os.path.basename(input_file))
    if (not overwrite) and os.path.exists(out_path):
        return out_path

    with open(input_file, "rb") as frb:
        data = frb.read()

    text = ""
    if data:
        # UTF-8 BOM 处理
        if data.startswith(b"\xef\xbb\xbf"):
            try:
                text = data.decode("utf-8-sig", errors="ignore")
            except Exception:
                text = data.decode("utf-8", errors="ignore")
        else:
            enc_info = detect(data) or {}
            enc = (enc_info.get("encoding") or "").lower()
            # 统一中文编码到 gb18030 以提高兼容性
            if enc in {"gb2312", "gbk"}:
                enc = "gb18030"
            candidates = [enc, "utf-8", "gb18030", "big5", "shift_jis", "latin-1"]
            tried = set()
            for cand in candidates:
                if not cand or cand in tried:
                    continue
                tried.add(cand)
                try:
                    text = data.decode(cand, errors="ignore")
                    break
                except Exception:
                    continue
            if not text:
                # 最后兜底
                text = data.decode("utf-8", errors="ignore")

    with open(out_path, "w", encoding="utf-8") as fw:
        fw.write(text)

    return out_path


# 读取文件内容，返回 string
def read_txt_file(input_file: str) -> str:
    with open(input_file, "r", encoding="utf-8") as fr:
        text = fr.read()
    return text


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


def get_texts(text):
    normalizer = TextNormalizer()

    section_texts = []
    for seg in text.split("\n"):
        seg = seg.strip()
        if not seg:
            continue
        cls = classify_text(seg)
        if cls == "中文":
            seg = format_text(seg)
            seg = "".join(normalizer.normalize(seg))
        section_texts.append((seg, cls))

    return section_texts


def remove_punctuation(self, text: str) -> str:
    """
    去除文本中的标点符号。

    Args:
        text (str): 原始文本。

    Returns:
        str: 去除标点符号后的文本。
    """
    # 保留中英文、数字、日文，去除其他字符
    # 中文：\u4e00-\u9fff，英文：a-zA-Z，数字：0-9，日文：\u3040-\u309f\u30a0-\u30ff
    pattern = r"[^\u4e00-\u9fffa-zA-Z0-9\u3040-\u309f\u30a0-\u30ff]"
    return re.sub(pattern, "", text)


if __name__ == "__main__":
    text = """

"""
    # text = format_text(text)
    sections = get_texts(text)
    for idx, section in enumerate(sections):
        print(idx, section)
