import argparse
import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional

# 允许作为脚本执行时仍可导入上级目录下的包
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from auto_task_util.auto_task_help_v3 import (
    convert_txt_file_to_utf8,
    format_text,
    is_chapter_title,
    read_txt_file,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_novel_to_tmp(book_path: str) -> str:
    """
    第一步：输入小说路径，保存 UTF-8 文本到 tmp/{novel_name}/{novel_name}.txt
    - 自动按原编码读取并转为 UTF-8
    - 文件命名为小说名（去扩展名）.txt
    返回：保存后的目标路径
    """
    if not os.path.isfile(book_path):
        raise FileNotFoundError(f"文件不存在: {book_path}")

    novel_name = os.path.splitext(os.path.basename(book_path))[0]
    tmp_dir = os.path.join("tmp", novel_name)
    _ensure_dir(tmp_dir)

    logging.info(f"[1/3] 转码并保存到 {tmp_dir}")
    # 先转码到 tmp_dir，文件名保持原名
    out_path = convert_txt_file_to_utf8(book_path, output_dir=tmp_dir, overwrite=True)

    # 若转码后的文件名与目标名不同，则重命名为 {novel_name}.txt
    target_path = os.path.join(tmp_dir, f"{novel_name}.txt")
    if os.path.abspath(out_path) != os.path.abspath(target_path):
        if os.path.exists(target_path):
            os.remove(target_path)
        os.replace(out_path, target_path)

    return target_path


def format_tmp_text(novel_txt_path: str) -> None:
    """
    第二步：格式化 tmp/{novel_name}/{novel_name}.txt（就地覆盖写回）。
    """
    logging.info("[2/3] 格式化文本（清洗标点、移除噪声、合并短行等）")
    text = read_txt_file(novel_txt_path)
    formatted = format_text(text)
    with open(novel_txt_path, "w", encoding="utf-8") as fw:
        fw.write(formatted)


def _parse_title_num(title: str) -> Optional[int]:
    """从章节标题中解析数字编号，如“第0123章…”, “Chapter 10 …”。解析失败返回 None。"""
    patterns = [
        re.compile(r"^\s*第\s*([0-9]{1,8})\s*章", re.IGNORECASE),
        re.compile(r"\bChapter\s*([0-9]{1,8})\b", re.IGNORECASE),
    ]
    for p in patterns:
        m = p.search(title)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
    return None


def _validate_and_report(novel_name: str, manifest: List[Dict[str, Any]]) -> None:
    """对章节清单做一致性校验，输出更清晰的缺失章节信息，并生成简要报告。
    检测项：
    - 相邻可解析数字间的缺失范围（e.g., 100 -> 105 => 缺失 101-104）
    - 重复编号（相邻且数字相等）
    - 逆序编号（相邻且后者小于前者）
    - 无法解析数字的章节统计
    """
    # 构造序列数据
    seq = []
    for ch in manifest:
        idx = ch.get("index")
        title = ch.get("title", "")
        num = _parse_title_num(title)
        seq.append(
            {
                "index": idx,
                "num": num,
                "title": title,
                "path": ch.get("path"),
                "start_line": ch.get("start_line"),
                "end_line": ch.get("end_line"),
            }
        )

    unknown = [x for x in seq if x["num"] is None]
    # 相邻检测
    gaps = []  # {from_num, to_num, missing: [..], left_idx, right_idx, left_title, right_title}
    duplicates = []  # 相邻重复
    decreases = []  # 相邻逆序
    prev = None
    for cur in seq:
        if prev is not None and prev["num"] is not None and cur["num"] is not None:
            a, b = prev["num"], cur["num"]
            if b > a + 1:
                missing = list(range(a + 1, b))
                gaps.append(
                    {
                        "from_num": a,
                        "to_num": b,
                        "missing": missing,
                        "left_idx": prev["index"],
                        "right_idx": cur["index"],
                        "left_title": prev["title"],
                        "right_title": cur["title"],
                    }
                )
            elif b == a:
                duplicates.append(
                    {
                        "at_left_idx": prev["index"],
                        "at_right_idx": cur["index"],
                        "num": a,
                        "left_title": prev["title"],
                        "right_title": cur["title"],
                    }
                )
            elif b < a:
                decreases.append(
                    {
                        "left_idx": prev["index"],
                        "right_idx": cur["index"],
                        "left_num": a,
                        "right_num": b,
                        "left_title": prev["title"],
                        "right_title": cur["title"],
                    }
                )
        prev = cur

    # 日志输出（清晰可读）
    if gaps:
        logging.warning("检测到缺失章节区间: 共 %d 处" % len(gaps))
        for i, g in enumerate(gaps[:50], 1):
            miss_preview = ", ".join(map(str, g["missing"][:10])) + (
                " …" if len(g["missing"]) > 10 else ""
            )
            logging.warning(
                f"  Gap#{i}: 在 index {g['left_idx']} → {g['right_idx']} 之间，编号 {g['from_num']} → {g['to_num']}，缺失 {len(g['missing'])} 章: {miss_preview}"
            )
            logging.info(f"    左侧章节: [{g['left_idx']}] {g['left_title'][:80]}")
            logging.info(f"    右侧章节: [{g['right_idx']}] {g['right_title'][:80]}")
    else:
        logging.info("未检测到相邻编号上的缺失区间")

    if duplicates:
        logging.warning("检测到相邻重复编号: 共 %d 处" % len(duplicates))
        for d in duplicates[:50]:
            logging.warning(
                f"  重复: index {d['at_left_idx']} → {d['at_right_idx']}，编号均为 {d['num']}"
            )
    if decreases:
        logging.warning("检测到相邻逆序编号: 共 %d 处" % len(decreases))
        for d in decreases[:50]:
            logging.warning(
                f"  逆序: index {d['left_idx']}({d['left_num']}) → {d['right_idx']}({d['right_num']})"
            )

    if unknown:
        logging.info("无法从标题解析编号的章节: %d 条，示例：" % len(unknown))
        for u in unknown[:10]:
            logging.info(f"  index={u['index']}, title={u['title'][:100]}")

    # 简要报告 JSON
    report = {
        "novel": novel_name,
        "total_chapters": len(seq),
        "unknown_count": len(unknown),
        "gaps_count": len(gaps),
        "duplicates_count": len(duplicates),
        "decreases_count": len(decreases),
        "gaps": gaps,
        "duplicates": duplicates,
        "decreases": decreases,
    }
    report_path = os.path.join("tmp", novel_name, "chapters_report.json")
    with open(report_path, "w", encoding="utf-8") as rf:
        json.dump(report, rf, ensure_ascii=False, indent=2)
    logging.info(f"已写出校验报告: {report_path}")


def split_chapters(novel_name: str, novel_txt_path: str) -> List[str]:
    """
    第三步：根据章节切分文本，输出到 tmp/{novel_name}/data/chapter_*.txt
    同时生成章节清单 tmp/{novel_name}/chapters.json
    返回：输出的章节文件路径列表
    """
    logging.info("[3/3] 按章节切分文本")

    with open(novel_txt_path, "r", encoding="utf-8") as fr:
        formatted = fr.read()

    lines = [ln.strip() for ln in formatted.splitlines() if ln.strip()]
    chapter_starts = [i for i, ln in enumerate(lines) if is_chapter_title(ln)]

    data_dir = os.path.join("tmp", novel_name, "data")
    _ensure_dir(data_dir)

    outputs: List[str] = []
    manifest = []

    if not chapter_starts:
        # 未识别到章节标题，整体保存为一章
        out_path = os.path.join(data_dir, "chapter_1.txt")
        with open(out_path, "w", encoding="utf-8") as fw:
            fw.write("\n".join(lines))
        outputs.append(out_path)
        manifest.append(
            {
                "index": 1,
                "title": lines[0] if lines else "",
                "path": out_path,
                "start_line": 0,
                "end_line": len(lines),
            }
        )
        # 写出清单
        manifest_path = os.path.join("tmp", novel_name, "chapters.json")
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(
                {"novel": novel_name, "chapters": manifest},
                mf,
                ensure_ascii=False,
                indent=2,
            )
        logging.warning(
            "未识别到章节标题，已将全文保存为 chapter_1.txt，并生成章节清单 chapters.json"
        )
        # 生成空的报告
        _validate_and_report(novel_name, manifest)
        return outputs

    # 按章节标题索引切分
    for idx, start in enumerate(chapter_starts):
        end = chapter_starts[idx + 1] if idx + 1 < len(chapter_starts) else len(lines)
        chunk_lines = lines[start:end]
        if not chunk_lines:
            continue
        out_path = os.path.join(data_dir, f"chapter_{idx + 1}.txt")
        with open(out_path, "w", encoding="utf-8") as fw:
            fw.write("\n".join(chunk_lines))
        outputs.append(out_path)
        manifest.append(
            {
                "index": idx + 1,
                "title": chunk_lines[0],
                "path": out_path,
                "start_line": start,
                "end_line": end,
            }
        )
        logging.info(f"保存章节 {idx + 1}: {chunk_lines[0][:60]}")

    # 写出章节清单
    manifest_path = os.path.join("tmp", novel_name, "chapters.json")
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(
            {"novel": novel_name, "chapters": manifest},
            mf,
            ensure_ascii=False,
            indent=2,
        )
    logging.info(f"已写出章节清单: {manifest_path}（共 {len(manifest)} 章）")

    # 校验并输出更清楚的缺失章节信息
    _validate_and_report(novel_name, manifest)

    return outputs


def process(book_path: str) -> None:
    novel_path = save_novel_to_tmp(book_path)
    novel_name = os.path.splitext(os.path.basename(novel_path))[0]
    format_tmp_text(novel_path)
    split_chapters(novel_name, novel_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="小说 txt 拆分（v2）")
    parser.add_argument("book_path", type=str, help="小说绝对路径或相对路径")
    args = parser.parse_args()

    try:
        process(args.book_path)
        logging.info("处理完成")
    except Exception as e:
        logging.exception(f"处理失败: {e}")
