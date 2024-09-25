import json
import os
from tools.auto_task_help import get_texts_v2, has_omission, format_line
import numpy as np
import pyloudnorm as pyln
import torch
from tqdm import tqdm
import soundfile as sf
import argparse
from GPT_SoVITS.inference import inference
from loguru import logger

logger.info(f"CUDA available: {torch.cuda.is_available()}")

# 用于存储相似度与音频的映射
pinyin_similarity_map = {}

rate = 32768


def normalize_loudness(audio, sample_rate):
    if isinstance(audio, torch.Tensor):
        audio = audio.numpy()
    if audio.ndim == 2:
        audio = audio.squeeze()
    meter = pyln.Meter(sample_rate)

    block_size = meter.block_size * sample_rate
    if len(audio) <= block_size:
        return torch.from_numpy(audio)

    loudness = meter.integrated_loudness(audio)
    normalized_audio = pyln.normalize.loudness(audio, loudness, -16.0)
    return torch.from_numpy(normalized_audio)


def prepare_audio(audio, sample_rate):
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)

    return normalize_loudness(audio.float(), sample_rate)  # 确保音频数据为浮点数


def process_text_line(index, texts, try_again, wav_list, spk):
    """
    处理单行文本
    :param index: 当前处理的文本行索引
    :param texts: 文本列表
    :param try_again: 重试次数
    :param wav_list: 音频列表
    :param spk: 说话人
    :return: 更新后的索引和重试次数
    """
    line = format_line(texts[index])
    global rate

    rate, output = inference(spk, line)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    is_continu, pinyin_similarity, gen_text_clean, text_clean = has_omission(
        output, line
    )
    logger.info(f"""
=========
原始文本：{line}
try_again: {try_again}
生成文本：{gen_text_clean}
输入文本：{text_clean}
相似度：{pinyin_similarity}
=========
    """)

    pinyin_similarity_map[pinyin_similarity] = output

    if is_continu:
        try_again += 1
        if try_again >= 5:
            best_similarity = max(pinyin_similarity_map.keys())
            best_audio = pinyin_similarity_map[best_similarity]
            pinyin_similarity_map.clear()
            try_again = 0
            index += 1
            # 将 best_audio 转换为 (1, length) 的 tensor 并连接到 wav_list
            best_audio_tensor = torch.from_numpy(best_audio).unsqueeze(0)
            wav_list.append(prepare_audio(best_audio_tensor, rate))
            # 生成停顿音频（tensor）并连接到 wav_list
            # silence_tensor = generate_silence(line)
            # wav_list.append(silence_tensor)
    else:
        pinyin_similarity_map.clear()
        try_again = 0
        index += 1
        # 将 output 转换为 (1, length) 的 tensor 并连接到 wav_list
        output_tensor = torch.from_numpy(output).unsqueeze(0)
        wav_list.append(prepare_audio(output_tensor, rate))
        # 生成停顿音频（tensor）并连接到 wav_list
        # silence_tensor = generate_silence(line)
        # wav_list.append(silence_tensor)

    return index, try_again


# def generate_silence(line):
#     """
#     生成停顿音频
#     :param line: 当前处理的文本行
#     :return: 停顿音频数据
#     """
#     random_duration = (
#         np.random.uniform(0.05, 0.08)
#         if line[-1] == "，"
#         else np.random.uniform(0.09, 0.13)
#     )
#     zero_wav = np.zeros(
#         int(rate * random_duration),
#         dtype=np.int16,
#     )
#     return torch.from_numpy(zero_wav).unsqueeze(0)


def process_chapter(book_name, idx):
    """
    处理章节
    :param book_name: 书名
    :param idx: 章节索引
    """
    file_path = f"./tmp/{book_name}/data/chapter_{idx}.txt"
    if not os.path.exists(file_path):
        logger.info(f"文件 {file_path} 不存在，跳过处理。")
        return

    with open(file_path, "r", encoding="utf-8") as file:
        file_content = file.read()
        # 获取第一行，文章标题
        title = file_content.split("\n")[0]

    texts = get_texts_v2(file_content)

    with open(f"./tmp/{book_name}/bookname2role.json", "r", encoding="utf-8") as f:
        bookname2role = json.load(f)

    with open(f"./tmp/{book_name}/role/chapter_{idx}.json", "r", encoding="utf-8") as f:
        line_map_list = json.load(f)
    line_map = {}
    for item in line_map_list:
        line_map.update(item)

    wav_list = []
    total_texts = len(texts)
    index = 0
    try_again = 0
    last_idx = index
    punctuation_marks = [
        "，",
        "。",
        "！",
        "？",
        "：",
        "；",
        "“",
        "”",
        "、",
        "（",
        "）",
        "《",
        "》",
        "……",
        "——",
        "‘",
        "’",
        '"',
        ".",
        ",",
        "!",
        "?",
    ]

    with tqdm(
        total=total_texts,
        desc=f"正在处理：{book_name}-{idx} - {title}",
        leave=True,  # 确保进度条在完成后不会被清除
    ) as pbar:
        while index < total_texts:
            line = texts[index].strip()
            if (
                line == "……"
                or line.strip() == ""
                or all(char in punctuation_marks for char in line)
            ):
                index += 1
                pbar.update(1)
                continue

            skp = "旁白1"
            content_to_process = []

            if "“" in line and "”" in line:
                # 获取所有 ”“ 之间的内容并按照顺序处理
                segments = line.split("“")
                for segment in segments:
                    if "”" in segment:
                        quote, rest = segment.split("”", 1)
                        content_to_process.append((quote, True))
                        if rest.strip():
                            content_to_process.append((rest.strip(), False))
                    else:
                        if segment.strip():
                            content_to_process.append((segment.strip(), False))
            else:
                content_to_process.append((line.strip(), False))

            for content, is_quote in content_to_process:
                if is_quote:
                    bookname = line_map.get(content, {}).get("role", "")
                    skp = bookname2role.get(bookname, "旁白1")
                index, try_again = process_text_line(
                    index, texts, try_again, wav_list, skp
                )
                if last_idx != index:
                    last_idx = index
                    pbar.update(1)

    # 检查 wav_list 是否为空
    if not wav_list:
        logger.info(f"章节 {idx} 没有生成有效音频，跳过保存。")
        return

    # 将wav_list中的音频数据连接成一个长的tensor，并确保其dtype为float32
    wav_list = torch.cat(wav_list, dim=-1).squeeze().numpy()  # 使用dim=-1进行连接

    output_path = os.path.join(f"./tmp/{book_name}/gen", f"{book_name}_{idx}.wav")
    sf.write(output_path, wav_list, rate, subtype="PCM_16")  # 设置为16位PCM格式

    logger.info(f"文件已保存到 {output_path}")

    process_file_path = f"./tmp/{book_name}/process.txt"
    with open(process_file_path, "a", encoding="utf-8") as process_file:
        process_file.write(f"Chapter {idx} processed and saved to {output_path}\n")


def main():
    """
    主函数
    """

    parser = argparse.ArgumentParser(
        description="Process book chapters to generate audio files."
    )
    parser.add_argument(
        "--book_name", type=str, help="Name of the book", default="诡秘之主"
    )
    parser.add_argument(
        "--start_idx", type=int, help="Starting chapter index", default=1
    )
    parser.add_argument("--end_idx", type=int, help="Ending chapter index", default=100)

    args = parser.parse_args()

    book_name = args.book_name
    start_idx = args.start_idx
    end_idx = args.end_idx

    os.makedirs(f"./tmp/{book_name}/gen", exist_ok=True)

    for idx in range(start_idx, end_idx + 1):
        process_chapter(book_name, idx)


if __name__ == "__main__":
    main()
