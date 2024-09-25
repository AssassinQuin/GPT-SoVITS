import argparse
import json
import os
import time
import numpy as np
import torch
import torchaudio
import pyloudnorm as pyln
from tqdm import tqdm
from loguru import logger
from GPT_SoVITS.inference import inference
from tools.auto_task_help_v2 import (
    get_texts,
    has_omission,
    clear_text,
)


logger.info(f"CUDA available: {torch.cuda.is_available()}")


# 用于存储相似度与音频的映射
pinyin_similarity_map = {}
sample_rate = 22050
book_name = ""


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


def process_text(text, audio_list, speaker):
    global sample_rate
    texts = get_texts(text, True)
    for text in texts:
        try_again = 0
        line = text
        clear_line = clear_text(text, True)
        if clear_line == "":
            continue

        while True:
            sample_rate, output = inference(speaker, line)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            is_continu, pinyin_similarity, gen_text_clean, text_clean = has_omission(
                output, line, sample_rate
            )
            logger.info(f"""
=========
原始文本：{line}
try_again: {try_again}
speaker: {speaker}
生成文本：{gen_text_clean}
输入文本：{text_clean}
相似度：{pinyin_similarity}
=========
            """)
            pinyin_similarity_map[pinyin_similarity] = output

            if is_continu:
                try_again += 1
                if try_again >= 3:
                    best_similarity = max(pinyin_similarity_map.keys())
                    best_audio = pinyin_similarity_map[best_similarity]
                    pinyin_similarity_map.clear()
                    try_again = 0
                    best_audio_tensor = torch.from_numpy(best_audio).unsqueeze(0)
                    best_audio = prepare_audio(best_audio_tensor, sample_rate)

                    audio_list.append(best_audio)
                    audio_list.append(generate_silence(line))
                    break
            else:
                pinyin_similarity_map.clear()
                try_again = 0
                output_tensor = torch.from_numpy(output).unsqueeze(0)
                tts_speech = prepare_audio(output_tensor, sample_rate)
                if try_again == 0:
                    save_wav(speaker, tts_speech, line)
                audio_list.append(tts_speech)
                audio_list.append(generate_silence(line))
                break

    return audio_list


skip_spk = ["旁白1"]


def save_wav(spk, input_audio, line):
    # 获取 tmp/gen/{spk} 目录路径
    spk_dir = f"./tmp/gen/{spk}"

    # 检查目录是否存在
    if os.path.exists(spk_dir):
        # 读取目录下的所有文件
        wav_files = [f for f in os.listdir(spk_dir) if f.endswith(".wav")]

        # 如果文件数目超过 5000，则将 spk 添加到 skip_spk 列表中
        if len(wav_files) > 5000:
            skip_spk.append(spk)

    # 如果 spk 在 skip_spk 列表中，则跳过保存操作
    if spk in skip_spk:
        return

    timestamp = time.strftime("%Y%m%d%H%M%S")
    tmp_name = f"{book_name}_{spk}_{timestamp}"
    temp_wav_path = f"./tmp/gen/{spk}/{tmp_name}.wav"
    os.makedirs(os.path.dirname(temp_wav_path), exist_ok=True)
    torchaudio.save(temp_wav_path, input_audio.unsqueeze(0), sample_rate)

    with open(f"./tmp/gen/{spk}/{tmp_name}.normalized.txt", "w", encoding="utf-8") as f:
        f.write(line)


def generate_silence(line):
    random_duration = (
        np.random.uniform(0.05, 0.08)
        if line[-1] == "，"
        else np.random.uniform(0.09, 0.13)
    )
    silence_wav = np.zeros(int(sample_rate * random_duration), dtype=np.float32)
    return torch.from_numpy(silence_wav).unsqueeze(0)


def load_or_initialize_json(file_path, default_value):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = default_value
    return data


def process_chapter(book_name, chapter_index):
    file_path = f"./tmp/{book_name}/data/chapter_{chapter_index}.txt"
    if not os.path.exists(file_path):
        logger.info(f"文件 {file_path} 不存在，跳过处理。")
        return

    with open(file_path, "r", encoding="utf-8") as file:
        file_content = file.read()
        title = file_content.split("\n")[0]

    bookname2role_file = f"./tmp/{book_name}/bookname2role.json"
    bookname2role = load_or_initialize_json(bookname2role_file, {})

    line_map_file = f"./tmp/{book_name}/role/chapter_{chapter_index}.json"
    line_map_list = load_or_initialize_json(line_map_file, [])
    line_map = {}

    for item in line_map_list:
        # 清除文本中的所有标点符号
        cleaned_item = {}
        for key, value in item.items():
            cleaned_key = clear_text(key, True)
            cleaned_item[cleaned_key] = value
        line_map.update(cleaned_item)

    texts = []
    if bookname2role == {}:
        texts = get_texts(file_content, True)
    else:
        texts = get_texts(file_content)

    audio_list = []
    total_texts = len(texts)
    index = 0

    logger.info(f"total_texts: {total_texts}")

    with tqdm(
        total=total_texts,
        desc=f"正在处理：{book_name}-{chapter_index} - {title}",
        leave=True,
    ) as pbar:
        while index < total_texts:
            line = texts[index]
            tmp_line = clear_text(line, True)
            speaker = "旁白1"
            bookname = line_map.get(tmp_line, {}).get("role", "")
            speaker = bookname2role.get(bookname, "旁白1")

            audio_list = process_text(line, audio_list, speaker)

            index += 1
            pbar.update(1)

    audio_list = [wav if wav.ndim == 2 else wav.unsqueeze(0) for wav in audio_list]
    audio_list = torch.cat(audio_list, dim=1)  # 修改 concat 为 cat

    formatted_index = (
        f"{chapter_index:02d}"  # 格式化 chapter_index，确保在 1-9 之间的索引前面加上 0
    )

    output_path = os.path.join(
        f"./tmp/{book_name}/gen", f"{book_name}_{formatted_index}.wav"
    )
    torchaudio.save(output_path, audio_list, sample_rate)
    logger.info(f"文件已保存到 {output_path}")

    process_file_path = f"./tmp/{book_name}/process.txt"
    with open(process_file_path, "a", encoding="utf-8") as process_file:
        process_file.write(
            f"Chapter {chapter_index} processed and saved to {output_path}\n"
        )


def main():
    global book_name
    parser = argparse.ArgumentParser(
        description="Process book chapters to generate audio files."
    )
    parser.add_argument(
        "book_name", type=str, help="Name of the book", default="诡秘之主"
    )
    parser.add_argument("start_idx", type=int, help="Starting chapter index", default=1)
    parser.add_argument("end_idx", type=int, help="Ending chapter index", default=100)

    args = parser.parse_args()

    book_name = args.book_name
    start_idx = args.start_idx
    end_idx = args.end_idx

    os.makedirs(f"./tmp/{book_name}/gen", exist_ok=True)

    for chapter_index in range(start_idx, end_idx + 1):
        process_chapter(book_name, chapter_index)


if __name__ == "__main__":
    main()
