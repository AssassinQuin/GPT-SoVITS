import json
import os
import re
import time
from typing import List, Dict, Any

import numpy as np
import torch
import torchaudio
import pyloudnorm as pyln
from tqdm import tqdm
from loguru import logger
from torchaudio import transforms

from GPT_SoVITS.inference import inference
from tools.auto_task_help_v2 import (
    get_texts,
    has_omission,
    clear_text,
)


# 全局变量
logger.info(f"CUDA available: {torch.cuda.is_available()}")

# 用于存储相似度与音频的映射
pinyin_similarity_map: Dict[float, np.ndarray] = {}
TARGET_SAMPLE_RATE: int = 22050
current_book_name: str = ""

# 跳过处理的说话人列表
skipped_speakers: List[str] = []


def resample_audio(
    audio_tensor: torch.Tensor, original_sr: int, target_sr: int
) -> torch.Tensor:
    """
    重采样音频到目标采样率。

    Args:
        audio_tensor (torch.Tensor): 原始音频张量。
        original_sr (int): 原始采样率。
        target_sr (int): 目标采样率。

    Returns:
        torch.Tensor: 重采样后的音频张量。
    """
    if original_sr == target_sr:
        return audio_tensor

    resampler = transforms.Resample(orig_freq=original_sr, new_freq=target_sr)
    audio_tensor = (
        audio_tensor.float() if audio_tensor.dtype != torch.float32 else audio_tensor
    )
    return resampler(audio_tensor)


def normalize_loudness(audio: np.ndarray, sample_rate: int) -> torch.Tensor:
    """
    归一化音频响度到 -16 LUFS。

    Args:
        audio (np.ndarray): 原始音频数组。
        sample_rate (int): 采样率。

    Returns:
        torch.Tensor: 归一化后的音频张量。
    """
    if audio.ndim == 2:
        audio = audio.squeeze()

    meter = pyln.Meter(sample_rate)
    block_size = meter.block_size * sample_rate

    if len(audio) <= block_size:
        return torch.from_numpy(audio)

    integrated_loudness = meter.integrated_loudness(audio)
    normalized_audio = pyln.normalize.loudness(audio, integrated_loudness, -16.0)
    return torch.from_numpy(normalized_audio)


def prepare_audio(audio_tensor: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """
    准备音频数据，包括维度调整和响度归一化。

    Args:
        audio_tensor (torch.Tensor): 原始音频张量。
        sample_rate (int): 采样率。

    Returns:
        torch.Tensor: 处理后的音频张量。
    """
    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)

    normalized_audio = normalize_loudness(audio_tensor.numpy(), sample_rate)
    return normalized_audio


def save_audio(spk: str, audio_tensor: torch.Tensor, text_line: str) -> None:
    """
    保存音频文件和对应的文本文件。

    Args:
        spk (str): 说话人名称。
        audio_tensor (torch.Tensor): 要保存的音频张量。
        text_line (str): 对应的文本行。
    """
    global current_book_name

    if len(text_line) < 15:
        return

    speaker_dir = f"./tmp/gen/{spk}"
    os.makedirs(speaker_dir, exist_ok=True)

    # 检查并跳过处理过多文件的说话人
    existing_wav_files = [f for f in os.listdir(speaker_dir) if f.endswith(".wav")]
    if len(existing_wav_files) > 5000:
        skipped_speakers.append(spk)
        return

    if spk in skipped_speakers:
        return

    timestamp = time.strftime("%Y%m%d%H%M%S")
    sanitized_book_name = re.sub(r"[^\w\-]", "_", current_book_name)
    sanitized_speaker = re.sub(r"[^\w\-]", "_", spk)
    temp_wav_filename = f"{sanitized_book_name}_{sanitized_speaker}_{timestamp}.wav"
    temp_wav_path = os.path.join(speaker_dir, temp_wav_filename)

    torchaudio.save(temp_wav_path, audio_tensor.unsqueeze(0), TARGET_SAMPLE_RATE)

    # 保存对应的文本文件
    temp_txt_path = os.path.splitext(temp_wav_path)[0] + ".normalized.txt"
    with open(temp_txt_path, "w", encoding="utf-8") as f:
        f.write(text_line)


def generate_silence(text_line: str) -> torch.Tensor:
    """
    生成一段静音音频，持续时间根据文本标点决定。

    Args:
        text_line (str): 文本行，用于决定静音持续时间。

    Returns:
        torch.Tensor: 静音音频张量。
    """
    if text_line.endswith("，"):
        duration = np.random.uniform(0.05, 0.08)
    else:
        duration = np.random.uniform(0.09, 0.13)

    silence_samples = int(TARGET_SAMPLE_RATE * duration)
    silence_audio = np.zeros(silence_samples, dtype=np.float32)
    return torch.from_numpy(silence_audio).unsqueeze(0)


def load_json(file_path: str, default: Any) -> Any:
    """
    加载 JSON 文件，如果不存在则返回默认值。

    Args:
        file_path (str): JSON 文件路径。
        default (Any): 默认返回值。

    Returns:
        Any: 加载的 JSON 数据或默认值。
    """
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def delete_files(wav_file: str) -> None:
    """
    删除指定的音频文件及其对应的文本文件。

    Args:
        wav_file (str): 音频文件路径。
    """
    normalized_txt = f"{os.path.splitext(wav_file)[0]}.normalized.txt"

    logger.info(f"删除文件: {wav_file}")
    os.remove(wav_file)

    if os.path.isfile(normalized_txt):
        logger.info(f"删除文件: {normalized_txt}")
        os.remove(normalized_txt)
    else:
        logger.warning(f"对应的 .normalized.txt 文件不存在: {normalized_txt}")


def process_text_line(
    text_line: str, audio_fragments: List[torch.Tensor], speaker: str
) -> List[torch.Tensor]:
    """
    处理单行文本，包括生成音频、处理相似度等。

    Args:
        text_line (str): 原始文本行。
        audio_fragments (List[torch.Tensor]): 音频片段列表，用于拼接最终音频。
        speaker (str): 说话人名称。

    Returns:
        List[torch.Tensor]: 更新后的音频片段列表。
    """
    global pinyin_similarity_map, TARGET_SAMPLE_RATE

    parsed_texts = get_texts(text_line, remove_punc=True)
    for text in parsed_texts:
        try_count = 0
        original_text = text
        cleaned_text = clear_text(text, remove_punc=True)

        if not cleaned_text:
            continue

        while True:
            # 调用 TTS 推理函数
            returned_sr, generated_audio = inference(speaker, original_text)

            # 重采样如果返回的采样率不同
            if returned_sr != TARGET_SAMPLE_RATE:
                audio_tensor = torch.from_numpy(generated_audio).unsqueeze(0)
                resampled_tensor = resample_audio(
                    audio_tensor, orig_sr=returned_sr, target_sr=TARGET_SAMPLE_RATE
                )
                generated_audio = resampled_tensor.squeeze(0).numpy()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 检查是否有遗漏
            needs_repeat, similarity, clean_generated_text, clean_input_text = (
                has_omission(generated_audio, original_text, TARGET_SAMPLE_RATE)
            )

            logger.info(f"""
==================
原始文本: {original_text}
尝试次数: {try_count}
说话人: {speaker}
生成文本: {clean_generated_text}
输入文本: {clean_input_text}
相似度: {similarity}
==================
            """)

            # 记录相似度与音频映射
            pinyin_similarity_map[similarity] = generated_audio

            if needs_repeat:
                try_count += 1
                if try_count >= 3:
                    # 获取最佳相似度的音频
                    best_similarity = max(pinyin_similarity_map.keys())
                    best_audio = pinyin_similarity_map[best_similarity]
                    pinyin_similarity_map.clear()
                    try_count = 0

                    # 处理最佳音频
                    best_audio_tensor = torch.from_numpy(best_audio).unsqueeze(0)
                    prepared_audio = prepare_audio(
                        best_audio_tensor, TARGET_SAMPLE_RATE
                    )

                    audio_fragments.append(prepared_audio)
                    audio_fragments.append(generate_silence(original_text))
                    break
            else:
                # 处理生成的音频
                pinyin_similarity_map.clear()
                try_count = 0
                generated_tensor = torch.from_numpy(generated_audio).unsqueeze(0)
                prepared_audio = prepare_audio(generated_tensor, TARGET_SAMPLE_RATE)

                # 保存音频和文本
                if try_count == 0:
                    save_audio(speaker, prepared_audio, original_text)

                audio_fragments.append(prepared_audio)
                audio_fragments.append(generate_silence(original_text))
                break

    return audio_fragments


def process_chapter(book_name: str, chapter_index: int) -> None:
    """
    处理指定章节的文本文件，生成对应的音频文件。

    Args:
        book_name (str): 书名。
        chapter_index (int): 章节索引。
    """
    file_path = f"./tmp/{book_name}/data/chapter_{chapter_index}.txt"
    if not os.path.exists(file_path):
        logger.info(f"文件 {file_path} 不存在，跳过处理。")
        return

    with open(file_path, "r", encoding="utf-8") as file:
        file_content = file.read()
        chapter_title = file_content.split("\n")[0]

    # 加载角色映射
    role_mapping_file = f"./tmp/{book_name}/bookname2role.json"
    bookname_to_role = load_json(role_mapping_file, default={})

    # 加载章节映射
    chapter_map_file = f"./tmp/{book_name}/role/chapter_{chapter_index}.json"
    chapter_map_list = load_json(chapter_map_file, default=[])
    chapter_map = {}

    for item in chapter_map_list:
        sanitized_item = {
            clear_text(key, remove_punc=True): value for key, value in item.items()
        }
        chapter_map.update(sanitized_item)

    # 获取所有文本行
    if not bookname_to_role:
        texts = get_texts(file_content, remove_punc=True)
    else:
        texts = get_texts(file_content)

    audio_fragments: List[torch.Tensor] = []
    total_texts = len(texts)
    current_index = 0

    logger.info(f"总文本数: {total_texts}")

    with tqdm(
        total=total_texts,
        desc=f"正在处理：{book_name}-第{chapter_index}章 - {chapter_title}",
        leave=True,
    ) as progress_bar:
        while current_index < total_texts:
            line = texts[current_index]
            cleaned_line = clear_text(line, remove_punc=True)
            speaker_role = "旁白1"

            # 获取角色对应的说话人
            book_role = chapter_map.get(cleaned_line, {}).get("role", "")
            speaker = bookname_to_role.get(book_role, "旁白1")

            # 处理文本行并生成音频
            audio_fragments = process_text_line(line, audio_fragments, speaker)

            current_index += 1
            progress_bar.update(1)

    # 拼接所有音频片段
    audio_fragments = [
        audio if audio.ndim == 2 else audio.unsqueeze(0) for audio in audio_fragments
    ]
    combined_audio = torch.cat(audio_fragments, dim=1)

    # 保存最终音频文件
    formatted_chapter_index = f"{chapter_index:02d}"
    output_path = os.path.join(
        f"./tmp/{book_name}/gen", f"{book_name}_chapter_{formatted_chapter_index}.wav"
    )
    torchaudio.save(output_path, combined_audio, TARGET_SAMPLE_RATE)
    logger.info(f"文件已保存到 {output_path}")

    # 记录处理日志
    process_log_path = f"./tmp/{book_name}/process.txt"
    with open(process_log_path, "a", encoding="utf-8") as process_log:
        process_log.write(
            f"Chapter {chapter_index} processed and saved to {output_path}\n"
        )


def main() -> None:
    """
    主函数，加载任务列表并处理每个任务中的章节。
    """
    global current_book_name

    # 加载任务列表
    task_list_file = "./tmp/task_list.json"
    if not os.path.exists(task_list_file):
        raise FileNotFoundError(f"任务列表文件 {task_list_file} 不存在")

    with open(task_list_file, "r", encoding="utf-8-sig") as f:
        task_list = json.load(f)

    for book_name, start_chapter_idx in task_list:
        current_book_name = book_name
        os.makedirs(f"./tmp/{book_name}/gen", exist_ok=True)
        data_dir = f"./tmp/{book_name}/data"

        # 获取所有章节文件，按章节编号排序
        chapter_files = sorted(
            [
                f
                for f in os.listdir(data_dir)
                if f.endswith(".txt") and not f.startswith(".")
            ],
            key=lambda x: int(re.findall(r"\d+", x)[0]) if re.findall(r"\d+", x) else 0,
        )
        total_chapters = len(chapter_files)

        for chapter_idx in range(start_chapter_idx, total_chapters + 1):
            process_chapter(book_name, chapter_idx)


if __name__ == "__main__":
    main()
