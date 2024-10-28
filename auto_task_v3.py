from concurrent.futures import ThreadPoolExecutor, as_completed
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

from GPT_SoVITS.inference_v2 import TTSGenerator

# from GPT_SoVITS.inference import inference
from tools.auto_task_help_v2 import (
    get_texts,
    has_omission,
    clear_text,
)

# 确保导入所有需要的函数和模块


class AudioProcessor:
    def __init__(self):
        # 初始化拼音相似度映射
        self.pinyin_similarity_map: Dict[float, np.ndarray] = {}
        # 目标采样率
        self.target_sample_rate: int = 22050
        # 跳过处理的说话人列表
        self.skipped_speakers: List[str] = []
        # 任务列表文件路径
        self.task_list_file = "./tmp/task_list.json"
        # 当前书籍名称
        self.current_book_name: str = ""
        self.default_spk = "旁白1"
        self.model = TTSGenerator()

    def load_task_list(self) -> List[Dict[str, Any]]:
        """
        加载任务列表，如果文件不存在则返回空列表。

        Returns:
            List[Dict[str, Any]]: 任务列表。
        """
        if not os.path.exists(self.task_list_file):
            logger.warning(
                f"任务列表文件 {self.task_list_file} 不存在，创建新的任务列表。"
            )
            return []

        with open(self.task_list_file, "r", encoding="utf-8-sig") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                logger.error(
                    f"任务列表文件 {self.task_list_file} 解析失败，返回空列表。"
                )
                return []

    def save_task_list(self, task_list: List[Dict[str, Any]]) -> None:
        """
        保存任务列表到文件。

        Args:
            task_list (List[Dict[str, Any]]): 任务列表。
        """
        with open(self.task_list_file, "w", encoding="utf-8") as f:
            json.dump(task_list, f, ensure_ascii=False, indent=4)

    @staticmethod
    def resample_audio(
        audio_tensor: torch.Tensor, original_sr: int, target_sr: int
    ) -> torch.Tensor:
        if original_sr == target_sr:
            return audio_tensor

        resampler = transforms.Resample(orig_freq=original_sr, new_freq=target_sr)
        audio_tensor = (
            audio_tensor.float()
            if audio_tensor.dtype != torch.float32
            else audio_tensor
        )
        return resampler(audio_tensor)

    @staticmethod
    def normalize_loudness(audio: np.ndarray, sample_rate: int) -> torch.Tensor:
        if audio.ndim == 2:
            audio = audio.squeeze()

        meter = pyln.Meter(sample_rate)
        block_size = meter.block_size * sample_rate

        if len(audio) <= block_size:
            return torch.from_numpy(audio)

        integrated_loudness = meter.integrated_loudness(audio)
        normalized_audio = pyln.normalize.loudness(audio, integrated_loudness, -16.0)
        return torch.from_numpy(normalized_audio)

    def prepare_audio(
        self, audio_tensor: torch.Tensor, sample_rate: int
    ) -> torch.Tensor:
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        normalized_audio = self.normalize_loudness(audio_tensor.numpy(), sample_rate)
        return normalized_audio

    def save_audio(self, spk: str, audio_tensor: torch.Tensor, text_line: str) -> None:
        if len(text_line) < 15:
            return

        speaker_dir = f"./tmp/gen/{spk}"
        os.makedirs(speaker_dir, exist_ok=True)

        # 检查并跳过处理过多文件的说话人
        existing_wav_files = [f for f in os.listdir(speaker_dir) if f.endswith(".wav")]
        if len(existing_wav_files) > 5000:
            self.skipped_speakers.append(spk)
            return

        if spk in self.skipped_speakers:
            return

        timestamp = time.strftime("%Y%m%d%H%M%S")
        sanitized_book_name = re.sub(r"[^\w\-]", "_", self.current_book_name)
        sanitized_speaker = re.sub(r"[^\w\-]", "_", spk)
        temp_wav_filename = f"{sanitized_book_name}_{sanitized_speaker}_{timestamp}.wav"
        temp_wav_path = os.path.join(speaker_dir, temp_wav_filename)

        torchaudio.save(
            temp_wav_path, audio_tensor.unsqueeze(0), self.target_sample_rate
        )

        # 保存对应的文本文件
        temp_txt_path = os.path.splitext(temp_wav_path)[0] + ".normalized.txt"
        with open(temp_txt_path, "w", encoding="utf-8") as f:
            f.write(text_line)

    @staticmethod
    def generate_silence(text_line: str, target_sample_rate: int) -> torch.Tensor:
        if text_line.endswith("，"):
            duration = np.random.uniform(0.05, 0.08)
        else:
            duration = np.random.uniform(0.09, 0.13)

        silence_samples = int(target_sample_rate * duration)
        silence_audio = np.zeros(silence_samples, dtype=np.float32)
        return torch.from_numpy(silence_audio).unsqueeze(0)

    @staticmethod
    def load_json(file_path: str, default: Any) -> Any:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return default

    @staticmethod
    def delete_files(wav_file: str) -> None:
        normalized_txt = f"{os.path.splitext(wav_file)[0]}.normalized.txt"

        logger.info(f"删除文件: {wav_file}")
        os.remove(wav_file)

        if os.path.isfile(normalized_txt):
            logger.info(f"删除文件: {normalized_txt}")
            os.remove(normalized_txt)
        else:
            logger.warning(f"对应的 .normalized.txt 文件不存在: {normalized_txt}")

    def process_text_line(
        self, text_line: str, audio_fragments: List[torch.Tensor], speaker: str
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
        parsed_texts = get_texts(text_line, ignore_punctuation=True)
        for text in parsed_texts:
            try_count = 0
            original_text = text
            cleaned_text = clear_text(text, ignore_punctuation=True)

            if not cleaned_text:
                continue

            while True:
                # 调用 TTS 推理函数
                returned_sr, generated_audio = self.model.inference(
                    speaker, original_text
                )

                # 重采样如果返回的采样率不同
                if returned_sr != self.target_sample_rate:
                    audio_tensor = torch.from_numpy(generated_audio).unsqueeze(0)
                    resampled_tensor = self.resample_audio(
                        audio_tensor,
                        original_sr=returned_sr,
                        target_sr=self.target_sample_rate,
                    )
                    generated_audio = resampled_tensor.squeeze(0).numpy()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # 检查是否有遗漏
                needs_repeat, similarity, clean_generated_text, clean_input_text = (
                    has_omission(
                        generated_audio, original_text, self.target_sample_rate
                    )
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
                self.pinyin_similarity_map[similarity] = generated_audio

                if needs_repeat:
                    try_count += 1
                    if try_count >= 3:
                        # 获取最佳相似度的音频
                        best_similarity = max(self.pinyin_similarity_map.keys())
                        best_audio = self.pinyin_similarity_map[best_similarity]
                        self.pinyin_similarity_map.clear()
                        try_count = 0

                        # 处理最佳音频
                        best_audio_tensor = torch.from_numpy(best_audio).unsqueeze(0)
                        prepared_audio = self.prepare_audio(
                            best_audio_tensor, self.target_sample_rate
                        )

                        audio_fragments.append(prepared_audio)
                        audio_fragments.append(
                            self.generate_silence(
                                original_text, self.target_sample_rate
                            )
                        )
                        break
                else:
                    # 处理生成的音频
                    self.pinyin_similarity_map.clear()
                    try_count = 0
                    generated_tensor = torch.from_numpy(generated_audio).unsqueeze(0)
                    prepared_audio = self.prepare_audio(
                        generated_tensor, self.target_sample_rate
                    )

                    # 保存音频和文本
                    if try_count == 0:
                        self.save_audio(speaker, prepared_audio, original_text)

                    audio_fragments.append(prepared_audio)
                    audio_fragments.append(
                        self.generate_silence(original_text, self.target_sample_rate)
                    )
                    break

        return audio_fragments

    def process_chapter(self, book_name: str, chapter_index: int) -> None:
        """
        处理指定章节的文本文件，生成对应的音频文件。

        Args:
            book_name (str): 书名。
            chapter_index (int): 章节索引。
        """
        try:
            file_path = f"./tmp/{book_name}/data/chapter_{chapter_index}.txt"
            if not os.path.exists(file_path):
                logger.warning(f"文件 {file_path} 不存在，跳过处理。")
                return

            with open(file_path, "r", encoding="utf-8") as file:
                file_content = file.read()
                chapter_title = file_content.split("\n")[0]

            # 加载角色映射
            role_mapping_file = f"./tmp/{book_name}/bookname2role.json"
            bookname_to_role = self.load_json(role_mapping_file, default={})

            # 加载章节映射
            chapter_map_file = f"./tmp/{book_name}/role/chapter_{chapter_index}.json"
            chapter_map_list = self.load_json(chapter_map_file, default=[])
            chapter_map = {}

            for item in chapter_map_list:
                sanitized_item = {
                    clear_text(key, ignore_punctuation=True): value
                    for key, value in item.items()
                }
                chapter_map.update(sanitized_item)

            # 获取所有文本行
            texts = (
                get_texts(file_content, ignore_punctuation=True)
                if not bookname_to_role
                else get_texts(file_content)
            )

            audio_fragments: List[torch.Tensor] = []
            total_texts = len(texts)

            logger.info(
                f"开始处理书籍：{book_name}，章节 {chapter_index}: {chapter_title}，总文本数：{total_texts}"
            )

            with tqdm(
                texts,
                desc=f"正在处理：{book_name}-第{chapter_index}章 - {chapter_title}",
                unit="行",
                leave=True,
            ) as progress_bar:
                for line in progress_bar:
                    cleaned_line = clear_text(line, ignore_punctuation=True)

                    # 获取角色对应的说话人
                    book_role = chapter_map.get(cleaned_line, {}).get("role", "")
                    speaker = bookname_to_role.get(book_role, self.default_spk)
                    if speaker != "旁白1":
                        print("speaker")

                    # 处理文本行并生成音频
                    audio_fragments = self.process_text_line(
                        line, audio_fragments, speaker
                    )

            # 拼接所有音频片段
            audio_fragments = [
                audio if audio.ndim == 2 else audio.unsqueeze(0)
                for audio in audio_fragments
            ]
            if not audio_fragments:
                logger.warning(f"章节 {chapter_index} 没有生成任何音频片段，跳过保存。")
                return

            combined_audio = torch.cat(audio_fragments, dim=1)

            # 保存最终音频文件
            formatted_chapter_index = f"{chapter_index:02d}"
            output_path = os.path.join(
                f"./tmp/{book_name}/gen", f"{book_name}_{formatted_chapter_index}.wav"
            )
            torchaudio.save(output_path, combined_audio, self.target_sample_rate)
            logger.info(f"章节 {chapter_index} 的音频文件已保存到 {output_path}")

            # 记录处理日志
            process_log_path = f"./tmp/{book_name}/process.txt"
            with open(process_log_path, "w", encoding="utf-8") as process_log:
                process_log.write(f"{chapter_index}")
            logger.info(f"已记录章节 {chapter_index} 的处理进度。")

        except Exception as e:
            logger.exception(f"处理章节 {chapter_index} 时发生异常：{e}")

    def run_task(self, task: Dict[str, Any], task_list: List[Dict[str, Any]]):
        book_name = task.get("book_name")
        status = task.get("status", "等待中")

        if status == "已完成":
            logger.info(f"书籍 {book_name} 已完成，跳过。")
            return

        # 更新任务状态为 '执行中'
        task["status"] = "执行中"
        self.default_spk = task.get("default_spk", "旁白1")
        self.save_task_list(task_list)
        self.current_book_name = book_name

        # 记录处理日志路径
        process_log_path = f"./tmp/{book_name}/process.txt"
        # 确保书籍处理目录存在
        os.makedirs(os.path.dirname(process_log_path), exist_ok=True)
        os.makedirs(f"./tmp/{book_name}/gen", exist_ok=True)
        data_dir = f"./tmp/{book_name}/data"

        # 读取开始的章节索引，如果日志文件存在
        if os.path.exists(process_log_path):
            with open(process_log_path, "r", encoding="utf-8") as process_log:
                last_processed_chapter = process_log.read().strip()
                if last_processed_chapter.isdigit():
                    start_chapter_idx = int(last_processed_chapter) + 1
                else:
                    start_chapter_idx = 1
        else:
            start_chapter_idx = 1

        logger.info(f"开始处理书籍：{book_name}，从章节 {start_chapter_idx} 开始。")

        # 获取所有章节文件，按章节编号排序
        try:
            chapter_files = sorted(
                [
                    f
                    for f in os.listdir(data_dir)
                    if f.endswith(".txt") and not f.startswith(".")
                ],
                key=lambda x: int(re.findall(r"\d+", x)[0])
                if re.findall(r"\d+", x)
                else 0,
            )
        except Exception as e:
            logger.error(f"获取书籍 {book_name} 的章节文件时发生错误：{e}")
            task["status"] = "failed"
            self.save_task_list(task_list)
            return

        total_chapters = len(chapter_files)

        if start_chapter_idx > total_chapters:
            logger.info(f"书籍 {book_name} 所有章节已处理完毕。")
            task["status"] = "已完成"
            self.save_task_list(task_list)
            return

        for chapter_idx in range(start_chapter_idx, total_chapters + 1):
            self.process_chapter(book_name, chapter_idx)

        # 更新任务状态为 '已完成' after all chapters are processed
        task["status"] = "已完成"
        self.save_task_list(task_list)
        logger.info(f"书籍 {book_name} 的所有章节已处理完毕。")

    def run(self):
        """
        运行音频处理任务，支持多线程并发。
        """
        task_list = self.load_task_list()

        if not task_list:
            logger.info("任务列表为空，等待新任务。")
            return

        max_workers = min(2, len(task_list))  # 根据任务数量调整线程数
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(self.run_task, task, task_list): task
                for task in task_list
                if task.get("status", "等待中") != "已完成"
            }

            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    future.result()
                except Exception as exc:
                    logger.error(f"任务 {task.get('book_name')} 发生异常：{exc}")
                    task["status"] = "failed"
                    self.save_task_list(task_list)


def main() -> None:
    """
    主函数，初始化并运行音频处理器。
    """
    processor = AudioProcessor()
    processor.run()


if __name__ == "__main__":
    main()
