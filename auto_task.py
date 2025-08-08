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

from api_v3 import GPTSoVITSWrapper

from auto_task_util.auto_task_help_v2 import (
    get_texts,
    clear_text,
)


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
        self.model = GPTSoVITSWrapper()

        self.task_list = self.load_task_list()
        self.current_task = None

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

        # 将音频数据转换为浮点数，并缩放到 [-1.0, 1.0] 范围
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        elif audio.dtype == np.uint8:
            audio = (audio.astype(np.float32) - 128) / 128.0
        elif audio.dtype not in [np.float32, np.float64]:
            raise ValueError(f"Unsupported audio dtype: {audio.dtype}")

        meter = pyln.Meter(sample_rate)
        block_size = meter.block_size * sample_rate

        if len(audio) <= block_size:
            return torch.from_numpy(audio)

        integrated_loudness = meter.integrated_loudness(audio)
        normalized_audio = pyln.normalize.loudness(audio, integrated_loudness, -16.0)
        return torch.from_numpy(normalized_audio.astype(np.float32))

    def prepare_audio(
        self, audio_tensor: torch.Tensor, sample_rate: int
    ) -> torch.Tensor:
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        # 确保音频数据是浮点数类型
        if audio_tensor.dtype != torch.float32 and audio_tensor.dtype != torch.float64:
            audio_tensor = audio_tensor.float() / 32768.0  # 假设原始数据为 int16

        normalized_audio = self.normalize_loudness(
            audio_tensor.cpu().numpy(), sample_rate
        )
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

        # 如果音频是浮点数类型，转换回 int16 以保存
        if audio_tensor.dtype in [torch.float32, torch.float64]:
            audio_np = audio_tensor.squeeze(0).numpy()
            audio_np = np.clip(audio_np, -1.0, 1.0)
            audio_np = (audio_np * 32767).astype(np.int16)
        else:
            audio_np = audio_tensor.squeeze(0).numpy()

        torchaudio.save(
            temp_wav_path,
            torch.from_numpy(audio_np).unsqueeze(0),
            self.target_sample_rate,
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

    def process_text_line(
        self,
        text_line: str,
        audio_fragments: List[torch.Tensor],
        speaker: str,
        subtitle_lines: List[str],
        current_time: float,
    ) -> List[torch.Tensor]:
        """
        处理单行文本，包括生成音频、处理相似度等，并生成字幕。

        Args:
            text_line (str): 原始文本行。
            audio_fragments (List[torch.Tensor]): 音频片段列表，用于拼接最终音频。
            speaker (str): 说话人名称。
            subtitle_lines (List[str]): 字幕行列表。
            current_time (float): 当前时间，单位为秒，用于计算字幕的时间戳。

        Returns:
            List[torch.Tensor]: 更新后的音频片段列表。
        """
        parsed_texts = get_texts(text_line, ignore_punctuation=True)
        for text_info in parsed_texts:
            (text, lang) = text_info
            try_count = 0
            original_text = text
            cleaned_text = clear_text(text, ignore_punctuation=True)

            if not cleaned_text:
                continue

            if original_text == "":
                break

            # 调用 TTS 推理函数
            returned_sr, generated_audio = self.model.inference_with_spk(
                speaker, original_text, lang
            )
            self.target_sample_rate = returned_sr

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 确保 generated_audio 是 PyTorch 张量
            if isinstance(generated_audio, np.ndarray):
                generated_audio = torch.from_numpy(generated_audio)

            # 将生成的音频转换为 NumPy 数组
            generated_audio_np = generated_audio.squeeze().cpu().numpy()

            # break
            self.pinyin_similarity_map.clear()
            try_count = 0
            generated_tensor = torch.from_numpy(generated_audio_np).unsqueeze(0)
            prepared_audio = self.prepare_audio(
                generated_tensor, self.target_sample_rate
            )

            # 保存音频和文本
            if try_count == 0:
                self.save_audio(speaker, prepared_audio, original_text)

            audio_fragments.append(prepared_audio)

            # 获取最后一个片段的时长
            fragment_duration = (
                prepared_audio.size(0) / self.target_sample_rate
            )  # 修复这一行，使用 size(0) 处理一维或二维张量
            start_time = current_time
            end_time = current_time + fragment_duration

            # 将字幕信息添加到字幕列表
            subtitle_lines.append(
                f"{len(subtitle_lines) + 1}\n"
                f"{self.format_time(start_time)} --> {self.format_time(end_time)}\n"
                f"{self.insert_line_breaks(original_text)}\n\n"
            )

            # 更新当前时间
            current_time = end_time

        return audio_fragments, subtitle_lines, current_time

    def insert_line_breaks(self, text: str) -> str:
        """
        在标点符号后插入换行符，并且每10个字符进行换行，
        若当前第11个字符为标点符号，则保留该标点符号在当前行。

        Args:
            text (str): 原始文本。

        Returns:
            str: 处理后的文本。
        """
        max_chars = 10  # 每行最大字符数
        # 定义需要换行的标点符号
        punctuations = ["，", "。", "；", "！", "？", "：", "、"]

        result = []
        temp_line = ""
        count = 0

        i = 0
        while i < len(text):
            char = text[i]
            temp_line += char
            count += 1

            # 如果字符数量达到最大字符数，检查下一个字符是否为标点
            if count == max_chars:
                if i + 1 < len(text) and text[i + 1] in punctuations:
                    # 如果下一个字符是标点符号，保留在当前行
                    temp_line += text[i + 1]
                    result.append(temp_line)
                    temp_line = ""
                    count = 0
                    i += 2  # 跳过标点符号
                    continue
                else:
                    # 其他情况，强制换行
                    result.append(temp_line)
                    temp_line = ""
                    count = 0

            # 如果当前字符是标点符号且不满10个字符，则在标点符号后换行
            elif char in punctuations and count < max_chars:
                # 不在行尾插入换行符，只有在文本后面时才插入
                if i + 1 < len(text):
                    result.append(temp_line)
                    temp_line = ""
                    count = 0

            i += 1

        # 将剩余部分加入结果
        if temp_line:
            result.append(temp_line)

        # 拼接所有行，并去除多余的空白行
        return "\n".join(result).strip()

    def process_chapter(self, book_name: str, chapter_index: int) -> None:
        """
        处理指定章节的文本文件，生成对应的音频文件，并创建字幕文件。

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
            subtitle_lines: List[str] = []
            total_texts = len(texts)
            current_time = 0.0  # 当前时间（秒）

            logger.info(
                f"开始处理书籍：{book_name}，章节 {chapter_index}: {chapter_title}，总文本数：{total_texts}"
            )

            with tqdm(
                texts,
                desc=f"正在处理：{book_name}-第{chapter_index}章 - {chapter_title}",
                unit="行",
                leave=True,
            ) as progress_bar:
                for idx, line_info in enumerate(progress_bar, start=1):
                    (line, _) = line_info
                    cleaned_line = clear_text(line, ignore_punctuation=True)

                    # 获取角色对应的说话人
                    book_role = chapter_map.get(cleaned_line, {}).get("role", "")
                    speaker = bookname_to_role.get(book_role, self.default_spk)

                    # 处理文本行并生成音频，更新字幕
                    audio_fragments, subtitle_lines, current_time = (
                        self.process_text_line(
                            line, audio_fragments, speaker, subtitle_lines, current_time
                        )
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
            output_audio_path = os.path.join(
                f"./tmp/{book_name}/gen", f"{book_name}_{formatted_chapter_index}.wav"
            )
            torchaudio.save(output_audio_path, combined_audio, self.target_sample_rate)
            logger.info(f"章节 {chapter_index} 的音频文件已保存到 {output_audio_path}")

            # 保存字幕文件
            output_subtitle_path = os.path.join(
                f"./tmp/{book_name}/gen", f"{book_name}_{formatted_chapter_index}.srt"
            )
            with open(output_subtitle_path, "w", encoding="utf-8") as subtitle_file:
                subtitle_file.writelines(subtitle_lines)
            logger.info(
                f"章节 {chapter_index} 的字幕文件已保存到 {output_subtitle_path}"
            )

        except Exception as e:
            logger.exception(f"处理章节 {chapter_index} 时发生异常：{e}")

    @staticmethod
    def format_time(seconds: float) -> str:
        """
        格式化时间为 SRT 的时间格式。

        Args:
            seconds (float): 秒数。

        Returns:
            str: 格式化后的时间字符串。
        """
        millis = int((seconds % 1) * 1000)
        seconds = int(seconds)
        minutes = seconds // 60
        hours = minutes // 60
        seconds = seconds % 60
        minutes = minutes % 60
        return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"

    def run(self):
        """
        运行音频处理任务。
        status：等待中，执行中，已完成
        """
        task_list = self.task_list

        if not task_list:
            logger.info("任务列表为空，等待新任务。")
            return

        for task in task_list:
            book_name = task.get("book_name")
            check_spk_status = task.get("check_spk_status", "等待中")

            if check_spk_status != "已完成":
                logger.info(
                    f"跳过处理书籍 '{book_name}'，当前状态为 '{check_spk_status}'。"
                )
                continue

            gen_tts_status = task.get("gen_tts_status", "等待中")

            if gen_tts_status == "已完成":
                logger.info(f"跳过处理书籍 '{book_name}'，当前状态为 '已完成'。")
                continue

            gen_tts_start_idx = task.get("gen_tts_start_idx", 1)
            self.default_spk = task.get("default_spk", "旁白1")

            # 更新任务状态为 '执行中'
            task["gen_tts_status"] = "执行中"
            task["default_spk"] = self.default_spk
            self.current_book_name = book_name

            # 记录处理日志路径
            os.makedirs(f"./tmp/{book_name}/gen", exist_ok=True)
            data_dir = f"./tmp/{book_name}/data"

            logger.info(f"开始处理书籍：{book_name}，从章节 {gen_tts_start_idx} 开始。")

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
                task["gen_tts_status"] = "失败"
                self.save_task_list(task_list)
                continue

            total_chapters = len(chapter_files)
            task["gen_tts_end_idx"] = total_chapters
            self.save_task_list(task_list)

            if gen_tts_start_idx > total_chapters:
                logger.info(f"书籍 {book_name} 所有章节已处理完毕。")
                task["gen_tts_status"] = "已完成"
                self.save_task_list(task_list)
                continue

            for chapter_idx in range(gen_tts_start_idx, total_chapters + 1):
                self.process_chapter(book_name, chapter_idx)
                task["gen_tts_start_idx"] = chapter_idx + 1
                self.save_task_list(task_list)

            task["gen_tts_status"] = "已完成"
            self.save_task_list(task_list)
            logger.info(f"书籍 {book_name} 的所有章节已处理完毕。")


def main() -> None:
    """
    主函数，初始化并运行音频处理器。
    """
    processor = AudioProcessor()
    processor.run()


if __name__ == "__main__":
    main()
