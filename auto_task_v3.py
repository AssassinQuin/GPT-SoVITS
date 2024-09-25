import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torchaudio
import pyloudnorm as pyln
from tqdm import tqdm
from loguru import logger

from GPT_SoVITS.inference import inference
from tools.auto_task_help_v2 import get_texts, has_omission, clear_text


class SkipSpeakerManager:
    """
    管理需要跳过的说话人列表，包括加载和保存操作。
    """

    def __init__(
        self, skip_spk_file: str, initial_skip_spk: Optional[List[str]] = None
    ):
        self.skip_spk_file = skip_spk_file
        self.skip_spk = set(initial_skip_spk) if initial_skip_spk else set()
        self._load_skip_spk()

    def _load_skip_spk(self):
        """从文件加载需要跳过的说话人列表。"""
        if os.path.exists(self.skip_spk_file):
            try:
                with open(self.skip_spk_file, "r", encoding="utf-8") as f:
                    for line in f:
                        spk = line.strip()
                        if spk:
                            self.skip_spk.add(spk)
                logger.info(f"成功加载跳过说话人列表：{self.skip_spk_file}")
            except Exception as e:
                logger.error(f"加载跳过说话人列表时出错: {e}")
        else:
            os.makedirs(os.path.dirname(self.skip_spk_file), exist_ok=True)
            self._save_initial_skip_spk()
            logger.info(f"已创建跳过说话人列表文件：{self.skip_spk_file}")

    def _save_initial_skip_spk(self):
        """保存初始的跳过说话人列表到文件。"""
        try:
            with open(self.skip_spk_file, "w", encoding="utf-8") as f:
                for spk in self.skip_spk:
                    f.write(f"{spk}\n")
        except Exception as e:
            logger.error(f"保存初始跳过说话人列表时出错: {e}")

    def add_speaker(self, spk: str):
        """将说话人添加到跳过列表并保存到文件。"""
        if spk not in self.skip_spk:
            self.skip_spk.add(spk)
            try:
                with open(self.skip_spk_file, "a", encoding="utf-8") as f:
                    f.write(f"{spk}\n")
                logger.info(f"已将说话人 '{spk}' 添加到跳过列表。")
            except Exception as e:
                logger.error(f"将说话人 '{spk}' 添加到跳过列表时出错: {e}")

    def is_skipped(self, spk: str) -> bool:
        """检查说话人是否在跳过列表中。"""
        return spk in self.skip_spk


class AudioProcessor:
    """
    处理音频相关的操作，如标准化、保存等。
    """

    def __init__(
        self, sample_rate: int, skip_manager: SkipSpeakerManager, book_name: str
    ):
        self.sample_rate = sample_rate
        self.skip_manager = skip_manager
        self.book_name = book_name

    @staticmethod
    def normalize_loudness(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """标准化音频响度。"""
        audio_np = audio.numpy()
        if audio_np.ndim == 2:
            audio_np = audio_np.squeeze()
        meter = pyln.Meter(sample_rate)
        loudness = meter.integrated_loudness(audio_np)
        normalized_audio = pyln.normalize.loudness(audio_np, loudness, -16.0)
        return torch.from_numpy(normalized_audio)

    def prepare_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """准备并标准化音频。"""
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        audio = audio.float()
        return self.normalize_loudness(audio, self.sample_rate)

    def generate_silence(self, line: str) -> torch.Tensor:
        """根据标点生成静音。"""
        if line.endswith("，"):
            duration = np.random.uniform(0.05, 0.08)
        else:
            duration = np.random.uniform(0.09, 0.13)
        silence = np.zeros(int(self.sample_rate * duration), dtype=np.float32)
        return torch.from_numpy(silence).unsqueeze(0)

    def save_wav(self, speaker: str, audio: torch.Tensor, text: str):
        """保存音频和对应的文本文件。"""
        if self.skip_manager.is_skipped(speaker):
            logger.info(f"跳过说话人 '{speaker}'，因为它在跳过列表中。")
            return

        speaker_dir = os.path.join("./tmp/gen", speaker)
        os.makedirs(speaker_dir, exist_ok=True)

        # 检查并更新跳过列表
        try:
            wav_count = sum(
                1 for fname in os.listdir(speaker_dir) if fname.lower().endswith(".wav")
            )
            if wav_count >= 5000:
                self.skip_manager.add_speaker(speaker)
                logger.info(
                    f"说话人 '{speaker}' 的 .wav 文件数达到或超过 5000，已添加到跳过列表。"
                )
                return
        except Exception as e:
            logger.error(f"访问目录 '{speaker_dir}' 时出错: {e}")
            return

        timestamp = time.strftime("%Y%m%d%H%M%S")
        tmp_name = f"{self.book_name}_{speaker}_{timestamp}"
        wav_path = os.path.join(speaker_dir, f"{tmp_name}.wav")
        txt_path = os.path.join(speaker_dir, f"{tmp_name}.normalized.txt")

        try:
            torchaudio.save(wav_path, audio.unsqueeze(0), self.sample_rate)
            logger.info(f"已保存音频文件：{wav_path}")
        except Exception as e:
            logger.error(f"保存音频文件 '{wav_path}' 时出错: {e}")
            return

        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            logger.info(f"已保存文本文件：{txt_path}")
        except Exception as e:
            logger.error(f"保存文本文件 '{txt_path}' 时出错: {e}")


class ChapterProcessor:
    """
    处理单个章节的文本和音频生成。
    """

    def __init__(
        self,
        book_name: str,
        chapter_index: int,
        audio_processor: AudioProcessor,
        bookname2role: Dict[str, Any],
        line_map: Dict[str, Any],
    ):
        self.book_name = book_name
        self.chapter_index = chapter_index
        self.audio_processor = audio_processor
        self.bookname2role = bookname2role
        self.line_map = line_map
        self.pinyin_similarity_map: Dict[float, np.ndarray] = {}

    def process_text(
        self, text: str, audio_list: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """处理单个文本段落，生成音频并添加到音频列表中。"""
        texts = get_texts(text, True)
        for line in texts:
            cleartext = clear_text(line, True)
            if not line.strip() or not cleartext:
                continue

            try_again = 0
            while True:
                sample_rate, output = inference(self.speaker, line)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                is_continu, similarity, gen_text, clean_text = has_omission(
                    output, line, sample_rate
                )
                logger.info(
                    f"""
=========
原始文本：{line}
try_again: {try_again}
speaker: {self.speaker}
生成文本：{gen_text}
输入文本：{clean_text}
相似度：{similarity}
=========
"""
                )
                self.pinyin_similarity_map[similarity] = output

                if is_continu:
                    try_again += 1
                    if try_again >= 3:
                        best_similarity = max(self.pinyin_similarity_map.keys())
                        best_audio = self.pinyin_similarity_map[best_similarity]
                        self.pinyin_similarity_map.clear()
                        try_again = 0
                        best_audio_tensor = torch.from_numpy(best_audio).unsqueeze(0)
                        best_audio_tensor = self.audio_processor.prepare_audio(
                            best_audio_tensor
                        )

                        audio_list.append(best_audio_tensor)
                        audio_list.append(self.audio_processor.generate_silence(line))
                        break
                else:
                    best_audio_tensor = torch.from_numpy(output).unsqueeze(0)
                    tts_speech = self.audio_processor.prepare_audio(best_audio_tensor)
                    self.audio_processor.save_wav(self.speaker, tts_speech, line)
                    audio_list.append(tts_speech)
                    audio_list.append(self.audio_processor.generate_silence(line))
                    break

        return audio_list

    def process(self, audio_list: List[torch.Tensor]):
        """处理整个章节的文本和音频。"""
        file_path = os.path.join(
            "./tmp", self.book_name, "data", f"chapter_{self.chapter_index}.txt"
        )
        if not os.path.exists(file_path):
            logger.warning(f"文件 {file_path} 不存在，跳过处理。")
            return

        with open(file_path, "r", encoding="utf-8") as file:
            file_content = file.read()

        title = file_content.split("\n")[0].strip()

        texts = get_texts(file_content, bool(self.bookname2role))
        total_texts = len(texts)
        logger.info(f"总文本数: {total_texts}")

        with tqdm(
            total=total_texts,
            desc=f"正在处理：{self.book_name}-{self.chapter_index} - {title}",
            leave=True,
        ) as pbar:
            for line in texts:
                tmp_line = clear_text(line, True)
                speaker = "旁白1"
                bookname = self.line_map.get(tmp_line, {}).get("role", "")
                speaker = self.bookname2role.get(bookname, "旁白1")
                self.speaker = speaker

                audio_list = self.process_text(line, audio_list)
                pbar.update(1)

        if not audio_list:
            logger.warning(f"章节 {self.chapter_index} 没有生成任何音频。")
            return

        # 统一处理音频维度
        audio_list = [wav if wav.ndim == 2 else wav.unsqueeze(0) for wav in audio_list]
        concatenated_audio = torch.cat(audio_list, dim=1)

        formatted_index = f"{self.chapter_index:02d}"
        output_dir = os.path.join("./tmp", self.book_name, "gen")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir, f"{self.book_name}_{formatted_index}.wav"
        )

        try:
            torchaudio.save(
                output_path, concatenated_audio, self.audio_processor.sample_rate
            )
            logger.info(f"文件已保存到 {output_path}")

            process_file_path = os.path.join("./tmp", self.book_name, "process.txt")
            with open(process_file_path, "a", encoding="utf-8") as process_file:
                process_file.write(
                    f"Chapter {self.chapter_index} processed and saved to {output_path}\n"
                )
        except Exception as e:
            logger.error(f"保存章节 {self.chapter_index} 的音频时出错: {e}")


def load_json(file_path: str, default: Any) -> Any:
    """加载 JSON 文件或返回默认值。"""
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载 JSON 文件 '{file_path}' 时出错: {e}")
            return default
    return default


def main():
    parser = argparse.ArgumentParser(description="处理书籍章节生成音频文件。")
    parser.add_argument(
        "book_name", type=str, help="书名", default="诡秘之主", nargs="?"
    )
    parser.add_argument(
        "start_idx", type=int, help="起始章节索引", default=1, nargs="?"
    )
    parser.add_argument(
        "end_idx", type=int, help="结束章节索引", default=100, nargs="?"
    )

    args = parser.parse_args()

    book_name = args.book_name
    start_idx = args.start_idx
    end_idx = args.end_idx

    logger.info(f"CUDA 可用: {torch.cuda.is_available()}")

    skip_spk_file = os.path.join("./tmp/gen", "skip_spk_list.txt")
    initial_skip_spk = []
    skip_manager = SkipSpeakerManager(skip_spk_file, initial_skip_spk)

    audio_processor = AudioProcessor(
        sample_rate=22050, skip_manager=skip_manager, book_name=book_name
    )

    book_dir = os.path.join("./tmp", book_name)
    gen_dir = os.path.join(book_dir, "gen")
    os.makedirs(gen_dir, exist_ok=True)

    bookname2role_file = os.path.join(book_dir, "bookname2role.json")
    bookname2role = load_json(bookname2role_file, {})

    for chapter_index in range(start_idx, end_idx + 1):
        line_map_file = os.path.join(book_dir, "role", f"chapter_{chapter_index}.json")
        line_map_list = load_json(line_map_file, [])
        line_map = {}

        for item in line_map_list:
            for key, value in item.items():
                cleaned_key = clear_text(key, True)
                line_map[cleaned_key] = value

        chapter_processor = ChapterProcessor(
            book_name=book_name,
            chapter_index=chapter_index,
            audio_processor=audio_processor,
            bookname2role=bookname2role,
            line_map=line_map,
        )
        chapter_processor.process(audio_list=[])


if __name__ == "__main__":
    main()
