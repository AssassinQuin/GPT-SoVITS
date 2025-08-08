import os
import json
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from pypinyin import pinyin, Style
from Levenshtein import distance
from typing import List
from loguru import logger


class ChineseNameDetector:
    """中文人名识别与处理一体化类"""

    def __init__(
        self,
        model_path: str,
        data_dir: str,
        output_dir: str,
        similarity_threshold: int = 2,
    ):
        """
        :param model_path: 模型路径
        :param data_dir: 输入数据目录
        :param output_dir: 输出目录
        :param similarity_threshold: 拼音相似度阈值
        """
        self.model_path = model_path
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.similarity_threshold = similarity_threshold
        self.seen_pinyins = set()

        # 初始化模型组件
        self._init_model()

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

    def _init_model(self):
        """初始化NER模型"""
        try:
            self.pipeline = pipeline(
                Tasks.named_entity_recognition, self.model_path, device="gpu"
            )
        except Exception as e:
            raise RuntimeError(f"模型初始化失败: {str(e)}")

    def _generate_pinyin(self, name: str) -> str:
        """生成拼音标识"""
        return "".join([item[0] for item in pinyin(name, style=Style.TONE3)])

    def _is_duplicate(self, pinyin_str: str) -> bool:
        """检查拼音相似性"""
        return any(
            pinyin_str in exist
            or exist in pinyin_str
            or distance(pinyin_str, exist) <= self.similarity_threshold
            for exist in self.seen_pinyins
        )

    def process_text(self, text: str) -> List[str]:
        """处理单个文本"""
        if not text.strip():
            return []

        try:
            self.seen_pinyins = set()
            text_list = text.split("\n")

            input_list = []
            valid_names = []
            skip = 5
            for idx in range(0, len(text_list), skip):
                min_idx = idx
                max_idx = min(len(text_list), idx + skip)
                item = "\n".join(text_list[min_idx:max_idx])
                if len(item) > 500:
                    for i in range(skip):
                        input_list.append(text_list[min_idx + i])
                else:
                    input_list.append(item)

            for item in input_list:
                entities = self.pipeline(item)

                for entity in entities["output"]:
                    if entity["type"] in ["PER", "TIT"]:
                        name = entity["span"]
                        pinyin_str = self._generate_pinyin(name)
                        if not self._is_duplicate(pinyin_str):
                            valid_names.append(name)
                            self.seen_pinyins.add(pinyin_str)

            return valid_names
        except Exception as e:
            logger.warning(f"文本处理异常: {str(e)}")
            return []

    def process_files(self):
        """批量处理文件"""
        processed_files = []

        filelist = os.listdir(self.data_dir)
        # 根据文件名排序
        filelist = sorted(filelist, key=lambda x: int(x.split(".")[0]))
        for filename in filelist:
            if filename.endswith(".txt"):
                input_path = os.path.join(self.data_dir, filename)
                output_path = os.path.join(
                    self.output_dir, f"name_{os.path.splitext(filename)[0]}.json"
                )

                try:
                    with open(input_path, "r", encoding="utf-8") as f:
                        text = f.read()

                    names = self.process_text(text)
                    logger.info(f"filename: {filename} - names: {names}")
                    self._save_results(names, output_path)
                    processed_files.append(output_path)
                except Exception as e:
                    logger.warning(f"文件处理失败 [{filename}]: {str(e)}")

        return processed_files

    def _save_results(self, data: List, path: str):
        """保存结果到文件"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


# 使用示例
if __name__ == "__main__":
    bookname = "趋吉避凶，从天师府开始"
    detector = ChineseNameDetector(
        model_path="iic/nlp_raner_named-entity-recognition_chinese-base-book",
        data_dir=f"/root/code/GPT-SoVITS/tmp/{bookname}/data",
        output_dir=f"/root/code/GPT-SoVITS/tmp/{bookname}/role",
    )

    result_files = detector.process_files()
    logger.info(f"处理完成，生成 {len(result_files)} 个结果文件")
