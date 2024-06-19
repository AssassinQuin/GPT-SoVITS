import os
import argparse
import numpy as np
from GPT_SoVITS.inference import inference
import soundfile as sf


# 执行生成音频, 输入长文本生成对应音频
def gen_wav(file_path, text, language):
    # 调用 inference 函数生成音频
    audio_generator = inference(text, language)

    # 创建输出目录
    output_dir = os.path.dirname(file_path)
    os.makedirs(output_dir, exist_ok=True)
    file_rate = 0
    file_audio = []

    # 保存生成的音频
    for rate, audio in audio_generator:
        # 将生成的音频保存为 WAV 文件
        file_rate = rate
        file_audio.append(audio)

    # 将音频块合并成一个一维的 numpy 数组
    file_audio = np.concatenate(file_audio)
    # 将音频数据保存为 WAV 文件
    sf.write(file_path, file_audio, file_rate)
    print(f"音频已保存到: {file_path}")


def main(novel_name, start_chapter, end_chapter):
    # text = "在他身后的图腾柱上那熊熊燃烧的火球中突然发出了一连串令人不安的噼啪爆鸣"
    base_dir = os.path.join(os.getcwd(), "tmp", novel_name)
    data_dir = os.path.join(base_dir, "data")
    gen_dir = os.path.join(base_dir, "gen")

    for chapter in range(start_chapter, end_chapter + 1):
        chapter_file = os.path.join(data_dir, f"chapter_{chapter}.txt")
        if os.path.exists(chapter_file):
            with open(chapter_file, "r", encoding="utf-8") as file:
                text = file.read()
            output_file = os.path.join(gen_dir, f"chapter_{chapter}.wav")
            gen_wav(output_file, text, "中文")
        else:
            print(f"章节文件 {chapter_file} 不存在")


def test():
    main("shyj", 21, 21)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description="Generate WAV files from text chapters."
    # )
    # parser.add_argument(
    #     "--novel_name", type=str, default="shyj", help="The name of the novel."
    # )
    # parser.add_argument(
    #     "--start", type=int, default=1, help="The starting chapter number."
    # )
    # parser.add_argument("--end", type=int, default=2, help="The ending chapter number.")

    # args = parser.parse_args()

    # main(args.novel_name, args.start, args.end)

    test()
