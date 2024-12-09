# -*- coding:utf-8 -*-

from datetime import datetime
import os
import traceback
import uuid
from funasr import AutoModel


def only_asr(input_audio, rate=22050):
    import torch
    import soundfile as sf

    try:
        # 检查输入是否是文件路径或 tensor
        if isinstance(input_audio, str):
            text = model.generate(input=input_audio)[0]["text"]
        else:
            # 确保输入是 torch.Tensor 并且是 1 维
            if isinstance(input_audio, torch.Tensor) and len(input_audio.shape) == 2:
                input_audio = input_audio.squeeze(0)

            # 将 tensor 保存为临时 wav 文件
            # 生成时间戳
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

            # 生成四位的 UUID 字符串
            uuid_str = str(uuid.uuid4())[:4]
            temp_wav_path = f"./tmp/gen_temp/{timestamp}_{uuid_str}.wav"
            sf.write(temp_wav_path, input_audio, rate)

            # 使用临时文件进行推理
            text = model.generate(input=temp_wav_path, language="auto")[0]["text"]
            os.remove(temp_wav_path)
    except:
        text = ""
        print(traceback.format_exc())
    return text


# 模型路径和版本
path_vad = "tools/asr/models/speech_fsmn_vad_zh-cn-16k-common-pytorch"
path_vad = (
    path_vad
    if os.path.exists(path_vad)
    else "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
)

vad_model_revision = punc_model_revision = "v2.0.4"

path_asr = "tools/asr/models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
path_asr = (
    path_asr
    if os.path.exists(path_asr)
    else "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
)
model_revision = "v2.0.4"

# 初始化模型
model = AutoModel(
    model=path_asr,
    model_revision=model_revision,
    vad_model=path_vad,
    vad_model_revision=vad_model_revision,
)

if __name__ == "__main__":
    res = only_asr("/root/code/GPT-SoVITS/tmp/fz/gen/1.wav")
    print(res)
