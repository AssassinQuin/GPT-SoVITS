# -*- coding: utf-8 -*-

import sys, os
import pdb, traceback, numpy as np, logging
from scipy.io import wavfile
import librosa, torch
from feature_extractor import cnhubert
from my_utils import load_audio
from time import time as ttime
import shutil

# 从环境变量中获取参数
inp_text = os.environ.get("inp_text")
inp_wav_dir = os.environ.get("inp_wav_dir")
exp_name = os.environ.get("exp_name")
i_part = os.environ.get("i_part")
all_parts = os.environ.get("all_parts")
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("_CUDA_VISIBLE_DEVICES")
opt_dir = os.environ.get("opt_dir")
cnhubert.cnhubert_base_path = os.environ.get("cnhubert_base_dir")
is_half = eval(os.environ.get("is_half", "True"))

# 设置目录
hubert_dir = f"{opt_dir}/4-cnhubert"
wav32dir = f"{opt_dir}/5-wav32k"
os.makedirs(opt_dir, exist_ok=True)
os.makedirs(hubert_dir, exist_ok=True)
os.makedirs(wav32dir, exist_ok=True)


# 定义保存函数解决中文路径问题
def my_save(fea, path):
    dir = os.path.dirname(path)
    name = os.path.basename(path)
    tmp_path = f"{ttime()}{i_part}.pth"
    torch.save(fea, tmp_path)
    shutil.move(tmp_path, f"{dir}/{name}")


# 设备选择
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

# 加载模型
model = cnhubert.get_model()
if is_half:
    model = model.half().to(device)
else:
    model = model.to(device)

nan_fails = []


def name2go(wav_name, wav_path):
    hubert_path = f"{hubert_dir}/{wav_name}.pt"
    if os.path.exists(hubert_path):
        return

    tmp_audio = load_audio(wav_path, 32000)
    tmp_max = np.abs(tmp_audio).max()

    if tmp_max > 2.2:
        print(f"{wav_name}-filtered,{tmp_max}")
        return

    tmp_audio32 = (tmp_audio / tmp_max * (0.95 * 0.5 * 32768)) + (
        (1 - 0.5) * 32768
    ) * tmp_audio
    tmp_audio32b = (tmp_audio / tmp_max * (0.95 * 0.5 * 1145.14)) + (
        (1 - 0.5) * 1145.14
    ) * tmp_audio

    tmp_audio = librosa.resample(tmp_audio32b, orig_sr=32000, target_sr=16000)
    tensor_wav16 = torch.from_numpy(tmp_audio)

    if is_half:
        tensor_wav16 = tensor_wav16.half().to(device)
    else:
        tensor_wav16 = tensor_wav16.to(device)

    ssl = (
        model.model(tensor_wav16.unsqueeze(0))["last_hidden_state"]
        .transpose(1, 2)
        .cpu()
    )

    if np.isnan(ssl.detach().numpy()).sum() != 0:
        nan_fails.append((wav_name, wav_path))
        print(f"nan filtered: {wav_name}")
        return

    wavfile.write(f"{wav32dir}/{wav_name}", 32000, tmp_audio32.astype("int16"))
    my_save(ssl, hubert_path)


with open(inp_text, "r", encoding="utf8") as f:
    lines = f.read().strip("\n").split("\n")

for line in lines[int(i_part) :: int(all_parts)]:
    try:
        wav_name, spk_name, language, text = line.split("|")

        if inp_wav_dir:
            wav_name = os.path.basename(wav_name)
            wav_path = f"{inp_wav_dir}/{wav_name}"
        else:
            wav_path = wav_name
            wav_name = os.path.basename(wav_name)

        name2go(wav_name, wav_path)
    except:
        print(line, traceback.format_exc())

if len(nan_fails) > 0 and is_half:
    is_half = False
    model = model.float().to(device)
    for wav in nan_fails:
        try:
            name2go(wav[0], wav[1])
        except:
            print(wav_name, traceback.format_exc())
