import os
import math
import traceback
import multiprocessing
import sys
import pdb
from random import shuffle
import torch.multiprocessing as mp
from glob import glob
from tqdm import tqdm
import logging
import librosa
import utils
import torch
from module.models import SynthesizerTrn

logging.getLogger("numba").setLevel(logging.WARNING)

# 从环境变量中获取参数
inp_text = os.environ.get("inp_text")
exp_name = os.environ.get("exp_name")
i_part = os.environ.get("i_part")
all_parts = os.environ.get("all_parts")
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("_CUDA_VISIBLE_DEVICES")
opt_dir = os.environ.get("opt_dir")
pretrained_s2G = os.environ.get("pretrained_s2G")
s2config_path = os.environ.get("s2config_path")
is_half = eval(os.environ.get("is_half", "True"))

# 设置目录
hubert_dir = f"{opt_dir}/4-cnhubert"
semantic_path = f"{opt_dir}/6-name2semantic-{i_part}.tsv"

if not os.path.exists(semantic_path):
    os.makedirs(opt_dir, exist_ok=True)

    # 设备选择
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # 加载配置和模型
    hps = utils.get_hparams_from_file(s2config_path)
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    )

    if is_half:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)

    vq_model.eval()

    # 加载预训练模型
    state_dict = torch.load(pretrained_s2G, map_location="cpu")
    print(vq_model.load_state_dict(state_dict["weight"], strict=False))

    def name2go(wav_name, lines):
        hubert_path = f"{hubert_dir}/{wav_name}.pt"
        if not os.path.exists(hubert_path):
            return

        ssl_content = torch.load(hubert_path, map_location="cpu")

        if is_half:
            ssl_content = ssl_content.half().to(device)
        else:
            ssl_content = ssl_content.to(device)

        codes = vq_model.extract_latent(ssl_content)
        semantic = " ".join([str(i) for i in codes[0, 0, :].tolist()])
        lines.append(f"{wav_name}\t{semantic}")

    with open(inp_text, "r", encoding="utf8") as f:
        lines = f.read().strip("\n").split("\n")

    lines1 = []
    for line in lines[int(i_part) :: int(all_parts)]:
        try:
            wav_name, spk_name, language, text = line.split("|")
            wav_name = os.path.basename(wav_name)
            name2go(wav_name, lines1)
        except:
            print(line, traceback.format_exc())

    with open(semantic_path, "w", encoding="utf8") as f:
        f.write("\n".join(lines1))
