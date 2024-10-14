import json
import os
import shutil
import sys
import torch
import yaml
from tools import my_utils
from subprocess import Popen
import traceback
import platform
import psutil
import signal
# from tools.asr.funasr_asr import execute_asr


from config import (
    python_exec,
    is_half,
    exp_root,
)

version = "v2"
os.environ["version"] = version
now_dir = os.getcwd()
sys.path.insert(0, now_dir)

SoVITS_weight_root = ["SoVITS_weights_v2", "SoVITS_weights"]
GPT_weight_root = ["GPT_weights_v2", "GPT_weights"]
for root in SoVITS_weight_root + GPT_weight_root:
    os.makedirs(root, exist_ok=True)


# 判断是否有能用来训练和加速推理的N卡
ok_gpu_keywords = {
    "10",
    "16",
    "20",
    "30",
    "40",
    "A2",
    "A3",
    "A4",
    "P4",
    "A50",
    "500",
    "A60",
    "70",
    "80",
    "90",
    "M4",
    "T4",
    "TITAN",
    "L4",
    "4060",
    "H",
}
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
ps1abc = []


set_gpu_numbers = set()
if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(value in gpu_name.upper() for value in ok_gpu_keywords):
            # A10#A100#V100#A40#P40#M40#K80#A4500
            if_gpu_ok = True  # 至少有一张能用的N卡
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            set_gpu_numbers.add(i)
            mem.append(
                int(
                    torch.cuda.get_device_properties(i).total_memory
                    / 1024
                    / 1024
                    / 1024
                    + 0.4
                )
            )

default_gpu_numbers = str(sorted(list(set_gpu_numbers))[0])

system = platform.system()


def check_for_exists(file_list=None, is_train=False, is_dataset_processing=False):
    missing_files = []
    if is_train is True and file_list:
        file_list.append(os.path.join(file_list[0], "2-name2text.txt"))
        file_list.append(os.path.join(file_list[0], "3-bert"))
        file_list.append(os.path.join(file_list[0], "4-cnhubert"))
        file_list.append(os.path.join(file_list[0], "5-wav32k"))
        file_list.append(os.path.join(file_list[0], "6-name2semantic.tsv"))
    for file in file_list:
        if os.path.exists(file):
            pass
        else:
            missing_files.append(file)
    if missing_files:
        print("缺少文件：", missing_files)


def fix_gpu_number(input):  # 将越界的number强制改到界内
    try:
        if int(input) not in set_gpu_numbers:
            return default_gpu_numbers
    except:
        return input
    return input


def kill_proc_tree(pid, including_parent=True):
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        # Process already terminated
        return

    children = parent.children(recursive=True)
    for child in children:
        try:
            os.kill(child.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass
    if including_parent:
        try:
            os.kill(parent.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass


def kill_process(pid):
    if system == "Windows":
        cmd = "taskkill /t /f /pid %s" % pid
        os.system(cmd)
    else:
        kill_proc_tree(pid)


def close1abc():
    global ps1abc
    if ps1abc != []:
        for p1abc in ps1abc:
            try:
                kill_process(p1abc.pid)
            except:
                traceback.print_exc()
        ps1abc = []
    return (
        "已终止所有一键三连进程",
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


def open1abc(
    inp_text,
    inp_wav_dir,
    exp_name,
    gpu_numbers1a,
    gpu_numbers1Ba,
    gpu_numbers1c,
    bert_pretrained_dir,
    ssl_pretrained_dir,
    pretrained_s2G_path,
):
    global ps1abc
    inp_text = my_utils.clean_path(inp_text)
    inp_wav_dir = my_utils.clean_path(inp_wav_dir)
    check_for_exists([inp_text, inp_wav_dir])
    if ps1abc == []:
        opt_dir = "%s/%s" % (exp_root, exp_name)
        try:
            #############################1a
            path_text = "%s/2-name2text.txt" % opt_dir
            if os.path.exists(path_text) == False or (
                os.path.exists(path_text) == True
                and len(
                    open(path_text, "r", encoding="utf8").read().strip("\n").split("\n")
                )
                < 2
            ):
                config = {
                    "inp_text": inp_text,
                    "inp_wav_dir": inp_wav_dir,
                    "exp_name": exp_name,
                    "opt_dir": opt_dir,
                    "bert_pretrained_dir": bert_pretrained_dir,
                    "is_half": str(is_half),
                }
                gpu_names = gpu_numbers1a.split("-")
                all_parts = len(gpu_names)
                for i_part in range(all_parts):
                    config.update(
                        {
                            "i_part": str(i_part),
                            "all_parts": str(all_parts),
                            "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                        }
                    )
                    os.environ.update(config)
                    cmd = '"%s" GPT_SoVITS/prepare_datasets/1-get-text.py' % python_exec
                    print(cmd)
                    p = Popen(cmd, shell=True)
                    ps1abc.append(p)
                yield (
                    "进度：1a-ing",
                    {"__type__": "update", "visible": False},
                    {"__type__": "update", "visible": True},
                )
                for p in ps1abc:
                    p.wait()

                opt = []
                for i_part in range(
                    all_parts
                ):  # txt_path="%s/2-name2text-%s.txt"%(opt_dir,i_part)
                    txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
                    with open(txt_path, "r", encoding="utf8") as f:
                        opt += f.read().strip("\n").split("\n")
                    os.remove(txt_path)
                with open(path_text, "w", encoding="utf8") as f:
                    f.write("\n".join(opt) + "\n")
                assert len("".join(opt)) > 0, "1Aa-文本获取进程失败"
            yield (
                "进度：1a-done",
                {"__type__": "update", "visible": False},
                {"__type__": "update", "visible": True},
            )
            ps1abc = []
            #############################1b
            config = {
                "inp_text": inp_text,
                "inp_wav_dir": inp_wav_dir,
                "exp_name": exp_name,
                "opt_dir": opt_dir,
                "cnhubert_base_dir": ssl_pretrained_dir,
            }
            gpu_names = gpu_numbers1Ba.split("-")
            all_parts = len(gpu_names)
            for i_part in range(all_parts):
                config.update(
                    {
                        "i_part": str(i_part),
                        "all_parts": str(all_parts),
                        "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                    }
                )
                os.environ.update(config)
                cmd = (
                    '"%s" GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py'
                    % python_exec
                )
                print(cmd)
                p = Popen(cmd, shell=True)
                ps1abc.append(p)
            yield (
                "进度：1a-done, 1b-ing",
                {"__type__": "update", "visible": False},
                {"__type__": "update", "visible": True},
            )
            for p in ps1abc:
                p.wait()
            yield (
                "进度：1a1b-done",
                {"__type__": "update", "visible": False},
                {"__type__": "update", "visible": True},
            )
            ps1abc = []
            #############################1c
            path_semantic = "%s/6-name2semantic.tsv" % opt_dir
            if os.path.exists(path_semantic) == False or (
                os.path.exists(path_semantic) == True
                and os.path.getsize(path_semantic) < 31
            ):
                config = {
                    "inp_text": inp_text,
                    "exp_name": exp_name,
                    "opt_dir": opt_dir,
                    "pretrained_s2G": pretrained_s2G_path,
                    "s2config_path": "GPT_SoVITS/configs/s2.json",
                }
                gpu_names = gpu_numbers1c.split("-")
                all_parts = len(gpu_names)
                for i_part in range(all_parts):
                    config.update(
                        {
                            "i_part": str(i_part),
                            "all_parts": str(all_parts),
                            "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                        }
                    )
                    os.environ.update(config)
                    cmd = (
                        '"%s" GPT_SoVITS/prepare_datasets/3-get-semantic.py'
                        % python_exec
                    )
                    print(cmd)
                    p = Popen(cmd, shell=True)
                    ps1abc.append(p)
                yield (
                    "进度：1a1b-done, 1cing",
                    {"__type__": "update", "visible": False},
                    {"__type__": "update", "visible": True},
                )
                for p in ps1abc:
                    p.wait()

                opt = ["item_name\tsemantic_audio"]
                for i_part in range(all_parts):
                    semantic_path = "%s/6-name2semantic-%s.tsv" % (opt_dir, i_part)
                    with open(semantic_path, "r", encoding="utf8") as f:
                        opt += f.read().strip("\n").split("\n")
                    os.remove(semantic_path)
                with open(path_semantic, "w", encoding="utf8") as f:
                    f.write("\n".join(opt) + "\n")
                yield (
                    "进度：all-done",
                    {"__type__": "update", "visible": False},
                    {"__type__": "update", "visible": True},
                )
            ps1abc = []
            yield (
                "一键三连进程结束",
                {"__type__": "update", "visible": True},
                {"__type__": "update", "visible": False},
            )
        except:
            traceback.print_exc()
            close1abc()
            yield (
                "一键三连中途报错",
                {"__type__": "update", "visible": True},
                {"__type__": "update", "visible": False},
            )
    else:
        yield (
            "已有正在进行的一键三连任务，需先终止才能开启下一次任务",
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
        )


def train_abc(model_list):
    for name in model_list:
        inp_text = f"/root/code/GPT-SoVITS/output/asr_opt/{name}.list"
        inp_wav_dir = f"/root/code/GPT-SoVITS/output/{name}"
        exp_name = name
        gpu_numbers1a = "0-0"
        gpu_numbers1Ba = "0-0"
        gpu_numbers1c = "0-0"
        bert_pretrained_dir = (
            "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
        )
        ssl_pretrained_dir = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
        pretrained_s2G_path = (
            "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
        )

        print(f"当前处理模型：{name}")

        for status in open1abc(
            inp_text,
            inp_wav_dir,
            exp_name,
            gpu_numbers1a,
            gpu_numbers1Ba,
            gpu_numbers1c,
            bert_pretrained_dir,
            ssl_pretrained_dir,
            pretrained_s2G_path,
        ):
            print(status)

        print(f"{name} 一键三连结束")


def fix_gpu_numbers(inputs):
    output = []
    try:
        for input in inputs.split(","):
            output.append(str(fix_gpu_number(input)))
        return ",".join(output)
    except:
        return inputs


torch.manual_seed(233333)
tmp = os.path.join(now_dir, "TEMP")
os.makedirs(tmp, exist_ok=True)
os.environ["TEMP"] = tmp
if os.path.exists(tmp):
    for name in os.listdir(tmp):
        if name == "jieba.cache":
            continue
        path = "%s/%s" % (tmp, name)
        delete = os.remove if os.path.isfile(path) else shutil.rmtree
        try:
            delete(path)
        except Exception as e:
            print(str(e))
            pass

p_train_SoVITS = None


# SoVITS训练：batch_size:4 total_epoch:25 exp_name:艾尔海森 text_low_lr_rate:0.4 if_save_latest:True if_save_every_weights:True save_every_epoch:25 gpu_numbers1Ba:0 pretrained_s2G:GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth pretrained_s2D:GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth
def open1Ba(
    batch_size,
    total_epoch,
    exp_name,
    text_low_lr_rate,
    if_save_latest,
    if_save_every_weights,
    save_every_epoch,
    gpu_numbers1Ba,
    pretrained_s2G,
    pretrained_s2D,
):
    global p_train_SoVITS
    if p_train_SoVITS is None:
        with open("GPT_SoVITS/configs/s2.json") as f:
            data = f.read()
            data = json.loads(data)
        s2_dir = "%s/%s" % (exp_root, exp_name)
        os.makedirs("%s/logs_s2" % (s2_dir), exist_ok=True)
        check_for_exists([s2_dir], is_train=True)
        if is_half is False:
            data["train"]["fp16_run"] = False
            batch_size = max(1, batch_size // 2)
        data["train"]["batch_size"] = batch_size
        data["train"]["epochs"] = total_epoch
        data["train"]["text_low_lr_rate"] = text_low_lr_rate
        data["train"]["pretrained_s2G"] = pretrained_s2G
        data["train"]["pretrained_s2D"] = pretrained_s2D
        data["train"]["if_save_latest"] = if_save_latest
        data["train"]["if_save_every_weights"] = if_save_every_weights
        data["train"]["save_every_epoch"] = save_every_epoch
        data["train"]["gpu_numbers"] = gpu_numbers1Ba
        data["model"]["version"] = version
        data["data"]["exp_dir"] = data["s2_ckpt_dir"] = s2_dir
        data["save_weight_dir"] = SoVITS_weight_root[-int(version[-1]) + 2]
        data["name"] = exp_name
        data["version"] = version
        tmp_config_path = "%s/tmp_s2.json" % tmp
        with open(tmp_config_path, "w") as f:
            f.write(json.dumps(data))

        cmd = '"%s" GPT_SoVITS/s2_train.py --config "%s"' % (
            python_exec,
            tmp_config_path,
        )
        yield (
            "SoVITS训练开始：%s" % cmd,
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
        )
        print(cmd)
        p_train_SoVITS = Popen(cmd, shell=True)
        p_train_SoVITS.wait()
        p_train_SoVITS = None
        yield (
            "SoVITS训练完成",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )
    else:
        yield (
            "已有正在进行的SoVITS训练任务，需先终止才能开启下一次任务",
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
        )


p_train_GPT = None


# GPT训练：batch_size:4 total_epoch:30 exp_name:艾尔海森 if_dpo:False if_save_latest:True if_save_every_weights:True save_every_epoch:30 gpu_numbers:0 pretrained_s1:GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt
def open1Bb(
    batch_size,
    total_epoch,
    exp_name,
    if_dpo,
    if_save_latest,
    if_save_every_weights,
    save_every_epoch,
    gpu_numbers,
    pretrained_s1,
):
    global p_train_GPT
    if p_train_GPT is None:
        with open(
            "GPT_SoVITS/configs/s1longer.yaml"
            if version == "v1"
            else "GPT_SoVITS/configs/s1longer-v2.yaml"
        ) as f:
            data = f.read()
            data = yaml.load(data, Loader=yaml.FullLoader)
        s1_dir = "%s/%s" % (exp_root, exp_name)
        os.makedirs("%s/logs_s1" % (s1_dir), exist_ok=True)
        check_for_exists([s1_dir], is_train=True)
        if is_half is False:
            data["train"]["precision"] = "32"
            batch_size = max(1, batch_size // 2)
        data["train"]["batch_size"] = batch_size
        data["train"]["epochs"] = total_epoch
        data["pretrained_s1"] = pretrained_s1
        data["train"]["save_every_n_epoch"] = save_every_epoch
        data["train"]["if_save_every_weights"] = if_save_every_weights
        data["train"]["if_save_latest"] = if_save_latest
        data["train"]["if_dpo"] = if_dpo
        data["train"]["half_weights_save_dir"] = GPT_weight_root[-int(version[-1]) + 2]
        data["train"]["exp_name"] = exp_name
        data["train_semantic_path"] = "%s/6-name2semantic.tsv" % s1_dir
        data["train_phoneme_path"] = "%s/2-name2text.txt" % s1_dir
        data["output_dir"] = "%s/logs_s1" % s1_dir
        # data["version"]=version

        os.environ["_CUDA_VISIBLE_DEVICES"] = fix_gpu_numbers(
            gpu_numbers.replace("-", ",")
        )
        os.environ["hz"] = "25hz"
        tmp_config_path = "%s/tmp_s1.yaml" % tmp
        with open(tmp_config_path, "w") as f:
            f.write(yaml.dump(data, default_flow_style=False))
        # cmd = '"%s" GPT_SoVITS/s1_train.py --config_file "%s" --train_semantic_path "%s/6-name2semantic.tsv" --train_phoneme_path "%s/2-name2text.txt" --output_dir "%s/logs_s1"'%(python_exec,tmp_config_path,s1_dir,s1_dir,s1_dir)
        cmd = '"%s" GPT_SoVITS/s1_train.py --config_file "%s" ' % (
            python_exec,
            tmp_config_path,
        )
        yield (
            "GPT训练开始：%s" % cmd,
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
        )
        print(cmd)
        p_train_GPT = Popen(cmd, shell=True)
        p_train_GPT.wait()
        p_train_GPT = None
        yield (
            "GPT训练完成",
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )
    else:
        yield (
            "已有正在进行的GPT训练任务，需先终止才能开启下一次任务",
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
        )


def train_1Bb(model_list):
    # GPT训练：batch_size:4 total_epoch:30 exp_name:艾尔海森 if_dpo:False if_save_latest:True if_save_every_weights:True save_every_epoch:30 gpu_numbers:0 pretrained_s1:GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt
    # def open1Bb
    for name in model_list:
        print(f"GPT训练：{name}")
        batch = 2
        epoch = 40
        exp_name = name
        dpo = False
        save_latest = True
        save_every_weights = True
        save_every_epoch = 40
        gpu_numbers = "0"
        pretrained_s1 = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
        for status in open1Bb(
            batch,
            epoch,
            exp_name,
            dpo,
            save_latest,
            save_every_weights,
            save_every_epoch,
            gpu_numbers,
            pretrained_s1,
        ):
            print(status)

        print(f"GPT: {name} done!")


# SoVITS训练：batch_size:4 total_epoch:25 exp_name:艾尔海森 text_low_lr_rate:0.4 if_save_latest:True if_save_every_weights:True save_every_epoch:25 gpu_numbers1Ba:0 pretrained_s2G:GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth pretrained_s2D:GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth
# def open1Ba(
def train_1Ba(model_list):
    for name in model_list:
        batch_size = 2
        total_epoch = 24
        exp_name = name
        text_low_lr_rate = 0.4
        if_save_latest = True
        if_save_every_weights = True
        save_every_epoch = 24
        gpu_numbers1Ba = "0"
        pretrained_s2G = (
            "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
        )
        pretrained_s2D = (
            "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth"
        )
        for status in open1Ba(
            batch_size,
            total_epoch,
            exp_name,
            text_low_lr_rate,
            if_save_latest,
            if_save_every_weights,
            save_every_epoch,
            gpu_numbers1Ba,
            pretrained_s2G,
            pretrained_s2D,
        ):
            print(status)

        print(f"sovits: 训练完成{name}")


if __name__ == "__main__":
    # 获取 output 目录下除开 asr_opt 的其他目录名
    output_dir = "/root/code/GPT-SoVITS/output"
    model_list = [
        name
        for name in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, name)) and name != "asr_opt"
    ]
    # # 处理数据
    train_abc(model_list)
    # model_list = ["丹恒"]
    # 训练 SoVITS
    train_1Ba(model_list)
    # 训练 GPT
    train_1Bb(model_list)
