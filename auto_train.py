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
from loguru import logger

# from tools.asr.funasr_asr import execute_asr
from tools.my_utils import check_details, check_for_existance


from config import (
    python_exec,
    is_half,
    exp_root,
)

version = "v4"
os.environ["version"] = version
now_dir = os.getcwd()
sys.path.insert(0, now_dir)

SoVITS_weight_root = [
    "SoVITS_weights",
    "SoVITS_weights_v2",
    "SoVITS_weights_v3",
    "SoVITS_weights_v4",
]
GPT_weight_root = ["GPT_weights", "GPT_weights_v2", "GPT_weights_v3", "GPT_weights_v4"]
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
        logger.info("缺少文件：", missing_files)


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
    if check_for_existance([inp_text, inp_wav_dir], is_dataset_processing=True):
        check_details([inp_text, inp_wav_dir], is_dataset_processing=True)
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
                    logger.info(cmd)
                    p = Popen(cmd, shell=True)
                    ps1abc.append(p)
                yield (
                    # i18n("进度") + ": 1A-Doing",
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
                # assert len("".join(opt)) > 0, process_info(process_name_1a, "failed")
            yield (
                # i18n("进度") + ": 1A-Done",
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
                logger.info(cmd)
                p = Popen(cmd, shell=True)
                ps1abc.append(p)
            yield (
                # i18n("进度") + ": 1A-Done, 1B-Doing",
                {"__type__": "update", "visible": False},
                {"__type__": "update", "visible": True},
            )
            for p in ps1abc:
                p.wait()
            yield (
                # i18n("进度") + ": 1A-Done, 1B-Done",
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
                    logger.info(cmd)
                    p = Popen(cmd, shell=True)
                    ps1abc.append(p)
                yield (
                    # i18n("进度") + ": 1A-Done, 1B-Done, 1C-Doing",
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
                    # i18n("进度") + ": 1A-Done, 1B-Done, 1C-Done",
                    {"__type__": "update", "visible": False},
                    {"__type__": "update", "visible": True},
                )
            ps1abc = []
            yield (
                # process_info(process_name_1abc, "finish"),
                {"__type__": "update", "visible": True},
                {"__type__": "update", "visible": False},
            )
        except:
            traceback.print_exc()
            close1abc()
            yield (
                # process_info(process_name_1abc, "failed"),
                {"__type__": "update", "visible": True},
                {"__type__": "update", "visible": False},
            )
    else:
        yield (
            # process_info(process_name_1abc, "occupy"),
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
        pretrained_s2G_path = "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth"

        logger.info(f"当前处理模型：{name}")

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
            logger.info(status)

        logger.info(f"{name} 一键三连结束")


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
            logger.info(str(e))
            pass

p_train_SoVITS = None


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
    if_grad_ckpt,
    lora_rank,
):
    global p_train_SoVITS
    if p_train_SoVITS == None:
        with open("GPT_SoVITS/configs/s2.json") as f:
            data = f.read()
            data = json.loads(data)
        s2_dir = "%s/%s" % (exp_root, exp_name)
        os.makedirs("%s/logs_s2_%s" % (s2_dir, version), exist_ok=True)
        if check_for_existance([s2_dir], is_train=True):
            check_details([s2_dir], is_train=True)
        if is_half == False:
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
        data["train"]["grad_ckpt"] = if_grad_ckpt
        data["train"]["lora_rank"] = lora_rank
        data["model"]["version"] = version
        data["data"]["exp_dir"] = data["s2_ckpt_dir"] = s2_dir
        data["save_weight_dir"] = SoVITS_weight_root[int(version[-1]) - 1]
        data["name"] = exp_name
        data["version"] = version
        tmp_config_path = "%s/tmp_s2.json" % tmp
        with open(tmp_config_path, "w") as f:
            f.write(json.dumps(data))
        if version in ["v1", "v2"]:
            cmd = '"%s" GPT_SoVITS/s2_train.py --config "%s"' % (
                python_exec,
                tmp_config_path,
            )
        else:
            cmd = '"%s" GPT_SoVITS/s2_train_v3_lora.py --config "%s"' % (
                python_exec,
                tmp_config_path,
            )
        yield ({"log_status": "begin"})
        logger.info(cmd)
        p_train_SoVITS = Popen(cmd, shell=True)
        p_train_SoVITS.wait()
        if p_train_SoVITS.returncode != 0:
            logger.warning(f"进程异常退出，代码 {p_train_SoVITS.returncode}")
            yield ({"log_status": "error"})
        p_train_SoVITS = None
        # SoVITS_dropdown_update, GPT_dropdown_update = change_choices()
        yield ({"log_status": "success"})
    else:
        yield ({"log_status": "waitting"})


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
    if p_train_GPT == None:
        with open(
            "GPT_SoVITS/configs/s1longer.yaml"
            if version == "v1"
            else "GPT_SoVITS/configs/s1longer-v2.yaml"
        ) as f:
            data = f.read()
            data = yaml.load(data, Loader=yaml.FullLoader)
        s1_dir = "%s/%s" % (exp_root, exp_name)
        os.makedirs("%s/logs_s1" % (s1_dir), exist_ok=True)
        if check_for_existance([s1_dir], is_train=True):
            check_details([s1_dir], is_train=True)
        if is_half == False:
            data["train"]["precision"] = "32"
            batch_size = max(1, batch_size // 2)
        data["train"]["batch_size"] = batch_size
        data["train"]["epochs"] = total_epoch
        data["pretrained_s1"] = pretrained_s1
        data["train"]["save_every_n_epoch"] = save_every_epoch
        data["train"]["if_save_every_weights"] = if_save_every_weights
        data["train"]["if_save_latest"] = if_save_latest
        data["train"]["if_dpo"] = if_dpo
        data["train"]["half_weights_save_dir"] = GPT_weight_root[int(version[-1]) - 1]
        data["train"]["exp_name"] = exp_name
        data["train_semantic_path"] = "%s/6-name2semantic.tsv" % s1_dir
        data["train_phoneme_path"] = "%s/2-name2text.txt" % s1_dir
        data["output_dir"] = "%s/logs_s1_%s" % (s1_dir, version)
        # data["version"]=version

        os.environ["_CUDA_VISIBLE_DEVICES"] = fix_gpu_numbers(
            gpu_numbers.replace("-", ",")
        )
        os.environ["hz"] = "25hz"
        tmp_config_path = "%s/tmp_s1.yaml" % tmp
        with open(tmp_config_path, "w") as f:
            f.write(yaml.dump(data, default_flow_style=False))
        cmd = '"%s" GPT_SoVITS/s1_train.py --config_file "%s" ' % (
            python_exec,
            tmp_config_path,
        )
        yield ({"log_status": "begin"})

        logger.info(cmd)
        p_train_GPT = Popen(cmd, shell=True)
        p_train_GPT.wait()
        if p_train_GPT.returncode != 0:
            logger.warning(f"进程异常退出，代码 {p_train_GPT.returncode}")
            yield ({"log_status": "error"})
        p_train_GPT = None
        yield ({"log_status": "success"})
    else:
        yield ({"log_status": "waiting"})


# GPT训练：batch_size:4 total_epoch:30 exp_name:艾尔海森 if_dpo:False if_save_latest:True if_save_every_weights:True save_every_epoch:30 gpu_numbers:0 pretrained_s1:GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt
def train_1Bb(model_list):
    idx = 0
    for name in model_list:
        logger.info(f"""
##################
GPT训练：{name}
已训练：{idx} 个模型
还有：{len(model_list) - idx} 个模型未训练
##################
""")

        original_batch = 8  # 初始batch_size
        current_batch = original_batch
        max_retries = 4
        retry_count = 0
        success = False
        while retry_count <= max_retries and not success:
            # 设置训练参数（每次重试都需要重新设置）
            epoch = 15
            exp_name = name
            dpo = False
            save_latest = True
            save_every_weights = True
            save_every_epoch = 15
            gpu_numbers = "0"
            pretrained_s1 = "GPT_SoVITS/pretrained_models/s1v3.ckpt"
            retry_flag = False
            for status in open1Bb(
                current_batch,
                epoch,
                exp_name,
                dpo,
                save_latest,
                save_every_weights,
                save_every_epoch,
                gpu_numbers,
                pretrained_s1,
            ):
                logger.info(status)
                # 检测到需要降低batch的指令
                if status.get("log_status") == "error":
                    retry_flag = True

            # 结果处理
            if not retry_flag:
                success = True
                logger.info(f"sovits: 训练完成{name}")
            else:
                if current_batch == 1:  # 已经是允许的最小值
                    logger.error(f"sovits: 训练失败{name}，batch_size已降至最低仍失败")
                    break

                current_batch = max(1, current_batch // 2)  # 每次降半
                retry_count += 1
                logger.warning(
                    f"第{retry_count}次重试，batch_size调整为: {current_batch}"
                )

                if retry_count > max_retries:
                    logger.error(
                        f"sovits: 训练失败{name}，超过最大重试次数{max_retries}"
                    )

        if not success:
            logger.error(f"sovits: 最终训练失败{name}")

        logger.info(f"GPT: {name} done!")
        idx += 1


# SoVITS训练：batch_size:4 total_epoch:25 exp_name:艾尔海森 text_low_lr_rate:0.4 if_save_latest:True if_save_every_weights:True save_every_epoch:25 gpu_numbers1Ba:0 pretrained_s2G:GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth pretrained_s2D:GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth
def train_1Ba(model_list):
    idx = 0
    for name in model_list:
        logger.info(f"""
#################      
SoVITS 训练：{name}
已训练：{idx} 个模型
还有：{len(model_list) - idx} 个模型未训练
#################
""")
        original_batch = 2  # 初始batch_size
        current_batch = original_batch
        max_retries = 2
        retry_count = 0
        success = False

        while retry_count <= max_retries and not success:
            # 设置训练参数（每次重试都需要重新设置）
            total_epoch = 4
            exp_name = name
            text_low_lr_rate = 0.4
            if_save_latest = True
            if_save_every_weights = True
            save_every_epoch = 4
            gpu_numbers1Ba = "0"
            pretrained_s2G = "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth"
            pretrained_s2D = "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Dv4.pth"
            if_grad_ckpt = False
            lora_rank = 32

            # 执行训练并捕获状态
            retry_flag = False
            for status in open1Ba(
                current_batch,  # 使用当前batch_size
                total_epoch,
                exp_name,
                text_low_lr_rate,
                if_save_latest,
                if_save_every_weights,
                save_every_epoch,
                gpu_numbers1Ba,
                pretrained_s2G,
                pretrained_s2D,
                if_grad_ckpt,
                lora_rank,
            ):
                logger.info(status)
                # 检测到需要降低batch的指令
                if status.get("log_status") == "error":
                    retry_flag = True

            # 结果处理
            if not retry_flag:
                success = True
                logger.info(f"sovits: 训练完成{name}")
            else:
                if current_batch == 1:  # 已经是允许的最小值
                    logger.error(f"sovits: 训练失败{name}，batch_size已降至最低仍失败")
                    break

                current_batch = max(1, current_batch // 2)  # 每次降半
                retry_count += 1
                logger.warning(
                    f"第{retry_count}次重试，batch_size调整为: {current_batch}"
                )

                if retry_count > max_retries:
                    logger.error(
                        f"sovits: 训练失败{name}，超过最大重试次数{max_retries}"
                    )

        if not success:
            logger.error(f"sovits: 最终训练失败{name}")
        idx += 1


if __name__ == "__main__":
    # 获取 output 目录下包含wav文件的子目录名（排除指定目录）
    output_dir = "/root/code/GPT-SoVITS/output"
    exclude_list = [
        "asr_opt",
    ]

    model_list = [
        name
        for name in os.listdir(output_dir)
        if (
            os.path.isdir(os.path.join(output_dir, name))  # 是目录
            and name not in exclude_list  # 不在排除列表
            and any(  # 包含至少一个wav文件
                f.endswith(".wav") for f in os.listdir(os.path.join(output_dir, name))
            )
        )
    ]

    logger.info(model_list)
    # 处理数据
    # train_abc(model_list)
    # 训练 SoVITS
    train_1Ba(model_list)
    # 训练 GPT
    train_1Bb(model_list)
