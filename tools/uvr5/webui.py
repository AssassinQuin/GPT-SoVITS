import os
import traceback, gradio as gr
import logging
from tools.i18n.i18n import I18nAuto
from tools.my_utils import clean_path

# 初始化国际化自动工具
i18n = I18nAuto()

# 配置日志记录器
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

import librosa, ffmpeg
import soundfile as sf
import torch
import sys
from mdxnet import MDXNetDereverb
from vr import AudioPre, AudioPreDeEcho
from bsroformer import BsRoformer_Loader

# 加载UVR5模型权重文件名
weight_uvr5_root = "tools/uvr5/uvr5_weights"
uvr5_names = []
for name in os.listdir(weight_uvr5_root):
    if name.endswith(".pth") or name.endswith(".ckpt") or "onnx" in name:
        uvr5_names.append(name.replace(".pth", "").replace(".ckpt", ""))

# 设置默认值
default_device = "cuda"
default_is_half = False
default_webui_port_uvr5 = 9873
default_is_share = False

# 获取命令行参数或使用默认值
device = sys.argv[1] if len(sys.argv) > 1 else default_device
is_half = eval(sys.argv[2]) if len(sys.argv) > 2 else default_is_half
webui_port_uvr5 = int(sys.argv[3]) if len(sys.argv) > 3 else default_webui_port_uvr5
is_share = eval(sys.argv[4]) if len(sys.argv) > 4 else default_is_share


# 定义UVR处理函数
def uvr(model_name, inp_root, save_root_vocal, paths, save_root_ins, agg, format0):
    infos = []
    try:
        inp_root = clean_path(inp_root)
        save_root_vocal = clean_path(save_root_vocal)
        save_root_ins = clean_path(save_root_ins)
        is_hp3 = "HP3" in model_name
        if model_name == "onnx_dereverb_By_FoxJoy":
            pre_fun = MDXNetDereverb(15)
        elif model_name == "Bs_Roformer" or "bs_roformer" in model_name.lower():
            func = BsRoformer_Loader
            pre_fun = func(
                model_path = os.path.join(weight_uvr5_root, model_name + ".ckpt"),
                device = device,
                is_half=is_half
            )
        else:
            func = AudioPre if "DeEcho" not in model_name else AudioPreDeEcho
            pre_fun = func(
                agg=int(agg),
                model_path=os.path.join(weight_uvr5_root, model_name + ".pth"),
                device=device,
                is_half=is_half,
            )
        if inp_root != "":
            paths = [os.path.join(inp_root, name) for name in os.listdir(inp_root)]
        else:
            paths = [path.name for path in paths]
        for path in paths:
            inp_path = os.path.join(inp_root, path)
            if os.path.isfile(inp_path) == False:
                continue
            need_reformat = 1
            done = 0
            try:
                info = ffmpeg.probe(inp_path, cmd="ffprobe")
                if (
                    info["streams"][0]["channels"] == 2
                    and info["streams"][0]["sample_rate"] == "44100"
                ):
                    need_reformat = 0
                    pre_fun._path_audio_(
                        inp_path, save_root_ins, save_root_vocal, format0, is_hp3
                    )
                    done = 1
            except:
                need_reformat = 1
                logger.error("音频信息获取失败，需重新格式化", exc_info=True)
            if need_reformat == 1:
                tmp_path = "%s/%s.reformatted.wav" % (
                    os.path.join(os.environ["TEMP"]),
                    os.path.basename(inp_path),
                )
                os.system(
                    f'ffmpeg -i "{inp_path}" -vn -acodec pcm_s16le -ac 2 -ar 44100 "{tmp_path}" -y'
                )
                inp_path = tmp_path
            try:
                if done == 0:
                    pre_fun._path_audio_(
                        inp_path, save_root_ins, save_root_vocal, format0, is_hp3
                    )
                infos.append("%s->Success" % (os.path.basename(inp_path)))
                yield "\n".join(infos)
            except:
                infos.append(
                    "%s->%s" % (os.path.basename(inp_path), traceback.format_exc())
                )
                logger.error("处理音频时出错", exc_info=True)
                yield "\n".join(infos)
    except:
        infos.append(traceback.format_exc())
        logger.error("处理过程中发生错误", exc_info=True)
        yield "\n".join(infos)
    finally:
        try:
            if model_name == "onnx_dereverb_By_FoxJoy":
                del pre_fun.pred.model
                del pre_fun.pred.model_
            else:
                del pre_fun.model
                del pre_fun
        except:
            logger.error("清理模型缓存时出错", exc_info=True)
        logger.info("清理GPU缓存")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    yield "\n".join(infos)


# 使用Gradio构建WebUI
with gr.Blocks(title="UVR5 WebUI") as app:
    gr.Markdown(
        value=i18n(
            "本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责. <br>如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录<b>LICENSE</b>."
        )
    )
    with gr.Tabs():
        with gr.TabItem(i18n("伴奏人声分离&去混响&去回声")):
            with gr.Group():
                gr.Markdown(
                    value=i18n("人声伴奏分离批量处理， 使用UVR5模型。") + "<br>" + \
                        i18n("合格的文件夹路径格式举例： E:\\codes\\py39\\vits_vc_gpu\\白鹭霜华测试样例(去文件管理器地址栏拷就行了)。")+ "<br>" + \
                        i18n("模型分为三类：") + "<br>" + \
                        i18n("1、保留人声：不带和声的音频选这个，对主人声保留比HP5更好。内置HP2和HP3两个模型，HP3可能轻微漏伴奏但对主人声保留比HP2稍微好一丁点；") + "<br>" + \
                        i18n("2、仅保留主人声：带和声的音频选这个，对主人声可能有削弱。内置HP5一个模型；") + "<br>" + \
                        i18n("3、去混响、去延迟模型（by FoxJoy）：") + "<br>  " + \
                        i18n("(1)MDX-Net(onnx_dereverb):对于双通道混响是最好的选择，不能去除单通道混响；") + "<br>&emsp;" + \
                        i18n("(234)DeEcho:去除延迟效果。Aggressive比Normal去除得更彻底，DeReverb额外去除混响，可去除单声道混响，但是对高频重的板式混响去不干净。") + "<br>" + \
                        i18n("去混响/去延迟，附：") + "<br>" + \
                        i18n("1、DeEcho-DeReverb模型的耗时是另外2个DeEcho模型的接近2倍；") + "<br>" + \
                        i18n("2、MDX-Net-Dereverb模型挺慢的；") + "<br>" + \
                        i18n("3、个人推荐的最干净的配置是先MDX-Net再DeEcho-Aggressive。")
                )
                with gr.Row():
                    with gr.Column():
                        dir_wav_input = gr.Textbox(
                            label=i18n("输入待处理音频文件夹路径"),
                            placeholder="C:\\Users\\Desktop\\todo-songs",
                        )
                        wav_inputs = gr.File(
                            file_count="multiple",
                            label=i18n("也可批量输入音频文件, 二选一, 优先读文件夹"),
                        )
                    with gr.Column():
                        model_choose = gr.Dropdown(
                            label=i18n("模型"), choices=uvr5_names
                        )
                        agg = gr.Slider(
                            minimum=0,
                            maximum=20,
                            step=1,
                            label=i18n("人声提取激进程度"),
                            value=10,
                            interactive=True,
                            visible=False,  # 先不开放调整
                        )
                        opt_vocal_root = gr.Textbox(
                            label=i18n("指定输出主人声文件夹"), value="output/uvr5_opt"
                        )
                        opt_ins_root = gr.Textbox(
                            label=i18n("指定输出非主人声文件夹"),
                            value="output/uvr5_opt",
                        )
                        format0 = gr.Radio(
                            label=i18n("导出文件格式"),
                            choices=["wav", "flac", "mp3", "m4a"],
                            value="flac",
                            interactive=True,
                        )
                    but2 = gr.Button(i18n("转换"), variant="primary")
                    vc_output4 = gr.Textbox(label=i18n("输出信息"))
                    but2.click(
                        uvr,
                        [
                            model_choose,
                            dir_wav_input,
                            opt_vocal_root,
                            wav_inputs,
                            opt_ins_root,
                            agg,
                            format0,
                        ],
                        [vc_output4],
                        api_name="uvr_convert",
                    )
app.queue(concurrency_count=511, max_size=1022).launch(
    server_name="0.0.0.0",
    inbrowser=True,
    share=is_share,
    server_port=webui_port_uvr5,
    quiet=True,
)
