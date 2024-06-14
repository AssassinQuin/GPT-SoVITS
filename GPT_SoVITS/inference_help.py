import re
import torch
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from module.mel_processing import spectrogram_torch
from my_utils import load_audio
from tools.i18n.i18n import I18nAuto

# 多语言
i18n = I18nAuto()

punctuation = set(["!", "?", "…", ",", ".", "-", " "])


# 定义语言对应的符号
language_punctuation = {
    "en": {
        "comma": ",",
        "period": ".",
        "newline": ".",
        "question_mark": "?",
        "exclamation_mark": "!",
        "ellipsis": "...",
        "tilde": "~",
        "colon": ":",
    },
    "zh": {
        "comma": "，",
        "period": "。",
        "newline": "。",
        "question_mark": "？",
        "exclamation_mark": "！",
        "ellipsis": "…",
        "tilde": "～",
        "colon": "：",
    },
}

dict_language = {
    i18n("中文"): "all_zh",  # 全部按中文识别
    i18n("英文"): "en",  # 全部按英文识别#######不变
    i18n("日文"): "all_ja",  # 全部按日文识别
    i18n("中英混合"): "zh",  # 按中英混合识别####不变
    i18n("日英混合"): "ja",  # 按日英混合识别####不变
    i18n("多语种混合"): "auto",  # 多语种启动切分识别语种
}

splits = {
    "，",
    "。",
    "？",
    "！",
    ",",
    ".",
    "?",
    "!",
    "~",
    ":",
    "：",
    "—",
    "…",
}


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


#
def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    return spec


# 格式化文本
def format_text(text, language="zh"):
    text = text.strip("\n")
    # 根据语言获取对应的符号
    punct = (
        language_punctuation["zh"] if "zh" in language else language_punctuation["en"]
    )

    # 替换规则
    text = re.sub(r" {2,}", punct["period"], text)  # 多个空格替换为句号
    text = re.sub(r"\n|\r", punct["newline"], text)  # 回车，换行符替换为句号
    text = re.sub(r" ", punct["comma"], text)  # 一个空格替换为逗号
    text = re.sub(r"[\"\'‘’“”]", "", text)

    # 替换所有非当前语言的符号为对应语言的符号
    if language == "en":
        text = re.sub(
            r"[，。？！…～：]",
            lambda match: (
                punct["comma"]
                if match.group(0) == "，"
                else punct["period"]
                if match.group(0) == "。"
                else punct["question_mark"]
                if match.group(0) == "？"
                else punct["exclamation_mark"]
                if match.group(0) == "！"
                else punct["ellipsis"]
                if match.group(0) == "…"
                else punct["tilde"]
                if match.group(0) == "～"
                else punct["colon"]
            ),
            text,
        )
    elif language == "zh":
        text = re.sub(
            r"[,.\?!~:]+",
            lambda match: (
                punct["comma"]
                if match.group(0) == ","
                else punct["period"]
                if match.group(0) == "."
                else punct["question_mark"]
                if match.group(0) == "?"
                else punct["exclamation_mark"]
                if match.group(0) == "!"
                else punct["ellipsis"]
                if match.group(0) == "..."
                else punct["tilde"]
                if match.group(0) == "~"
                else punct["colon"]
            ),
            text,
        )

    # 确保文本开头没有标点符号
    text = re.sub(r"^[，。？！…～：]|^[,.?!~:]", "", text)

    # 保留多个连续标点符号中的第一个
    def remove_consecutive_punctuation(text):
        result = []
        i = 0
        while i < len(text):
            if text[i] in punct.values():
                result.append(text[i])
                while i + 1 < len(text) and text[i + 1] in punct.values():
                    i += 1
            else:
                result.append(text[i])
            i += 1
        return "".join(result)

    text = remove_consecutive_punctuation(text)

    return text


#
def clean_text_inf(text, language):
    phones, word2ph, norm_text = clean_text(text, language)
    phones = cleaned_text_to_sequence(phones)
    return phones, word2ph, norm_text


#
def get_first(text):
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text


def merge_short_text_in_array(texts, threshold):
    """
    合并短文本，直到文本长度达到指定的阈值。

    参数:
    texts (list): 包含短文本的列表。
    threshold (int): 合并文本的阈值长度。

    返回:
    list: 合并后的文本列表。
    """
    # 如果文本列表长度小于2，直接返回原列表
    if len(texts) < 2:
        return texts

    result = []
    current_text = ""

    for text in texts:
        # 将当前文本添加到合并文本中
        current_text += text
        # 如果合并文本长度达到阈值，将其添加到结果列表中，并重置合并文本
        if len(current_text) >= threshold:
            result.append(current_text)
            current_text = ""

    # 处理剩余的文本
    if current_text:
        # 如果结果列表为空，直接添加剩余文本
        if not result:
            result.append(current_text)
        else:
            # 否则，将剩余文本添加到结果列表的最后一个元素中
            result[-1] += current_text

    return result


def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts


def cut1(inp):
    """
    将输入文本按每4个元素一组进行切割，并去除包含标点符号的组。

    参数:
    inp (str): 输入文本。

    返回:
    str: 切割后的文本，每组之间用换行符分隔。
    """
    inp = inp.strip("\n")
    inps = inp.split()
    split_idx = list(range(0, len(inps), 4))
    split_idx[-1] = None
    opts = [" ".join(inps[i:j]) for i, j in zip([None] + split_idx, split_idx + [None])]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut2(inp):
    """
    将输入文本按长度切割，每段不超过50个字符，并去除包含标点符号的段。

    参数:
    inp (str): 输入文本。

    返回:
    str: 切割后的文本，每段之间用换行符分隔。
    """
    inp = inp.strip("\n")
    inps = inp.split()
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += " " + inps[i] if tmp_str else inps[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str.strip())
            tmp_str = ""
    if tmp_str:
        opts.append(tmp_str.strip())
    if len(opts) > 1 and len(opts[-1]) < 50:
        opts[-2] += " " + opts[-1]
        opts = opts[:-1]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut3(inp):
    """
    将输入文本按句号进行切割，并去除包含标点符号的句。

    参数:
    inp (str): 输入文本。

    返回:
    str: 切割后的文本，每句之间用换行符分隔。
    """
    inp = inp.strip("\n")
    opts = [
        item
        for item in inp.strip("。").split("。")
        if not set(item).issubset(punctuation)
    ]
    return "\n".join(opts)


def cut4(inp):
    """
    将输入文本按英文句号进行切割，并去除包含标点符号的句。

    参数:
    inp (str): 输入文本。

    返回:
    str: 切割后的文本，每句之间用换行符分隔。
    """
    inp = inp.strip("\n")
    opts = [
        item
        for item in inp.strip(".").split(".")
        if not set(item).issubset(punctuation)
    ]
    return "\n".join(opts)


def cut5(inp):
    """
    将输入文本按标点符号进行切割，并去除包含标点符号的段。

    参数:
    inp (str): 输入文本。

    返回:
    str: 切割后的文本，每段之间用换行符分隔。
    """
    inp = inp.strip("\n")
    punds = r"[,.;?!、，。？！;：…]"
    items = re.split(f"({punds})", inp)
    mergeitems = ["".join(group) for group in zip(items[::2], items[1::2])]
    if len(items) % 2 == 1:
        mergeitems.append(items[-1])
    opt = [item for item in mergeitems if not set(item).issubset(punctuation)]
    return "\n".join(opt)


def process_text(texts):
    """
    处理输入的文本列表，移除无效的文本条目。

    参数:
    texts (list): 包含文本条目的列表。

    返回:
    list: 处理后的有效文本列表。

    异常:
    ValueError: 当输入的文本列表中全是无效条目时抛出。
    """
    # 判断列表中是否全是无效的文本条目
    if all(text in [None, " ", "\n", ""] for text in texts):
        raise ValueError("请输入有效文本")

    # 过滤并返回有效的文本条目
    return [text for text in texts if text not in [None, " ", "", "\n"]]


if __name__ == "__main__":
    text = """
第一章 上班第一天就准备辞职
    辞职报告
    尊敬的局领导:
    今天是正式入职第一天，我很高兴自己在这个时候向局里正式提出辞职。
    进入单位进行初任培训也已经半年了，在这半年里，也没得到局里什么帮助，每天上班像上坟，一到周一就心里发堵。
    在这里我收获了繁琐的审批流程，收获了无所不在的官僚主义，就是没收获多少薪水。
    实在不想在这份自己并不适合的工作中浪费生命，也想换一下环境，看看诗和远方。
    爱谁谁吧。
    当然，并不是因为单位的工资低，也不是因为工作环境危险，更不是因为正式入职就把我调到了后勤支援部门。
    离开异常局，很舍不得，舍不得领导们的官僚主义裙带关系，舍不得同事之间的拍马溜须。
    我很高兴不能为领导们辉煌的明天贡献自己的力量了。
    另外建议以后给年轻人画大饼的时候，不要用“领导都看在眼里”这种话术。
    时间长了容易给大家领导得了白内障的错觉。
    此致
    敬礼!
    辞职人：李凡
    将笔放下，李凡轻轻弹了弹刚写完的辞职信，露出如释重负的笑容。
    顺手把“培训定级E，精神力等级E，定岗支援中心解剖处见习调查员”的通知给丢在垃圾桶里。
    重生到这个世界已经一个多月了。
    他一个前世的古董商，穿越成了什么异常局西南分局的见习调查员。
    还没有继承对方的记忆，直接就进入了培训。
    封闭式培训了一个多月，整个人都是懵的，最后直接定了个最低等级，分了个最差的部门。
    如果不是封闭式培训的时候不让出来，也为了摸清他这个前身到底是什么状况，他早就已经辞职了。
    今天是结束培训正式入职工作的第一天，也是他彻底辞职的一天。
    实在是不适合这种每天心惊胆战，朝九晚五做噩梦的生活。
    有编制也不要了。
    将辞职报告放进信封里折好，李凡走出公寓宿舍，迈着轻松的步伐向办公楼走去。
    全局新人入职欢迎大会即将在礼堂召开，远远已经能看到会议召开的电子横幅。
    院子里还有一些警示性标语，诸如：
    “一旦心空，立刻报告！警惕清洁协会！”
    “内心一尘不染的往往不是人类！”
    “内心越平静，越远离人类！”
    来到办公楼，把辞职报告塞进局长信箱，李凡感觉心里的大石头彻底落下。
    接下来这里的一切都和他无关了，以后他就彻底自由自在了。
    随后转身去单位餐厅吃早餐。
    进了餐厅，里面已经是熙熙攘攘的人。
    虽然前些天大家还在一起培训，但今天分了部门和岗位之后，餐厅里吃饭的人已经分成了泾渭分明的几拨。
    靠近窗户，最敞亮舒服的地方，是那些进入一线调查部，最有希望成为觉醒者的人。
    这些人身穿挺拔的制服，声音洪亮，目光明亮，相互交谈着，不时发出哄堂大笑，人虽然不多，声音却遮盖了整个餐厅。
    有几个原本和李凡熟识的，在看到李凡的时候只是目光一瞥，仿佛根本不认识一样。
    阶层就这么拉开了。
    然后是中间区域，这里坐着的都是一脸官僚气息的政工部门成员，虽然精神力达不到觉醒者的标准，却是实权部门。
    最后则是靠近边缘角落的区域。
    这里都是被分到支援中心的人，可能这辈子都是见习调查员了，职级低工资低待遇低，三低人员。
    这些支援中心的见习调查员们也都在低调的吃饭，没什么声音。
    一群败犬。
    “凡哥，这里，这里！”一个声音传来。
    刚打了一份油条豆浆的李凡循声看去，就见赵雷正一脸兴高采烈地在支援中心的就餐区朝他招手。
    李凡端着餐盘过去坐下，赵雷已经迫不及待地开始八卦：
    “凡哥，分到哪个部门了？我分到支援中心装备处了，据说张雅晴分到调查部三大队二组了，哎，没想到她那么厉害，不知道她来领装备的时候还能不能再见面。”
    李凡吃一块腐乳说道：“我分到解剖处了。”
    赵雷一口豆浆差点喷出来，呛得猛咳一通。
    旁边的几个同事也都稍稍把盘子往后撤了撤，下意识离李凡远了点。
    解剖处，那可不是个好地方，基本上是异常局里最烂的部门了。
    每天就是和那些怪异的尸体打交道，升职无望，加薪很难，工作据说又极为繁忙。
    而且据说和那些尸体待久了人都容易变态，都疯疯癫癫的。
    解剖处减员很少，一般都是自杀，每年收不了几个人，所以也很少听谁过去。
    旁边几个原本还觉得同病相怜的同事看向李凡的目光中，都带上了一点优越感。
    败犬中的败犬。
    “凡哥，你这到底是考了多少分？不过解剖处也没啥，起码……起码安稳。”赵雷一时有些不知道怎么安慰。
    周围几人也都纷纷出言安慰，什么起点不是终点、好好干领导都看在眼里之类的话说了一堆。
    李凡倒也不在意，毕竟他的辞职报告都递交了。
    墙上的电视里，新一期的“异常简报”已经开始播放。
    “枫叶谷市发生群体性异常感染事件，异常局东北分局迅速处置……”
    “幻灵党袭击墨西哥军警车队，造成131人阵亡……”
    “北美镇魂局发现近百座印第安亡灵教堂墓地，正在艰难镇压……”
    “一个月前，清洁协会十二骑士之一的‘收藏家’突袭东南亚降临会曼谷总会，当场毙杀三十五名降灵师，包括七名异常附体的大降灵师，头颅全无……”
    李凡吹了个口哨。
    清洁协会不愧是最大的觉醒者犯罪组织，实在是太邪性了，不，应该说“收藏家”太邪性了。
    三十五名降灵师，就是三十五名觉醒者，渣都不剩，什么概念！
    不愧是凶名赫赫，在觉醒者通缉榜单里排名前几的人物，据说被他暗杀过的南美和非洲国家总统就有七个。
    好在自己辞职之后，安安稳稳做个古董商，这辈子都不可能遇到收藏家了。
    对面一个同属装备处的男子小声说道：
    “我听说这个收藏家做事还挺有原则的，从来不对平民出手，目标全都是觉醒者或者各国政要……你们说咱们局长和收藏家谁厉害？”
    他说的局长，就是异常局西南分局的局长赵逸峰。
    赵雷同样低声道：
    “那还用说，赵逸峰局长可是顶级觉醒者，评级据说达到了A！我估计收藏家要杀他，起码得三个回合吧？”
    旁边一个模样俊俏的少女捂嘴笑道：
    “你也太损了，咱们局长就不能投降了？我听说收藏家超级帅气，但是谁也没见过他的真面目……”
    赵雷道：“悖论悖论，既然没人见过，那怎么知道他帅气的？哎，这清洁协会实力扩张的有点太厉害了……你说是吧凡哥？”
    李凡摇头道：“收藏家有些强过头了，功高震主，对清洁协会来说不是什么好事情。”
    此时电视上现出“觉醒者通缉榜”几个字样，随后是排名前十的觉醒者罪犯，第一名的赫然就是只有剪影的收藏家！
    “卧槽，收藏家的排名又上升了！都飙到第一了！”
    “好像连降临会都对他发出了通缉，黑白两道都出动了。”
    “清洁协会这么有面子吗？十二骑士里面随便一个都这么强……”
    一片惊呼中，李凡吃完早餐，和赵雷打个招呼，回到了宿舍，开始收拾行李。
    新人入职欢迎会还有十几分钟就要召开了，不过他也懒得参加。
    反正已经辞职了，再和人虚与委蛇的寒暄也没意思。
    收拾了一会儿，电话突然响起，是门口传达室打来的。
    “李凡，你的父母来看望你了，按规定必须由内部人员自己接人进去。”
    李凡不由一愣，目光都变得柔和，看向摆在桌子上的一个相框。
    相框里是一家三口的合影，和前世不同，这一世他是有自己的父母家人的。
    虽然没有继承记忆，但光是看合影中那个略显严肃的父亲和那个眉目慈祥的母亲，就能知道这是个温馨的家庭。
    之前只通过电话，他还没有真正见过自己的父母，辞职的事情也没有沟通，希望他们能理解吧。
    收拾好心情，李凡在传达室接到了自己的父母。
    “小凡，最近是不是又没好好吃饭？怎么瘦了？”穿着碎花长裙的母亲心疼的说道，“快带我和你爸去宿舍看看，给你带了一堆好吃的。”
    “这小子看起来倒是有劲儿了，看样子培训期间表现不错。”父亲微笑着拍了拍李凡的肩膀满意道，“长大了……”
    李凡心中一暖，这一世，他不再是孤家寡人了。
    “爸，妈，咱们去宿舍。”李凡接过沉重的行礼包裹笑着说道。
    两人人手一个拉杆箱，还有个大背包，带了不少东西。
    甚至隐隐能闻到腊肉腊肠的味道。
    迈着轻快的步伐带着父母走过异常局的大院，李凡心中安定了许多。
    有家人的感觉，真的很好。
    哪怕是为了家人，他也应该辞职，离开异常局，过上安稳的普通人的生活。
    就是不知道二老会不会在意编制的问题。
    李凡决定到了宿舍好好跟父母解释解释自己的决定，希望他们能理解。
    “今天局里开入职迎新大会，大家都在大礼堂呢。”李凡笑着说。
    父母相视一眼，依然是笑吟吟的没有说什么。
    只是偶尔感叹大院里连个警卫都没有。
    很快到了宿舍，李凡打开门放下行礼，正准备给父母倒水，却没想到原本笑吟吟的父母表情瞬间严肃。
    难道他们发现自己要辞职的事情了？
    两人将门关好，毕恭毕敬地站在李凡面前。
    看他们的神情，竟然对李凡十分畏惧，眼神之中则带着狂热。
    正在李凡疑惑的时候，两人突然齐齐将双手举过头顶，向李凡行礼，同时低声吟唱道：
    “人类的灵魂终将净化，污浊的尘世终将清洁，深渊之主终将降临！”
    紧接着父亲向李凡低头沉声说道：
    “报告组长，杜鹃计划第一阶段已经布置完毕，异常局西南分局323处精神炸弹全部设置完毕，一旦启动，将在短时间内毁掉异常局西南分局百分之六十的行动力，同时释放本地收容的异常物品造成大面积感染。杜鹃小队的潜伏者已经全部待命，随时可以发动总攻。”
    母亲接着上前一步，恭敬地垂手说道：
    “请问是否要立刻启动第一阶段计划？只需要一次爆炸，所有的调查员都将得到净化，这实在是一个完美无缺的计划，请接受属下的赞美，愿尘世清洁，愿深渊之主降临。”
    “另外，您的武器装备也已经准备好了……”
    一个大行李箱被打开，露出里面挂成几排的人头，每一个头颅都做了防腐处理，风干缩水后只有巴掌大小。
    其中一些还有绚丽的纹面，赫然是今天电视上看到的那些降临会降灵师！
    父亲和母亲的脸上带着讨好和兴奋的笑容，齐声说道：
    “……用生命捍卫您的命令……”
    “……尊敬的收藏家大人！”
"""
    # print(format_text(text))
    lan = "en"
    text = format_text(text, lan)

    if "zh" in lan:
        text = cut3(text)
    else:
        text = cut4(text)
    texts = text.split("\n")
    texts = process_text(texts)
    texts = merge_short_text_in_array(texts, 5)
    for i in range(len(texts)):
        print(f"{i+1}/{len(texts)}: {texts[i]}")
