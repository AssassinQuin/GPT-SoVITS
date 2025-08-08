import os
import re
from time import sleep
import requests
import json
from tqdm import tqdm
from auto_task_help_v2 import clear_text
from typing import Dict, Any, Optional, List
from loguru import logger
from pathlib import Path
from pypinyin import pinyin, Style
from Levenshtein import distance

# 常量定义
API_URL = "http://localhost:11434/api/generate"
# MODEL_NAME = "goekdenizguelmez/JOSIEFIED-Qwen3"
# MODEL_NAME = "huihui_ai/qwen3-abliterated:1.7b"
# MODEL_NAME = "gemma3:4b"

model_name_map = {
    1: "goekdenizguelmez/JOSIEFIED-Qwen3",
    2: "huihui_ai/qwen3-abliterated:1.7b",
    3: "gemma3:4b",
    4: "shmily_006/Qw3:4b_4bit_think",
}

CHECK_SPK_LIST_PATH = "tmp/task_list.json"
DEFAULT_START_IDX = 1
DEFAULT_TOP_N = 25


# def extract_json_from_response(response: str) -> dict:
#     """从LLM响应中提取JSON部分"""
#     match = re.search(r"```json\s*([\s\S]*?)\s*```", response)
#     if match:
#         try:
#             return json.loads(match.group(1))
#         except json.JSONDecodeError:
#             return {}
#     return {}


def extract_json_from_response(response: str) -> dict:
    # 优化正则表达式（允许无 json 标签的代码块）
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
    if match:
        # 修正后的 JSON 字符串（删除多余逗号）
        json_str = match.group(1).replace(",,", ",")
        return json.loads(json_str)


def post_request_with_retries(
    prompt: str, model_type: int = 4, max_retries: int = 3, timeout: int = 60
) -> Optional[str]:
    """
    发送非流式 POST 请求并支持重试，返回完整响应内容。

    :param prompt: 用户输入的提示词
    :param max_retries: 最大重试次数
    :param timeout: 单次请求的超时时间（秒）
    :return: 模型返回的文本；所有重试失败时返回 None
    """
    MODEL_NAME = model_name_map.get(model_type, "gemma3:4b")
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
    }

    headers = {"Content-Type": "application/json"}

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                API_URL, json=payload, headers=headers, timeout=timeout
            )
            response.raise_for_status()

            # 如果接口返回 JSON 格式：
            try:
                data = response.json()
                # 以 key "response" 或者实际字段为准
                if isinstance(data, dict) and "response" in data:
                    return data["response"]
                # 如果所有内容就在一个字段里，比如 "completion"
                if isinstance(data, dict) and "completion" in data:
                    return data["completion"]
                # 否则直接把整个 JSON 转为字符串返回
                return json.dumps(data, ensure_ascii=False)
            except json.JSONDecodeError:
                # 如果返回的是纯文本
                return response.text

        except requests.exceptions.RequestException as e:
            logger.error(f"请求失败 (尝试 {attempt}/{max_retries})：{e}")
            if attempt < max_retries:
                sleep(2**attempt)  # 指数退避
            else:
                logger.error("重试次数耗尽，返回 None")
                return None

    return None


# def post_request_with_retries_stream(
#     prompt: str, max_retries: int = 3
# ) -> Optional[str]:
#     """发送流式POST请求并支持重试，返回完整响应"""
#     payload = {"model": MODEL_NAME, "prompt": prompt, "stream": True}
#     for attempt in range(max_retries):
#         try:
#             with requests.post(API_URL, json=payload, stream=True) as response:
#                 response.raise_for_status()
#                 full_response = ""
#                 for line in response.iter_lines():
#                     if line:
#                         try:
#                             json_data = json.loads(line.decode("utf-8"))
#                             if not json_data.get("done", False):
#                                 chunk = json_data.get("response", "")
#                                 if chunk:
#                                     full_response += chunk
#                             else:
#                                 break
#                         except (json.JSONDecodeError, UnicodeDecodeError) as e:
#                             logger.error(f"数据解析失败: {e}, 原始数据: {line[:50]}...")
#                             continue
#                 return full_response
#         except requests.exceptions.RequestException as e:
#             logger.error(f"请求失败 ({attempt + 1}/{max_retries}): {e}")
#             if attempt < max_retries - 1:
#                 sleep(10)
#         except Exception as e:
#             logger.error(f"意外错误: {e}")
#             break
#     logger.error("所有重试均失败")
#     return None


def get_top_roles(
    book_data: Dict[str, Dict[str, Any]], current_chapter: int, top_n: int
) -> str:
    """获取最近章节中频率最高的前N个角色名"""
    filtered_roles = [
        (name, data["count"], data["last_chapter"])
        for name, data in book_data.items()
        if data["last_chapter"] + 5 >= current_chapter
    ]
    sorted_roles = sorted(filtered_roles, key=lambda x: (x[1], x[2]), reverse=True)
    unique_roles = {name: count for name, count, _ in sorted_roles[:top_n]}

    if len(unique_roles) < top_n:
        remaining = sorted(
            (
                (name, data["count"])
                for name, data in book_data.items()
                if name not in unique_roles
            ),
            key=lambda x: x[1],
            reverse=True,
        )
        for name, count in remaining:
            if len(unique_roles) >= top_n:
                break
            unique_roles[name] = count

    return list(unique_roles.keys())


def role_name_clear_prompt(top_roles: str) -> str:
    prompt = f"""
根据给出来的角色名列表，过滤可能重复的角色名，返回非重复的角色名
角色列表: {top_roles}
返回格式：
```json
{{
  "role_name": ["名称1", "名称2"]
}}
```
"""
    return prompt


# def construct_prompt(sentence: str, context: str, top_roles: str) -> str:
#     """构造综合提示词，指导LLM完成角色分析"""
#     prompt = f"""
# 请根据以下信息，识别说话者的角色、性别和年龄段，优先从给定的角色列表中选择。若无法确定，则填“未知”。

# 输入格式：
# - 上下文（context）：【{context}】
# - 句子（sentence）：【{sentence}】
# - 候选角色（roles）：【{",".join(top_roles)}】

# 分析流程（务必依次执行并记录要点）：
# 1. 提取线索：从句子和上下文中捕捉身份、职业、关系等提示。
# 2. 匹配角色：优先将线索与候选角色对照，选出最契合者。
# 3. 校验合理性：检查所选角色与上下文是否一致，如有矛盾，回到第2步。
# 4. 性别与年龄：结合语言风格、社会身份等信息，判断“性别”（男/女/未知）和“年龄”（青年/中年/老年/未知）。

# 输出格式（严格JSON）：
# ```json
# {{
#     "role": "最终角色名或未知",
#     "gender": "男／女／未知",
#     "age": "青年／中年／老年／未知"
# }}

# ```
# """
#     return prompt


def age_and_gender_prompt(role_name: str, context: str) -> str:
    """构造用于API的提示语（Prompt）。"""
    prompt = f"""
【原文】：
【{context}】

【任务描述】：
根据原文，判断角色【{role_name}】的年龄与性别。

年龄选择仅：青年/中年/老年/未知
性别选择仅：男/女/未知


输出格式（严格JSON）：
```json
{{
    "age": "青年/中年/老年/未知",
    "gender": "男/女/未知"
}}
```
### 注意事项：
- 请确保每一步都清晰、明确，避免任何模糊或歧义。
- 所有输出必须为有效的 JSON 格式，且字段名称和数据格式严格遵守上述规范。
"""
    return prompt


def construct_prompt(sentence: str, context: str, top_roles: str) -> str:
    """构造用于API的提示语（Prompt）。"""
    prompt = f"""
【原文】：
{context}

【任务描述】：
请根据给定的句子和上下文内容，判断该句子由原文中的哪个角色说出。请严格按照以下步骤执行：

### 步骤 1：角色识别
1.1 分析句子【{sentence}】及其在上下文中的位置，推断说出该句子的角色。**请勿直接使用句子中的任何角色名称。**

1.2 如果无法确定具体角色或句子没有特定角色，请返回以下 role 返回 “未知”

### 步骤 2：角色匹配
2.1 从给定的角色列表中【{top_roles}】查找与步骤 1 中推断的角色名称相似的角色名。

2.2 如果找到相似的角色名，请使用列表中的角色名作为最终结果；否则，保留步骤 1 推断的角色名。

输出格式（严格JSON）：
```json
{{
    "role": "最终角色名或未知",
}}
```
### 注意事项：
- 请确保每一步都清晰、明确，避免任何模糊或歧义。
- 所有输出必须为有效的 JSON 格式，且字段名称和数据格式严格遵守上述规范。
"""
    return prompt


def construct_check_prompt(
    context: str, sentence: str, first_role_name: str, top_roles: str
) -> str:
    """构造用于二次检查的提示语（Prompt）。"""
    prompt = f"""
【原文】：
【{context}】

【句子】：
【{sentence}】

【已识别角色】：
第一判断的角色名称为【{first_role_name}】。

【角色列表】：
{top_roles}

【任务描述】：
请根据原文和句子的上下文，验证步骤 1 中识别的角色是否正确。请严格按照以下步骤执行：

### 步骤 1：验证角色
1.1 判断句子是否由角色【{first_role_name}】说出。

1.2 如果是，请确认并返回 {first_role_name}】

1.3 如果不是，请从角色列表中【{top_roles}】重新判断句子的说话角色。

### 步骤 2：重新识别
2.1 如果重新识别出新的角色，请返回 重新判断的角色名

2.2 如果依然无法确定，请返回以 “未知”

### 步骤 3：输出格式
```json
{{
    "role": "最终角色名或未知",
}}
```
### 注意事项：
- 确保所有输出为有效的 JSON 格式。
- 在重新识别角色时，优先考虑与上下文高度相关的角色。
"""
    return prompt


def construct_comprehensive_prompt(
    context: str,
    sentence: str,
    first_role: Dict[str, str],
    second_role: Dict[str, str],
    top_roles: str,
) -> str:
    """构造用于综合判断的提示语（Prompt）。"""
    prompt = f"""
【原文】：
{context}

【句子】：
【{sentence}】

【第一判断角色】：
【{first_role["role"]}】

【第二判断角色】：
【{second_role["role"]}】

【角色列表】：
【{top_roles}】

【任务描述】：
请综合第一判断和第二判断的结果，结合原文内容和已有的角色信息，确定句子【{sentence}】的最终说话者。请严格按照以下步骤执行：

### 步骤 1：角色综合评估
1.1 分析第一判断的角色和第二判断的角色在上下文中的相关性和适用性。

1.2 确定哪个角色更有可能是该句子的说话者。

### 步骤 4：输出格式
```json
{{
    "role": "最终角色名或未知",
}}
```
### 注意事项：
- 所有输出必须为有效的 JSON 格式。
- 确保评估过程基于上下文中的明确线索，避免主观判断。
"""
    return prompt


def process_sentence(
    sentence: str, context1: str, context2: str, context3: str, top_roles: str
) -> Dict[str, Dict[str, str]]:
    """处理单个句子，识别角色和性别，采用三层思维逻辑。"""
    result = {}
    try:
        # 第一层思维：初步识别
        prompt1 = construct_prompt(sentence, context1, top_roles)
        response1 = post_request_with_retries(prompt1, model_type=2)
        json_data = extract_json_from_response(response1)
        if not json_data:
            return result
        first_role_name = json_data.get("role", "未知")
        logger.info(f"初步识别角色: {first_role_name}")

        # 第二层思维：验证与扩展
        check_prompt = construct_check_prompt(
            context2, sentence, first_role_name, top_roles
        )
        response2 = post_request_with_retries(check_prompt, model_type=4)
        json_data = extract_json_from_response(response2)
        if not json_data:
            return result
        second_role_name = json_data.get("role", "未知")
        logger.info(f"第二层结果: {second_role_name}")

        # 如果第一层和第二层结果不一致，进行第三层思维
        if first_role_name != second_role_name:
            # 第三层思维：综合判断
            comprehensive_prompt = construct_comprehensive_prompt(
                context3,
                sentence,
                {"role": first_role_name},
                {"role": second_role_name},
                top_roles,
            )
            response3 = post_request_with_retries(comprehensive_prompt, model_type=1)
            json_data = extract_json_from_response(response3)
            if json_data:
                final_role = json_data.get("role", "未知")
            else:
                final_role = "未知"
            logger.info(f"第三层结果: {final_role}")
        else:
            # 如果一致，直接使用第一层结果
            final_role = first_role_name

        age_and_gender = age_and_gender_prompt(final_role, context3)
        response4 = post_request_with_retries(age_and_gender, model_type=4)
        json_data = extract_json_from_response(response4)
        if not json_data:
            return result
        final_age = json_data.get("age", "未知")
        final_gender = json_data.get("gender", "未知")

        # 填充最终结果
        result[clear_text(sentence, True)] = {
            "role": final_role,
            "gender": final_gender,
            "age": final_age,
            "content": context1,
        }
        logger.info(
            f"句子处理完成: {sentence} - {final_role} - {final_gender} - {final_age}"
        )

    except Exception as e:
        logger.error(f"处理句子时出错: {e}")

    return result


def post_process_results(
    results: List[Dict[str, Dict[str, str]]],
) -> List[Dict[str, Dict[str, str]]]:
    """后处理结果，确保角色性别和年龄一致性"""
    role_info = {}
    for result in results:
        for sentence, info in result.items():
            role = info["role"]
            if role != "未知":
                if role not in role_info:
                    role_info[role] = {"gender": info["gender"], "age": info["age"]}
                elif role_info[role]["gender"] == "未知" and info["gender"] != "未知":
                    role_info[role]["gender"] = info["gender"]
                elif role_info[role]["age"] == "未知" and info["age"] != "未知":
                    role_info[role]["age"] = info["age"]

    for result in results:
        for sentence, info in result.items():
            role = info["role"]
            if role in role_info:
                info["gender"] = role_info[role]["gender"]
                info["age"] = role_info[role]["age"]
    return results


def merge_chapter_roles(book_name, current_chapter):
    """合并相邻章节的角色列表"""
    # 初始化参数
    book_dir = Path(f"/root/code/GPT-SoVITS/tmp/{book_name}")
    role_dir = book_dir / "role"

    # 计算章节范围
    min_chapter = max(1, current_chapter - 1)  # 确保最小值不小于1
    max_chapter = current_chapter + 1

    top_roles = []
    seen_roles = set()  # 用于去重

    for chapter_idx in range(min_chapter, max_chapter + 1):
        json_path = role_dir / f"name_chapter_{chapter_idx}.json"

        # 验证文件存在性
        if not json_path.exists():
            continue

        try:
            # 读取并验证文件格式
            with open(json_path, "r", encoding="utf-8") as f:
                chapter_roles = json.load(f)

            if not isinstance(chapter_roles, list):
                print(f"文件格式错误：{json_path} 内容不是数组")
                continue

            # 合并并去重
            for role in chapter_roles:
                pinyin_name = "".join(
                    [item[0] for item in pinyin(role, style=Style.NORMAL)]
                )
                if pinyin_name not in seen_roles:
                    top_roles.append(role)
                    seen_roles.add(pinyin_name)

        except json.JSONDecodeError:
            print(f"JSON解析失败：{json_path}")
        except Exception as e:
            print(f"处理 {json_path} 时发生异常：{str(e)}")

    # 处理 top_roles
    prompt = role_name_clear_prompt(top_roles)
    response = post_request_with_retries(prompt, 1)
    json_data = extract_json_from_response(response)
    top_roles = json_data.get("role_name", [])
    logger.info(f"top_roles: {top_roles}")

    return top_roles


def generate_pinyin(name: str) -> str:
    """生成拼音标识"""
    return "".join([item[0] for item in pinyin(name, style=Style.TONE3)])


def is_duplicate(pinyin_str: str, seen_pinyins: set) -> bool:
    """检查拼音相似性"""
    return any(
        pinyin_str in exist or exist in pinyin_str or distance(pinyin_str, exist) <= 2
        for exist in seen_pinyins
    )


def process_file(
    file_path: str,
    output_dir: str,
    book_data_dir: str,
    current_chapter: int,
    top_n: int = DEFAULT_TOP_N,
    book_name: str = "",
):
    """处理单个文件，提取并分析对话"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        logger.error(f"读取文件 {file_path} 时出错: {e}")
        return False

    book_data_file = os.path.join(book_data_dir, "book_data.json")
    book_data = (
        json.load(open(book_data_file, "r", encoding="utf-8"))
        if os.path.exists(book_data_file)
        else {}
    )

    lines = content.split("\n")
    results = []

    top_roles = merge_chapter_roles(book_name, current_chapter)
    # top_roles = get_top_roles(book_data, current_chapter, top_n)

    seen_pinyins = set()
    for name in top_roles:
        py = generate_pinyin(name)
        if not is_duplicate(py, seen_pinyins):
            top_roles.append(name)
            seen_pinyins.add(py)

    for idx, line in enumerate(lines):
        for sentence in re.findall(r"“([^”]*)”", line):
            sentence = (
                sentence.replace("……", "。")
                .replace("。。", "。")
                .replace("、", "，")
                .strip()
            )
            if not sentence or sentence[-1] not in "，。！？":
                continue
            context1 = "".join(
                lines[max(0, idx - 5) : idx + 1]
                + lines[idx + 1 : min(len(lines), idx + 5)]
            )
            context2 = "".join(
                lines[max(0, idx - 8) : idx + 1]
                + lines[idx + 1 : min(len(lines), idx + 8)]
            )
            context3 = "".join(
                lines[max(0, idx - 12) : idx + 1]
                + lines[idx + 1 : min(len(lines), idx + 12)]
            )
            result = process_sentence(sentence, context1, context2, context3, top_roles)
            if result:
                results.append(result)
                role_key = result[next(iter(result))]["role"]
                if role_key != "未知":
                    book_data[role_key] = book_data.get(
                        role_key, {"count": 0, "last_chapter": current_chapter}
                    )
                    book_data[role_key]["count"] += 1
                    book_data[role_key]["last_chapter"] = current_chapter
            py_name = generate_pinyin(result[next(iter(result))]["role"])
            if not is_duplicate(py_name, seen_pinyins):
                seen_pinyins.add(py_name)

    results = post_process_results(results)
    output_file = os.path.join(
        output_dir, os.path.basename(file_path).replace(".txt", ".json")
    )
    try:
        with open(output_file, "w", encoding="utf-8") as out_f:
            json.dump(results, out_f, ensure_ascii=False, indent=4)
        with open(book_data_file, "w", encoding="utf-8") as out_f:
            json.dump(book_data, out_f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        logger.error(f"写入文件时出错: {e}")
        return False


def process_directory(
    input_dir: str,
    output_dir: str,
    book_data_dir: str,
    start_idx: int,
    top_n: int,
    book_name: str,
    check_spk_list: List[Dict[str, Any]],
):
    """处理目录中的所有txt文件"""
    os.makedirs(output_dir, exist_ok=True)
    txt_files = sorted(
        [
            f
            for f in os.listdir(input_dir)
            if f.endswith(".txt") and not f.startswith(".")
        ],
        key=lambda x: int(re.findall(r"\d+", x)[0]) if re.findall(r"\d+", x) else 0,
    )

    processed = False
    for txt_file in tqdm(txt_files, desc="处理文件"):
        file_idx = (
            int(re.findall(r"\d+", txt_file)[0]) if re.findall(r"\d+", txt_file) else 0
        )
        if file_idx >= start_idx:
            file_path = os.path.join(input_dir, txt_file)
            success = process_file(
                file_path, output_dir, book_data_dir, file_idx, top_n, book_name
            )
            if success:
                start_idx = file_idx + 1
                processed = True
                update_status(check_spk_list, book_name, "执行中", start_idx=start_idx)
                with open(CHECK_SPK_LIST_PATH, "w", encoding="utf-8-sig") as f:
                    json.dump(check_spk_list, f, ensure_ascii=False, indent=4)
            else:
                break
    return processed


def update_status(
    check_spk_list: List[Dict[str, Any]],
    book_name: str,
    status: str,
    start_idx: Optional[int] = None,
    top_n: Optional[int] = None,
):
    """更新任务列表中书籍的状态"""
    for book in check_spk_list:
        if book.get("book_name") == book_name:
            book["check_spk_status"] = status
            if start_idx is not None:
                book["check_spk_start_idx"] = start_idx
            if top_n is not None:
                book["top_n"] = top_n
            return
    check_spk_list.append(
        {
            "book_name": book_name,
            "check_spk_status": status,
            "check_spk_start_idx": start_idx,
            "top_n": top_n,
        }
    )


def main():
    """主函数，处理任务列表中的书籍"""
    if not os.path.exists(CHECK_SPK_LIST_PATH):
        logger.error(f"任务列表不存在: {CHECK_SPK_LIST_PATH}")
        return

    with open(CHECK_SPK_LIST_PATH, "r", encoding="utf-8-sig") as f:
        check_spk_list = json.load(f)

    # check_spk_list = [
    #     {
    #         "book_name": "原来我是修仙大佬",
    #         "check_spk_start_idx": 1,
    #         "top_n": 25,
    #     }
    # ]

    for book in check_spk_list:
        book.setdefault("check_spk_status", "等待中")
        book_name = book.get("book_name")
        if not book_name or book["check_spk_status"] == "已完成":
            continue

        logger.info(f"开始处理书籍: {book_name}")
        update_status(
            check_spk_list,
            book_name,
            "执行中",
            book.get("check_spk_start_idx", DEFAULT_START_IDX),
            book.get("top_n", DEFAULT_TOP_N),
        )
        with open(CHECK_SPK_LIST_PATH, "w", encoding="utf-8-sig") as f:
            json.dump(check_spk_list, f, ensure_ascii=False, indent=4)

        book_data_dir = os.path.join("tmp", book_name)
        input_dir, output_dir = (
            os.path.join(book_data_dir, "data"),
            os.path.join(book_data_dir, "role"),
        )

        if not os.path.exists(book_data_dir):
            logger.error(f"书籍数据目录不存在: {book_data_dir}")
            update_status(check_spk_list, book_name, "等待中")
            continue

        processed = process_directory(
            input_dir,
            output_dir,
            book_data_dir,
            book.get("check_spk_start_idx", DEFAULT_START_IDX),
            book.get("top_n", DEFAULT_TOP_N),
            book_name,
            check_spk_list,
        )
        update_status(check_spk_list, book_name, "已完成" if processed else "执行中")
        with open(CHECK_SPK_LIST_PATH, "w", encoding="utf-8-sig") as f:
            json.dump(check_spk_list, f, ensure_ascii=False, indent=4)

    logger.info("所有书籍处理完成.")


if __name__ == "__main__":
    main()
