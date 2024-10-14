import os
import re
import requests
import json
import logging
from tqdm import tqdm
from tools.auto_task_help_v2 import clear_text
import argparse
from typing import Dict, Any

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 常量
API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "wangshenzhi/gemma2-9b-chinese-chat"


def post_request_with_retries(
    prompt: str, max_retries: int = 3, timeout: int = 10
) -> Any:
    """发送POST请求，并在失败时重试。"""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "keep_alive": "5s",
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, json=payload, timeout=timeout)
            response.raise_for_status()  # 检查响应状态
            response_json = response.json()
            response_content = (
                response_json.get("response", "")
                .replace("```", "")
                .replace("json", "")
                .strip()
            )
            return json.loads(response_content)
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            logger.error(f"请求错误: {e}")
        logger.info(f"重试中 ({attempt + 1}/{max_retries})...")
    return None


def get_top_roles(
    book_data: Dict[str, Dict[str, Any]], current_chapter: int, top_n: int
) -> str:
    """根据出现次数获取前N个角色名。"""
    # 过滤最近章节出现过的角色
    filtered_roles = [
        (name, data["count"])
        for name, data in book_data.items()
        if data["last_chapter"] + 3 >= current_chapter
    ]
    # 按出现次数排序
    sorted_filtered_roles = sorted(filtered_roles, key=lambda x: x[1], reverse=True)
    unique_roles = {name: count for name, count in sorted_filtered_roles[:top_n]}

    # 如果角色数量不足，从出现频率最高的角色中添加
    if len(unique_roles) < top_n:
        remaining_roles = sorted(
            (
                (name, data["count"])
                for name, data in book_data.items()
                if name not in unique_roles
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        for name, count in remaining_roles:
            if len(unique_roles) >= top_n:
                break
            unique_roles[name] = count

    return ",".join(unique_roles.keys())


def construct_prompt(sentence: str, context: str, top_roles: str) -> str:
    """构造用于API的提示语（Prompt）。"""
    prompt = f"""
【原文】：
{context}

【任务描述】：
请根据给定的句子和上下文内容，判断该句子由原文中的哪个角色说出，并确定该角色的姓名和性别。请严格按照以下步骤执行：

### 步骤 1：角色识别
1.1 分析句子【{sentence}】及其在上下文中的位置，推断说出该句子的角色。**请勿直接使用句子中的任何角色名称。**

1.2 如果无法确定具体角色或句子没有特定角色，请返回以下 JSON 格式：
{{"role": "未知", "gender": "未知"}}

### 步骤 2：角色匹配
2.1 从给定的角色列表中【{top_roles}】查找与步骤 1 中推断的角色名称相似的角色名。

2.2 如果找到相似的角色名，请使用列表中的角色名作为最终结果；否则，保留步骤 1 推断的角色名。

### 步骤 3：性别判定
3.1 根据上下文对角色的描述，推断该角色的性别。性别结果必须为 "男"、"女" 或 "未知"。

3.2 如果无法从上下文中推断出性别，请返回以下 JSON 格式：
{{"role": "角色姓名", "gender": "未知"}}

### 步骤 4：输出格式
无论结果如何，输出都必须严格按照以下 JSON 格式返回：
{{"role": "角色姓名", "gender": "男/女/未知"}}

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
{context}

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

1.2 如果是，请确认并返回以下 JSON 格式：
{{"role": "{first_role_name}", "gender": "{{相应性别}}" }}

1.3 如果不是，请从角色列表中【{top_roles}】重新判断句子的说话角色。

### 步骤 2：重新识别
2.1 如果重新识别出新的角色，请返回以下 JSON 格式：
{{"role": "新角色名称", "gender": "{{相应性别}}" }}

2.2 如果依然无法确定，请返回以下 JSON 格式：
{{"role": "未知", "gender": "未知"}}

### 步骤 3：输出格式
无论结果如何，输出都必须严格按照以下 JSON 格式返回：
{{"role": "角色姓名", "gender": "男/女/未知"}}

### 注意事项：
- 确保所有输出为有效的 JSON 格式。
- 在重新识别角色时，优先考虑与上下文高度相关的角色。
"""
    return prompt


def construct_clarification_prompt(
    context: str, first_role_name: str, second_role_name: str, sentence: str
) -> str:
    """构造用于澄清的提示语（Prompt）。"""
    prompt = f"""
【原文】：
{context}

【句子】：
【{sentence}】

【已识别角色】：
- 第一判断的角色名称为【{first_role_name}】。
- 第二判断的角色名称为【{second_role_name}】。

【任务描述】：
请根据原文内容，决定哪个角色名称更符合句子【{sentence}】的说话者，并确认该角色的性别。请严格按照以下步骤执行：

### 步骤 1：角色评估
1.1 比较【{first_role_name}】和【{second_role_name}】在上下文中的相关性和适用性。

1.2 确定哪个角色更有可能是该句子的说话者。

### 步骤 2：性别确认
2.1 根据上下文中对该角色的描述，确认角色的性别。性别结果必须为 "男"、"女" 或 "未知"。

2.2 如果无法从上下文中推断出性别，请返回 "未知"。

### 步骤 3：输出格式
无论结果如何，输出都必须严格按照以下 JSON 格式返回：
{{"role": "最终角色姓名", "gender": "男/女/未知"}}

### 注意事项：
- 所有输出必须为有效的 JSON 格式。
- 确保评估过程基于上下文中的明确线索，避免主观判断。
"""
    return prompt


def process_sentence(
    sentence: str, context: str, top_roles: str
) -> Dict[str, Dict[str, str]]:
    """处理单个句子，识别角色和性别。"""
    result = {}
    try:
        # 第一次请求
        prompt = construct_prompt(sentence, context, top_roles)
        response_content = post_request_with_retries(prompt)
        if response_content:
            first_role_name = response_content.get("role", "未知")
            gender = response_content.get("gender", "未知")
            result[clear_text(sentence, True)] = {
                "role": first_role_name,
                "gender": gender,
                "content": context,
            }

            # 第二次检查
            check_prompt = construct_check_prompt(
                context, sentence, first_role_name, top_roles
            )
            check_response_content = post_request_with_retries(check_prompt)
            if check_response_content:
                second_role_name = check_response_content.get("role", "未知")
                second_gender = check_response_content.get("gender", "未知")

                # 如果两个角色名不同，则请求澄清
                if first_role_name != second_role_name:
                    clarification_prompt = construct_clarification_prompt(
                        context, first_role_name, second_role_name, sentence
                    )
                    clarification_response = post_request_with_retries(
                        clarification_prompt
                    )
                    if clarification_response:
                        final_role_name = clarification_response.get("role", "未知")
                        final_gender = clarification_response.get("gender", "未知")
                        result[clear_text(sentence, True)]["role"] = final_role_name
                        result[clear_text(sentence, True)]["gender"] = final_gender

    except Exception as e:
        logger.error(f"处理句子时出错: {e}")

    return result


def process_file(
    file_path: str,
    output_dir: str,
    book_data_dir: str,
    current_chapter: int,
    top_n: int = 15,
):
    """处理单个文件以提取对话并分析发言者。"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        logger.error(f"文件未找到: {file_path}")
        return
    except Exception as e:
        logger.error(f"读取文件 {file_path} 时出错: {e}")
        return

    book_data_file = os.path.join(book_data_dir, "book_data.json")
    if not os.path.exists(book_data_file):
        with open(book_data_file, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=4)

    try:
        with open(book_data_file, "r", encoding="utf-8") as f:
            book_data = json.load(f)
    except json.JSONDecodeError:
        logger.error(f"解析 JSON 文件失败: {book_data_file}")
        book_data = {}
    except Exception as e:
        logger.error(f"读取书籍数据文件时出错: {e}")
        book_data = {}

    lines = content.split("\n")
    results = []

    for idx, line in enumerate(lines):
        for sentence in re.findall(r"“([^”]*)”", line):
            sentence = sentence.strip()
            if not sentence:
                continue
            context = "".join(
                lines[max(0, idx - 7) : idx]
                + [line]
                + lines[idx + 1 : min(len(lines), idx + 8)]
            )
            top_roles = get_top_roles(book_data, current_chapter, top_n)
            result = process_sentence(sentence, context, top_roles)
            if result:
                results.append(result)
                # 更新书籍数据
                role_key = result[next(iter(result))]["role"]
                book_data[role_key] = book_data.get(
                    role_key, {"count": 0, "last_chapter": current_chapter}
                )
                book_data[role_key]["count"] += 1
                book_data[role_key]["last_chapter"] = current_chapter

                logger.info(
                    f"处理句子: {sentence}, 角色: {role_key}, 性别: {result[next(iter(result))]['gender']}"
                )

    # 将结果写入输出文件
    output_file = os.path.join(
        output_dir, os.path.basename(file_path).replace(".txt", ".json")
    )
    try:
        with open(output_file, "w", encoding="utf-8") as out_f:
            json.dump(results, out_f, ensure_ascii=False, indent=4)
    except Exception as e:
        logger.error(f"写入输出文件 {output_file} 时出错: {e}")

    # 更新书籍数据文件
    try:
        with open(book_data_file, "w", encoding="utf-8") as out_f:
            json.dump(book_data, out_f, ensure_ascii=False, indent=4)
    except Exception as e:
        logger.error(f"更新书籍数据文件 {book_data_file} 时出错: {e}")


def process_directory(
    input_dir: str, output_dir: str, book_data_dir: str, start_idx: int, top_n: int = 15
):
    """处理目录中的所有.txt文件。"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"创建输出目录: {output_dir}")

    try:
        txt_files = sorted(
            [
                f
                for f in os.listdir(input_dir)
                if f.endswith(".txt") and not f.startswith(".")
            ],
            key=lambda x: int(re.findall(r"\d+", x)[0]) if re.findall(r"\d+", x) else 0,
        )
    except Exception as e:
        logger.error(f"列出目录 {input_dir} 中的文件时出错: {e}")
        return

    for txt_file in tqdm(txt_files, desc="处理文件"):
        try:
            file_idx = (
                int(re.findall(r"\d+", txt_file)[0])
                if re.findall(r"\d+", txt_file)
                else 0
            )
        except (IndexError, ValueError):
            logger.warning(f"无法解析文件名中的数字部分: {txt_file}")
            continue

        if file_idx >= start_idx:
            file_path = os.path.join(input_dir, txt_file)
            process_file(file_path, output_dir, book_data_dir, file_idx, top_n=top_n)


def main():
    # book_lisk = [
    #     # ("藏海花", 1),
    #     ("藏海花2", 1),
    # ]
    parser = argparse.ArgumentParser(description="小说 txt 文件处理程序")
    parser.add_argument("book_name", type=str, help="小说名")
    parser.add_argument(
        "--start_idx", type=int, help="起始章节", default=1, required=False
    )
    parser.add_argument(
        "--top_n", type=int, help="获取前N个角色", default=25, required=False
    )

    args = parser.parse_args()
    book_name = args.book_name
    start_idx = args.start_idx  # 从指定的章节开始处理
    top_n = args.top_n

    book_data_dir = os.path.join("tmp", book_name)
    input_dir = os.path.join(book_data_dir, "data")
    output_dir = os.path.join(book_data_dir, "role")

    if not os.path.exists(book_data_dir):
        logger.error(f"书籍数据目录不存在: {book_data_dir}")
        return

    process_directory(input_dir, output_dir, book_data_dir, start_idx, top_n=top_n)


if __name__ == "__main__":
    main()
