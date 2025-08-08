import os
import re
import json
from time import sleep
from pypinyin import Style, pinyin
import requests
from typing import Optional
from loguru import logger
from Levenshtein import distance
from concurrent.futures import ThreadPoolExecutor, as_completed

# 常量定义
API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "huihui_ai/qwen3-abliterated:1.7b"
CHECK_SPK_LIST_PATH = "tmp/task_list.json"
DEFAULT_TOP_N = 25

# 目录配置
book_name = "趋吉避凶，从天师府开始"  # 请替换为实际的book_name
data_dir = f"/root/code/GPT-SoVITS/tmp/{book_name}/data"
output_dir = f"/root/code/GPT-SoVITS/tmp/{book_name}/role"
os.makedirs(output_dir, exist_ok=True)


def extract_json_from_response(response: str) -> dict:
    match = re.search(r"```json\s*([\s\S]*?)\s*```", response)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return {}
    return {}


def post_request_with_retries(
    prompt: str, max_retries: int = 3, timeout: int = 60
) -> Optional[str]:
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
    headers = {"Content-Type": "application/json"}

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                API_URL, json=payload, headers=headers, timeout=timeout
            )
            response.raise_for_status()
            try:
                data = response.json()
                if isinstance(data, dict) and "response" in data:
                    return data["response"]
                if isinstance(data, dict) and "completion" in data:
                    return data["completion"]
                return json.dumps(data, ensure_ascii=False)
            except json.JSONDecodeError:
                return response.text

        except requests.exceptions.RequestException as e:
            logger.error(f"请求失败 (尝试 {attempt}/{max_retries})：{e}")
            if attempt < max_retries:
                sleep(2**attempt)
            else:
                logger.error("重试次数耗尽，返回 None")
                return None
    return None


def role_name_clear_prompt(top_roles: str) -> str:
    return f"""
根据给出来的角色名列表，过滤可能重复的角色名，返回非重复的角色名
角色列表: {top_roles}
返回格式：
```json
{{
  "role_name": ["名称1", "名称2"]
}}
```"""


def construct_prompt(context: str) -> str:
    return f"""
## 角色定义
你是一个专业的文本信息抽取引擎，专门从非结构化文本中识别并提取人物实体。返回全部人物实体名,处理人物实体名，保证返回内容不重复。

## 处理文本
【{context}】

## 输出格式
请在回答的最后以JSON格式输出结果，并用```json和```包围。示例如下：
```json
{{
    "role_list": ["name1", "name2"]
}}
```"""


def generate_pinyin(name: str) -> str:
    return "".join([item[0] for item in pinyin(name, style=Style.NORMAL)])


def is_duplicate(seen_pinyins: set, pinyin_str: str) -> bool:
    return any(
        pinyin_str in exist or exist in pinyin_str or distance(pinyin_str, exist) <= 2
        for exist in seen_pinyins
    )


def process_file(filename: str) -> None:
    if not filename.endswith(".txt"):
        return

    base = os.path.splitext(filename)[0]
    output_path = os.path.join(output_dir, f"name_{base}.json")
    if os.path.exists(output_path):
        logger.info(f"{output_path} 已存在，跳过")
        return

    seen_pinyins = set()
    names = []
    file_path = os.path.join(data_dir, filename)
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    # 分批处理
    skip = DEFAULT_TOP_N
    for i in range(0, len(lines), skip):
        chunk = "\n".join(lines[i : i + skip])
        prompt = construct_prompt(chunk)
        resp = post_request_with_retries(prompt)
        if not resp:
            continue
        json_data = extract_json_from_response(resp)
        for name in json_data.get("role_list", []):
            py = generate_pinyin(name)
            if not is_duplicate(seen_pinyins, py):
                names.append(name)
                seen_pinyins.add(py)
        logger.info(f"{filename} current_names: {names}")

    # 二次去重
    clear_prompt = role_name_clear_prompt(names)
    resp = post_request_with_retries(clear_prompt)
    top_roles = extract_json_from_response(resp).get("role_name", [])
    logger.info(f"{filename} top_roles: {top_roles}")

    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(names, out, ensure_ascii=False, indent=4)
    logger.info(f"已处理 {filename}，发现角色: {names}")


def main():
    filenames = sorted(os.listdir(data_dir))
    workers = min(2, len(filenames))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_file, fn): fn for fn in filenames}
        for future in as_completed(futures):
            fn = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"处理 {fn} 时出错: {e}")


if __name__ == "__main__":
    main()
