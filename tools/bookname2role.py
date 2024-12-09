import os
import json
import argparse
from itertools import cycle


def generate_role(role_json_path):
    # 读取角色配置文件，并处理可能的UTF-8 BOM
    with open(role_json_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    role_map = {}

    def get_files(path):
        files = os.listdir(path)

        def find_file(extension):
            try:
                return next(f for f in files if f.endswith(extension))
            except StopIteration:
                raise FileNotFoundError(f"No {extension} file found in {path}")

        wav_file = find_file(".wav")
        pth_file = find_file(".pth")
        ckpt_file = find_file(".ckpt")

        return {
            "ref_audio_path": os.path.join(path, wav_file),
            "ref_text": os.path.splitext(wav_file)[0],  # wav 文件名（不包括扩展名）
            "gpt_model": os.path.join(path, ckpt_file),
            "sovits_model": os.path.join(path, pth_file),
        }

    # 遍历所有角色并保存到角色映射
    for role_category, role_list in data.items():
        if isinstance(role_list, list):
            for role in role_list:
                role_name = role.get("name", "")
                role_path = role.get("path", "")
                role_age = role.get("age", "")
                try:
                    role_map[role_name] = get_files(role_path)
                    role_map[role_name]["role"] = role_category
                    role_map[role_name]["age"] = role_age
                except FileNotFoundError as e:
                    print(f"Error processing {role_name}: {e}")
        else:
            role_name = role_list.get("name", "")
            role_path = role_list.get("path", "")
            role_age = role.get("age", "")

            try:
                role_map[role_name] = get_files(role_path)
                role_map[role_name]["role"] = role_category
                role_map[role_name]["age"] = role_age

            except FileNotFoundError as e:
                print(f"Error processing {role_name}: {e}")

    return role_map


def select_role_by_gender_and_age(gender, age, role_map, role_cyclers):
    if gender == "女":
        if age == "未知":
            age = "青年"
        female_roles = [
            key
            for key, role in role_map.items()
            if role["role"] == "女配" and role["age"] == age
        ]
        if len(female_roles) == 0:
            female_roles = [
                key for key, role in role_map.items() if role["role"] == "女配"
            ]
        if female_roles:
            if "female" not in role_cyclers:
                role_cyclers["female"] = cycle(female_roles)
            return next(role_cyclers["female"])
        else:
            print("No female roles available.")
    else:
        if age == "未知":
            age = "青年"
        male_roles = [
            key
            for key, role in role_map.items()
            if role["role"] == "男配" and role["age"] == age
        ]
        if len(male_roles) == 0:
            male_roles = [
                key for key, role in role_map.items() if role["role"] == "男配"
            ]
        if male_roles:
            if "male" not in role_cyclers:
                role_cyclers["male"] = cycle(male_roles)
            return next(role_cyclers["male"])
        else:
            print("No male roles available.")


def main(base_directory, book_name):
    role_json_path = os.path.join(base_directory, f"model/role_{book_name}.json")

    # 检查 role_json_path 是否存在
    if not os.path.exists(role_json_path):
        # 定义模板 JSON 文件的路径
        template_json_path = os.path.join(base_directory, "model/role.json")

        # 检查模板 JSON 文件是否存在
        if os.path.exists(template_json_path):
            # 读取模板 JSON 文件
            with open(template_json_path, "r", encoding="utf-8-sig") as template_file:
                template_data = json.load(template_file)

            # 生成新的 JSON 文件
            with open(role_json_path, "w", encoding="utf-8") as new_file:
                json.dump(template_data, new_file, ensure_ascii=False, indent=4)
            print(f"Generated {role_json_path} from {template_json_path}")
        else:
            print(f"Template file {template_json_path} does not exist.")
    else:
        print(f"{role_json_path} already exists.")

    role_map = generate_role(role_json_path)

    role_mapping_path = os.path.join(
        base_directory, f"tmp/{book_name}/bookname2role.json"
    )

    # 检查文件是否存在
    if not os.path.exists(role_mapping_path):
        # 如果文件不存在则创建一个空的字典
        data = {}
        # 创建必要的目录结构
        os.makedirs(os.path.dirname(role_mapping_path), exist_ok=True)
        # 将空字典写入文件
        with open(role_mapping_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    else:
        # 如果文件存在则读取其内容
        with open(role_mapping_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    # 读取 book_name 目录下所有 chapter_{idx}.json 文件，按照 idx 升序排列
    role_directory_path = os.path.join(base_directory, f"tmp/{book_name}/role")

    # 如果目录不存在则创建
    os.makedirs(role_directory_path, exist_ok=True)

    # 获取目录下所有文件
    files = [
        f
        for f in os.listdir(role_directory_path)
        if f.startswith("chapter_") and f.endswith(".json")
    ]

    # 按照 idx 升序排序
    files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

    # 初始化角色轮询器
    role_cyclers = {}

    # 读取所有文件内容并存储在列表中
    for file in files:
        file_path = os.path.join(role_directory_path, file)
        with open(file_path, "r", encoding="utf-8") as f:
            content = json.load(f)
            if isinstance(content, list):
                for item in content:
                    for key, value in item.items():
                        if data.get(value.get("role", "")) is None:
                            role_name = select_role_by_gender_and_age(
                                value.get("gender", "男"),
                                value.get("age", "青年"),
                                role_map,
                                role_cyclers,
                            )
                            data[value["role"]] = role_name

    # 将数据写入文件
    with open(role_mapping_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Role generation script.")
    parser.add_argument(
        "book_name",
        type=str,
        help="Name of the book.",
        default="深海余烬",
    )
    parser.add_argument(
        "--base_directory",
        type=str,
        default="/root/code/GPT-SoVITS",
        help="Base directory path.",
    )

    args = parser.parse_args()

    main(args.base_directory, args.book_name)
