#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from collections import defaultdict

# 全局角色列表与轮询器
male_role_list = []
female_role_list = []
index_tracker = defaultdict(int)  # 格式: { "gender_age": index }


def select_role_by_gender_and_age(gender, age):
    """
    基于性别和年龄的持久化轮询选择器
    :param gender: 目标性别（"男"/"女"）
    :param age: 目标年龄段（"青年"/"中年"/"老年"）
    :return: 符合条件角色名（优先匹配年龄，无匹配时返回全体）
    """
    # 确定目标列表
    target_list = female_role_list if gender == "女" else male_role_list

    # 生成分类键（例如 female_青年）
    category_key = f"{gender}_{age}"

    # 筛选精确匹配或全体
    filtered = [r for r in target_list if r.get("age") == age]
    if not filtered:
        filtered = target_list
        category_key = f"{gender}_any"  # 使用全体分类键

    # 获取并更新索引（保证线程安全）
    current_idx = index_tracker[category_key] % len(filtered)
    selected = filtered[current_idx]["name"]

    # 持久化索引（确保下次调用时递增）
    index_tracker[category_key] = current_idx + 1

    return selected


def cache_default_role_map(base_directory, book_name):
    """
    初始化角色列表并创建轮询器
    """
    global male_role_list, female_role_list, index_tracker

    # 路径处理
    book_role_map_path = os.path.join(
        base_directory, "model", f"{book_name}_role_info.json"
    )
    default_role_map_path = os.path.join(base_directory, "model", "role_info.json")

    # 副本创建逻辑
    if not os.path.exists(book_role_map_path):
        try:
            with open(default_role_map_path, "r", encoding="utf-8") as src:
                default_data = json.load(src)
            with open(book_role_map_path, "w", encoding="utf-8") as dst:
                json.dump(default_data, dst, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Error creating role map: {str(e)}")
            return

    # 加载角色数据
    try:
        with open(book_role_map_path, "r", encoding="utf-8") as f:
            role_map = json.load(f)
    except Exception as e:
        print(f"Error loading role map: {str(e)}")
        return

    index_tracker.clear()

    # 分类角色
    male_role_list.clear()
    female_role_list.clear()
    for role, info in role_map.items():
        if info.get("hide", False):
            continue
        info["name"] = role
        if info.get("gender") == "男":
            male_role_list.append(info)
        else:
            female_role_list.append(info)


def get_role_info_map(base_directory, book_name):
    """
    生成角色统计特征图谱
    """
    role_directory = os.path.join(base_directory, "tmp", book_name, "role")
    os.makedirs(role_directory, exist_ok=True)

    # 获取章节文件
    files = []
    try:
        files = [
            f
            for f in os.listdir(role_directory)
            if f.startswith("chapter_") and f.endswith(".json")
        ]
        files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
    except Exception as e:
        print(f"Error listing files: {str(e)}")
        return {}

    # 统计分析
    role_stats = {}
    for file in files:
        file_path = os.path.join(role_directory, file)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = json.load(f)
        except Exception as e:
            print(f"Error reading {file}: {str(e)}")
            continue

        # 处理嵌套结构
        def process_item(item):
            for key, value in item.items():
                role = value.get("role", "")
                if not role:
                    continue

                # 初始化统计项
                if role not in role_stats:
                    role_stats[role] = {
                        "gender": {"male": 0, "female": 0, "unknown": 0},
                        "age": {"young": 0, "middle": 0, "old": 0, "unknown": 0},
                    }

                # 性别统计
                gender = str(value.get("gender", "")).strip()
                if "男" in gender:
                    role_stats[role]["gender"]["male"] += 1
                elif "女" in gender:
                    role_stats[role]["gender"]["female"] += 1
                else:
                    role_stats[role]["gender"]["unknown"] += 1

                # 年龄统计
                age = str(value.get("age", "")).strip()
                if "青" in age:
                    role_stats[role]["age"]["young"] += 1
                elif "中" in age:
                    role_stats[role]["age"]["middle"] += 1
                elif "老" in age:
                    role_stats[role]["age"]["old"] += 1
                else:
                    role_stats[role]["age"]["unknown"] += 1

        # 处理不同数据结构
        if isinstance(content, list):
            for item in content:
                process_item(item)
        elif isinstance(content, dict):
            process_item(content)

    # 推断最终属性
    result = {}
    for role, stats in role_stats.items():
        # 性别推断
        male = stats["gender"]["male"]
        female = stats["gender"]["female"]
        gender = "男" if male >= female else "女"

        # 年龄推断
        ages = stats["age"]
        max_age = max(ages["young"], ages["middle"], ages["old"])
        if max_age == 0:
            age = "青年"
        else:
            if ages["young"] == max_age:
                age = "青年"
            elif ages["middle"] == max_age:
                age = "中年"
            else:
                age = "老年"

        result[role] = {"gender": gender, "age": age}

    return result


def main(base_directory, book_name):
    # 初始化角色库
    cache_default_role_map(base_directory, book_name)

    # 生成特征图谱
    role_info_map = get_role_info_map(base_directory, book_name)

    # 创建映射关系
    mapping = {}

    output_path = os.path.join(base_directory, "tmp", book_name, "bookname2role.json")
    # 读取已有数据（若文件存在）
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        with open(output_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)  # 读取已有内容
    else:
        mapping = {}  # 初始化空字典
    for role, info in role_info_map.items():
        if role not in mapping:
            mapping[role] = select_role_by_gender_and_age(info["gender"], info["age"])

    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error saving result: {str(e)}")


if __name__ == "__main__":
    base_dir = "/root/code/GPT-SoVITS"
    book_name = "趋吉避凶，从天师府开始"
    main(base_dir, book_name)
