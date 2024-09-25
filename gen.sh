#!/bin/bash

# 定义源目录和目标目录
src_directory="model"
dst_directory="model"

# 遍历源目录中的所有文件
for file in "$src_directory"/*; do
    # 提取文件名（不包括路径）
    filename=$(basename "$file")
    
    # 提取文件名中的角色名 (假设角色名在下划线前面)
    character=$(echo "$filename" | awk -F'_' '{print $1}')
    
    # 构建目标目录路径
    target_dir="$dst_directory/$character"
    
    # 检查目标目录是否存在
    if [ ! -d "$target_dir" ]; then
        echo "目标目录不存在: $target_dir"
        continue
    fi
    
    # 移动文件到目标目录
    mv "$file" "$target_dir/"
    echo "移动 $filename 到 $target_dir/"
done
