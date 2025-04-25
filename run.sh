#!/bin/bash

# 检查是否有参数传入
if [ -z "$1" ]; then
    echo "请输入参数："
    echo "  a (auto_task) - 运行 auto_task.py"
    echo "  c (check_spk) - 运行 check_spk.py"
    echo "  e (stop)      - 停止所有相关进程"
    exit 1
fi

# 根据参数选择操作
if [[ $1 == "c" ]]; then
    SCRIPT="tools/check_spk.py"
    echo "运行 check_spk.py"
elif [[ $1 == "a" ]]; then
    SCRIPT="auto_task.py"
    echo "运行 auto_task.py"
elif [[ $1 == "e" ]]; then
    echo "停止所有相关进程..."
    # 杀掉所有相关的 Python 进程
    pkill -f "auto_task.py"
    pkill -f "tools/check_spk.py"
    echo "已停止所有相关进程"
    exit 0
else
    echo "无效参数，请输入以下参数之一："
    echo "  a (auto_task) - 运行 auto_task.py"
    echo "  c (check_spk) - 运行 check_spk.py"
    echo "  e (stop)      - 停止所有相关进程"
    exit 1
fi

# 检查是否有同名进程在运行
if pgrep -f "$SCRIPT" > /dev/null; then
    pgrep -f "$SCRIPT" | xargs kill -9
    echo "发现 $SCRIPT 正在运行，杀掉进程"
else
    echo "未运行 $SCRIPT"
fi

# 启动新的进程
nohup /root/miniconda3/envs/GPTSoVits/bin/python "$SCRIPT" > out.log 2>&1 &