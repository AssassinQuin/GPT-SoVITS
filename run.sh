#!/bin/bash

if pgrep -f "auto_task*" > /dev/null
then
    pgrep -f "auto_task*" | xargs kill -9
    echo "运行 auto_task.py，正在杀掉进程"
else
    echo "未运行 auto_task.py"
fi

# 在后台运行新的脚本实例，并将输出重定向到 out.log
nohup /root/miniconda3/envs/GPTSoVits/bin/python auto_task_v2.py 谜海归巢 1 57 > out.log 2>&1 &
# nohup /root/miniconda3/envs/GPTSoVits/bin/python auto_task_v2.py --book_name 我在精神病院学斩神-三九音域 --start_idx 829 --end_idx 900 > out.log 2>&1 &
# nohup /root/miniconda3/envs/GPTSoVits/bin/python auto_task_v3.py  诡秘之主  174  300 > out.log 2>&1 &
