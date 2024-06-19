# 终止已经运行的相同脚本实例
if pgrep -f "python auto_task/auto_task.py" > /dev/null
then
    pgrep -f "python auto_task/auto_task.py" | xargs kill -9
    echo "Terminated existing instances of auto_task.py"
else
    echo "No existing instances of auto_task.py found"
fi

# 在后台运行新的脚本实例，并将输出重定向到 out.log
nohup python auto_task/auto_task.py --novel_name shyj --start 21 --end 30 > out.log 2>&1 &
echo "Started new instance of auto_task.py"
