import re

book_path = "/root/code/GPT-SoVITS/tmp/埃隆·马斯克传/埃隆·马斯克传.txt"

# 读取 a.txt 文件内容
with open(book_path, "r", encoding="utf-8") as file:
    text = file.read()

# 正则表达式匹配 "第XX节"
pattern = r"\d+ \d+"


# 替换函数：将 "节" 替换为 "章"
def replace_section_to_chapter(match):
    return match.group().replace(" ", "")


# 使用 re.sub 进行替换
result = re.sub(pattern, replace_section_to_chapter, text)

# 将结果写回 a.txt 文件（可选）
with open(book_path, "w", encoding="utf-8") as file:
    file.write(result)
