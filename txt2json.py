import json
#将chars.txt 转为 chars.json
# 1. 读取 txt 内容
input_file = 'chars.txt'      # 替换为你实际的 txt 路径
output_file = 'train_chars.json'      # 输出的 json 文件名

with open(input_file, 'r', encoding='utf-8') as f:
    text = f.read().strip()  # 去掉首尾空白

# 2. 遍历每个字符，构建字典
char_dict = {}
for char in text:
    code_point = ord(char)  # 取出字符的 Unicode 十进制码点
    char_dict[str(code_point)] = char  # 用字符串形式保存键（避免整数类型问题）

# 3. 保存为 JSON
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(char_dict, f, indent=4, ensure_ascii=False)

print(f"✅ 已成功生成 JSON，共 {len(char_dict)} 个字符。")
