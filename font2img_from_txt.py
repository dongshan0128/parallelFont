from PIL import Image, ImageDraw, ImageFont
import os
import pathlib
import argparse

# 设置命令行参数
parser = argparse.ArgumentParser(description='Obtaining characters from .ttf')
parser.add_argument('--ttf_path', type=str, default='data_examples/test', help='ttf directory')
parser.add_argument('--chara', type=str, default='chars.txt', help='characters')
parser.add_argument('--save_path', type=str, default='data_examples/test/xingshu_content_train', help='images directory')
parser.add_argument('--img_size', type=int, default=128, help='The size of generated images')
parser.add_argument('--chara_size', type=int, default=100, help='The size of generated characters')
args = parser.parse_args()

# 读取字符文件
with open(args.chara, encoding='utf-8') as file_object:
    characters = file_object.read()

# 绘制单个字符的图像
def draw_single_char(ch, font, canvas_size, x_offset, y_offset):
    img = Image.new("L", (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(img)
    draw.text((x_offset, y_offset), ch, 0, font=font)
    return img

# 使用指定字体和大小绘制字符
def draw_example(ch, src_font, canvas_size, x_offset, y_offset):
    src_img = draw_single_char(ch, src_font, canvas_size, x_offset, y_offset)
    return src_img

# 获取所有.ttf文件路径
data_dir = args.ttf_path
data_root = pathlib.Path(data_dir)

all_ttf_paths = list(data_root.glob('*.TTF')) + list(data_root.glob('*.ttf'))
all_ttf_paths = [str(path) for path in all_ttf_paths]

# 遍历所有.ttf文件，生成字符图像
for ttf_path in all_ttf_paths:
    ttf_filename = os.path.splitext(os.path.basename(ttf_path))[0]  # 获取字体文件名（去掉扩展名）
    path_full = os.path.join(args.save_path, ttf_filename)

    # 如果保存路径不存在，则创建目录
    if not os.path.exists(path_full):
        os.makedirs(path_full)

    # 使用当前ttf字体文件加载字体
    src_font = ImageFont.truetype(ttf_path, size=args.chara_size)

    # 遍历所有字符并生成图像
    for chara in characters:
        try:
            img = draw_example(chara, src_font, args.img_size, (args.img_size - args.chara_size) / 2,
                               (args.img_size - args.chara_size) / 2)

            # 创建新的文件名，格式为 风格名 + 字符名
            # 例如：FZHTJW--GB1-0+一.jpg
            valid_chara = "".join([c for c in chara if c.isalnum() or c in (' ', '.', '_')])  # 确保字符有效
            if not valid_chara:
                valid_chara = "default"  # 如果无效，则使用默认字符名

            # 拼接新文件名
            # filename = f"{ttf_filename}+{valid_chara}.jpg"
            filename = f"{valid_chara}.jpg"
            
            # 保存图片
            img.save(os.path.join(path_full, filename))
            print(f"Saved: {os.path.join(path_full, filename)}")

        except Exception as e:
            print(f"Error saving character '{chara}': {e}")
