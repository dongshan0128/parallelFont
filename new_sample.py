import os
import cv2
import time
import random
import numpy as np
from PIL import Image
import json

import torch
import torchvision.transforms as transforms
from accelerate.utils import set_seed

from src import (FontDiffuserDPMPipeline,
                 FontDiffuserModelDPM,
                 build_ddpm_scheduler,
                 build_unet,
                 build_unet_with_parallel_layers,
                 build_content_encoder,
                 build_style_encoder)
from utils import (ttf2im,
                   load_ttf,
                   is_char_in_font,
                   save_args_to_yaml,
                   save_single_image,
                   save_result_image,
                   save_image_with_content_style)

# from gpu_monitor import GPUMonitor

def arg_parse():
    from configs.fontdiffuser import get_parser

    parser = get_parser()
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--controlnet", type=bool, default=False, 
                        help="If in demo mode, the controlnet can be added.")
    parser.add_argument("--character_input", action="store_true")
    parser.add_argument("--test_type", type=str, default="char", help="The type of test.")
    parser.add_argument("--content_character", type=str, default=None)
    parser.add_argument("--content_image_path", type=str, default=None)
    parser.add_argument("--style_image_path", type=str, default=None)
    parser.add_argument("--style_ttf_path", type=str, default=None, help="Style TTF file path")
    parser.add_argument("--style_content", type=str, default=None, help="Characters to extract from style TTF")
    parser.add_argument("--save_image", action="store_true")
    parser.add_argument("--save_image_dir", type=str, default=None,
                        help="The saving directory.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ttf_path", type=str, default="ttf/KaiXinSongA.ttf")
    args = parser.parse_args()
    style_image_size = args.style_image_size
    content_image_size = args.content_image_size
    args.save_image_dir = os.path.join(args.save_image_dir, args.test_type)
    args.style_image_size = (style_image_size, style_image_size)
    args.content_image_size = (content_image_size, content_image_size)

    return args


def image_process(args, content_image=None, style_image=None):
    if not args.demo:
        # Read content image and style image
        if args.character_input:
            assert args.content_character is not None, "The content_character should not be None."
            if not is_char_in_font(font_path=args.ttf_path, char=args.content_character):
                return None, None
            font = load_ttf(ttf_path=args.ttf_path)
            content_image = ttf2im(font=font, char=args.content_character)
            content_image_pil = content_image.copy()
        else:
            content_image = Image.open(args.content_image_path).convert('RGB')
            content_image_pil = None
        style_image = Image.open(args.style_image_path).convert('RGB')
    else:
        assert style_image is not None, "The style image should not be None."
        if args.character_input:
            assert args.content_character is not None, "The content_character should not be None."
            if not is_char_in_font(font_path=args.ttf_path, char=args.content_character):
                return None, None
            font = load_ttf(ttf_path=args.ttf_path)
            content_image = ttf2im(font=font, char=args.content_character)
        else:
            assert content_image is not None, "The content image should not be None."
        content_image_pil = None
        
    ## Dataset transform
    content_inference_transforms = transforms.Compose(
        [transforms.Resize(args.content_image_size, \
                            interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])])
    style_inference_transforms = transforms.Compose(
        [transforms.Resize(args.style_image_size, \
                           interpolation=transforms.InterpolationMode.BILINEAR),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    content_image = content_inference_transforms(content_image)[None, :]
    style_image = style_inference_transforms(style_image)[None, :]

    return content_image, style_image, content_image_pil

# 加载模型
def load_fontdiffuer_pipeline(args):
    # Load the model state_dict
    # unet = build_unet(args=args)
    unet = build_unet_with_parallel_layers(args=args)
    unet.load_state_dict(torch.load(f"{args.ckpt_dir}/unet.pth"))
    style_encoder = build_style_encoder(args=args)
    style_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/style_encoder.pth"))
    content_encoder = build_content_encoder(args=args)
    content_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/content_encoder.pth"))
    model = FontDiffuserModelDPM(
        unet=unet,
        style_encoder=style_encoder,
        content_encoder=content_encoder)
    model.to(args.device)
    print("Loaded the model state_dict successfully!")

    # Load the training ddpm_scheduler.
    train_scheduler = build_ddpm_scheduler(args=args)
    print("Loaded training DDPM scheduler sucessfully!")

    # Load the DPM_Solver to generate the sample.
    pipe = FontDiffuserDPMPipeline(
        model=model,
        ddpm_train_scheduler=train_scheduler,
        model_type=args.model_type,
        guidance_type=args.guidance_type,
        guidance_scale=args.guidance_scale,
    )
    print("Loaded dpm_solver pipeline sucessfully!")

    return pipe


def sampling(args, pipe, content_image=None, style_image=None):
    if not args.demo:
        os.makedirs(args.save_image_dir, exist_ok=True)
        # saving sampling config

        save_args_to_yaml(args=args, output_file=f"{args.save_image_dir}/sampling_config.yaml")

    if args.seed:
        set_seed(seed=args.seed)
    
    content_image, style_image, content_image_pil = image_process(args=args, 
                                                                  content_image=content_image, 
                                                                  style_image=style_image)
    if content_image == None:
        print(f"The content_character you provided is not in the ttf. \
                Please change the content_character or you can change the ttf.")
        return None

    with torch.no_grad():
        content_image = content_image.to(args.device)
        style_image = style_image.to(args.device)
        print(f"Sampling by DPM-Solver++ ......")
        start = time.time()
        images = pipe.generate(
            content_images=content_image,
            style_images=style_image,
            batch_size=1,
            order=args.order,
            num_inference_step=args.num_inference_steps,
            content_encoder_downsample_size=args.content_encoder_downsample_size,
            t_start=args.t_start,
            t_end=args.t_end,
            dm_size=args.content_image_size,
            algorithm_type=args.algorithm_type,
            skip_type=args.skip_type,
            method=args.method,
            correcting_x0_fn=args.correcting_x0_fn)
        end = time.time()

        if args.save_image:
            print(f"Saving the image ......")
            save_single_image(save_dir=args.save_image_dir, image=images[0])
            if args.character_input:
                save_image_with_content_style(save_dir=args.save_image_dir,
                                            image=images[0],
                                            content_image_pil=content_image_pil,
                                            content_image_path=None,
                                            style_image_path=args.style_image_path,
                                            resolution=args.resolution)
            else:
                save_image_with_content_style(save_dir=args.save_image_dir,
                                            image=images[0],
                                            content_image_pil=None,
                                            content_image_path=args.content_image_path,
                                            style_image_path=args.style_image_path,
                                            resolution=args.resolution)
            print(f"Finish the sampling process, costing time {end - start}s")
        return images[0]


def load_controlnet_pipeline(args,
                             config_path="lllyasviel/sd-controlnet-canny", 
                             ckpt_path="runwayml/stable-diffusion-v1-5"):
    from diffusers import ControlNetModel, AutoencoderKL
    # load controlnet model and pipeline
    from diffusers import StableDiffusionControlNetPipeline, UniPCMultistepScheduler
    controlnet = ControlNetModel.from_pretrained(config_path, 
                                                 torch_dtype=torch.float16,
                                                 cache_dir=f"{args.ckpt_dir}/controlnet")
    print(f"Loaded ControlNet Model Successfully!")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(ckpt_path, 
                                                             controlnet=controlnet, 
                                                             torch_dtype=torch.float16,
                                                             cache_dir=f"{args.ckpt_dir}/controlnet_pipeline")
    # faster
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    print(f"Loaded ControlNet Pipeline Successfully!")

    return pipe


def controlnet(text_prompt, 
               pil_image,
               pipe):
    image = np.array(pil_image)
    # get canny image
    image = cv2.Canny(image=image, threshold1=100, threshold2=200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    
    seed = random.randint(0, 10000)
    generator = torch.manual_seed(seed)
    image = pipe(text_prompt, 
                 num_inference_steps=50, 
                 generator=generator, 
                 image=canny_image,
                 output_type='pil').images[0]
    return image


def load_instructpix2pix_pipeline(args,
                                  ckpt_path="timbrooks/instruct-pix2pix"):
    from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(ckpt_path, 
                                                                  torch_dtype=torch.float16)
    pipe.to(args.device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    return pipe

def instructpix2pix(pil_image, text_prompt, pipe):
    image = pil_image.resize((512, 512))
    seed = random.randint(0, 10000)
    generator = torch.manual_seed(seed)
    image = pipe(prompt=text_prompt, image=image, generator=generator, 
                 num_inference_steps=20, image_guidance_scale=1.1).images[0]

    return image



def batch_sampling(args, pipe):
    os.makedirs(args.save_image_dir, exist_ok=True)
    # 只保存一次配置
    # save_args_to_yaml(args=args, output_file=f"{args.save_image_dir}/sampling_config.yaml")

    # 获取所有content图片路径
    content_dir = args.content_image_path
    # 获取content图片路径列表
    content_image_paths = []
    
    # 检查是否提供了content_character参数
    if hasattr(args, 'content_character') and args.content_character and os.path.exists(args.content_character):
        # 如果提供了content.json文件，则根据其中的值来选择content_image
        print(f"Using content.json file: {args.content_character}")
        try:
            with open(args.content_character, 'r', encoding='utf-8') as f:
                content_data = json.load(f)
            
            # 遍历content.json中的所有值（字符）
            for char_code, character in content_data.items():
                # 构建图像路径: content_dir + character + .png
                img_path = os.path.join(content_dir, f"{character}.png")
                if os.path.exists(img_path):
                    content_image_paths.append(img_path)
                else:
                    # 如果PNG不存在，尝试其他格式
                    for ext in ['.jpg', '.jpeg', '.bmp']:
                        img_path = os.path.join(content_dir, f"{character}{ext}")
                        if os.path.exists(img_path):
                            content_image_paths.append(img_path)
                            break
                    else:
                        print(f"Warning: Content image for character '{character}' not found in {args.content_image_path}")
        except Exception as e:
            print(f"Error loading content file {args.content_character}: {e}")
            # 如果加载content.json失败，回退到遍历所有图片
            content_image_paths = [os.path.join(content_dir, fname)
                                 for fname in os.listdir(content_dir)
                                 if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    else:
        # 如果没有提供content.json，则遍历content_image_path下的所有图片
        content_image_paths = [os.path.join(content_dir, fname)
                             for fname in os.listdir(content_dir)
                             if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]


    for img_path in sorted(content_image_paths):
        args.content_image_path = img_path
        # 只保存生成图片，不保存content-style图
        content_image, style_image, _ = image_process(args=args)
        if content_image is None:
            print(f"Skip {img_path}: content_image is None.")
            continue
        with torch.no_grad():
            content_image = content_image.to(args.device)
            style_image = style_image.to(args.device)
            images = pipe.generate(
                content_images=content_image,
                style_images=style_image,
                batch_size=1,
                order=args.order,
                num_inference_step=args.num_inference_steps,
                content_encoder_downsample_size=args.content_encoder_downsample_size,
                t_start=args.t_start,
                t_end=args.t_end,
                dm_size=args.content_image_size,
                algorithm_type=args.algorithm_type,
                skip_type=args.skip_type,
                method=args.method,
                correcting_x0_fn=args.correcting_x0_fn)
            # 使用原图片名保存
            base_name = os.path.basename(img_path)
            save_path = os.path.join(args.save_image_dir, base_name)
            save_result_image(save_dir=save_path, image=images[0])
            print(f"Saved: {save_path}")

    args.content_image_path = content_dir  # 恢复原始路径

def extract_characters_from_ttf(ttf_path, output_dir, characters, image_size=128, char_size=100):
    """
    从 TTF 文件中生成指定字符的 PNG 图像。
    
    Args:
        ttf_path (str): TTF 文件的完整路径。
        output_dir (str): 生成的 PNG 图像的保存目录。
        characters (str or list): 要提取的字符。
        image_size (int): 生成图像的尺寸。
        char_size (int): 字符的尺寸。
        
    Returns:
        bool: 是否成功生成字符图像
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有TTF文件路径
        ttf_files = []
        if os.path.isfile(ttf_path) and ttf_path.lower().endswith(('.ttf', '.otf')):
            # 单个TTF文件
            ttf_files = [ttf_path]
        elif os.path.isdir(ttf_path):
            # 目录中的所有TTF文件
            ttf_files = [os.path.join(ttf_path, f) for f in os.listdir(ttf_path) 
                        if f.lower().endswith(('.ttf', '.otf'))]
        
        if not ttf_files:
            print(f"No TTF files found in {ttf_path}")
            return False
        
        # 确保characters是列表格式
        if isinstance(characters, str):
            characters = list(characters)
        
        total_generated = 0
        
        # 处理每个TTF文件
        for ttf_file in ttf_files:
            try:
                # 获取TTF文件名（不含扩展名）作为字体名称
                font_name = os.path.splitext(os.path.basename(ttf_file))[0]
                
                # 加载字体
                font = ImageFont.truetype(ttf_file, size=char_size)
                
                # 遍历所有需要生成的字符
                for character in characters:
                    try:
                        # 创建图像
                        img = Image.new("L", (image_size, image_size), 255)
                        draw = ImageDraw.Draw(img)
                        
                        # 计算字符位置使其居中
                        x_offset = (image_size - char_size) / 2
                        y_offset = (image_size - char_size) / 2
                        
                        # 绘制字符
                        draw.text((x_offset, y_offset), character, 0, font=font)
                        
                        # 保存图像
                        # 处理特殊字符，避免文件名问题
                        safe_char = "".join(c for c in character if c.isalnum() or c in (' ', '.', '_', '-'))
                        if not safe_char:
                            safe_char = f"char_{ord(character)}" if len(character) == 1 else "multi_char"
                        
                        # 文件名格式: 字体名+字符.png
                        output_path = os.path.join(output_dir, f"{font_name}+{safe_char}.png")
                        img.save(output_path)
                        total_generated += 1
                        print(f"Saved character '{character}' from font '{font_name}' to {output_path}")
                        
                    except Exception as e:
                        print(f"警告: 生成字符 '{character}' 时出错 (font: {font_name}): {e}")
                
            except Exception as e:
                print(f"处理TTF文件 {ttf_file} 时出错: {e}")
                continue
        
        print(f"成功从 {len(ttf_files)} 个TTF文件中生成了 {total_generated} 个字符图像")
        return total_generated > 0
        
    except Exception as e:
        print(f"从 TTF 生成字符时出错: {e}")
        return False

def batch_style_sampling(args, pipe):
    """
    遍历style目录下的图片，循环调用batch_sampling实现自动采样
    """
    # 检查是否提供了TTF文件路径
    if hasattr(args, 'style_ttf_path') and args.style_ttf_path and os.path.exists(args.style_ttf_path):
        # 如果提供了TTF文件路径
        style_ttf_path = args.style_ttf_path
        
        # 获取要提取的字符
        characters = getattr(args, 'style_content', '永')  # 默认字符
        if not characters:
            characters = '永'
        
        # 创建存储路径: style_ttf_path同级目录下的styleImg文件夹
        if os.path.isfile(style_ttf_path):
            style_img_dir = os.path.join(os.path.dirname(style_ttf_path), "styleImg")
        else:  # 目录
            style_img_dir = os.path.join(style_ttf_path, "styleImg")
        
        os.makedirs(style_img_dir, exist_ok=True)
        
        # 提取字符图像
        if extract_characters_from_ttf(style_ttf_path, style_img_dir, characters):
            # 更新style_image_path为生成的图像目录
            args.style_image_path = style_img_dir
            print(f"Extracted characters to {style_img_dir}, updated style_image_path")
        else:
            print(f"Failed to extract characters from {style_ttf_path}")
            return
    elif hasattr(args, 'style_image_path') and args.style_image_path:
        # 如果直接提供了style_image_path，使用它
        pass
    else:
        print("No style path provided")
        return
    
    # 获取style目录路径
    style_dir = args.style_image_path
    if not style_dir or not os.path.exists(style_dir):
        print(f"Warning: Style directory not found at {style_dir}")
        return
    
    # 获取所有图片文件 - 使用与batch_sampling相同的风格
    style_image_paths = [os.path.join(style_dir, fname)
                        for fname in os.listdir(style_dir)
                        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    
    if not style_image_paths:
        print(f"Warning: No image files found in {style_dir}")
        return
    
    # 保存原始的save_image_dir
    original_save_dir = args.save_image_dir
    
    # 对每个风格图像进行采样
    for style_img_path in sorted(style_image_paths):
        try:
            # 获取图片文件名（不含扩展名）
            filename = os.path.splitext(os.path.basename(style_img_path))[0]
            
            # 提取"+"号前的内容作为风格名称
            if '+' in filename:
                style_name = filename.split('+')[0]
            else:
                style_name = filename
                
            print(f"Processing style: {style_name} from {style_img_path}")
            
            # 设置当前风格图像路径
            args.style_image_path = style_img_path
            
            # 设置保存路径: save_image_dir + 风格名称
            args.save_image_dir = os.path.join(original_save_dir, style_name)
            os.makedirs(args.save_image_dir, exist_ok=True)
            
            print(f"Saving results to: {args.save_image_dir}")
            
            # 调用batch_sampling进行采样
            batch_sampling(args, pipe)
            
        except Exception as e:
            print(f"Error processing style image {style_img_path}: {e}")
            continue
    
    # 恢复原始参数
    args.save_image_dir = original_save_dir

if __name__=="__main__":
    args = arg_parse()
    
    # 开始总计时
    total_start = time.time()

    # gpu_monitor = GPUMonitor(interval=0.05)  # 每0.05秒采样一次
    # gpu_monitor.start_monitoring()

    # try:
    #     # 加载fontdiffuser pipeline
    #     pipe = load_fontdiffuer_pipeline(args=args)
    #     batch_style_sampling(args=args, pipe=pipe)
    #     # sampling(args=args, pipe=pipe)
    # finally:
    #     # 停止GPU监控并打印结果
    #     gpu_monitor.print_results()

    # 加载fontdiffuser pipeline
    pipe = load_fontdiffuer_pipeline(args=args)
    batch_style_sampling(args=args, pipe=pipe)
    # sampling(args=args, pipe=pipe)
    

    # 结束总计时
    total_end = time.time()
    print(f"总执行时间: {total_end - total_start}秒")

