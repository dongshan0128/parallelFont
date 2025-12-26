"""
Phase 1 Inference Script for ParallelFont
仿照 sample.py 的结构，专门用于测试第一阶段训练的模型
支持 DDPM 和 DPM-Solver++ 两种采样方式
"""
import os
import time
import yaml
import argparse
from PIL import Image

import torch
import torchvision.transforms as transforms
from accelerate. utils import set_seed
from tqdm import tqdm

from src import (
    FontDiffuserModel,
    FontDiffuserModelDPM,
    FontDiffuserDPMPipeline,
    build_ddpm_scheduler,
    build_unet_with_parallel_layers,
    build_content_encoder,
    build_style_encoder
)
from utils import (
    ttf2im,
    load_ttf,
    is_char_in_font,
    save_args_to_yaml,
    save_single_image,
    save_image_with_content_style,
    reNormalize_img
)


def setup_yaml_with_python_types():
    """设置 YAML 加载器以支持 Python 类型（如 tuple）"""
    def tuple_constructor(loader, node):
        """自定义元组构造器"""
        return tuple(loader. construct_sequence(node))
    
    # 为 safe_load 添加元组构造器
    yaml. SafeLoader.add_constructor(
        'tag:yaml.org,2002:python/tuple',
        tuple_constructor
    )


def arg_parse():
    """参数解析，仿照 sample.py 的风格"""
    from configs.fontdiffuser import get_parser

    parser = get_parser()
    
    # 添加 Phase 1 特有的参数
    parser.add_argument("--ckpt_dir", type=str, default=None,
                        help="Phase 1 checkpoint directory")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Path to training config yaml file (optional, will auto-search if not provided)")
    parser.add_argument("--character_input", action="store_true",
                        help="Use character as content input instead of image")
    parser.add_argument("--content_character", type=str, default=None,
                        help="Content character (if character_input is True)")
    parser.add_argument("--content_image_path", type=str, default=None,
                        help="Content image path (if character_input is False)")
    parser.add_argument("--style_image_path", type=str, default=None,
                        help="Style reference image path")
    parser.add_argument("--save_image", action="store_true",
                        help="Save the generated image")
    parser.add_argument("--save_image_dir", type=str, default="outputs/phase1_inference",
                        help="Directory to save generated images")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run inference on")
    parser.add_argument("--ttf_path", type=str, default="ttf/KaiXinSongA.ttf",
                        help="TTF font path for character rendering")
    parser.add_argument("--sampler", type=str, default="dpm-solver++", choices=["ddpm", "dpm-solver++"],
                        help="Sampling method:  ddpm or dpm-solver++")
    
    args = parser.parse_args()
    
    # 设置 YAML 加载器以支持 Python 类型
    setup_yaml_with_python_types()
    
    # 加载训练配置
    args = load_training_config(args)
    
    # 确保图像尺寸是元组格式
    if isinstance(args.style_image_size, int):
        args.style_image_size = (args.style_image_size, args.style_image_size)
    if isinstance(args.content_image_size, int):
        args.content_image_size = (args.content_image_size, args.content_image_size)
    if isinstance(args.unet_channels, list):
        args.unet_channels = tuple(args.unet_channels)
    
    return args


def load_training_config(args):
    """加载训练配置文件"""
    config_path = args.config_path
    
    if config_path is None and args. ckpt_dir: 
        config_path = find_config_file(args. ckpt_dir)
    
    if config_path and os.path.exists(config_path):
        print(f"📋 Loading training config from: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
        except yaml.constructor.ConstructorError:
            print("  ⚠ Using yaml.unsafe_load() due to Python-specific types")
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.unsafe_load(f)
        
        key_configs = [
            'resolution', 'unet_channels', 'style_image_size', 'content_image_size',
            'style_start_channel', 'content_start_channel', 'content_encoder_downsample_size',
            'channel_attn', 'beta_scheduler'
        ]
        
        for key in key_configs: 
            if key in config_dict:
                value = config_dict[key]
                if key == 'unet_channels' and isinstance(value, list):
                    value = tuple(value)
                setattr(args, key, value)
        
        print(f"  ✓ Loaded key configurations:")
        print(f"    - Resolution: {args.resolution}")
        print(f"    - UNet channels: {args.unet_channels}")
        print(f"    - Style image size: {args.style_image_size}")
        print(f"    - Content image size: {args.content_image_size}")
    else:
        print(f"⚠ Config file not found, using default configuration")
    
    return args


def find_config_file(ckpt_dir):
    """在检查点目录中查找配置文件"""
    possible_names = [
        "FontDiffuser_training_phase_1_config.yaml",
        "fontdiffuer_training_config.yaml",
        "train_config.yaml",
        "config.yaml",
    ]
    
    for name in possible_names:
        config_path = os.path.join(ckpt_dir, name)
        if os.path. exists(config_path):
            return config_path
    
    parent_dir = os.path.dirname(ckpt_dir)
    for name in possible_names:
        config_path = os.path. join(parent_dir, name)
        if os.path.exists(config_path):
            return config_path
    
    return None


def image_process(args, content_image=None, style_image=None):
    """图像预处理，完全仿照 sample.py"""
    if args.character_input:
        assert args.content_character is not None, "The content_character should not be None."
        if not is_char_in_font(font_path=args.ttf_path, char=args.content_character):
            return None, None, None
        font = load_ttf(ttf_path=args.ttf_path)
        content_image = ttf2im(font=font, char=args.content_character)
        content_image_pil = content_image.copy()
    else:
        assert args.content_image_path is not None, "The content_image_path should not be None."
        content_image = Image.open(args.content_image_path).convert('RGB')
        content_image_pil = None
    
    style_image = Image.open(args.style_image_path).convert('RGB')
    
    content_inference_transforms = transforms.Compose([
        transforms.Resize(args.content_image_size,
                         interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    style_inference_transforms = transforms. Compose([
        transforms.Resize(args.style_image_size,
                         interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms. Normalize([0.5], [0.5])
    ])
    
    content_image = content_inference_transforms(content_image)[None, :]
    style_image = style_inference_transforms(style_image)[None, :]
    
    return content_image, style_image, content_image_pil


def load_phase1_model_ddpm(args):
    """加载 Phase 1 模型（DDPM 采样）"""
    print(f"🔧 Building Phase 1 model for DDPM sampling...")
    
    unet = build_unet_with_parallel_layers(args=args)
    style_encoder = build_style_encoder(args=args)
    content_encoder = build_content_encoder(args=args)
    
    print(f"📦 Loading checkpoint from {args.ckpt_dir}...")
    unet.load_state_dict(torch.load(f"{args. ckpt_dir}/unet. pth", map_location="cpu"))
    print("  ✓ Loaded unet. pth")
    style_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/style_encoder.pth", map_location="cpu"))
    print("  ✓ Loaded style_encoder.pth")
    content_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/content_encoder.pth", map_location="cpu"))
    print("  ✓ Loaded content_encoder.pth")
    
    model = FontDiffuserModel(
        unet=unet,
        style_encoder=style_encoder,
        content_encoder=content_encoder
    )
    model.to(args.device)
    model.eval()
    
    scheduler = build_ddpm_scheduler(args=args)
    print("✓ Phase 1 model (DDPM) loaded successfully!")
    
    return model, scheduler


def load_phase1_pipeline_dpmsolver(args):
    """加载 Phase 1 Pipeline（DPM-Solver++ 采样），完全仿照 sample.py"""
    print(f"🔧 Building Phase 1 pipeline for DPM-Solver++ sampling...")
    
    unet = build_unet_with_parallel_layers(args=args)
    style_encoder = build_style_encoder(args=args)
    content_encoder = build_content_encoder(args=args)
    
    print(f"📦 Loading checkpoint from {args.ckpt_dir}...")
    unet.load_state_dict(torch.load(f"{args.ckpt_dir}/unet.pth", map_location="cpu"))
    print("  ✓ Loaded unet.pth")
    style_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/style_encoder.pth", map_location="cpu"))
    print("  ✓ Loaded style_encoder.pth")
    content_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/content_encoder.pth", map_location="cpu"))
    print("  ✓ Loaded content_encoder.pth")
    
    # 使用 DPM 版本的模型
    model = FontDiffuserModelDPM(
        unet=unet,
        style_encoder=style_encoder,
        content_encoder=content_encoder
    )
    model.to(args.device)
    model.eval()
    print("  ✓ Loaded model successfully!")
    
    # 加载训练调度器
    train_scheduler = build_ddpm_scheduler(args=args)
    print("  ✓ Loaded training DDPM scheduler!")
    
    # 创建 DPM-Solver Pipeline
    pipe = FontDiffuserDPMPipeline(
        model=model,
        ddpm_train_scheduler=train_scheduler,
        model_type=args.model_type,
        guidance_type=args.guidance_type,
        guidance_scale=args.guidance_scale,
    )
    print("✓ Phase 1 pipeline (DPM-Solver++) loaded successfully!")
    
    return pipe


def sampling_ddpm(args, model, scheduler, content_image=None, style_image=None):
    """使用 DDPM 采样"""
    if args.save_image:
        os.makedirs(args.save_image_dir, exist_ok=True)
        save_args_to_yaml(args=args, output_file=f"{args.save_image_dir}/phase1_sampling_config.yaml")
    
    if args.seed: 
        set_seed(seed=args.seed)
        print(f"🎲 Random seed:  {args.seed}")
    
    content_image, style_image, content_image_pil = image_process(
        args=args, 
        content_image=content_image, 
        style_image=style_image
    )
    
    if content_image is None:
        print(f"⚠ Content character not in TTF font")
        return None
    
    print(f"🖼️ Input images:")
    print(f"  - Content:  {content_image.shape}")
    print(f"  - Style: {style_image.shape}")
    
    with torch.no_grad():
        content_image = content_image.to(args.device)
        style_image = style_image.to(args.device)
        
        print(f"🎨 Sampling with DDPM...")
        start = time.time()
        
        num_steps = args.num_inference_steps
        scheduler.set_timesteps(num_steps, device=args.device)
        print(f"  - Inference steps: {num_steps}")
        
        batch_size = content_image.shape[0]
        latent_shape = (batch_size, 3, args.resolution, args.resolution)
        latents = torch.randn(latent_shape, device=args.device)
        print(f"  - Latent shape: {latents.shape}")
        
        for i, t in enumerate(tqdm(scheduler.timesteps, desc="  DDPM Sampling")):
            timesteps = t.unsqueeze(0).expand(batch_size) if t.dim() == 0 else t
            
            noise_pred, _ = model(
                x_t=latents,
                timesteps=timesteps,
                style_images=style_image,
                content_images=content_image,
                content_encoder_downsample_size=args.content_encoder_downsample_size,
            )
            
            step_output = scheduler.step(noise_pred, t, latents)
            latents = step_output.prev_sample
        
        images_tensor = reNormalize_img(latents. clamp(-1, 1))
        image_array = (images_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype('uint8')
        output_image = Image.fromarray(image_array, mode='RGB')
        
        end = time.time()
        
        if args.save_image:
            print(f"💾 Saving generated image...")
            save_single_image(save_dir=args.save_image_dir, image=output_image)
            
            if args.character_input:
                save_image_with_content_style(
                    save_dir=args.save_image_dir,
                    image=output_image,
                    content_image_pil=content_image_pil,
                    content_image_path=None,
                    style_image_path=args.style_image_path,
                    resolution=args.resolution
                )
            else:
                save_image_with_content_style(
                    save_dir=args.save_image_dir,
                    image=output_image,
                    content_image_pil=None,
                    content_image_path=args.content_image_path,
                    style_image_path=args.style_image_path,
                    resolution=args.resolution
                )
            
            print(f"  ✓ Saved to:  {args.save_image_dir}")
        
        print(f"✨ Sampling completed!  Time: {end - start:.2f}s")
        
        return output_image


def sampling_dpmsolver(args, pipe, content_image=None, style_image=None):
    """使用 DPM-Solver++ 采样，完全仿照 sample. py"""
    if args.save_image:
        os.makedirs(args.save_image_dir, exist_ok=True)
        save_args_to_yaml(args=args, output_file=f"{args.save_image_dir}/phase1_sampling_config.yaml")
    
    if args.seed:
        set_seed(seed=args.seed)
        print(f"🎲 Random seed: {args.seed}")
    
    content_image, style_image, content_image_pil = image_process(
        args=args,
        content_image=content_image,
        style_image=style_image
    )
    
    if content_image is None:
        print(f"⚠ Content character not in TTF font")
        return None
    
    print(f"🖼️ Input images:")
    print(f"  - Content: {content_image.shape}")
    print(f"  - Style: {style_image.shape}")
    
    with torch. no_grad():
        content_image = content_image.to(args.device)
        style_image = style_image.to(args.device)
        
        print(f"🎨 Sampling with DPM-Solver++...")
        start = time.time()
        
        images = pipe. generate(
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
            correcting_x0_fn=args.correcting_x0_fn
        )
        
        end = time.time()
        output_image = images[0]
        
        if args.save_image:
            print(f"💾 Saving generated image...")
            save_single_image(save_dir=args.save_image_dir, image=output_image)
            
            if args.character_input:
                save_image_with_content_style(
                    save_dir=args.save_image_dir,
                    image=output_image,
                    content_image_pil=content_image_pil,
                    content_image_path=None,
                    style_image_path=args. style_image_path,
                    resolution=args.resolution
                )
            else:
                save_image_with_content_style(
                    save_dir=args.save_image_dir,
                    image=output_image,
                    content_image_pil=None,
                    content_image_path=args. content_image_path,
                    style_image_path=args.style_image_path,
                    resolution=args.resolution
                )
            
            print(f"  ✓ Saved to: {args.save_image_dir}")
        
        print(f"✨ Sampling completed! Time: {end - start:.2f}s")
        
        return output_image


if __name__ == "__main__": 
    print("="*60)
    print("Phase 1 Inference for ParallelFont")
    print("="*60)
    
    args = arg_parse()
    
    # 验证必需参数
    if args. ckpt_dir is None: 
        raise ValueError("--ckpt_dir is required!")
    if args.style_image_path is None: 
        raise ValueError("--style_image_path is required!")
    if not args.character_input and args.content_image_path is None:
        raise ValueError("Either --content_image_path or --character_input must be provided!")
    
    # 根据采样方法选择不同的处理流程
    if args.sampler == "ddpm":
        print(f"📌 Using DDPM sampling")
        model, scheduler = load_phase1_model_ddpm(args=args)
        output_image = sampling_ddpm(args=args, model=model, scheduler=scheduler)
    else:  # dpm-solver++
        print(f"📌 Using DPM-Solver++ sampling")
        pipe = load_phase1_pipeline_dpmsolver(args=args)
        output_image = sampling_dpmsolver(args=args, pipe=pipe)
    
    print("="*60)
    print("✅ Done!")
    print("="*60)