import os
import argparse
from typing import Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from configs.fontdiffuser import get_parser
from src import (
    FontDiffuserModel,
    build_unet_with_parallel_layers,
    build_style_encoder,
    build_content_encoder,
    build_ddpm_scheduler,
)
from utils import reNormalize_img  # 用于[-1,1] -> [0,1]可视化


def load_image(path: str, size: Tuple[int, int], normalize=True) -> torch.Tensor:
    tfms = [
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ]
    if normalize:
        tfms.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))  # 三通道
    tfms = transforms.Compose(tfms)
    img = Image.open(path).convert("RGB")  # 改为RGB
    return tfms(img)  # 返回 [3,H,W]


@torch.no_grad()
def infer_phase1(
    ckpt_dir: str,
    content_path: str,
    style_path: str,
    out_path: str,
    seed: int = 123,
    num_inference_steps: int = None,
    device: str = "cuda",
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 1) 读取训练配置（保持与训练一致）
    args = get_parser().parse_args([])

    # 统一为 (H, W) tuple（训练时在 get_args() 里也做了类似转换）
    if isinstance(args.style_image_size, int):
        args.style_image_size = (args.style_image_size, args.style_image_size)
    if isinstance(args.content_image_size, int):
        args.content_image_size = (args.content_image_size, args.content_image_size)

    style_size = args.style_image_size
    content_size = args.content_image_size

    # 2) 构建模型组件（使用 parallel 版本的 UNet）
    unet = build_unet_with_parallel_layers(args=args).to(device)
    style_encoder = build_style_encoder(args=args).to(device)
    content_encoder = build_content_encoder(args=args).to(device)
    scheduler = build_ddpm_scheduler(args)

    # 3) 加载 Phase 1 的权重
    unet.load_state_dict(torch.load(os.path.join(ckpt_dir, "unet.pth"), map_location="cpu"))
    style_encoder.load_state_dict(torch.load(os.path.join(ckpt_dir, "style_encoder.pth"), map_location="cpu"))
    content_encoder.load_state_dict(torch.load(os.path.join(ckpt_dir, "content_encoder.pth"), map_location="cpu"))

    # 4) 组装推理模型包装
    model = FontDiffuserModel(unet=unet, style_encoder=style_encoder, content_encoder=content_encoder).to(device)
    model.eval()

    # 5) 读取输入图片并规范化到训练时的分布
    content_img = load_image(content_path, content_size).unsqueeze(0).to(device)   # [1,1,H,W], [-1,1]
    style_img = load_image(style_path, style_size).unsqueeze(0).to(device)         # [1,1,H,W], [-1,1]

    # 6) DDPM 采样
    torch.manual_seed(seed)
    scheduler.set_timesteps(num_inference_steps or scheduler.config.num_train_timesteps, device=device)

    # 初始噪声（形状需与训练目标一致）
    # 通道数取决于你的数据集/UNet配置；通常是1通道字体图。如果你的UNet是4通道，请改为 (1,4,H,W) 并按训练方式处理。
    B, _, H, W = content_img.shape
    x = torch.randn((B, unet.config.out_channels if hasattr(unet, "config") and hasattr(unet.config, "out_channels") else content_img.shape[1], H, W), device=device)

    for t in scheduler.timesteps:
        # 模型预测噪声（训练里 forward 的参数保持一致）
        # 你的 FontDiffuserModel 在训练中是:
        # noise_pred, _ = model(x_t, timesteps, style_images, content_images, content_encoder_downsample_size)
        noise_pred, _ = model(
            x_t=x,
            timesteps=t,
            style_images=style_img,
            content_images=content_img,
            content_encoder_downsample_size=args.content_encoder_downsample_size,
        )
        # 根据扩散调度器更新 x
        step_out = scheduler.step(noise_pred, t, x)
        x = step_out.prev_sample

    # 7) 反归一化并保存
    # x 当前在 [-1,1] 分布（与训练一致），reNormalize_img -> [0,1]
    # 反归一化后
    x_img01 = reNormalize_img(x.clamp(-1, 1))  # [B,C,H,W], C=3
    out = (x_img01[0].permute(1,2,0).cpu().numpy() * 255.0).clip(0,255).astype("uint8")
    Image.fromarray(out).save(out_path)
    print(f"[OK] Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser("Phase-1 only inference")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Path to Phase-1 checkpoint folder (contains unet.pth, style_encoder.pth, content_encoder.pth)")
    parser.add_argument("--content", type=str, required=True, help="Path to a content image")
    parser.add_argument("--style", type=str, required=True, help="Path to a style reference image")
    parser.add_argument("--output", type=str, required=True, help="Path to save the generated image")
    parser.add_argument("--steps", type=int, default=None, help="Number of inference steps, default uses scheduler's num_train_timesteps")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    infer_phase1(
        ckpt_dir=args.ckpt_dir,
        content_path=args.content,
        style_path=args.style,
        out_path=args.output,
        seed=args.seed,
        num_inference_steps=args.steps,
        device=args.device,
    )


if __name__ == "__main__":
    main()