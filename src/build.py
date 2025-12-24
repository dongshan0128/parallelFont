from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from src import (ContentEncoder, 
                 StyleEncoder, 
                 UNet,
                 SCR)


def build_unet(args):
    unet = UNet(
        sample_size=args.resolution,
        in_channels=3,
        out_channels=3,
        flip_sin_to_cos=True,
        freq_shift=0,
        down_block_types=('DownBlock2D', 
                          'MCADownBlock2D',
                          'MCADownBlock2D', 
                          'DownBlock2D'),
        up_block_types=('UpBlock2D', 
                        'StyleRSIUpBlock2D',
                        'StyleRSIUpBlock2D', 
                        'UpBlock2D'),
        block_out_channels=args.unet_channels, 
        layers_per_block=2,
        downsample_padding=1,
        mid_block_scale_factor=1,
        act_fn='silu',
        norm_num_groups=32,
        norm_eps=1e-05,
        cross_attention_dim=args.style_start_channel * 16,
        attention_head_dim=1,
        channel_attn=args.channel_attn,
        content_encoder_downsample_size=args.content_encoder_downsample_size,
        content_start_channel=args.content_start_channel,
        reduction=32)
    
    return unet

def build_unet_with_parallel_layers(args):
    """构建包含平行处理层的5层UNet"""
    
    unet = UNet(
        sample_size=args. resolution,
        in_channels=3,
        out_channels=3,
        flip_sin_to_cos=True,
        freq_shift=0,
        
        # 5层下采样，中间插入平行处理层
        down_block_types=('DownBlock2D',           # Layer 0: 64×64 → 32×32  
                          'MCADownBlock2D',        # Layer 1: 32×32 → 16×16
                          'ParallelDownBlock2D',   # Layer 2: 16×16 → 16×16 (平行精炼)
                          'MCADownBlock2D',        # Layer 3: 16×16 → 8×8
                          'DownBlock2D'),          # Layer 4: 8×8 → 4×4
        
        # 5层上采样，对应插入平行处理层  
        up_block_types=('UpBlock2D',               # Layer 0: 4×4 → 8×8
                        'StyleRSIUpBlock2D',       # Layer 1: 8×8 → 16×16  
                        'ParallelUpBlock2D',       # Layer 2: 16×16 → 16×16 (平行精炼)
                        'StyleRSIUpBlock2D',       # Layer 3: 16×16 → 32×32
                        'UpBlock2D'),              # Layer 4: 32×32 → 64×64
        
        # 通道配置：平行层保持通道数不变
        block_out_channels=args.unet_channels ,  # (64, 128, 128, 256, 512)
        
        # # 平行层配置
        # parallel_layer_indices=[2],  # 第2层是平行处理层
        
        layers_per_block=2,
        downsample_padding=1,
        mid_block_scale_factor=1,
        act_fn='silu',
        norm_num_groups=32,
        norm_eps=1e-05,
        cross_attention_dim=args.style_start_channel * 16,
        attention_head_dim=1,
        channel_attn=args.channel_attn,
        content_encoder_downsample_size=args.content_encoder_downsample_size,
        content_start_channel=args.content_start_channel,
        reduction=32,
        window_size=4)
    
    print("Built UNet with parallel processing layers!")
    
    
    return unet

def build_style_encoder(args):
    style_image_encoder = StyleEncoder(
        G_ch=args.style_start_channel,
        resolution=args.style_image_size[0])
    print("Get CG-GAN Style Encoder!")
    return style_image_encoder


def build_content_encoder(args):
    content_image_encoder = ContentEncoder(
        G_ch=args.content_start_channel,
        resolution=args.content_image_size[0])
    print("Get CG-GAN Content Encoder!")
    return content_image_encoder


def build_scr(args):
    scr = SCR(
        temperature=args.temperature,
        mode=args.mode,
        image_size=args.scr_image_size)
    print("Loaded SCR module for supervision successfully!")
    return scr


def build_ddpm_scheduler(args):
    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule=args.beta_scheduler,
        trained_betas=None,
        variance_type="fixed_small",
        clip_sample=True)
    return ddpm_scheduler