import argparse
import json
import os
from pathlib import Path
import shutil
import warnings

# 只过滤特定的警告信息
warnings.filterwarnings("ignore", message=".*pretrained.*deprecated.*", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", message=".*weights.*deprecated.*", category=UserWarning, module="torchvision")



import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm



from lpips import LPIPS
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from scipy.linalg import sqrtm
import torch.nn.functional as F

# 全局LPIPS模型实例，避免重复加载
_lpips_model = None

def get_lpips_model(device):
    """获取LPIPS模型实例，确保只初始化一次"""
    global _lpips_model
    if _lpips_model is None:
        print("Loading LPIPS model...")
        _lpips_model = LPIPS(net='vgg').to(device)
        print("LPIPS model loaded successfully!")
    return _lpips_model
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_metrics(real_images_batch, generated_images_batch, device, image_size):
    """
    计算一组图像的平均SSIM、L1 Loss、MSE、LPIPS。
    Args:
        real_images_batch (torch.Tensor): 真实图像的 Batch Tensor (N, C, H, W)。
        generated_images_batch (torch.Tensor): 生成图像的 Batch Tensor (N, C, H, W)。
        device (torch.device): 计算设备。
        image_size (int): 图像尺寸。
    Returns:
        tuple: ( avg_ssim, avg_l1, avg_mse, avg_lpips)
    """
    real_images_batch = real_images_batch.to(device)
    generated_images_batch = generated_images_batch.to(device)

    ssim_scores = []
    l1_losses = []
    mse_losses = []  # 添加MSE指标
    lpips_scores = []

        # 获取LPIPS模型（只初始化一次）
    lpips_model = get_lpips_model(device)

    # --- 逐对计算 SSIM, L1, MSE, LPIPS ---
    for i in range(real_images_batch.shape[0]):
        real_img = real_images_batch[i].unsqueeze(0) # shape (1, C, H, W)
        generated_img = generated_images_batch[i].unsqueeze(0) # shape (1, C, H, W)

        # SSIM
        real_img_np = real_img.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # C, H, W -> H, W, C
        generated_img_np = generated_img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        
        # 确保图像值在[0,1]范围内
        real_img_np = np.clip(real_img_np, 0, 1)
        generated_img_np = np.clip(generated_img_np, 0, 1)
        
        # 根据图像尺寸确定合适的win_size
        min_dim = min(real_img_np.shape[0], real_img_np.shape[1])
        win_size = min(7, min_dim)
        if win_size >= 3:  # 确保最小为3
            if win_size % 2 == 0:  # 确保是奇数
                win_size -= 1
        else:
            win_size = 3  # 最小值
            
        try:
            ssim_score = ssim(
                real_img_np, 
                generated_img_np, 
                data_range=1.0,
                win_size=win_size,
                channel_axis=2 if real_img_np.ndim == 3 else None
            )
        except Exception as e:
            print(f"SSIM calculation error: {e}, using fallback value")
            ssim_score = 0.0
        ssim_scores.append(ssim_score)

        # L1 Loss
        l1_loss = mean_absolute_error(real_img.cpu().numpy().flatten(), generated_img.cpu().numpy().flatten())
        l1_losses.append(l1_loss)

        # MSE Loss
        real_flat = real_img.cpu().numpy().flatten()
        generated_flat = generated_img.cpu().numpy().flatten()
        mse_loss = np.mean((real_flat - generated_flat) ** 2)
        mse_losses.append(mse_loss)

        # LPIPS
        lpips_score = lpips_model(real_img, generated_img).item()
        lpips_scores.append(lpips_score)

    # 计算平均值
    avg_ssim = np.mean(ssim_scores) if ssim_scores else 0
    avg_l1 = np.mean(l1_losses) if l1_losses else 0
    avg_mse = np.mean(mse_losses) if mse_losses else 0  # MSE平均值
    avg_lpips = np.mean(lpips_scores) if lpips_scores else 0

    return  avg_ssim, avg_l1, avg_mse, avg_lpips  # 返回MSE

def calculate_fid(real_images_batch, generated_images_batch, device, batch_size=32, eps=1e-6):
    """
    计算FID分数
    
    Args:
        real_images_batch (torch.Tensor): 真实图像批次
        generated_images_batch (torch.Tensor): 生成图像批次
        device (torch.device): 计算设备
        batch_size (int): 批处理大小
    
    Returns:
        float: FID分数
    """
    try:
        from pytorch_fid.inception import InceptionV3
        from pytorch_fid.fid_score import calculate_frechet_distance
    except ImportError:
        print("Warning: pytorch_fid not installed. Please install it with 'pip install pytorch-fid'")
        return None
    
    # 初始化Inception模型 (使用默认的block index)
    dims = 2048
    try:
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    except (AttributeError, KeyError):
        # 如果找不到,使用默认值3 (对应pool3, 2048维)
        block_idx = 3
        print(f"Using default block_idx={block_idx} for dims={dims}")

    inception_model = InceptionV3([block_idx]).to(device)
    inception_model.eval()
    
    # 预处理函数
    def preprocess_images(images):
        """
        预处理图像以适配Inception模型
        输入: [N, C, H, W], 值域[0, 1]
        输出: [N, C, 299, 299], 值域[-1, 1]
        """
        # 1. 确保值域在[0, 1]
        if images.max() > 1.0:
            print(f"Warning: Image values exceed 1.0 (max={images.max().item():.2f}), normalizing...")
            images = torch.clamp(images, 0, 1)
        
        # 2. 调整尺寸到299x299
        if images. shape[-1] != 299 or images.shape[-2] != 299:
            images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        
        # 3. 标准化到[-1, 1] (Inception v3的要求)
        images = images * 2.0 - 1.0
        
        return images
    
    # 提取特征的函数
    def get_activations(images, model, batch_size=32):
        """提取Inception特征"""
        activations = []
        n = len(images)
        
        with torch.no_grad():
            for i in range(0, n, batch_size):
                batch = images[i:i+batch_size]. to(device)
                batch = preprocess_images(batch)
                
                # 前向传播
                pred = model(batch)[0]
                
                # 全局平均池化
                if pred.size(2) != 1 or pred.size(3) != 1:
                    pred = torch.nn.AdaptiveAvgPool2d((1, 1))(pred)
                
                # 展平
                pred = pred.squeeze(3).squeeze(2).cpu().numpy()
                activations.append(pred)
        
        return np.concatenate(activations, axis=0)
    
    # 获取真实和生成图像的特征
    real_features = get_activations(real_images_batch.to(device), inception_model, batch_size)
    gen_features = get_activations(generated_images_batch.to(device), inception_model, batch_size)
    
    # 计算统计信息
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    mu_gen = np.mean(gen_features, axis=0)
    sigma_gen = np.cov(gen_features, rowvar=False)
    
    # 添加数值稳定性
    sigma_real = sigma_real + np.eye(sigma_real.shape[0]) * eps
    sigma_gen = sigma_gen + np.eye(sigma_gen.shape[0]) * eps
    # 计算FID
    try:
        fid_score = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    except Exception as e:
        print(f"FID calculation error: {e}")
        return None
    
    return float(fid_score)

# def calculate_activation_statistics(images, device, batch_size=50):
#     """计算图像激活统计信息并返回均值和协方差矩阵。"""
#     # 将原来的:
#     # model = inception_v3.InceptionV3().to(device)
    
#     # 替换为:
#     model = inception.InceptionV3().to(device)
#     model.eval()
#     n_samples = images.shape[0]
#     n_batches = int(np.ceil(float(n_samples) / batch_size))
#     act = []
#     with torch.no_grad():
#         for i in range(n_batches):
#             start = i * batch_size
#             end = min(start + batch_size, n_samples)
#             batch = torch.from_numpy(images[start:end]).float().to(device)
#             batch = batch / 255.0 * 2 - 1  # 归一化到 [-1, 1]
#             pred = model(batch)[0]  # 使用 InceptionV3 得到 features (batch_size, 2048)
#             act.append(pred.cpu().numpy())
#     act = np.concatenate(act, axis=0)
#     mu = np.mean(act, axis=0)
#     sigma = np.cov(act, rowvar=False)
#     return mu, sigma



def image_to_tensor(image_path, image_size):
    """
    加载图像并转换为tensor，进行必要的预处理。
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  #  这里先不进行normalize，在calculate_activation_statistics中进行归一化
    ])
    return transform(img).unsqueeze(0)  # 添加batch维度


def extract_characters_from_ttf(ttf_path, output_dir, content_json_path, image_size=128, char_size=100):
    """
    从 TTF 文件中生成 content.json 中指定的所有字符的 PNG 图像。
    Args:
        ttf_path (str): TTF 文件的完整路径。
        output_dir (str): 生成的 PNG 图像的保存目录。
        content_json_path (str): content.json 文件的路径。
        image_size (int): 生成图像的尺寸。
        char_size (int): 字符的尺寸。
    Returns:
        int: 成功生成并保存的字符数量。
    """
    try:
        # 加载 content.json 文件
        content_data = load_json(content_json_path)
        characters = list(content_data.values())  # 获取所有字符值
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载字体
        font = ImageFont.truetype(ttf_path, size=char_size)
        
        generated_count = 0
        # 遍历所有需要生成的字符
        for character in characters:
            try:
                # 创建图像
                img = Image. new("RGB", (image_size, image_size), (255, 255, 255))  # 白色背景
                draw = ImageDraw.Draw(img)
                
                # 计算字符位置使其居中
                x_offset = (image_size - char_size) / 2
                y_offset = (image_size - char_size) / 2
                
                # 绘制字符
                draw.text((x_offset, y_offset), character, 0, font=font)
                
                # 保存图像
                output_path = os.path.join(output_dir, f"{character}.png")
                img.save(output_path)
                generated_count += 1
                
            except Exception as e:
                print(f"警告: 生成字符 '{character}' 时出错: {e}")
        
        print(f"成功从 {ttf_path} 中生成了 {generated_count}/{len(characters)} 个字符")
        return generated_count
        
    except FileNotFoundError as e:
        print(f"错误: 文件未找到 - {e}")
        return 0
    except Exception as e:
        print(f"从 TTF 生成字符时出错: {e}")
        return 0

def main(args):
    """
    主函数：处理参数，加载数据，进行评估，并保存结果。
    """
    device = torch.device(args.device)
    image_size = args.image_size

    # 加载JSON文件
    content_json = load_json(args.content_path)
    with open(args.style_path, "r", encoding="utf-8") as f:
        style_list = json.load(f)

    results_list = []
    all_ssims = []
    all_l1s = []
    all_mses = []  # 添加MSE列表
    all_lpips = []


    style_content_value = args.style_content  # 这是一个内容值，例如 'A'

    # 构建保存结果的目录结构
    if style_content_value == "":
        save_dir = os.path.join(args.save_result_dir, args.type)
    else:
        save_dir = os.path.join(args.save_result_dir, f"{style_content_value}", args.type)
    os.makedirs(save_dir, exist_ok=True)  # 创建输出目录

    # 创建临时目录用于存储目标图像
    temp_dir = os.path.join(save_dir, "temp_images")  #  在 `save_dir` 下创建临时目录
    os.makedirs(temp_dir, exist_ok=True)

    # ← 改进: 收集所有style的所有图像,统一计算FID
    all_real_images = []
    all_generated_images = []

    # 对每一个风格进行评估
    for style_name in tqdm(style_list, desc="Evaluating Styles"):  # 直接迭代风格名称列表

        # 假设 style_name 在 style.json 中就是 TTF 文件名（不带扩展名）
        # 支持多种扩展名
        ttf_extensions = [".TTF", ".ttf", ".OTF", ".otf"]
        ttf_path = None
        for ext in ttf_extensions:
            potential_path = os.path.join(args.style_ttf_path, style_name + ext)
            if os.path.exists(potential_path):
                ttf_path = potential_path
                break

        # 检查TTF文件是否存在
        if not os.path.exists(ttf_path):
            print(f"Warning: TTF file not found at '{ttf_path}' for style '{style_name}'. Skipping.")
            continue

        # 为当前风格创建临时目录
        style_temp_dir = os.path.join(temp_dir, style_name)
        print(f"Processing style: {style_name}, TTF path: {ttf_path}, Temp dir: {style_temp_dir}")
        os.makedirs(style_temp_dir, exist_ok=True)

        # 从当前风格的 TTF 文件中提取所有目标字符图像并保存到临时目录
        print(f"正在从 TTF 文件 {style_name} 中提取所有字符...")
        extracted_count = extract_characters_from_ttf(
            ttf_path=ttf_path,
            output_dir=style_temp_dir,
            content_json_path=args.content_path
        )

        if extracted_count == 0:
            print(f"未能从 TTF 文件 {os.path.basename(ttf_path)} 中提取任何字符，跳过该风格。")
            # 删除临时目录
            try:
                if os.path.exists(style_temp_dir):
                    shutil.rmtree(style_temp_dir)
                    print(f"Removed temporary directory: {style_temp_dir}")
            except OSError as e:
                print(f"Error deleting temporary directory: {e}")
            continue

        # 收集该风格下所有字符的图像用于整体评估
        real_images_list = []
        generated_images_list = []

        # 遍历 content.json 中的所有字符进行评估
        for char_code, character in content_json.items():
            # 定义目标图像的临时保存路径
            temp_target_image_path = os.path.join(style_temp_dir, f"{character}.png")

            # 检查字符图像是否存在
            if not os.path.exists(temp_target_image_path):
                print(f"警告: 字符 '{character}' 的图像在临时目录中未找到，跳过。")
                continue

            # 加载临时目标图像
            real_image = image_to_tensor(temp_target_image_path, image_size)
            if real_image is None:
                print(
                    f"Warning: Could not load temporary target image for style '{style_name}' and character '{character}'. Skipping."
                )
                continue

            # 获取生成图像路径 - 支持多种格式
            generated_image_extensions = [".jpg", ".jpeg", ".png"]
            generated_image_path = None
            
            for ext in generated_image_extensions:
                potential_path = os.path.join(args.generated_image_path, style_name, f"{character}{ext}")
                if os.path.exists(potential_path):
                    generated_image_path = potential_path
                    break

            # 检查路径是否存在
            if not os.path.exists(generated_image_path):
                print(
                    f"Warning: Generated image not found for style '{style_name}' and character '{character}' at {generated_image_path}. Skipping."
                )
                continue

            # 加载生成图像
            generated_image = image_to_tensor(generated_image_path, image_size)
            if generated_image is None:
                print(
                    f"Warning: Could not load generated image for style '{style_name}' and character '{character}'. Skipping."
                )
                continue

            # 收集到全局列表
            all_real_images.append(real_image)
            all_generated_images.append(generated_image)

            # 收集图像用于批量处理
            real_images_list.append(real_image)
            generated_images_list.append(generated_image)

        # 如果有图像需要评估
        if real_images_list and generated_images_list:
            # 合并所有图像
            real_images_batch = torch.cat(real_images_list, dim=0)
            generated_images_batch = torch.cat(generated_images_list, dim=0)
            
            # 计算该风格的整体指标（包含MSE）
            ssim_score, l1_loss, mse_loss, lpips_score = calculate_metrics(
                real_images_batch, generated_images_batch, device, image_size
            )

            # 记录每个风格的结果（而不是每个字符的结果）
            results_list.append(
                {
                    "style_name": style_name,
                    "SSIM": float(ssim_score),
                    "L1": float(l1_loss),
                    "MSE": float(mse_loss),  # 添加MSE指标
                    "LPIPS": float(lpips_score),
                }
            )


            all_ssims.append(float(ssim_score))
            all_l1s.append(float(l1_loss))
            all_mses.append(float(mse_loss))  # 添加MSE到列表
            all_lpips.append(float(lpips_score))

            print(f"Style: {style_name},  SSIM: {ssim_score:.4f}, L1: {l1_loss:.4f}, MSE: {mse_loss:.4f}, LPIPS: {lpips_score:.4f}")
        else:
            print(f"Style: {style_name} - No valid images found for evaluation")
            
        # 评估完毕后删除当前风格的临时目录
        try:
            if os.path.exists(style_temp_dir):
                shutil.rmtree(style_temp_dir)
                print(f"Removed temporary directory for style {style_name}: {style_temp_dir}")
        except OSError as e:
            print(f"Error deleting temporary directory for style {style_name}: {e}")
        
    # ← 新增: 统一计算全局FID
    if all_real_images and all_generated_images:
        print(f"\n计算全局FID (样本数: {len(all_real_images)})...")
        all_real_batch = torch.cat(all_real_images, dim=0)
        all_gen_batch = torch.cat(all_generated_images, dim=0)
        
        global_fid = calculate_fid(all_real_batch, all_gen_batch, device)

    # 计算所有风格的平均分数
    if all_ssims:  # 避免在没有成功评估任何风格时发生除以零的错误
        avg_ssim = float(np.mean(all_ssims))
        avg_l1 = float(np.mean(all_l1s))
        avg_mse = float(np.mean(all_mses))  # 计算MSE平均值
        avg_lpips = float(np.mean(all_lpips))

        # 记录平均结果
        results_list.append(
            {
                "style_name": "average",
                "SSIM": float(avg_ssim),
                "L1": float(avg_l1),
                "MSE": float(avg_mse),  # 添加MSE指标
                "LPIPS": float(avg_lpips),
                "FID": float(global_fid),
            }
        )
        print(
            f"Average SSIM: {avg_ssim:.4f}, Average L1: {avg_l1:.4f}, Average MSE: {avg_mse:.4f}, Average LPIPS: {avg_lpips:.4f}"
        )
    else:
        print("No styles were successfully evaluated. Skipping average calculation and result saving.")

    # 保存结果
    save_path_json = os.path.join(save_dir, "evaluation_results.json")
    with open(save_path_json, "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=4)
    print(f"Results saved to {save_path_json}")

    #  删除临时目录及其内容
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Removed temporary directory: {temp_dir}")
    except OSError as e:
        print(f"Error deleting temporary directory: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate image generation results.")
    parser.add_argument("--type", type=str, required=True, help="Type of evaluation.")
    parser.add_argument(
        "--style_content", type=str, required=True, help="The character to extract and evaluate (e.g., 'A')."
    )
    parser.add_argument(
        "--style_ttf_path", type=str, required=True,
        help="Base path to the directory containing style TTF files (usually zip archives).",
    )
    parser.add_argument(
        "--generated_image_path", type=str, required=True,
        help="Path to the directory containing generated character images.",
    )
    parser.add_argument(
        "--content_path", type=str, required=True,
        help="Path to the content JSON file (though not directly used in this logic, kept for consistency).",
    )
    parser.add_argument(
        "--style_path", type=str, required=True,
        help="Path to the style JSON file, which contains TTF filenames.",
    )
    parser.add_argument(
        "--save_result_dir", type=str, default="outputs",
        help="Directory to save the evaluation results. Results will be saved to save_result_dir/<style_content>/<type>.",
    )
    parser.add_argument("--image_size", type=int, default=128, help="Image size for resizing during evaluation.")
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to run the evaluation on (e.g., cuda:0, cpu)."
    )
    args = parser.parse_args()
    main(args)

