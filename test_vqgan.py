import os
import sys
import torch
import math
import torchio as tio
from omegaconf import OmegaConf
from vqgan.vqgan import *
from utils.trainer import *
from dataset import *
import warnings
import torch.nn.functional as F
from monai.metrics import SSIMMetric
from torchvision.utils import save_image
from pathlib import Path
import numpy as np

ssim_metric = SSIMMetric(spatial_dims=2, data_range=1.0)

seed = 44

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# 忽略所有用户警告
warnings.filterwarnings("ignore", category=UserWarning)


def load_config(config_path="config/residual_vq_gan_2d.yaml"):  # config/vq_gan_3d.yaml config/residual_vq_gan_3d.yaml
    # 加载基础配置
    cfg = OmegaConf.load(config_path)
    # 允许通过命令行覆盖参数
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_cfg)  # 合并命令行参数
    return cfg


def run(cfg, ckpt_path):
    print(torch.cuda.is_available())
    print(cfg)

    path = r'/media/xxx/Data/ULF_ENC'
    dataset = load_Multi_modal_dataset(path, is_train=True, load_getitem=False, out_min_max=(0, 1))
    # dataset = load_val_dataset(path, is_train=False, out_min_max=(0, 1))
    dataloader = patch_train_maskdataloader(dataset, patch_size=(224, 224, 5), batch_size=8, samples_per_volume=64)
    # 加载模型
    model = VQGAN2.load_from_checkpoint(ckpt_path).cuda().eval()

    mean_list = []
    var_list = []
    min_list = []
    max_list = []
    l1_loss_list = []
    ssim_list = []

    all_zs = []
    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            Tup_T1_raw = batch['Tup_T1'][tio.DATA].cuda()
            Tup_T2_raw = batch['Tup_T2'][tio.DATA].cuda()
            Tup_FLAIR_raw = batch['Tup_FLAIR'][tio.DATA].cuda()
            Tlow_FLAIR_raw = batch['Tlow_FLAIR'][tio.DATA].cuda()
            Tlow_T1_raw = batch['Tlow_T1'][tio.DATA].cuda()
            Tlow_T2_raw = batch['Tlow_T2'][tio.DATA].cuda()
            # (B, 1, H, W, 3) -> (B, H, W, 3) -> (B, 3, H, W)
            Tup_T1 = Tup_T1_raw.squeeze(1).permute(0, 3, 1, 2)
            Tup_T2 = Tup_T2_raw.squeeze(1).permute(0, 3, 1, 2)
            Tup_FLAIR = Tup_FLAIR_raw.squeeze(1).permute(0, 3, 1, 2)
            Tlow_T1 = Tlow_T1_raw.squeeze(1).permute(0, 3, 1, 2)
            Tlow_T2 = Tlow_T2_raw.squeeze(1).permute(0, 3, 1, 2)
            Tlow_FLAIR = Tlow_FLAIR_raw.squeeze(1).permute(0, 3, 1, 2)
            # concat
            input_images = torch.cat([Tup_T1, Tup_T2, Tup_FLAIR], dim=1).cuda()
            tlow_images = torch.cat([Tlow_T1, Tlow_T2, Tlow_FLAIR], dim=1).cuda()
            z = model.encode(input_images)
            z2 = model.encode(tlow_images)
            print("Latent codes shape, min and max:", z.shape, z.min().item(), z.max().item())
            recon_img = model.decode(z)
            recon_img2 = model.decode(z2)
            mae1 = F.l1_loss(recon_img, input_images).item()
            mae2 = F.l1_loss(recon_img2, tlow_images).item()
            print(f'Batch L1 Loss (Up): {mae1}')
            print(f'Batch L1 Loss (Low): {mae2}')

            out_dir = Path("sample")
            out_dir.mkdir(exist_ok=True)
            recon1_flat = recon_img[0:1, :].reshape(-1, recon_img.size(2), recon_img.size(3)).unsqueeze(1)
            recon2_flat = recon_img2[0:1, :].reshape(-1, recon_img2.size(2), recon_img2.size(3)).unsqueeze(1)
            save_image(recon1_flat, out_dir / f"recon_img_grid_{i}.png", nrow=recon_img.size(1))
            save_image(recon2_flat, out_dir / f"recon_img2_grid_{i}.png", nrow=recon_img2.size(1))

            z_e_flat = z.flatten(start_dim=1)  # [batch, channel*h*w]
            all_zs.append(z_e_flat.cpu())

        q1 = torch.quantile(z, 0.1)
        q3 = torch.quantile(z, 0.9)
        batch_mean = z.mean().item()
        batch_var = z.var().item()
        batch_std = z.std().item()
        batch_min = z.min().item()
        batch_max = z.max().item()
        l1_loss = F.l1_loss(recon_img, input_images).item()
        ssim = ssim_metric(recon_img, input_images).mean()
        print(f'Batch L1 Loss: {l1_loss}, SSIM: {ssim}')
        print(f'Batch Mean: {batch_mean}, Variance: {batch_var}, Std: {batch_std}, Min: {batch_min}, Max: {batch_max}, Q1: {q1.item()}, Q3: {q3.item()}')
        l1_loss_list.append(l1_loss)
        ssim_list.append(ssim)
        mean_list.append(batch_mean)
        var_list.append(batch_var)
        min_list.append(batch_min)
        max_list.append(batch_max)

    all_zs = torch.cat(all_zs, dim=0)
    global_mean = all_zs.mean().item()
    global_std = all_zs.std().item()
    print(f"global_mean: {global_mean:.6f}")
    print(f"global_std: {global_std:.6f}")

    overall_mean = sum(mean_list) / len(mean_list)
    overall_var = sum(var_list) / len(var_list)
    overall_std = math.sqrt(overall_var)
    overall_min = min(min_list)
    overall_max = max(max_list)
    overall_l1_loss = sum(l1_loss_list) / len(l1_loss_list)
    overall_ssim = sum(ssim_list) / len(ssim_list)
    print(f'Average L1 Loss: {overall_l1_loss}, Average SSIM: {overall_ssim}')
    print(f'Overall Mean: {overall_mean}, Overall Variance: {overall_var}, Overall Std: {overall_std}, Overall Min: {overall_min}, Overall Max: {overall_max}')


def calculate_rq_codebook_usage():
    print(torch.cuda.is_available())

    path = r'/media/xxx/Data/ULF_ENC'
    val_dataset = load_val_dataset(path, is_train=False, out_min_max=(0, 1))
    dataloader = patch_train_maskdataloader(val_dataset, patch_size=(224, 224, 5), batch_size=8, samples_per_volume=64)

    model = ResidualVQGAN.load_from_checkpoint('checkpoints/vqgan/last.ckpt').cuda().eval()

    num_quantizers = 8  # from config
    codebook_size = 1024

    # 全局统计：每层一个集合，记录所有用过的码子
    global_used_indices = [set() for _ in range(num_quantizers)]

    # 同时收集所有 batch 的 indices 用于算全局 perplexity
    all_indices_per_layer = [[] for _ in range(num_quantizers)]

    for batch in dataloader:
        with torch.no_grad():
            # ... 数据预处理相同 ...
            Tup_T1_raw = batch['Tup_T1'][tio.DATA].cuda()
            Tup_T2_raw = batch['Tup_T2'][tio.DATA].cuda()
            Tup_FLAIR_raw = batch['Tup_FLAIR'][tio.DATA].cuda()

            Tup_T1 = Tup_T1_raw.squeeze(1).permute(0, 3, 1, 2)
            Tup_T2 = Tup_T2_raw.squeeze(1).permute(0, 3, 1, 2)
            Tup_FLAIR = Tup_FLAIR_raw.squeeze(1).permute(0, 3, 1, 2)
            x = torch.cat([Tup_T1, Tup_T2, Tup_FLAIR], dim=1).cuda()

            z = model.encoder(x)
            z, indices, quantization_losses = model.vector_quantize(z)
            # indices: [B, num_tokens, num_quant]

            B, num_tokens, num_quant = indices.shape

            for q in range(num_quant):
                layer_indices = indices[..., q].flatten().cpu().numpy()

                # 累积到全局集合
                global_used_indices[q].update(layer_indices.tolist())

                # 累积用于 perplexity
                all_indices_per_layer[q].append(layer_indices)

    # 全局统计
    print(f"\n{'=' * 50}")
    print("Global Codebook Usage (corrected):")
    print(f"{'=' * 50}")

    for q in range(num_quantizers):
        # 1. 全局使用率
        global_usage = len(global_used_indices[q]) / codebook_size

        # 2. 全局困惑度（合并所有 batch）
        all_idx = np.concatenate(all_indices_per_layer[q])
        counts = np.bincount(all_idx, minlength=codebook_size)
        probs = counts / counts.sum()
        perplexity = np.exp(-np.sum(probs * np.log(probs + 1e-10)))

        # 3. 有效 perplexity（只算用过的码子）
        used_probs = probs[counts > 0]
        effective_perplexity = np.exp(-np.sum(used_probs * np.log(used_probs + 1e-10)))

        print(f"\nLayer {q}:")
        print(f"  Global Usage Ratio: {global_usage:.4f} ({len(global_used_indices[q])}/1024)")
        print(f"  Global Perplexity: {perplexity:.2f}")
        print(f"  Effective Perplexity (used only): {effective_perplexity:.2f}")
        print(f"  Most frequent code used: {counts.max()} times ({counts.max() / counts.sum() * 100:.2f}%)")

    # 跨层分析
    print(f"\n{'=' * 50}")
    print("Cross-layer analysis:")
    overlap_01 = len(global_used_indices[0] & global_used_indices[1]) / codebook_size
    print(f"Layer 0-1 overlap: {overlap_01:.4f}")

    # 所有层的并集
    all_used = set()
    for s in global_used_indices:
        all_used.update(s)
    print(f"Total unique codes used (all layers): {len(all_used)}/1024 = {len(all_used) / 1024:.4f}")


def calculate_vq_codebook_usage(ckpt_path):
    print(torch.cuda.is_available())

    path = r'/media/xxx/Data/ULF_ENC'
    val_dataset = load_val_dataset(path, is_train=False, out_min_max=(0, 1))
    dataloader = patch_train_maskdataloader(val_dataset, patch_size=(224, 224, 5), batch_size=8, samples_per_volume=64)

    # 加载普通VQ模型（注意checkpoint路径改为你的普通VQ模型）
    model = VQGAN2.load_from_checkpoint(
        'new_checkpoints/vqgan/last.ckpt'
    ).cuda().eval()

    codebook_size = 8192  # from config

    # global_used_indices
    global_used_indices = set()

    # all_indices
    all_indices = []

    for batch in dataloader:
        with torch.no_grad():
            # 数据预处理（保持相同）
            Tup_T1_raw = batch['Tup_T1'][tio.DATA].cuda()
            Tup_T2_raw = batch['Tup_T2'][tio.DATA].cuda()
            Tup_FLAIR_raw = batch['Tup_FLAIR'][tio.DATA].cuda()

            Tup_T1 = Tup_T1_raw.squeeze(1).permute(0, 3, 1, 2)
            Tup_T2 = Tup_T2_raw.squeeze(1).permute(0, 3, 1, 2)
            Tup_FLAIR = Tup_FLAIR_raw.squeeze(1).permute(0, 3, 1, 2)
            x = torch.cat([Tup_T1, Tup_T2, Tup_FLAIR], dim=1).cuda()

            z = model.encoder(x)

            z, indices, quantization_losses = model.vector_quantize(z)

            if indices.dim() == 3:
                # [B, H, W] -> [B*H*W]
                indices_flat = indices.flatten().cpu().numpy()
            else:
                # [B, num_tokens] -> [B*num_tokens]
                indices_flat = indices.flatten().cpu().numpy()

            global_used_indices.update(indices_flat.tolist())

            all_indices.append(indices_flat)

    print(f"\n{'=' * 50}")
    print("Global Codebook Usage (Vanilla VQ):")
    print(f"{'=' * 50}")

    global_usage = len(global_used_indices) / codebook_size

    all_idx = np.concatenate(all_indices)
    counts = np.bincount(all_idx, minlength=codebook_size)
    probs = counts / counts.sum()
    perplexity = np.exp(-np.sum(probs * np.log(probs + 1e-10)))

    used_probs = probs[counts > 0]
    effective_perplexity = np.exp(-np.sum(used_probs * np.log(used_probs + 1e-10)))

    dead_codes = codebook_size - len(global_used_indices)

    print(f"Global Usage Ratio: {global_usage:.4f} ({len(global_used_indices)}/{codebook_size})")
    print(f"Dead Codes: {dead_codes} ({dead_codes / codebook_size * 100:.2f}%)")
    print(f"Global Perplexity: {perplexity:.2f}")
    print(f"Effective Perplexity (used only): {effective_perplexity:.2f}")
    print(f"Most frequent code used: {counts.max()} times ({counts.max() / counts.sum() * 100:.2f}%)")

    print(f"\n{'=' * 50}")
    print("Usage Distribution:")
    used_counts = counts[counts > 0]
    print(f"Mean usage per code: {used_counts.mean():.2f}")
    print(f"Median usage per code: {np.median(used_counts):.2f}")
    print(f"Std usage per code: {used_counts.std():.2f}")


def read_epoch(ckpt_path):
    checkpoint = torch.load(ckpt_path)
    epoch = checkpoint['epoch']
    print(f"模型在检查点 {ckpt_path} 中的训练轮次: {epoch}")
    return epoch


if __name__ == '__main__':
    ckpt_path = 'checkpoints/vqgan/last.ckpt'
    cfg = load_config()
    read_epoch(ckpt_path)
    run(cfg=cfg, ckpt_path=ckpt_path)
    calculate_vq_codebook_usage(ckpt_path=ckpt_path)