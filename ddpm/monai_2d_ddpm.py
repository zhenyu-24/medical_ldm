# from generative.inferers import LatentDiffusionInferer, DiffusionInferer
# from generative.losses import PatchAdversarialLoss, PerceptualLoss
# from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator, ControlNet
# from generative.networks.schedulers import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
import torchvision
from torch.amp import autocast
import lightning as L
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler, DDIMScheduler
from monai.inferers import DiffusionInferer
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import math


class DiffusionModel(L.LightningModule):
    def __init__(self, num_training_steps=None):
        super().__init__()
        self.num_training_steps = num_training_steps

        self.model = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=6,
            out_channels=3,
            num_res_blocks=2,
            channels=(64, 128, 256, 256, 512),
            attention_levels=(False, False, False, False, True),
            num_head_channels=32,
            norm_num_groups=32,
            resblock_updown=False,
        )
        self.scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta")
        self.inferer = DiffusionInferer(self.scheduler)

        self.val_scheduler = DDIMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta")
        self.val_inferer = DiffusionInferer(self.val_scheduler)

        self.sample_steps = 100000
        self.inference_steps = 100

    def configure_optimizers(self):
        # 1. 定义优化器
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=1e-4)

        # 2. 定义学习率调度器
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_training_steps,
                                                                  eta_min=1e-7)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def forward(self, x):
        noise = torch.randn_like(x)
        samples = self.sample(noise, x)
        samples = samples.clamp(0, 1)
        return samples

    def training_step(self, batch, batch_idx):
        # (B, 1, H, W, N）
        Tlow_T1_raw = batch['Tlow_T1'][tio.DATA]
        Tlow_T2_raw = batch['Tlow_T2'][tio.DATA]
        Tlow_FLAIR_raw = batch['Tlow_FLAIR'][tio.DATA]
        Tup_T1_raw = batch['Tup_T1'][tio.DATA]
        Tup_T2_raw = batch['Tup_T2'][tio.DATA]
        Tup_FLAIR_raw = batch['Tup_FLAIR'][tio.DATA]
        # (B, 1, H, W, N) -> (B, H, W, N) -> (B, N, H, W)
        Tlow_T1 = Tlow_T1_raw.squeeze(1).permute(0, 3, 1, 2)
        Tlow_T2 = Tlow_T2_raw.squeeze(1).permute(0, 3, 1, 2)
        Tlow_FLAIR = Tlow_FLAIR_raw.squeeze(1).permute(0, 3, 1, 2)
        Tup_T1 = Tup_T1_raw.squeeze(1).permute(0, 3, 1, 2)
        Tup_T2 = Tup_T2_raw.squeeze(1).permute(0, 3, 1, 2)
        Tup_FLAIR = Tup_FLAIR_raw.squeeze(1).permute(0, 3, 1, 2)

        # 沿通道维度拼接，得到 (B, img_channel, H, W) 的输入和目标
        input_tensor = torch.cat([Tlow_T1, Tlow_T2, Tlow_FLAIR], dim=1)  # (B, img_channel, H, W)
        target_tensor = torch.cat([Tup_T1, Tup_T2, Tup_FLAIR], dim=1)  # (B, img_channel, H, W)
        print(input_tensor.min(), input_tensor.max(), target_tensor.min(), target_tensor.max())

        condition_input = input_tensor
        # 噪声
        noise = torch.randn_like(target_tensor)
        # 时间步长
        timesteps = torch.randint(0, self.inferer.scheduler.num_train_timesteps, (target_tensor.shape[0],),
                                  device=self.device).long()
        # Get model prediction
        noise_pred = self.inferer(
            inputs=target_tensor,
            diffusion_model=self.model,
            noise=noise,
            timesteps=timesteps,
            condition=condition_input,
            mode="concat",
        )
        # 损失
        loss = F.mse_loss(noise_pred, noise)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    @torch.no_grad()
    def sample(self, noise, condition, num_inference_steps=1000, mode="concat", infer_type="ddpm"):
        if infer_type == "ddpm":
            self.scheduler.set_timesteps(num_inference_steps=num_inference_steps)
            image = self.inferer.sample(input_noise=noise, diffusion_model=self.model, scheduler=self.scheduler,
                                        conditioning=condition, mode=mode)
        elif infer_type == "ddim":
            self.val_scheduler.set_timesteps(num_inference_steps=num_inference_steps)
            image = self.val_inferer.sample(input_noise=noise, diffusion_model=self.model, scheduler=self.val_scheduler,
                                            conditioning=condition, mode=mode)
        return image