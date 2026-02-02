import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
import pytorch_lightning as pl
import torchvision
from torch.amp import autocast
from monai.networks.nets import DiffusionModelUNet, SPADEDiffusionModelUNet, ControlNet
from monai.networks.schedulers import DDPMScheduler, DDIMScheduler
from monai.inferers import DiffusionInferer
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import math
from vqgan.vqgan import *
from monai.utils import optional_import
from functools import partial
from typing import Optional, Tuple, List
import torch
from tqdm import tqdm
import warnings
from dataset import *
from utils.trainer import *
from monai.inferers import ControlNetDiffusionInferer

# 忽略所有用户警告
warnings.filterwarnings("ignore", category=UserWarning)


class ControlNetModel(pl.LightningModule):
    def __init__(self, num_training_steps=None, vae_path=None, condition_encoder_path=None):
        super().__init__()
        # ddpm
        self.ddpm = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=16,
            out_channels=8,
            num_res_blocks=2,
            channels=(64, 128, 256, 512),
            attention_levels=(False, False, True, True),
            num_head_channels=32,
            norm_num_groups=32,
        )
        checkpoint = torch.load("checkpoints/ldm/last.ckpt")
        state_dict = checkpoint.get("state_dict", checkpoint)
        model_weights = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}
        missing_model = self.ddpm.load_state_dict(model_weights, strict=False)
        print("Loaded ddpm weights:", missing_model)
        # controlnet
        self.controlnet = ControlNet(
            spatial_dims=3,
            in_channels=3,
            channels=(32, 64, 128, 256),
            attention_levels=(False, False, True, True),
            num_res_blocks=2,
            conditioning_embedding_in_channels=3,
            conditioning_embedding_num_channels=(32, 64, 128),
        )
        missing_model = self.controlnet.load_state_dict(self.ddpm.state_dict(), strict=False)
        print("ControlNet loaded weights from DDPM:", missing_model)
        # 冻结ddpm参数
        self.ddpm.eval()
        for p in self.ddpm.parameters():
            p.requires_grad = False
        # VAE模型
        self.vae = VQGAN2.load_from_checkpoint(vae_path).eval()
        # 参数
        self.num_training_steps = num_training_steps
        self.scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0205, clip_sample=False)
        self.inferer = ControlNetDiffusionInferer(self.scheduler)
        self.ddpm_inferer = DiffusionInferer(self.scheduler)
        self.inference_steps = 1000

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=list(self.controlnet.parameters()), lr=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_training_steps, eta_min=1e-7)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def training_step(self, batch, batch_idx):
        Tlow_T1_raw = batch['Tlow_T1'][tio.DATA] / 255.0
        Tlow_T2_raw = batch['Tlow_T2'][tio.DATA] / 255.0
        Tlow_FLAIR_raw = batch['Tlow_FLAIR'][tio.DATA] / 255.0
        Tup_T1_raw = batch['Tup_T1'][tio.DATA] / 255.0
        Tup_T2_raw = batch['Tup_T2'][tio.DATA] / 255.0
        Tup_FLAIR_raw = batch['Tup_FLAIR'][tio.DATA] / 255.0

        # concat
        input_tensor = torch.cat([Tlow_T1_raw, Tlow_T2_raw, Tlow_FLAIR_raw], dim=1)
        target_tensor = torch.cat([Tup_T1_raw, Tup_T2_raw, Tup_FLAIR_raw], dim=1)

        # latent
        with torch.no_grad():
            latent_tensor = self.vae.encode(target_tensor, quantize=True)
            latent_condition = self.vae.encode(input_tensor, quantize=True)

        # noise
        noise = torch.randn_like(latent_tensor)

        # timesteps
        timesteps = torch.randint(0, self.inferer.scheduler.num_train_timesteps, (latent_tensor.shape[0],),
                                  device=self.device).long()
        # Get model prediction
        noise_pred = self.inferer(
            inputs=latent_tensor,
            diffusion_model=self.ddpm,
            controlnet=self.controlnet,
            noise=noise,
            timesteps=timesteps,
            condition=latent_condition,
            cn_cond=input_tensor,
            mode="concat"
        )
        # 损失
        loss = F.mse_loss(noise_pred, noise)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    @torch.no_grad()
    def sample(self, noise, condition, cn_cond, num_inference_steps=1000, mode="crossattn"):  # , mode="crossattn"
        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps)
        with torch.no_grad():
            image = self.inferer.sample(input_noise=noise, diffusion_model=self.ddpm, controlnet=self.controlnet,
                                        scheduler=self.scheduler, conditioning=condition, cn_cond=cn_cond,
                                        mode=mode)
        return image