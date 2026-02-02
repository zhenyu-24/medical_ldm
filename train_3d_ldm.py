import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
import pytorch_lightning as pl
import lightning as L
import torchvision
from torch.amp import autocast
from monai.networks.schedulers import DDPMScheduler, DDIMScheduler
from torch.optim.lr_scheduler import LambdaLR
import math
from monai.utils import optional_import
from functools import partial
from typing import Optional, Tuple, List
import torch
from tqdm import tqdm
import warnings
from monai.networks.nets import DiffusionModelUNet, SPADEDiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler, DDIMScheduler
from monai.inferers import DiffusionInferer, LatentDiffusionInferer
from dataset import *
from utils.ddpm_trainer import *
from vqgan.wat_encoder import *
from vqgan.vqgan import *

warnings.filterwarnings("ignore", category=UserWarning)


class DiffusionModel(L.LightningModule):
    def __init__(self, num_training_steps=None):
        super().__init__()
        self.save_hyperparameters()
        # 条件encoder
        self.con_encoder = WatEncoder(in_channels=3, basedim=32, downdeepth=2, model_type='3d')
        self.embedding_dim = 8
        # diffusion model
        self.model = SPADEDiffusionModelUNet(  # DiffusionModelUNet/SPADEDiffusionModelUNet
            spatial_dims=3,
            in_channels=self.embedding_dim * 2,
            out_channels=self.embedding_dim,
            num_res_blocks=2,
            channels=(64, 128, 256, 512),
            attention_levels=(False, False, True, True),
            norm_num_groups=32,
            label_nc=128,
        )
        # VAE 模型
        self.vae = VQGAN2.load_from_checkpoint('checkpoints/vqgan_3d/last.ckpt').eval()
        for p in self.vae.parameters():
            p.requires_grad = False
        # 采样/训练参数
        self.num_training_steps = num_training_steps
        self.scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta",
                                       prediction_type="epsilon", beta_start=0.0015, beta_end=0.0205, clip_sample=False)
        # self.scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", prediction_type="v_prediction", beta_start=0.0015, beta_end=0.0205, clip_sample=False)
        # self.scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="cosine", clip_sample=False)
        self.scale_factor = 1
        self.inferer = DiffusionInferer(self.scheduler)
        self.inference_steps = 250
        self.val_scheduler = DDIMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta",
                                           prediction_type="epsilon", beta_start=0.0015, beta_end=0.0205,
                                           clip_sample=False)
        # self.val_scheduler = DDIMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", prediction_type="v_prediction", beta_start=0.0015, beta_end=0.0205, clip_sample=False)
        # self.val_scheduler = DDIMScheduler(
        #     num_train_timesteps=1000,
        #     schedule="cosine",
        #     clip_sample=False
        # )
        self.val_inferer = DiffusionInferer(self.val_scheduler)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=list(self.model.parameters()) + list(self.con_encoder.parameters()), lr=1e-4)  #
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_training_steps, eta_min=1e-7)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def training_step(self, batch, batch_idx):
        Tlow_T1_raw = batch['Tlow_T1'][tio.DATA]
        Tlow_T2_raw = batch['Tlow_T2'][tio.DATA]
        Tlow_FLAIR_raw = batch['Tlow_FLAIR'][tio.DATA]
        Tup_T1_raw = batch['Tup_T1'][tio.DATA]
        Tup_T2_raw = batch['Tup_T2'][tio.DATA]
        Tup_FLAIR_raw = batch['Tup_FLAIR'][tio.DATA]
        # 沿通道维度拼接
        input_tensor = torch.cat([Tlow_T1_raw, Tlow_T2_raw, Tlow_FLAIR_raw], dim=1)
        target_tensor = torch.cat([Tup_T1_raw, Tup_T2_raw, Tup_FLAIR_raw], dim=1)
        # 潜在空间表示
        with torch.no_grad():
            latent_tensor = self.vae.encode(target_tensor) * self.scale_factor
            condition_tensor1 = self.vae.encode(input_tensor) * self.scale_factor
        condition_tensor2 = self.con_encoder(input_tensor)
        # 噪声
        noise = torch.randn_like(latent_tensor)
        # 时间步长
        timesteps = torch.randint(0, self.inferer.scheduler.num_train_timesteps, (latent_tensor.shape[0],),
                                  device=self.device).long()

        if self.scheduler.prediction_type == "v_prediction":
            # Use v-prediction parameterization
            target = self.scheduler.get_velocity(latent_tensor, noise, timesteps)
            print("Using v-prediction")
        elif self.scheduler.prediction_type == "epsilon":
            target = noise
        # Get model prediction
        noise_pred = self.inferer(
            inputs=latent_tensor,
            diffusion_model=self.model,
            noise=noise,
            timesteps=timesteps,
            condition=condition_tensor1,
            mode="concat",              # "concat"/"crossattn"
            seg=condition_tensor2,
        )
        # 损失
        loss = F.mse_loss(noise_pred, target)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('global_step', self.global_step, on_step=True, prog_bar=True, logger=False)
        return loss

    def validation_step(self, batch, batch_idx):
        Tlow_T1_raw = batch['Tlow_T1'][tio.DATA]
        Tlow_T2_raw = batch['Tlow_T2'][tio.DATA]
        Tlow_FLAIR_raw = batch['Tlow_FLAIR'][tio.DATA]
        Tup_T1_raw = batch['Tup_T1'][tio.DATA]
        Tup_T2_raw = batch['Tup_T2'][tio.DATA]
        Tup_FLAIR_raw = batch['Tup_FLAIR'][tio.DATA]
        # concat
        input_tensor = torch.cat([Tlow_T1_raw, Tlow_T2_raw, Tlow_FLAIR_raw], dim=1)
        target_tensor = torch.cat([Tup_T1_raw, Tup_T2_raw, Tup_FLAIR_raw], dim=1)
        # latent
        latent_tensor = self.vae.encode(target_tensor) * self.scale_factor
        condition_tensor1 = self.vae.encode(input_tensor) * self.scale_factor
        condition_tensor2 = self.con_encoder(input_tensor)
        # noise
        noise = torch.randn_like(latent_tensor)
        # sample
        samples = self.sample(noise, condition=condition_tensor1, num_inference_steps=self.inference_steps,
                              mode="concat", seg=condition_tensor2)
        # log
        mse = F.mse_loss(samples, latent_tensor)
        self.log('val/latent_mse', mse, on_epoch=True, prog_bar=True)
        generated_tensor = self.vae.decode(samples / self.scale_factor)
        generated_tensor = torch.clamp(generated_tensor, 0.0, 1.0)
        loss = F.l1_loss(generated_tensor, target_tensor)
        self.log('val/loss', loss, on_epoch=True, prog_bar=True)
        recon = self.vae.decode(latent_tensor / self.scale_factor)
        recon = torch.clamp(recon, 0.0, 1.0)
        recon_mae = F.l1_loss(recon, target_tensor)
        self.log('val/recon_mae', recon_mae, on_epoch=True, prog_bar=True)
        # save
        try:
            save_target_vis = target_tensor[0, :, :, :, :]
            save_recon_vis = recon[0, :, :, :, :]
            save_pred_vis = generated_tensor[0, :, :, :, :]
            self.save_3d_image_slices("val/target", save_target_vis, self.global_step)
            self.save_3d_image_slices("val/pred", save_pred_vis, self.global_step)
            self.save_3d_image_slices("val/recon", save_recon_vis, self.global_step)
        except Exception as e:
            print(f"Warning: Logging images failed: {e}")
        return loss

    @torch.no_grad()
    def sample(self, noise, condition, num_inference_steps=100, mode="concat", seg=None):
        print(f'num_inference_steps: {num_inference_steps}')
        self.val_scheduler.set_timesteps(num_inference_steps=num_inference_steps)
        image = self.val_inferer.sample(input_noise=noise, diffusion_model=self.model, scheduler=self.val_scheduler,
                                        conditioning=condition, mode=mode, seg=seg)
        return image

    @torch.no_grad()
    def sample2(self, noise, condition, num_inference_steps=1000, mode="concat", seg=None):
        print(f'num_inference_steps: {num_inference_steps}')
        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps)
        image = self.inferer.sample(input_noise=noise, diffusion_model=self.model, scheduler=self.scheduler,
                                    conditioning=condition, mode=mode, seg=seg)
        return image

    def forward(self, x, z_size, num_inference_steps=1000, infer_type='ddpm'):
        B = x.shape[0]
        condition_tensor1 = self.vae.encode(x) * self.scale_factor
        condition_tensor2 = self.con_encoder(x)
        noise = torch.randn(B, *z_size, device=x.device)
        if infer_type == 'ddim':
            sampled_latent = self.sample(noise, condition=condition_tensor1, num_inference_steps=num_inference_steps,
                                         mode="concat", seg=condition_tensor2)
        elif infer_type == 'ddpm':
            sampled_latent = self.sample2(noise, condition=condition_tensor1, num_inference_steps=num_inference_steps,
                                          mode="concat", seg=condition_tensor2)
        else:
            raise ValueError(f"infer_type error: {infer_type}")
        reconstructed = self.vae.decode(sampled_latent / self.scale_factor)
        return reconstructed

    def save_3d_image_slices(self, tag, image_3d, global_step):
        """
        :param image_3d: 3D 图像张量，形状为 (C, D, H, W)
        :param tag: 图像的标签（用于日志记录）
        :param global_step: 当前训练步数
        """
        # 获取中间切片
        depth = image_3d.shape[1]
        height = image_3d.shape[2]
        width = image_3d.shape[3]

        axial_slice = image_3d[:, depth // 2, :, :]  # 轴向切片，形状为 (C, H, W)
        sagittal_slice = image_3d[:, :, height // 2, :]  # 矢状面切片，形状为 (C, D, W)
        coronal_slice = image_3d[:, :, :, width // 2]  # 冠状面切片，形状为 (C, D, H)

        # 拼接：形状为 (3, max_dim, max_dim)
        concat_slice = torch.cat([axial_slice, sagittal_slice, coronal_slice], dim=0)  # (3, max_dim, max_dim)
        concat_slice = concat_slice.unsqueeze(1)  # (3, 1, max_dim, max_dim)

        # 堆叠并生成网格
        grid = torchvision.utils.make_grid(
            concat_slice,
            nrow=3,
            normalize=True,
            value_range=(0, 1),
        )

        # 记录图像
        self.logger.experiment.add_image(tag, grid, global_step)
        return


def run():
    print(torch.cuda.is_available())

    path = r'/media/xxxx/Data/ULF_ENC'
    dataset = load_Multi_modal_dataset(path, is_train=True, load_getitem=False, out_min_max=(0, 1))
    dataloader = patch_train_maskdataloader(dataset, patch_size=(128, 128, 128), batch_size=4, samples_per_volume=16)
    # 加载模型
    num_training_steps = 1000000
    model = DiffusionModel(num_training_steps=num_training_steps)
    print('加载成功')

    trainer = create_ddpmtrainer(name='ldm_3d', save_dir="./new_logs", checkpoint_dir="./new_checkpoints/ldm_3d",
                                 precision='bf16', max_epoch=100000, monitor='train_loss', strategy="ddp",
                                 check_val_every_n_epoch=500)  # ddp_find_unused_parameters_true
    # 训练
    trainer.fit(model, dataloader, ckpt_path='last')


if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES=0 python train_2d_ldm.py 2>&1 | tee out.log
    # tensorboard --logdir xxxx
    run()