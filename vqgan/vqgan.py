import math
import random
import argparse
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision
import torchio as tio
from typing import Union
from vector_quantize_pytorch import FSQ, ResidualVQ, LFQ, VectorQuantize
from monai.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator, ControlNet, VQVAE
from monai.losses import PerceptualLoss
from monai.networks.layers import Act
from typing import *
from einops import rearrange
from monai.metrics import SSIMMetric

def shift_dim(x, src_dim=-1, dest_dim=-1, make_contiguous=True):
    n_dims = len(x.shape)
    if src_dim < 0:
        src_dim = n_dims + src_dim
    if dest_dim < 0:
        dest_dim = n_dims + dest_dim

    assert 0 <= src_dim < n_dims and 0 <= dest_dim < n_dims

    dims = list(range(n_dims))
    del dims[src_dim]

    permutation = []
    ctr = 0
    for i in range(n_dims):
        if i == dest_dim:
            permutation.append(src_dim)
        else:
            permutation.append(dims[ctr])
            ctr += 1
    x = x.permute(permutation)
    if make_contiguous:
        x = x.contiguous()
    return x

def r_reg(d_out, x_in):
    """
    计算R1正则化损失。

    参数:
        d_out: 判别器输出。
        x_in: 输入图像。

    返回:
        reg: 正则化损失。
    """
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.reshape(batch_size, -1).sum(1).mean(0)
    return reg

class GANLossComps(nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge',
            'wgan-logistic-ns'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self,
                 gan_type: str,
                 real_label_val: float = 1.0,
                 fake_label_val: float = 0.0,
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'wgan-logistic-ns':
            self.loss = self._wgan_logistic_ns_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(
                f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input: torch.Tensor, target: bool) -> torch.Tensor:
        """wgan loss.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def _wgan_logistic_ns_loss(self, input: torch.Tensor,
                               target: bool) -> torch.Tensor:
        """WGAN loss in logistically non-saturating mode.

        This loss is widely used in StyleGANv2.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """

        return F.softplus(-input).mean() if target else F.softplus(
            input).mean()

    def get_target_label(self, input: torch.Tensor,
                         target_is_real: bool) -> Union[bool, torch.Tensor]:
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise, \
                return Tensor.
        """

        if self.gan_type in ['wgan', 'wgan-logistic-ns']:
            return target_is_real
        target_val = (
            self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self,
                input: torch.Tensor,
                target_is_real: bool,
                is_disc: bool = False) -> torch.Tensor:
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight

def calculate_adaptive_weight(nll_loss, g_loss, last_layer):
    # https://github.com/kakaobrain/rq-vae-transformer/blob/main/rqvae/trainers/trainer_rqvae.py
    nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
    g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
    return d_weight

def Normalize(in_channels, norm_type='group', num_groups=16):
    assert norm_type in ['group', 'instance', 'batch']
    if norm_type == 'group':
        return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == 'batch':
        return torch.nn.SyncBatchNorm(in_channels)
    elif norm_type == 'instance':
        return torch.nn.InstanceNorm3d(in_channels, affine=True, track_running_stats=False)

def extend_by_dim(krnlsz, model_type='3d', half_dim=1):
    if model_type == '2d':
        outsz = [krnlsz] * 2
    elif model_type == '3d':
        outsz = [krnlsz] * 3
    elif model_type == '2.5d':
        outsz = [(np.array(krnlsz) * 0 + 1) * half_dim] + [krnlsz] * 2
    else:
        outsz = [krnlsz]
    return tuple(outsz)

def build_end_activation(input, activation='linear', alpha=None):
    if activation == 'softmax':
        output = F.softmax(input, dim=1)
    elif activation == 'sigmoid':
        output = torch.sigmoid(input)
    elif activation == 'elu':
        if alpha is None: alpha = 0.01
        output = F.elu(input, alpha=alpha)
    elif activation == 'lrelu':
        if alpha is None: alpha = 0.01
        output = F.leaky_relu(input, negative_slope=alpha)
    elif activation == 'relu':
        output = F.relu(input)
    elif activation == 'tanh':
        output = F.tanh(input)  # * (3-2.0*torch.relu(1-torch.relu(input*100)))
    else:
        output = input
    return output

class Autoencoder(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.cfg = cfg
        self.startd_step = cfg.model.startd_step
        self.sample_step = 5000

        self.model = AutoencoderKL(
            spatial_dims=2,
            in_channels=3,
            out_channels=3,
            channels=(128, 256),
            norm_num_groups=32,
            latent_channels=4,
            num_res_blocks=2,
            attention_levels=(False, True),
        )
        # 判别器
        self.discriminator = PatchDiscriminator(spatial_dims=2, in_channels=3,
                                                channels=64, out_channels=3,
                                                num_layers_d=3)
        # loss权重
        self.kl_weight = 1e-6
        self.gan_feat_weight = cfg.model.gan_feat_weight
        self.adv_weight = cfg.model.adv_weight
        self.recon_weight = cfg.model.recon_weight
        self.ganloss = GANLossComps(gan_type=cfg.model.gan_type, loss_weight=self.adv_weight)

        self.gradient_clip_val = cfg.gradient_clip_val

    def configure_optimizers(self):
        lr = self.cfg.model.lr
        opt_g = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr, betas=(0.9, 0.999))
        return [opt_g, opt_d], []

    def training_step(self, batch):
        optimizer_g, optimizer_d = self.optimizers()
        Tup_T1_raw = batch['Tup_T1'][tio.DATA]
        Tup_T2_raw = batch['Tup_T2'][tio.DATA]
        Tup_FLAIR_raw = batch['Tup_FLAIR'][tio.DATA]

        # (B, 1, H, W, 3) -> (B, H, W, 3) -> (B, 3, H, W)
        Tup_T1 = Tup_T1_raw.squeeze(1).permute(0, 3, 1, 2)
        Tup_T2 = Tup_T2_raw.squeeze(1).permute(0, 3, 1, 2)
        Tup_FLAIR = Tup_FLAIR_raw.squeeze(1).permute(0, 3, 1, 2)
        x = torch.concat([Tup_T1, Tup_T2, Tup_FLAIR], dim=1)

        # =================================================================================== #
        #                               2. 训练生成器                                          #
        # =================================================================================== #
        self.toggle_optimizer(optimizer_g)
        x_recon, z_mu, z_sigma = self.model(x)
        # 计算重建损失
        recon_loss = F.l1_loss(x_recon, x)
        # 计算KL散度损失
        kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        # features_loss and adv_loss
        if self.global_step > self.startd_step:
            # 计算感知loss和对抗loss
            out_recon = self.discriminator(x_recon)
            logits_recon = out_recon[-1]
            features_recon = out_recon[:-1]
            out_orig = self.discriminator(x)
            features_real = out_orig[:-1]
            adv_loss = self.ganloss(logits_recon, True, is_disc=False)

            features_loss = torch.tensor(0.0, device=x.device)
            if self.gan_feat_weight > 0:
                for i in range(0, len(features_recon)):
                    features_loss += F.l1_loss(features_recon[i], features_real[i].detach())
            self.log("train/adv_loss", adv_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/features_loss", features_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        else:
            adv_loss = torch.tensor(0.0, device=x.device)
            features_loss = torch.tensor(0.0, device=x.device)
        # 总生成器损失
        g_loss = recon_loss * self.recon_weight + (kl_loss * self.kl_weight) + adv_loss * self.adv_weight + features_loss * self.gan_feat_weight
        self.manual_backward(g_loss)  # 手动反向传播
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)  # 关闭优化器
        self.log("train_recon_loss", recon_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/kl_losss", kl_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/g_loss", g_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        # =================================================================================== #
        #                             3. 训练判别器                                            #
        # =================================================================================== #
        if self.global_step > self.startd_step:
            self.toggle_optimizer(optimizer_d)
            out_real = self.discriminator(x)
            logits_real = out_real[-1]
            x_fake = x_recon.detach()
            out_fake = self.discriminator(x_fake)
            logits_fake = out_fake[-1]
            d_loss = (self.ganloss(logits_real, True, is_disc=True) + self.ganloss(logits_fake, False, is_disc=True)) / 2
            self.clip_gradients(optimizer_d, gradient_clip_val=self.gradient_clip_val, gradient_clip_algorithm="norm")
            optimizer_d.step()
            optimizer_d.zero_grad()
            self.untoggle_optimizer(optimizer_d)
            self.log("train/d_loss", d_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return g_loss

    def validation_step(self, batch):
        pass

class VQGAN(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.cfg = cfg
        self.startd_step = cfg.model.startd_step
        self.sample_step = 5000

        self.model = VQVAE(
            spatial_dims=2,
            in_channels=15,
            out_channels=15,
            channels=(128, 256),
            num_res_channels=(128, 256),
            downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1)),
            upsample_parameters=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
            num_res_layers=3,
            num_embeddings=8192,
            embedding_dim=8,
            commitment_cost=0.25,
            decay=0.99,
        )
        # 判别器
        self.discriminator = PatchDiscriminator(spatial_dims=2, in_channels=15,
                                                channels=64, out_channels=15,
                                                num_layers_d=3)
        # loss权重
        self.gan_feat_weight = cfg.model.gan_feat_weight
        self.adv_weight = cfg.model.adv_weight
        self.recon_weight = cfg.model.recon_weight
        self.ganloss = GANLossComps(gan_type=cfg.model.gan_type, loss_weight=self.adv_weight)

        self.gradient_clip_val = cfg.gradient_clip_val

    def configure_optimizers(self):
        lr = self.cfg.model.lr
        opt_g = torch.optim.Adam(self.model.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        return [opt_g, opt_d], []

    def training_step(self, batch):
        optimizer_g, optimizer_d = self.optimizers()
        # (B, 1, H, W, 5）
        Tup_T1_raw = batch['Tup_T1'][tio.DATA]
        Tup_T2_raw = batch['Tup_T2'][tio.DATA]
        Tup_FLAIR_raw = batch['Tup_FLAIR'][tio.DATA]
        # (B, 1, H, W, 5) -> (B, H, W, 5) -> (B, 5, H, W)
        Tup_T1 = Tup_T1_raw.squeeze(1).permute(0, 3, 1, 2)
        Tup_T2 = Tup_T2_raw.squeeze(1).permute(0, 3, 1, 2)
        Tup_FLAIR = Tup_FLAIR_raw.squeeze(1).permute(0, 3, 1, 2)
        x = torch.concat([Tup_T1, Tup_T2, Tup_FLAIR], dim=1)

        # =================================================================================== #
        #                               2. 训练生成器                                          #
        # =================================================================================== #
        self.toggle_optimizer(optimizer_g)
        x_recon, quantization_losses = self.model(images=x)
        # 计算重建损失
        recon_loss = F.l1_loss(x_recon, x)
        # features_loss and adv_loss
        if self.global_step > self.startd_step:
            # 计算感知loss和对抗loss
            out_recon = self.discriminator(x_recon)
            logits_recon = out_recon[-1]
            features_recon = out_recon[:-1]
            out_orig = self.discriminator(x)
            features_real = out_orig[:-1]
            adv_loss = self.ganloss(logits_recon, True, is_disc=False)

            features_loss = torch.tensor(0.0, device=x.device)
            if self.gan_feat_weight > 0:
                for i in range(0, len(features_recon) - 1):
                    features_loss += F.l1_loss(features_recon[i], features_real[i].detach())
            self.log("train/adv_loss", adv_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/features_loss", features_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        else:
            adv_loss = torch.tensor(0.0, device=x.device)
            features_loss = torch.tensor(0.0, device=x.device)

        g_loss = quantization_losses + recon_loss * self.recon_weight + adv_loss * self.adv_weight + features_loss * self.gan_feat_weight
        self.manual_backward(g_loss)  # 手动反向传播
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)  # 关闭优化器
        self.log("train_recon_loss", recon_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/quantization_losses", quantization_losses, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/g_loss", g_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        # =================================================================================== #
        #                             3. 训练判别器                                            #
        # =================================================================================== #
        if self.global_step > self.startd_step:
            self.toggle_optimizer(optimizer_d)
            out_real = self.discriminator(x)
            logits_real = out_real[-1]
            x_fake = x_recon.detach()
            out_fake = self.discriminator(x_fake)
            logits_fake = out_fake[-1]
            d_loss = (self.ganloss(logits_real, True, is_disc=True) + self.ganloss(logits_fake, False, is_disc=True)) / 2
            self.clip_gradients(optimizer_d, gradient_clip_val=self.gradient_clip_val, gradient_clip_algorithm="norm")
            self.manual_backward(d_loss)
            optimizer_d.step()
            optimizer_d.zero_grad()
            self.untoggle_optimizer(optimizer_d)
            self.log("train/d_loss", d_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        # --- 可视化 ---
        if self.global_step % self.sample_step == 0:
            try:
                save_target_vis = x[0:1, :, :, :].permute(1, 0, 2, 3)  # (9, 1, H, W)
                save_pred_vis = x_recon[0:1, :, :, :].permute(1, 0, 2, 3)  # (9, 1, H, W)
                self.logger.experiment.add_images("train/target", save_target_vis, self.global_step)
                self.logger.experiment.add_images("train/pred", save_pred_vis, self.global_step)
            except Exception as e:
                print(f"Warning: Logging images/SSIM failed: {e}")
        return g_loss

class VQGAN2(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.cfg = cfg
        self.startd_step = cfg.model.startd_step
        self.sample_step = 5000
        self.encoder = Encoder(in_channels=cfg.model.in_channels, basedim=cfg.model.basedim,
                               downdeepth=cfg.model.downdeepth, model_type=cfg.model.model_type,
                               embedding_dim=cfg.model.embedding_dim, num_groups=cfg.model.num_groups)

        self.decoder = Decoder(out_channels=cfg.model.out_channels, basedim=cfg.model.basedim,
                               downdeepth=cfg.model.downdeepth, model_type=cfg.model.model_type,
                               embedding_dim=cfg.model.embedding_dim, num_groups=cfg.model.num_groups)
        # codebook
        self.codebook = VectorQuantize(dim=cfg.model.embedding_dim, codebook_size=cfg.model.n_codes, decay=0.99,
                                       commitment_weight=0.25, kmeans_init=True, threshold_ema_dead_code=2)
        # 判别器
        self.discriminator = PatchDiscriminator(spatial_dims=2, in_channels=cfg.model.in_channels,
                                                channels=cfg.model.disc_channels,
                                                out_channels=cfg.model.out_channels,
                                                num_layers_d=cfg.model.disc_downdeepth)

        # loss权重
        self.gan_feat_weight = cfg.model.gan_feat_weight
        self.adv_weight = cfg.model.adv_weight
        self.recon_weight = cfg.model.recon_weight
        self.perceptual_weight = cfg.model.perceptual_weight
        self.ganloss = GANLossComps(gan_type=cfg.model.gan_type, loss_weight=self.adv_weight)

        self.gradient_clip_val = cfg.model.gradient_clip_val
        self.ssim_metric = SSIMMetric(spatial_dims=2, data_range=1.0)

    def configure_optimizers(self):
        lr = self.cfg.model.lr
        opt_g = torch.optim.Adam(list(self.encoder.parameters()) +
                                 list(self.codebook.parameters()) +
                                 list(self.decoder.parameters()),
                                 lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr)
        return opt_g, opt_d

    def encode(self, x, quantize=True):
        z = self.encoder(x)
        if quantize:
            z, indices, _ = self.vector_quantize(z)
        return z

    def decode(self, z, quantize=True):
        if quantize:
            z, indices, _ = self.vector_quantize(z)
        x_recon = self.decoder(z)
        return x_recon

    def vector_quantize(self, z):
        # Step 1: 转换为 (B, H*W*D, C)
        B, C, H, W = z.shape
        z_flat = rearrange(z, 'b c h w -> b (h w) c')

        # Step 2: 通过 codebook
        quantized_flat, indices, commit_loss = self.codebook(z_flat)

        # Step 3: 还原为 (B, C, H, W, D)
        quantized = rearrange(quantized_flat, 'b (h w) c -> b c h w', h=H, w=W)

        return quantized, indices, commit_loss.mean()

    def forward(self, x):
        z = self.encoder(x)
        z, indices, _ = self.vector_quantize(z)
        x_recon = self.decoder(z)
        return x_recon

    def get_last_layer(self):
        return self.decoder.final_conv.weight

    def training_step(self, batch):
        optimizer_g, optimizer_d = self.optimizers()
        # (B, 1, H, W, 3）
        Tup_T1_raw = batch['Tup_T1'][tio.DATA]
        Tup_T2_raw = batch['Tup_T2'][tio.DATA]
        Tup_FLAIR_raw = batch['Tup_FLAIR'][tio.DATA]

        # (B, 1, H, W, 3) -> (B, H, W, 3) -> (B, 3, H, W)
        Tup_T1 = Tup_T1_raw.squeeze(1).permute(0, 3, 1, 2)
        Tup_T2 = Tup_T2_raw.squeeze(1).permute(0, 3, 1, 2)
        Tup_FLAIR = Tup_FLAIR_raw.squeeze(1).permute(0, 3, 1, 2)

        x = torch.concat([Tup_T1, Tup_T2, Tup_FLAIR], dim=1)
        # =================================================================================== #
        #                               2. 训练生成器                                          #
        # =================================================================================== #
        self.toggle_optimizer(optimizer_g)
        z = self.encoder(x)
        z, indices, quantization_losses = self.vector_quantize(z)
        x_recon = self.decoder(z)

        # 计算重建损失
        recon_loss = F.l1_loss(x_recon, x)

        # perceptual loss
        # perceptual_loss = self.perceptualloss(x_recon, x)

        # features_loss and adv_loss
        if self.global_step > self.startd_step:
            # 计算感知loss和对抗loss
            out_recon = self.discriminator(x_recon)
            logits_recon = out_recon[-1]
            features_recon = out_recon[:-1]
            # 获取真实图像的特征（用于特征匹配）
            with torch.no_grad():
                out_real = self.discriminator(x)
                features_real = out_real[:-1]

            adv_loss = self.ganloss(logits_recon, True, is_disc=False)

            if self.gan_feat_weight > 0:
                features_loss = torch.tensor(0.0, device=x.device)
                for feat_recon, feat_real in zip(features_recon, features_real):
                    features_loss += F.l1_loss(feat_recon, feat_real.detach())
            self.log("train/adv_loss", adv_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/features_loss", features_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        else:
            adv_loss = torch.tensor(0.0, device=x.device)
            features_loss = torch.tensor(0.0, device=x.device)

        # 计算总损失
        g_loss = recon_loss * self.recon_weight + quantization_losses + adv_loss + features_loss * self.gan_feat_weight  # + perceptual_loss * self.perceptual_weight
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)  # 关闭优化器
        self.log("train_recon_loss", recon_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        # self.log("train/perceptual_loss", perceptual_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("g_loss", g_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/quantization_losses", quantization_losses, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        # =================================================================================== #
        #                             3. 训练判别器                                            #
        # =================================================================================== #
        if self.global_step > self.startd_step:
            self.toggle_optimizer(optimizer_d)
            # x_real = x.detach().requires_grad_(True)
            x_real = x.detach()
            out_real = self.discriminator(x_real)
            logits_real = out_real[-1]
            x_fake = x_recon.detach()
            out_fake = self.discriminator(x_fake)
            logits_fake = out_fake[-1]
            d_loss = (self.ganloss(logits_real, True, is_disc=True) + self.ganloss(logits_fake, False, is_disc=True)) / 2
            self.log("train/d_loss", d_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

            # r1_penalty = r_reg(logits_real, x_real)
            # d_loss = d_loss + r1_penalty
            # self.log("train/r_penalty", r1_penalty, prog_bar=True, logger=True, on_step=True, on_epoch=True)

            self.manual_backward(d_loss)
            self.clip_gradients(optimizer_d, gradient_clip_val=self.gradient_clip_val, gradient_clip_algorithm="norm")
            optimizer_d.step()
            optimizer_d.zero_grad()
            self.untoggle_optimizer(optimizer_d)

        # --- 可视化 ---
        if self.global_step % self.sample_step == 0:
            try:
                save_target_vis = x[0:1, :, :, :].permute(1, 0, 2, 3)  # (9, 1, H, W)
                save_pred_vis = x_recon[0:1, :, :, :].permute(1, 0, 2, 3)  # (9, 1, H, W)
                self.logger.experiment.add_images("train/target", save_target_vis, self.global_step)
                self.logger.experiment.add_images("train/pred", save_pred_vis, self.global_step)

                ssim_value = self.ssim_metric(x_recon, x).mean()
                self.log("train/SSIM", ssim_value, on_step=True, prog_bar=True, logger=True, on_epoch=True)
            except Exception as e:
                print(f"Warning: Logging images/SSIM failed: {e}")
        return g_loss

    def save_3d_image_slices(self, image_3d, tag, global_step):
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

class FSQGAN(pl.LightningModule):
    def __init__(self, cfg):
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.cfg = cfg
        self.startd_step = cfg.model.startd_step
        self.sample_step = 5000
        self.encoder = Encoder(in_channels=cfg.model.in_channels, basedim=cfg.model.basedim,
                               downdeepth=cfg.model.downdeepth, model_type=cfg.model.model_type,
                               embedding_dim=cfg.model.embedding_dim, num_groups=cfg.model.num_groups)

        self.decoder = Decoder(out_channels=cfg.model.out_channels, basedim=cfg.model.basedim,
                               downdeepth=cfg.model.downdeepth, model_type=cfg.model.model_type,
                               embedding_dim=cfg.model.embedding_dim, num_groups=cfg.model.num_groups)
        self.codebook = FSQ(levels = [8, 8, 8, 5, 5, 5], dim=cfg.model.embedding_dim, channel_first = True,)
        # 判别器
        self.discriminator = PatchDiscriminator(spatial_dims=2, in_channels=cfg.model.in_channels,
                                                channels=cfg.model.disc_channels,
                                                out_channels=cfg.model.out_channels,
                                                num_layers_d=cfg.model.disc_downdeepth)

    def configure_optimizers(self):
        lr = self.cfg.model.lr

        opt_g = torch.optim.Adam(list(self.encoder.parameters()) +
                                 list(self.codebook.parameters()) +
                                 list(self.decoder.parameters()),
                                 lr=lr, betas=(0.5, 0.9))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr, betas=(0.5, 0.9))
        return [opt_g, opt_d], []

    def training_step(self, batch):
        optimizer_g, optimizer_d = self.optimizers()

        x = batch['Tlow'][tio.DATA]
        # =================================================================================== #
        #                               2. 训练生成器                                          #
        # =================================================================================== #
        self.toggle_optimizer(optimizer_g)
        z = self.encoder(x)
        z, indices = self.codebook(z)
        x_recon = self.decoder(z)

        # 计算重建损失
        recon_loss = F.l1_loss(x_recon, x)

        # features_loss and adv_loss
        if self.global_step > self.startd_step:
            # 计算感知loss和对抗loss
            out_recon = self.discriminator(x_recon)
            logits_recon = out_recon[-1]
            features_recon = out_recon[:-1]
            # 获取真实图像的特征（用于特征匹配）
            with torch.no_grad():
                out_real = self.discriminator(x)
                features_real = out_real[:-1]

            adv_loss = self.ganloss(logits_recon, True, is_disc=False)

            if self.gan_feat_weight > 0:
                features_loss = torch.tensor(0.0, device=x.device)
                for feat_recon, feat_real in zip(features_recon, features_real):
                    features_loss += F.l1_loss(feat_recon, feat_real.detach())
            self.log("train/adv_loss", adv_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/features_loss", features_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        else:
            adv_loss = torch.tensor(0.0, device=x.device)
            features_loss = torch.tensor(0.0, device=x.device)

        # 计算总损失
        g_loss = recon_loss * self.recon_weight + adv_loss + features_loss * self.gan_feat_weight
        self.manual_backward(g_loss)  # 手动反向传播
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)  # 关闭优化器
        self.log("train_recon_loss", recon_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/g_loss", g_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        # =================================================================================== #
        #                             3. 训练判别器                                            #
        # =================================================================================== #
        if self.global_step > self.startd_step:
            self.toggle_optimizer(optimizer_d)
            # x_real = x.detach().requires_grad_(True)
            x_real = x.detach()
            out_real = self.discriminator(x_real)
            logits_real = out_real[-1]
            x_fake = x_recon.detach()
            out_fake = self.discriminator(x_fake)
            logits_fake = out_fake[-1]
            d_loss = (self.ganloss(logits_real, True, is_disc=True) + self.ganloss(logits_fake, False, is_disc=True)) / 2
            self.log("train/d_loss", d_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

            # r1_penalty = r_reg(logits_real, x_real)
            # d_loss = d_loss + r1_penalty
            # self.log("train/r_penalty", r1_penalty, prog_bar=True, logger=True, on_step=True, on_epoch=True)

            self.manual_backward(d_loss)
            self.clip_gradients(optimizer_d, gradient_clip_val=self.gradient_clip_val, gradient_clip_algorithm="norm")
            optimizer_d.step()
            optimizer_d.zero_grad()
            self.untoggle_optimizer(optimizer_d)

    def save_3d_image_slices(self, image_3d, tag, global_step):
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

class ResidualVQGAN(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.cfg = cfg
        self.startd_step = cfg.model.startd_step
        self.sample_step = 5000
        self.encoder = Encoder(in_channels=cfg.model.in_channels, basedim=cfg.model.basedim,
                               downdeepth=cfg.model.downdeepth, model_type=cfg.model.model_type,
                               embedding_dim=cfg.model.embedding_dim, num_groups=cfg.model.num_groups)

        self.decoder = Decoder(out_channels=cfg.model.out_channels, basedim=cfg.model.basedim,
                               downdeepth=cfg.model.downdeepth, model_type=cfg.model.model_type,
                               embedding_dim=cfg.model.embedding_dim, num_groups=cfg.model.num_groups)
        # codebook
        self.codebook = ResidualVQ(dim=cfg.model.embedding_dim, codebook_size=cfg.model.n_codes,
                                   num_quantizers=cfg.model.num_quantizers, decay=0.99, commitment_weight=0.25,
                                   kmeans_init=True, threshold_ema_dead_code=2, rotation_trick=True)
        # 判别器
        self.discriminator = PatchDiscriminator(spatial_dims=2, in_channels=cfg.model.in_channels,
                                                channels=cfg.model.disc_channels,
                                                out_channels=cfg.model.out_channels,
                                                num_layers_d=cfg.model.disc_downdeepth)

        # loss权重
        self.gan_feat_weight = cfg.model.gan_feat_weight
        self.adv_weight = cfg.model.adv_weight
        self.recon_weight = cfg.model.recon_weight
        self.perceptual_weight = cfg.model.perceptual_weight
        self.ganloss = GANLossComps(gan_type=cfg.model.gan_type, loss_weight=self.adv_weight)
        self.ssim_metric = SSIMMetric(spatial_dims=2, data_range=1.0)
        print("loss weights:", self.recon_weight, self.adv_weight, self.gan_feat_weight, self.perceptual_weight)

    def configure_optimizers(self):
        lr = self.cfg.model.lr
        opt_g = torch.optim.Adam(list(self.encoder.parameters()) +
                                 list(self.codebook.parameters()) +
                                 list(self.decoder.parameters()),
                                 lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr)
        return opt_g, opt_d

    def encode(self, x, quantize=True):
        z = self.encoder(x)
        if quantize:
            z, indices, _ = self.vector_quantize(z)
        return z

    def decode(self, z, quantize=True):
        if quantize:
            z, indices, _ = self.vector_quantize(z)
        x_recon = self.decoder(z)
        return x_recon

    def vector_quantize(self, z):
        # Step 1: 转换为 (B, H*W*D, C)
        B, C, H, W = z.shape
        z_flat = rearrange(z, 'b c h w -> b (h w) c')

        # Step 2: 通过 codebook
        quantized_flat, indices, commit_loss = self.codebook(z_flat)

        # Step 3: 还原为 (B, C, H, W, D)
        quantized = rearrange(quantized_flat, 'b (h w) c -> b c h w', h=H, w=W)

        return quantized, indices, commit_loss.mean()

    def forward(self, x):
        z = self.encoder(x)
        z, indices, _ = self.vector_quantize(z)
        x_recon = self.decoder(z)
        return x_recon

    def get_last_layer(self):
        return self.decoder.final_conv.weight

    def training_step(self, batch):
        optimizer_g, optimizer_d = self.optimizers()
        # 原始形状: (B, 1, H, W, 3）
        Tup_T1_raw = batch['Tup_T1'][tio.DATA]
        Tup_T2_raw = batch['Tup_T2'][tio.DATA]
        Tup_FLAIR_raw = batch['Tup_FLAIR'][tio.DATA]

        # 形状变换: (B, 1, H, W, 3) -> (B, H, W, 3) -> (B, 3, H, W)
        Tup_T1 = Tup_T1_raw.squeeze(1).permute(0, 3, 1, 2)
        Tup_T2 = Tup_T2_raw.squeeze(1).permute(0, 3, 1, 2)
        Tup_FLAIR = Tup_FLAIR_raw.squeeze(1).permute(0, 3, 1, 2)

        x = torch.concat([Tup_T1, Tup_T2, Tup_FLAIR], dim=1)
        print(x.min(), x.max(), x.shape)
        # =================================================================================== #
        #                               2. 训练生成器                                          #
        # =================================================================================== #
        self.toggle_optimizer(optimizer_g)
        z = self.encoder(x)
        z, indices, quantization_losses = self.vector_quantize(z)
        print('z shape:', z.shape, 'global step:', self.global_step)
        x_recon = self.decoder(z)

        # 计算重建损失
        recon_loss = F.l1_loss(x_recon, x)

        # perceptual loss
        # perceptual_loss = self.perceptualloss(x_recon, x)

        # features_loss and adv_loss
        if self.global_step > self.startd_step:
            # 计算感知loss和对抗loss
            out_recon = self.discriminator(x_recon)
            logits_recon = out_recon[-1]
            features_recon = out_recon[:-1]
            # 获取真实图像的特征（用于特征匹配）
            with torch.no_grad():
                out_real = self.discriminator(x)
                features_real = out_real[:-1]

            adv_loss = self.ganloss(logits_recon, True, is_disc=False)

            if self.gan_feat_weight > 0:
                features_loss = torch.tensor(0.0, device=x.device)
                for feat_recon, feat_real in zip(features_recon, features_real):
                    features_loss += F.l1_loss(feat_recon, feat_real.detach())
            self.log("train/adv_loss", adv_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/features_loss", features_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        else:
            adv_loss = torch.tensor(0.0, device=x.device)
            features_loss = torch.tensor(0.0, device=x.device)

        # 计算总损失
        g_loss = recon_loss * self.recon_weight + quantization_losses + adv_loss + features_loss * self.gan_feat_weight  # + perceptual_loss * self.perceptual_weight
        self.manual_backward(g_loss)  # 手动反向传播
        # self.clip_gradients(optimizer_g, gradient_clip_val=1, gradient_clip_algorithm="norm")
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)  # 关闭优化器
        self.log("train_recon_loss", recon_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        # self.log("train/perceptual_loss", perceptual_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("g_loss", g_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/quantization_losses", quantization_losses, prog_bar=True, logger=True, on_step=True,
                 on_epoch=True)

        # =================================================================================== #
        #                             3. 训练判别器                                            #
        # =================================================================================== #
        if self.global_step > self.startd_step:
            self.toggle_optimizer(optimizer_d)
            x_real = x.detach().requires_grad_(True)
            # x_real = x.detach()
            out_real = self.discriminator(x_real)
            logits_real = out_real[-1]
            x_fake = x_recon.detach()
            out_fake = self.discriminator(x_fake)
            logits_fake = out_fake[-1]

            d_loss = (self.ganloss(logits_real, True, is_disc=True) + self.ganloss(logits_fake, False,
                                                                                   is_disc=True)) / 2
            self.log("train/d_loss", d_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

            # r1_penalty = r_reg(logits_real, x_real)
            # d_loss = d_loss + r1_penalty
            # self.log("train/r_penalty", r1_penalty, prog_bar=True, logger=True, on_step=True, on_epoch=True)

            self.manual_backward(d_loss)
            self.clip_gradients(optimizer_d, gradient_clip_val=0.0015, gradient_clip_algorithm="norm")
            optimizer_d.step()
            optimizer_d.zero_grad()
            self.untoggle_optimizer(optimizer_d)

        # --- 可视化 (调整以适应 9 通道) ---
        if self.global_step % self.sample_step == 0:
            try:
                save_target_vis = x[0:1, :, :, :].permute(1, 0, 2, 3)  # (9, 1, H, W)
                save_pred_vis = x_recon[0:1, :, :, :].permute(1, 0, 2, 3)  # (9, 1, H, W)
                self.logger.experiment.add_images("train/target", save_target_vis, self.global_step)
                self.logger.experiment.add_images("train/pred", save_pred_vis, self.global_step)

                ssim_value = self.ssim_metric(x_recon, x).mean()
                self.log("train/SSIM", ssim_value, on_step=True, prog_bar=True, logger=True, on_epoch=True)
            except Exception as e:
                print(f"Warning: Logging images/SSIM failed: {e}")
        return g_loss

    def save_3d_image_slices(self, image_3d, tag, global_step):
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

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='same', mid_channels=None,
                 model_type='3d', residualskip=True, num_groups=16, norm_type='group'):
        super(ResBlock, self).__init__()

        self.change_dimension = in_channels != out_channels
        self.model_type = model_type
        self.residualskip = residualskip
        padding = {'same': kernel_size // 2, 'valid': 0}[padding] if padding in ['same', 'valid'] else padding
        mid_channels = out_channels if mid_channels is None else mid_channels

        if self.model_type == '3d':
            self.Conv = nn.Conv3d
        elif self.model_type == '2.5d':
            self.Conv = nn.Conv3d
        else:
            self.Conv = nn.Conv2d

        def extdim(krnlsz, halfdim=1):
            return extend_by_dim(krnlsz, model_type=model_type.lower(), half_dim=halfdim)

        self.short_cut_conv = self.Conv(in_channels, out_channels, 1, extdim(stride))
        self.norm0 = Normalize(out_channels, norm_type, num_groups=num_groups)
        self.conv1 = self.Conv(in_channels, mid_channels, extdim(kernel_size, 3), extdim(stride),
                               padding=extdim(padding, 1), padding_mode='reflect')
        self.norm1 = Normalize(mid_channels, norm_type, num_groups=num_groups)
        self.silu1 = nn.SiLU()
        self.conv2 = self.Conv(mid_channels, out_channels, extdim(kernel_size, 3), extdim(1),
                               padding=extdim(padding, 1), padding_mode='reflect')
        self.norm2 = Normalize(out_channels, norm_type, num_groups=num_groups)
        self.silu2 = nn.SiLU()

    def forward(self, x):
        if self.residualskip and self.change_dimension:
            short_cut_conv = self.norm0(self.short_cut_conv(x))
        else:
            short_cut_conv = x
        o_c1 = self.silu1(self.norm1(self.conv1(x)))
        o_c2 = self.norm2(self.conv2(o_c1))
        if self.residualskip:
            out_res = self.silu2(o_c2 + short_cut_conv)
        else:
            out_res = self.silu2(o_c2)
        return out_res

class StackConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='same', mid_channels=None,
                 model_type='3d', residualskip=False):
        super(StackConvBlock, self).__init__()

        self.change_dimension = in_channels != out_channels
        self.model_type = model_type
        self.residualskip = residualskip
        padding = {'same': kernel_size // 2, 'valid': 0}[padding] if padding in ['same', 'valid'] else padding
        mid_channels = out_channels if mid_channels is None else mid_channels

        if self.model_type == '3d':
            self.ConvBlock, self.InstanceNorm = nn.Conv3d, nn.InstanceNorm3d
        elif self.model_type == '2.5d':
            self.ConvBlock, self.InstanceNorm = nn.Conv3d, nn.InstanceNorm3d
        else:
            self.ConvBlock, self.InstanceNorm = nn.Conv2d, nn.InstanceNorm2d

        def extdim(krnlsz, halfdim=1):
            return extend_by_dim(krnlsz, model_type=model_type.lower(), half_dim=halfdim)

        self.short_cut_conv = self.ConvBlock(in_channels, out_channels, 1, extdim(stride))
        self.norm0 = self.InstanceNorm(out_channels, affine=True)
        self.conv1 = self.ConvBlock(in_channels, mid_channels, extdim(kernel_size, 3), extdim(stride),
                                    padding=extdim(padding, 1), padding_mode='reflect')
        self.norm1 = self.InstanceNorm(mid_channels, affine=True)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.conv2 = self.ConvBlock(mid_channels, out_channels, extdim(kernel_size, 3), extdim(1),
                                    padding=extdim(padding, 1), padding_mode='reflect')
        self.norm2 = self.InstanceNorm(out_channels, affine=True, track_running_stats=False)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        if self.residualskip and self.change_dimension:
            short_cut_conv = self.norm0(self.short_cut_conv(x))
        else:
            short_cut_conv = x
        o_c1 = self.relu1(self.norm1(self.conv1(x)))
        o_c2 = self.norm2(self.conv2(o_c1))
        if self.residualskip:
            out_res = self.relu2(o_c2 + short_cut_conv)
        else:
            out_res = self.relu2(o_c2)
        return out_res

class Encoder(nn.Module):
    def __init__(self, in_channels=1, basedim=32, downdeepth=2, num_res_layers=2, model_type='3d', embedding_dim=8,
                 num_groups=32):
        super().__init__()
        if model_type == '3d':
            self.conv = nn.Conv3d
        else:
            self.conv = nn.Conv2d

        self.begin_conv = ResBlock(in_channels=in_channels, out_channels=basedim, model_type=model_type,
                                   num_groups=num_groups)
        self.encoding_block = nn.ModuleList()
        for convidx in range(0, downdeepth):
            for layeridx in range(0, num_res_layers - 1):
                self.encoding_block.append(
                    ResBlock(in_channels=basedim * 2 ** convidx, out_channels=basedim * 2 ** convidx,
                             model_type=model_type, ))
            self.encoding_block.append(ResBlock(basedim * 2 ** convidx, basedim * 2 ** (convidx + 1), 3, 2,
                                                model_type=model_type, num_groups=num_groups))
        self.enc_out = basedim * 2 ** downdeepth
        self.pre_vq_conv = self.conv(self.enc_out, embedding_dim, 3, 1, 1)

    def forward(self, x):
        x = self.begin_conv(x)
        for block in self.encoding_block:
            x = block(x)
        x = self.pre_vq_conv(x)
        return x

class Decoder(nn.Module):
    def __init__(self, out_channels=1, basedim=32, downdeepth=2, num_res_layers=2,
                 model_type='3d', embedding_dim=8, num_groups=32):
        super().__init__()
        self.model_type = model_type
        if self.model_type == '3d':
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = nn.Conv3d
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = nn.Conv2d

        self.enc_out = basedim * 2 ** downdeepth
        self.post_vq_conv = ResBlock(embedding_dim, self.enc_out, 3, 1, model_type=model_type, num_groups=num_groups)
        self.decoding_blocks = nn.ModuleList()

        for convidx in reversed(range(1, downdeepth + 1)):
            before_up_channels = basedim * 2 ** convidx
            after_up_channels = basedim * 2 ** (convidx - 1)
            block = nn.Module()
            block.upsample = self.up
            block.post_upsample_blocks = nn.ModuleList()

            for layer_idx in range(num_res_layers):
                if layer_idx == 0:
                    block.post_upsample_blocks.append(
                        ResBlock(
                            before_up_channels,
                            after_up_channels,
                            kernel_size=3,
                            stride=1,
                            model_type=self.model_type,
                            num_groups=num_groups
                        )
                    )
                else:
                    block.post_upsample_blocks.append(
                        ResBlock(
                            after_up_channels,
                            after_up_channels,
                            kernel_size=3,
                            stride=1,
                            model_type=self.model_type,
                            num_groups=num_groups
                        )
                    )
            self.decoding_blocks.append(block)
        self.final_conv = self.conv(basedim, out_channels, 3, 1, 1)

    def forward(self, x):
        x = self.post_vq_conv(x)
        for i, block in enumerate(self.decoding_blocks):
            x = block.upsample(x)
            for j, res_block in enumerate(block.post_upsample_blocks):
                x = res_block(x)
        x = self.final_conv(x)
        return x