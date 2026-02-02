from diffusers import UNet2DModel, UNet3DConditionModel, UNet2DConditionModel, PNDMScheduler, DDPMScheduler, \
    DDIMScheduler
from diffusers import PNDMScheduler, DDPMScheduler, DDIMScheduler, DDPMPipeline, DDIMPipeline, StableDiffusionPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
import torchvision


class diffusion_model(pl.LightningModule):
    def __init__(self, config, num_training_steps):
        super().__init__()
        self.modality_len = config.modality_len
        self.num_training_steps = num_training_steps

        self.model = UNet2DModel(
            in_channels=6,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 256, 512, 512),
            attention_head_dim=32,
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",  # a regular ResNet upsampling block
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
            ),
        )
        self.scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, betas=(0.95, 0.999),
                                      weight_decay=1e-6, eps=1e-08)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=1000,  # 预热步数
            num_training_steps=(self.num_training_steps),  # 总训练步数
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def training_step(self, batch):
        T1 = batch['T1']  # shape: [B, 1, H, W]
        T2 = batch['T2']  # shape: [B, 1, H, W]
        PD = batch['PD']  # shape: [B, 1, H, W]
        clean_images = torch.cat([T1, T2, PD], dim=1)

        # Sample noise to add to the images
        noise = torch.randn(clean_images.shape, device=self.device)
        bs = clean_images.shape[0]

        con_input = self.simulation_missing(bs, clean_images)  # [B, C, H, W]

        timesteps = torch.randint(
            0, 1000, (bs,), device=self.device,
            dtype=torch.int64
        )

        noisy_images = self.scheduler.add_noise(clean_images, noise, timesteps)

        model_input = torch.cat([noisy_images, con_input], dim=1)  # [B, 6, H, W]

        noise_pred = self.model(model_input, timesteps, return_dict=False)[0]
        loss = F.mse_loss(noise_pred, noise)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("learning_rate", self.optimizers().param_groups[0]['lr'], on_step=True, on_epoch=True, prog_bar=True,
                 logger=True)
        return loss

    @torch.no_grad()
    def sample(self, noise, con_input, num_inference_steps=1000):
        image = noise
        model_input = torch.cat((image, con_input), dim=1)  # [B, 6, H, W]
        self.scheduler.set_timesteps(num_inference_steps)

        for t in tqdm(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.model(model_input, t).sample

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image).prev_sample
            model_input = torch.cat((image, con_input), dim=1)

        # 3. to (0, 1)
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    def simulation_missing(self, bs, clean_images):
        C = self.modality_len  # 模态种类数（如 3）
        num_missing = torch.randint(1, C, (bs,), device=self.device)  # [B] # 随机决定每个样本缺失几个模态：1 ~ C-1
        # 生成每样本的随机通道排列（模拟 randperm）
        rand = torch.rand(bs, C, device=self.device)  # [B, C]
        _, shuffle_idx = rand.topk(C, dim=1)  # [B, C]，每行是 0~C-1 的随机排列
        # 创建 mask: 前 num_missing[i] 个通道设为 0（mask），其余为 1
        mask = torch.arange(C, device=self.device).expand(bs, C)  # [B, C]
        mask = mask >= num_missing.unsqueeze(1)  # [B, C]，前 k 个 False，其余 True
        # 根据随机顺序 scatter：把前 k 个 True 分配给随机选中的通道
        mask = torch.gather(mask, 1, shuffle_idx)  # 按 shuffle_idx 重排 mask
        mask = mask.float().view(bs, C, 1, 1)  # 转为 float 并扩展到空间维度
        return clean_images * mask  # 缺失位置变为 0

