import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
from lightning.pytorch import LightningModule
import torchvision
from torch.amp import autocast
from monai.networks.nets import DiffusionModelUNet, SPADEDiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler, DDIMScheduler
from monai.inferers import DiffusionInferer, LatentDiffusionInferer
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange

model = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=8,
    out_channels=8,
    num_res_blocks=2,
    channels=(32, 64, 128, 256, 512),
    attention_levels=(False, False, False, True, True),
    num_head_channels=32,
    norm_num_groups=32,
    with_conditioning=True,
    cross_attention_dim=32,
).cuda()

if __name__ == "__main__":
    x = torch.randn(1, 8, 32, 32).cuda()
    t = torch.randint(0, 1000, (1,)).cuda()
    noise = torch.randn(1, 8, 32, 32).cuda()
    condition = torch.randn(1, 32, 32, 32).cuda()
    context = rearrange(condition, 'b c h w -> b (h w) c')
    scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta")
    inferer = DiffusionInferer(scheduler)
    noise_pred = inferer(
        inputs=x,
        diffusion_model=model,
        noise=noise,
        timesteps=t,
        condition=context,
        mode="crossattn",
    )
    print(noise_pred.shape)
    image = inferer.sample(input_noise=noise, diffusion_model=model, scheduler=scheduler,
                                conditioning=condition, mode="crossattn")
    print(image.shape)