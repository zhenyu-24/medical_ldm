import os
import sys
import torch
import torchio as tio
from ddpm.diffusers_2d_ddpm import diffusion_model
from ddpm.monai_2d_ddpm import DiffusionModel
from utils.ddpm_trainer import *
from monai.transforms import *
import warnings
from dataset import *
import warnings
from dataclasses import dataclass

# 忽略所有用户警告
warnings.filterwarnings("ignore", category=UserWarning)


def run():
    print(torch.cuda.is_available())
    path = r'/media/xxxx/Data/ULF_ENC'

    dataset = load_Multi_modal_dataset(path, is_train=True, load_getitem=True, out_min_max=(-1, 1))
    dataloader = patch_train_maskdataloader(dataset, patch_size=(224, 224, 1), batch_size=16, samples_per_volume=128)

    num_training_steps = 1000000
    # model = diffusion_model(num_training_steps=num_training_steps)
    model = DiffusionModel(num_training_steps=num_training_steps)
    print('加载成功')

    trainer = create_ddpmtrainer(name='2d_ddpm_monai', save_dir="./logs", checkpoint_dir="./checkpoints/2d_ddpm_monai",
                                 precision='16', max_epoch=10000, monitor='train_loss', strategy="auto", )
    trainer.fit(model, dataloader, ckpt_path='last')


if __name__ == '__main__':
    run()
