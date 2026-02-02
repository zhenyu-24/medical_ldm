import os
import sys
import torch
import torchio as tio
from omegaconf import OmegaConf
from vqgan.vqgan import *  # vq_gan/vqgan_2d.py
from utils.trainer import create_trainer
from dataset import *
import warnings

# 忽略所有用户警告
warnings.filterwarnings("ignore", category=UserWarning)


def load_config(config_path="config/vqgan_3d.yaml"):
    cfg = OmegaConf.load(config_path)
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_cfg)  # 合并命令行参数
    return cfg


def run(cfg):
    print(torch.cuda.is_available())
    print(cfg)

    path = r'/media/xxxx/Data/ULF_ENC'
    dataset = load_Multi_modal_dataset(path, is_train=True, out_min_max=(0, 1))
    dataloader = patch_train_maskdataloader(dataset, patch_size=(128, 128, 128), batch_size=cfg.batch_size, samples_per_volume=8)

    # 加载模型
    # model = VQGAN(cfg)
    model = VQGAN2(cfg)
    # model = FSQGAN(cfg)
    # model = ResidualVQGAN(cfg)

    trainer = create_trainer(name=cfg.name, save_dir=cfg.save_dir, checkpoint_dir=cfg.checkpoint_dir,
                             precision=cfg.precision, max_epoch=cfg.max_epochs, monitor='train_recon_loss',
                             strategy=cfg.strategy, )
    # 训练
    trainer.fit(model, dataloader, ckpt_path='last')

if __name__ == '__main__':
    # export CUDA_VISIBLE_DEVICES=1 python train_2d_vqgan.py
    # tensorboard --logdir /xxxx
    cfg = load_config()
    run(cfg=cfg)
