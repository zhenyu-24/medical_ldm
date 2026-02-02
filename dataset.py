import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torchio as tio
import pandas as pd
from multiprocessing.dummy import Pool

#############################torchio dataset
def read_csv(fold_path, is_train):
    if is_train:
        csv_path = os.path.join(fold_path, 'train.csv')
    else:
        csv_path = os.path.join(fold_path, 'my_model.csv')
    df = pd.read_csv(csv_path)
    subjects = df['Subject'].tolist()
    return subjects

def create_Multi_subjects(fold_path, is_train):
    new_subjects = []
    subjects = read_csv(fold_path, is_train)
    for idx, subject in enumerate(subjects):
        # 读取图像和标签
        image_Tlow_T1 = tio.ScalarImage(os.path.join(fold_path, str(subject), '64mT', f'{subject}_T1.nii.gz'))
        image_Tlow_T2 = tio.ScalarImage(os.path.join(fold_path, str(subject), '64mT', f'{subject}_T2.nii.gz'))
        image_Tlow_FLAIR = tio.ScalarImage(os.path.join(fold_path, str(subject), '64mT', f'{subject}_FLAIR.nii.gz'))
        image_Tup_T1 = tio.ScalarImage(os.path.join(fold_path, str(subject), '3T', f'{subject}_T1.nii.gz'))
        image_Tup_T2 = tio.ScalarImage(os.path.join(fold_path, str(subject), '3T', f'{subject}_T2.nii.gz'))
        image_Tup_FLAIR = tio.ScalarImage(os.path.join(fold_path, str(subject), '3T', f'{subject}_FLAIR.nii.gz'))
        # image_seg = tio.LabelMap(os.path.join(fold_path, str(subject), '3T', f'{subject}_T1_seg.nii.gz'))
        mask = tio.LabelMap(os.path.join(fold_path, 'mask', str(subject), '3T', f'{subject}_skull.nii.gz'))
        # image_seg2 = tio.LabelMap(os.path.join(fold_path, str(subject), '3T', f'{subject}_T2_seg.nii.gz'))
        # image_seg3 = tio.LabelMap(os.path.join(fold_path, str(subject), '3T', f'{subject}_FLAIR_seg.nii.gz'))
        # 创建 Subject 对象
        combined_subject = tio.Subject(
            name=str(subject),
            Tlow_T1 = image_Tlow_T1,
            Tlow_T2 = image_Tlow_T2,
            Tlow_FLAIR = image_Tlow_FLAIR,
            Tup_T1 = image_Tup_T1,
            Tup_T2 = image_Tup_T2,
            Tup_FLAIR = image_Tup_FLAIR,
            # seg = image_seg,
            # seg2 = image_seg2,
            # seg3 = image_seg3,
            mask = mask,
        )
        new_subjects.append(combined_subject)
    return new_subjects

def create_Multi_transform(out_min_max=(0, 1), is_train=True):
    # 数据处理和增强
    if is_train:
        print('train augument')
        transform = tio.Compose([
            tio.RandomElasticDeformation(
                num_control_points=5,
                locked_borders=2,
                p=0.05,
                include=['Tlow_T1', 'Tlow_T2', 'Tlow_FLAIR', 'Tup_T1', 'Tup_T2', 'Tup_FLAIR', 'seg', 'seg2', 'seg3', 'mask']
            ),
            tio.RandomFlip(
                axes=(0, 1, 2),
                p=0.2,
                include=['Tlow_T1', 'Tlow_T2', 'Tlow_FLAIR', 'Tup_T1', 'Tup_T2', 'Tup_FLAIR', 'seg', 'seg2', 'seg3', 'mask']
            ),
            tio.RandomAffine(
                scales=0.02,
                degrees=(2, 2, 2),
                translation=(2, 2, 2),
                isotropic=True,
                p=0.8,
                include=['Tlow_T1', 'Tlow_T2', 'Tlow_FLAIR', 'Tup_T1', 'Tup_T2', 'Tup_FLAIR', 'seg', 'seg2', 'seg3', 'mask']
            ),
            tio.RescaleIntensity(out_min_max=out_min_max),
        ])
    else:
        transform = tio.Compose([
            tio.RescaleIntensity(out_min_max=out_min_max),
        ])
    return transform


def load_Multi_modal_dataset(fold_path, is_train=True, out_min_max=(0, 1), load_getitem=False):
    # 加载所有的数据
    subjects = create_Multi_subjects(fold_path, is_train)
    if is_train:
        dataset = tio.SubjectsDataset(subjects, transform=create_Multi_transform(out_min_max, is_train), load_getitem=load_getitem)
    else:
        dataset = tio.SubjectsDataset(subjects, load_getitem=load_getitem)
    return dataset

def patch_train_dataloader(dataset, patch_size=(224, 224, 1), batch_size=8, samples_per_volume=150):
    """
    PatchSampler:基类
    UniformSampler:以均匀的概率从体积中随机提取补丁
    WeightedSample:根据概率图从体积中随机提取补丁
    LabelSampler:提取中心带有标记体素的随机斑块
    GridSampler：网格采样器可用于使用体积中的所有块进行推理。它通常与GridAggregator一起使用
    Queue：也继承自 PyTorch Dataset。在这个排队系统中，采样器充当生成器
    """
    sampler = tio.UniformSampler(patch_size=patch_size)
    patches = tio.Queue(
        subjects_dataset=dataset,
        max_length=samples_per_volume,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=4,
        shuffle_subjects=True,
        shuffle_patches=True,
    )
    # 使用 tio.SubjectsLoader 来加载数据
    patch_loader = tio.SubjectsLoader(patches, batch_size=batch_size, num_workers=0)
    return patch_loader

def patch_train_maskdataloader(dataset, patch_size=(224, 224, 1), batch_size=8, samples_per_volume=128):
    """
    PatchSampler:基类
    UniformSampler:以均匀的概率从体积中随机提取补丁
    WeightedSample:根据概率图从体积中随机提取补丁
    LabelSampler:提取中心带有标记体素的随机斑块
    GridSampler：网格采样器可用于使用体积中的所有块进行推理。它通常与GridAggregator一起使用
    Queue：也继承自 PyTorch Dataset。在这个排队系统中，采样器充当生成器
    """
    sampler = tio.LabelSampler(patch_size=patch_size, label_name='mask')
    patches = tio.Queue(
        subjects_dataset=dataset,
        max_length=256,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=4,
        shuffle_subjects=False,
        shuffle_patches=False,
    )
    # 使用 tio.SubjectsLoader 来加载数据
    patch_loader = tio.SubjectsLoader(patches, batch_size=batch_size, num_workers=0, pin_memory=True)
    return patch_loader