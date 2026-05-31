"""
NYUv2 数据集加载器

复用 mtan-reference 的数据加载逻辑，适配 UAV-MTM-Inference 项目。
"""

import os
from pathlib import Path
import fnmatch
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class RandomScaleCrop(object):
    """
    随机缩放裁剪数据增强
    
    Credit to Jialong Wu from https://github.com/lorenmt/mtan/issues/34.
    """
    def __init__(self, scale=[1.0, 1.2, 1.5]):
        self.scale = scale

    def __call__(self, img, label, depth, normal):
        height, width = img.shape[-2:]
        sc = self.scale[random.randint(0, len(self.scale) - 1)]
        h, w = int(height / sc), int(width / sc)
        i = random.randint(0, height - h)
        j = random.randint(0, width - w)
        img_ = F.interpolate(img[None, :, i:i + h, j:j + w], size=(height, width), mode='bilinear', align_corners=True).squeeze(0)
        label_ = F.interpolate(label[None, None, i:i + h, j:j + w], size=(height, width), mode='nearest').squeeze(0).squeeze(0)
        depth_ = F.interpolate(depth[None, :, i:i + h, j:j + w], size=(height, width), mode='nearest').squeeze(0)
        normal_ = F.interpolate(normal[None, :, i:i + h, j:j + w], size=(height, width), mode='bilinear', align_corners=True).squeeze(0)
        return img_, label_, depth_ / sc, normal_


class NYUv2Dataset(Dataset):
    """
    NYUv2 多任务学习数据集
    
    数据格式：.npy 文件（由 mtan-reference 预处理生成）
    
    任务：
    - Semantic Segmentation (13 类)
    - Depth Estimation
    - Surface Normal Estimation
    
    Args:
        root: 数据集根目录（包含 train/ 和 val/ 子目录）
        split: 'train' 或 'val'
        augmentation: 是否应用数据增强（随机缩放、水平翻转）
    """
    def __init__(self, root: str, split: str = 'train', augmentation: bool = False):
        self.root = Path(root).expanduser()
        self.split = split
        self.augmentation = augmentation
        
        # 数据路径
        if split == 'train':
            self.data_path = self.root / 'train'
        elif split == 'val':
            self.data_path = self.root / 'val'
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'.")
        
        # 检查数据是否存在
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data path {self.data_path} not found. "
                f"Please ensure NYUv2 dataset is placed at {self.root}."
            )
        
        # 计算数据数量
        image_dir = self.data_path / 'image'
        self.data_len = len(fnmatch.filter(os.listdir(image_dir), '*.npy'))
        
        if self.data_len == 0:
            raise ValueError(f"No .npy files found in {image_dir}.")
        
        print(f"[INFO] Loaded NYUv2 {split} split: {self.data_len} samples")
    
    def __getitem__(self, index):
        """
        返回：
            image: [3, H, W] RGB 图像
            semantic: [H, W] 语义分割标签（0-12）
            depth: [1, H, W] 深度图
            normal: [3, H, W] 表面法向量
        """
        # 加载预处理的 .npy 文件
        image = torch.from_numpy(np.moveaxis(
            np.load(self.data_path / f'image/{index}.npy'), -1, 0
        ))
        semantic = torch.from_numpy(
            np.load(self.data_path / f'label/{index}.npy')
        )
        depth = torch.from_numpy(np.moveaxis(
            np.load(self.data_path / f'depth/{index}.npy'), -1, 0
        ))
        normal = torch.from_numpy(np.moveaxis(
            np.load(self.data_path / f'normal/{index}.npy'), -1, 0
        ))
        
        # 应用数据增强（训练时可选）
        if self.augmentation:
            image, semantic, depth, normal = RandomScaleCrop()(image, semantic, depth, normal)
            if torch.rand(1) < 0.5:
                image = torch.flip(image, dims=[2])
                semantic = torch.flip(semantic, dims=[1])
                depth = torch.flip(depth, dims=[2])
                normal = torch.flip(normal, dims=[2])
                normal[0, :, :] = -normal[0, :, :]  # 法向量 x 分量翻转
        
        return image.float(), semantic.float(), depth.float(), normal.float()
    
    def __len__(self):
        return self.data_len


def nyuv2_collate_fn(batch):
    """
    NYUv2 专用 collate_fn（必须是模块顶层函数，才能在 Windows 的多进程 DataLoader 中被 pickle）。

    Args:
        batch: List[Tuple[image, semantic, depth, normal]]

    Returns:
        dict batch:
            - image:   [B, 3, H, W]
            - semantic:[B, H, W] (Long)
            - depth:   [B, 1, H, W]
            - normal:  [B, 3, H, W]
    """
    images, semantics, depths, normals = zip(*batch)
    return {
        'image': torch.stack(images, dim=0),
        'semantic': torch.stack(semantics, dim=0).long(),  # Long 用于分类
        'depth': torch.stack(depths, dim=0),
        'normal': torch.stack(normals, dim=0),
    }


def create_nyuv2_dataloader(
    root: str,
    split: str = 'val',
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 4,
    augmentation: bool = False,
    pin_memory: bool = True
) -> DataLoader:
    """
    创建 NYUv2 DataLoader（包装成字典格式）
    
    Args:
        root: 数据集根目录
        split: 'train' 或 'val'
        batch_size: Batch size
        shuffle: 是否打乱
        num_workers: 数据加载线程数
        augmentation: 是否数据增强
        pin_memory: 是否使用 pinned memory（CUDA 下通常更快；若系统不稳定可设为 False）
    
    Returns:
        dataloader: 返回字典格式的 batch：
            {
                'image': [B, 3, H, W],
                'semantic': [B, H, W],
                'depth': [B, 1, H, W],
                'normal': [B, 3, H, W]
            }
    """
    dataset = NYUv2Dataset(root, split=split, augmentation=augmentation)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=nyuv2_collate_fn,
        pin_memory=pin_memory
    )
    
    return dataloader
