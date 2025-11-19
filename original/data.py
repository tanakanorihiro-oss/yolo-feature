# data.py
# データセットとデータローダーの定義

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# config から IMG_SIZE をインポート
from config import IMG_SIZE 

# -------------------------
# Dataset
# -------------------------
class WeatherDataset(Dataset):
    def __init__(self, noise_dir, gt_dir, img_size=IMG_SIZE):
        self.noise_dir = noise_dir
        self.gt_dir = gt_dir
        self.img_names = sorted([f for f in os.listdir(noise_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))])
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.img_names)
    def __getitem__(self, idx):
        name = self.img_names[idx]
        noise_path = os.path.join(self.noise_dir, name)
        gt_path = os.path.join(self.gt_dir, name)
        noise = cv2.imread(noise_path)
        gt = cv2.imread(gt_path)
        if noise is None or gt is None:
            raise FileNotFoundError(f"Missing image for {name}: {noise_path} or {gt_path}")
        noise = cv2.cvtColor(noise, cv2.COLOR_BGR2RGB)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        noise_t = self.transform(noise)
        gt_t = self.transform(gt)
        return noise_t, gt_t, name

# -------------------------
# Custom Collate Function 
# -------------------------
def dynamic_collate_fn(batch):
    # バッチ内の最大サイズを決定 (U-Netのため32の倍数に調整)
    max_h = max([item[0].shape[1] for item in batch])
    max_w = max([item[0].shape[2] for item in batch])
    
    target_h = ((max_h + 31) // 32) * 32
    target_w = ((max_w + 31) // 32) * 32

    noise_imgs = []
    gt_imgs = []
    names = []
    
    for noise_t, gt_t, name in batch:
        h, w = noise_t.shape[1], noise_t.shape[2]
        pad_h = target_h - h
        pad_w = target_w - w
        
        # 反射パディングを適用 (左, 右, 上, 下の順)
        noise_padded = F.pad(noise_t, (0, pad_w, 0, pad_h), mode='reflect')
        gt_padded = F.pad(gt_t, (0, pad_w, 0, pad_h), mode='reflect')
        
        noise_imgs.append(noise_padded)
        gt_imgs.append(gt_padded)
        names.append(name)

    return torch.stack(noise_imgs), torch.stack(gt_imgs), names