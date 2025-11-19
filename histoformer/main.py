#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a Histoformer restorer with YOLO-backbone feature alignment loss.
- Loads Histoformer definition from your provided histoformer_arch.py
- Loads pretrained weights from PRETRAIN_RESTORER (net_g_best.pth)
- Uses ultralytics YOLOv8 for backbone feature extraction (no detection loss API)
- Loss: L_total = L1 + lambda_ssim*(1-SSIM) + lambda_det * L_feat
- lambda_det is linearly warmed up for WARMUP_EPOCHS
- TensorBoard logging + checkpoint saving
"""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# ultralytics YOLOv8
from ultralytics import YOLO

# -------------------------
# === USER CONFIG ========
# -------------------------
# Paths - edit if different on your machine
HISTOFORMER_ARCH_DIR = "/home/usrs/tananori/Study/yolov8_end"   # directory containing histoformer_arch.py
PRETRAIN_RESTORER = "/home/usrs/tananori/Study/yolov8_end/net_g_best.pth"

TRAIN_NOISE_DIR = "/home/usrs/tananori/Study/yolov8_end/dataset/bdd100k_clear/noise/train"
TRAIN_GT_DIR    = "/home/usrs/tananori/Study/yolov8_end/dataset/bdd100k_clear/images/train"
VAL_NOISE_DIR   = "/home/usrs/tananori/Study/yolov8_end/dataset/bdd100k_clear/noise/val"
VAL_GT_DIR      = "/home/usrs/tananori/Study/yolov8_end/dataset/bdd100k_clear/images/val"

YOLO_WEIGHTS = "yolov8n.pt"   # or yolov8x.pt etc; keep consistent with your setup

OUT_DIR = Path("./output_histoformer_yolo")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = OUT_DIR / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)
LOG_DIR = OUT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2           # Histoformer is memory-heavy; adjust to your GPU
VAL_BATCH_SIZE = 2
IMG_SIZE = 320
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4

# Loss weights & schedule
LAMBDA_SSIM = 1.0
LAMBDA_DET_MAX = 2.0       # final target for lambda_det (tune if needed)
WARMUP_EPOCHS = 10         # linear warmup epochs for lambda_det

SAVE_EVERY = 1
PRINT_FREQ = 100
NUM_WORKERS = 4

# -------------------------
# Add histoformer_arch to path, import Histoformer
# -------------------------
sys.path.append(HISTOFORMER_ARCH_DIR)
try:
    from histoformer_arch import Histoformer
except Exception as e:
    raise ImportError(f"Could not import Histoformer from {HISTOFORMER_ARCH_DIR}/histoformer_arch.py: {e}")

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
            transforms.Resize((img_size, img_size)),
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
# SSIM module (returns SSIM in [0,1])
# -------------------------
class SSIMModule(nn.Module):
    def __init__(self, window_size=3):
        super().__init__()
        self.window_size = window_size
    def forward(self, x, y):
        # expects x,y in [0,1], shape (B,C,H,W)
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        mu_x = F.avg_pool2d(x, self.window_size, 1, self.window_size//2)
        mu_y = F.avg_pool2d(y, self.window_size, 1, self.window_size//2)
        sigma_x = F.avg_pool2d(x * x, self.window_size, 1, self.window_size//2) - mu_x ** 2
        sigma_y = F.avg_pool2d(y * y, self.window_size, 1, self.window_size//2) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y, self.window_size, 1, self.window_size//2) - mu_x * mu_y
        ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        ssim_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
        ssim_map = ssim_n / (ssim_d + 1e-12)
        return torch.clamp(ssim_map.mean(), 0.0, 1.0)

# -------------------------
# YOLO backbone extraction helper and feature loss util
# -------------------------
class YOLOv8FeatureExtractor(nn.Module):
    def __init__(self, yolo_model, layers=[4, 6, 9]):
        super().__init__()
        max_layer = max(layers)

        try:
            blocks = yolo_model.model.model[:max_layer + 1]
            self.backbone_blocks = nn.Sequential(*blocks)
        except Exception as e:
            raise RuntimeError(f"YOLO model structure changed: {e}")

        self.layers = set(layers)

        for p in yolo_model.model.parameters():
            p.requires_grad = False

        self.eval()

    def forward(self, x):
        features = []
        for i, module in enumerate(self.backbone_blocks):
            x = module(x)
            if i in self.layers:
                features.append(x)
        return features

def feature_mse_loss(fr, fg):
    # support tuple/list of features or single tensor
    if isinstance(fr, (tuple, list)) and isinstance(fg, (tuple, list)):
        loss = 0.0
        count = 0
        for a, b in zip(fr, fg):
            if a.shape[2:] != b.shape[2:]:
                b = F.interpolate(b, size=a.shape[2:], mode='bilinear', align_corners=False)
            loss = loss + F.mse_loss(a, b)
            count += 1
        return loss / max(1, count)
    else:
        if fr.shape[2:] != fg.shape[2:]:
            fg = F.interpolate(fg, size=fr.shape[2:], mode='bilinear', align_corners=False)
        return F.mse_loss(fr, fg)

# -------------------------
# Prepare datasets / loaders
# -------------------------
train_dataset = WeatherDataset(TRAIN_NOISE_DIR, TRAIN_GT_DIR, IMG_SIZE)
val_dataset = WeatherDataset(VAL_NOISE_DIR, VAL_GT_DIR, IMG_SIZE)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# -------------------------
# Build models, load pretrain weights
# -------------------------
print("Building Histoformer...")
restorer = Histoformer().to(DEVICE)

if PRETRAIN_RESTORER and os.path.exists(PRETRAIN_RESTORER):
    ck = torch.load(PRETRAIN_RESTORER, map_location=DEVICE)
    loaded = False
    # try common key patterns
    if isinstance(ck, dict):
        # try common keys
        for key in ['params_ema', 'state_dict', 'model', 'restorer_state', 'net_g', 'params']:
            if key in ck:
                try:
                    restorer.load_state_dict(ck[key], strict=False) 
                    print(f"Loaded pretrain weights from {PRETRAIN_RESTORER} using key '{key}'")
                    loaded = True
                    break
                except Exception:
                    pass
        if not loaded:
            # try ck itself as state_dict (sometimes ck is the state_dict)
            try:
                restorer.load_state_dict(ck, strict=False)
                print(f"Loaded pretrain weights from {PRETRAIN_RESTORER} directly as state_dict (strict=False)")
                loaded = True
            except Exception as e:
                print(f"Warning: Checkpoint auto-load failed. Error: {e}")
    else:
        # ck is not dict: try direct load
        try:
            restorer.load_state_dict(ck, strict=False)
            loaded = True
            print(f"Loaded pretrain weights from {PRETRAIN_RESTORER} (strict=False)")
        except Exception as e:
            print(f"Warning: failed to load checkpoint (strict=False): {e}")

    if not loaded:
        print("Proceeding with randomly initialized restorer (please check checkpoint compatibility).")
else:
    print("No pretrained restorer found - proceeding from scratch.")

# -------------------------
# YOLO backbone
# -------------------------
print("Loading YOLO and extracting multi-scale backbone...")
yolo_model = YOLO(YOLO_WEIGHTS)
yolo_model.model.to(DEVICE)
backbone = YOLOv8FeatureExtractor(yolo_model, layers=[4, 6, 9]).to(DEVICE)

# -------------------------
# Losses, optimizer, misc
# -------------------------
l1_loss = nn.L1Loss()
ssim_module = SSIMModule()

optimizer = torch.optim.AdamW(restorer.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
writer = SummaryWriter(LOG_DIR)

use_amp = torch.cuda.is_available()
scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

global_step = 0
best_val = float('inf')

# -------------------------
# Training loop
# -------------------------
for epoch in range(1, EPOCHS+1):
    # linear warmup for lambda_det
    if epoch <= WARMUP_EPOCHS:
        lambda_det = float(LAMBDA_DET_MAX) * (epoch-1) / max(1, (WARMUP_EPOCHS-1))
    else:
        lambda_det = float(LAMBDA_DET_MAX)
    print(f"Epoch {epoch}/{EPOCHS} | lambda_det={lambda_det:.6f}")

    restorer.train()
    epoch_l1 = epoch_ssim = epoch_feat = epoch_total = 0.0
    t0 = time.time()

    for i, (noise_img, gt_img, name) in enumerate(train_loader):
        noise_img = noise_img.to(DEVICE, non_blocking=True)
        gt_img = gt_img.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=use_amp):
            restored = restorer(noise_img)  # [B,3,H,W] in [0,1] expected

            # pixel losses
            l1 = l1_loss(restored, gt_img)
            ssim_val = ssim_module(restored, gt_img)  # SSIM in [0,1]
            ssim_loss_val = 1.0 - ssim_val

            # feature loss via YOLO backbone (gt features computed without grad)
            with torch.no_grad():
                feat_gt = backbone(gt_img)
            feat_rest = backbone(restored)
            feat_loss = feature_mse_loss(feat_rest, feat_gt)

            total_loss = l1 + LAMBDA_SSIM * ssim_loss_val + lambda_det * feat_loss

        # backward + step
        if use_amp:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        epoch_l1 += l1.item()
        epoch_ssim += ssim_loss_val.item()
        epoch_feat += feat_loss.item()
        epoch_total += total_loss.item()
        global_step += 1

        if (i+1) % PRINT_FREQ == 0 or (i+1) == len(train_loader):
            print(f"  [{i+1}/{len(train_loader)}] total {total_loss.item():.4f} l1 {l1.item():.4f} ssim {ssim_loss_val.item():.4f} feat {feat_loss.item():.4f}")

    # epoch logging
    n_batch = len(train_loader)
    writer.add_scalar("train/total_loss", epoch_total / n_batch, epoch)
    writer.add_scalar("train/l1", epoch_l1 / n_batch, epoch)
    writer.add_scalar("train/ssim", epoch_ssim / n_batch, epoch)
    writer.add_scalar("train/feat", epoch_feat / n_batch, epoch)
    writer.add_scalar("train/lambda_det", lambda_det, epoch)

    # validation (L1 + SSIM only)
    restorer.eval()
    val_l1 = val_ssim = val_feat = val_count = 0.0
    with torch.no_grad():
        for noise_img, gt_img, _ in val_loader:
            noise_img = noise_img.to(DEVICE)
            gt_img = gt_img.to(DEVICE)
            restored = restorer(noise_img)

            feat_gt = backbone(gt_img)
            feat_rest = backbone(restored)
            feat_loss = feature_mse_loss(feat_rest, feat_gt) 

            val_l1 += l1_loss(restored, gt_img).item() * restored.shape[0]
            ssim_v = ssim_module(restored, gt_img)
            val_ssim += (1.0 - ssim_v).item() * restored.shape[0]
            val_feat += feat_loss.item() * restored.shape[0]
            val_count += restored.shape[0]
            
    val_l1 /= max(1, val_count)
    val_ssim /= max(1, val_count)
    val_feat /= max(1, val_count)

    if epoch <= WARMUP_EPOCHS:
        current_lambda_det = float(LAMBDA_DET_MAX) * (epoch-1) / max(1, (WARMUP_EPOCHS-1))
    else:
        current_lambda_det = float(LAMBDA_DET_MAX)
    
    val_total = val_l1 + LAMBDA_SSIM * val_ssim + current_lambda_det * val_feat
    
    writer.add_scalar("val/total_loss_weighted", val_total, epoch)
    writer.add_scalar("val/feat", val_feat, epoch)
    
    if val_total < best_val:
        if epoch >= WARMUP_EPOCHS:
            best_val = val_total
            print(f"--- 💾 NEW BEST (Epoch {epoch}): Val Total Loss improved to {best_val:.4f}. Saving best model. ---")
            torch.save({
                "epoch": epoch,
                "restorer_state": restorer.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val": best_val
            }, CKPT_DIR / "restorer_BEST.pth")

    writer.add_scalar("val/l1", val_l1, epoch)
    writer.add_scalar("val/ssim", val_ssim, epoch)

    epoch_time = time.time() - t0
    print(f"Epoch {epoch} done in {epoch_time:.1f}s | train_total {epoch_total/n_batch:.4f} | val_l1 {val_l1:.4f} val_ssim {val_ssim:.4f} val_feat {val_feat:.4f} val_total {val_total:.4f}")

    # save checkpoint
    if epoch % SAVE_EVERY == 0:
        torch.save({
            "epoch": epoch,
            "restorer_state": restorer.state_dict(),
            "optimizer": optimizer.state_dict()
        }, CKPT_DIR / f"restorer_epoch{epoch:03d}.pth")

writer.close()
print("Training finished. Checkpoints and logs saved to", OUT_DIR)