# models.py
# モデル定義ファイル(U-Net Restorer, YOLO Backbone抽出, Losses)

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

# ----------------------------------------
# --- U-Net カスタムモデルの定義 (UNetRestorer) ---
# ----------------------------------------
class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, padding=1, bias=True):
        super(UNetConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size, padding=padding, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_size, out_size, kernel_size, padding=padding, bias=bias),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class UNetRestorer(nn.Module):
    """
    A simplified U-Net architecture for image restoration.
    """
    def __init__(self, in_ch=3, out_ch=3, base_ch=32):
        super(UNetRestorer, self).__init__()
        
        self.enc1 = UNetConvBlock(in_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = UNetConvBlock(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = UNetConvBlock(base_ch * 2, base_ch * 4)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.bottleneck = UNetConvBlock(base_ch * 4, base_ch * 8)
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = UNetConvBlock(base_ch * 4 * 2, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = UNetConvBlock(base_ch * 2 * 2, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = UNetConvBlock(base_ch * 2, base_ch)
        self.out = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, x):
        identity = x 
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        restored_delta = self.out(d1)
        return restored_delta + identity 

# -------------------------
# YOLO backbone extraction helper
# -------------------------
class YOLOv8FeatureExtractor(nn.Module):
    def __init__(self, yolo_model, layers=[4, 6, 9]):
        super().__init__()
        max_layer = max(layers)
        try:
            # YOLOv8のモデル構造からバックボーンブロックを抽出
            blocks = yolo_model.model.model[:max_layer + 1]
            self.backbone_blocks = nn.Sequential(*blocks)
        except Exception as e:
            raise RuntimeError(f"YOLO model structure changed: {e}")
        self.layers = set(layers)
        # 勾配計算を停止
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

# -------------------------
# Losses
# -------------------------
class SSIMModule(nn.Module):
    def __init__(self, window_size=3):
        super().__init__()
        self.window_size = window_size
    def forward(self, x, y):
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

def feature_mse_loss(fr, fg):
    if isinstance(fr, (tuple, list)) and isinstance(fg, (tuple, list)):
        loss = 0.0
        count = 0
        for a, b in zip(fr, fg):
            # 特徴マップのサイズが異なる場合、補間して合わせる (U-Netのskip接続のサイズ不一致対策)
            if a.shape[2:] != b.shape[2:]:
                b = F.interpolate(b, size=a.shape[2:], mode='bilinear', align_corners=False)
            loss = loss + F.mse_loss(a, b)
            count += 1
        return loss / max(1, count)
    else:
        if fr.shape[2:] != fg.shape[2:]:
            fg = F.interpolate(fg, size=fr.shape[2:], mode='bilinear', align_corners=False)
        return F.mse_loss(fr, fg)