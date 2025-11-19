# 作った復元モデルで推論
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inference script for Image Restoration using the trained UNetRestorer.
Loads the best checkpoint, processes noisy images, and saves the restored images.
"""

import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm
import numpy as np

# ==========================================================
# === CONFIGURATION (設定) =================================
# ==========================================================
# 訓練で定義したモデルパスとデータパス
CHECKPOINT_PATH = Path("/home/usrs/tananori/Study/yolov8_end/original/113/output_unet_restorer_yolo/checkpoints/restorer_BEST.pth")
INPUT_DIR = Path("/home/usrs/tananori/Study/yolov8_end/dataset/DAWN/Snow")
OUTPUT_DIR = Path("/home/usrs/tananori/Study/yolov8_end/original/113/DAWN/Snow")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1 # 推論用のバッチサイズ (画像サイズが異なるため1を推奨)
NUM_WORKERS = 4 # 推論時もワーカー数は確保する

# ==========================================================
# === MODEL DEFINITIONS (モデル定義) ========================
# ==========================================================
# 訓練スクリプトで使用したクラス定義を再利用
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

# ==========================================================
# === INFERENCE DATASET (推論用データセット) ==================
# ==========================================================
class InferenceDataset(Dataset):
    def __init__(self, noise_dir):
        self.noise_dir = Path(noise_dir)
        self.img_paths = sorted([p for p in self.noise_dir.glob('*.jpg')] + 
                                [p for p in self.noise_dir.glob('*.jpeg')] + 
                                [p for p in self.noise_dir.glob('*.png')])
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        noise = cv2.imread(str(path))
        if noise is None:
            raise FileNotFoundError(f"Missing image: {path}")
        noise = cv2.cvtColor(noise, cv2.COLOR_BGR2RGB)
        h, w, _ = noise.shape
        noise_t = self.transform(noise)
        return noise_t, path.name, (h, w)

# ==========================================================
# === MAIN INFERENCE LOGIC (メイン推論処理) ==================
# ==========================================================
def main(): # main関数の定義を追加

    # 1. 出力ディレクトリの作成
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"出力ディレクトリを作成しました: {OUTPUT_DIR}")

    # 2. モデルの構築とチェックポイントのロード
    restorer = UNetRestorer(base_ch=32).to(DEVICE)
    if not CHECKPOINT_PATH.exists():
        print(f"エラー: チェックポイントファイルが見つかりません: {CHECKPOINT_PATH}")
        return

    # チェックポイントのロード
    print(f"チェックポイントをロード中: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    restorer.load_state_dict(checkpoint['restorer_state'])
    restorer.eval() # 推論モードに設定
    print("モデルのロードが完了しました。")

    # 3. データローダーの準備
    inference_dataset = InferenceDataset(INPUT_DIR)
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=BATCH_SIZE, # BATCH_SIZE=1 を使用
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    print(f"推論対象の画像数: {len(inference_dataset)}")

    # 4. 推論ループ
    print("--- 推論を開始します ---")
    with torch.no_grad():
        for batch in tqdm(inference_loader, desc="Restoring Images"):
            noise_imgs, names, sizes = batch # noise_imgs: (B, C, H, W), names: list[str], sizes: tuple of tensors (H, W)

            # バッチ内の各画像に対して処理を行う (BATCH_SIZE=1 なので実質1つずつ)
            for i in range(noise_imgs.size(0)):
                img = noise_imgs[i].unsqueeze(0).to(DEVICE) # (1, C, H, W)
                name = names[i]
                
                # サイズ情報を取り出す (sizes[0]がH、sizes[1]がW)
                h = sizes[0][i].item() # 元画像の高さ
                w = sizes[1][i].item() # 元画像の幅

                # --- サイズを32の倍数にパディング（U-Netのダウンサンプル階層に合わせる） ---
                divisor = 32 
                pad_h = (divisor - h % divisor) % divisor
                pad_w = (divisor - w % divisor) % divisor
                
                # パディングを適用 (左, 右, 上, 下)
                img_padded = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')

                # --- 推論 ---
                restored_padded = restorer(img_padded)
                
                # パディングを除去して元のサイズに戻す
                # restored_padded.squeeze(0) は (C, H_padded, W_padded)
                # [..., :h, :w] で元の高さ h と幅 w の部分のみを取り出す
                restored = restored_padded.squeeze(0)[..., :h, :w] # (C, H_original, W_original)

                # --- 保存 ---
                img_np = restored.permute(1, 2, 0).cpu().numpy() # (H, W, C)
                img_np = (img_np.clip(0, 1) * 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                save_path = OUTPUT_DIR / name
                cv2.imwrite(str(save_path), img_bgr)
                
if __name__ == '__main__':
    main()