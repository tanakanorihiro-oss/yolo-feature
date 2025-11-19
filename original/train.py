# train.py
# 訓練を実行するスクリプト

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main Training script for Image Restoration (U-Net + YOLO Feature Loss).
Loads modules from separate files for clean structure.
"""

import time
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 外部ファイルから必要なものをインポート
# 注: このスクリプトを実行するには、他のファイル (models.py, data.py) が同じディレクトリに存在する必要があります。
from config import *
from models import UNetRestorer, YOLOv8FeatureExtractor, SSIMModule, feature_mse_loss
from data import WeatherDataset, dynamic_collate_fn


def main():
    # 1. ディレクトリ準備
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    
    # 2. データローダー準備
    train_dataset = WeatherDataset(TRAIN_NOISE_DIR, TRAIN_GT_DIR, IMG_SIZE)
    val_dataset = WeatherDataset(VAL_NOISE_DIR, VAL_GT_DIR, IMG_SIZE)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, # config.PHYSICAL_BATCH_SIZE を使用
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=True,
        collate_fn=dynamic_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=VAL_BATCH_SIZE,
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=True,
        collate_fn=dynamic_collate_fn 
    )

    # 3. モデル構築
    print("Building UNet Restorer...")
    restorer = UNetRestorer(base_ch=32).to(DEVICE)
    print("Proceeding with randomly initialized restorer (from scratch).")

    # 4. YOLO backbone (Feature Loss用)
    print("Loading YOLO and extracting multi-scale backbone...")
    from ultralytics import YOLO 
    yolo_model = YOLO(YOLO_WEIGHTS)
    yolo_model.model.to(DEVICE)
    backbone = YOLOv8FeatureExtractor(yolo_model, layers=[4, 6, 9]).to(DEVICE)

    # 5. 損失、最適化、その他
    l1_loss = nn.L1Loss()
    ssim_module = SSIMModule()

    optimizer = torch.optim.AdamW(restorer.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    writer = SummaryWriter(LOG_DIR)

    use_amp = DEVICE == "cuda" and torch.cuda.is_available() # amp利用可能性チェック
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    global_step = 0
    best_val = float('inf')

    # 6. 訓練ループ
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

            # optimizer.zero_grad() は更新ステップでのみ実行

            with torch.amp.autocast('cuda', enabled=use_amp):
                restored = restorer(noise_img)

                # pixel losses
                l1 = l1_loss(restored, gt_img)
                ssim_val = ssim_module(restored, gt_img)
                ssim_loss_val = 1.0 - ssim_val

                # feature loss via YOLO backbone
                with torch.no_grad():
                    feat_gt = backbone(gt_img)
                feat_rest = backbone(restored)
                feat_loss = feature_mse_loss(feat_rest, feat_gt)

                total_loss = l1 + LAMBDA_SSIM * ssim_loss_val + lambda_det * feat_loss

                # === 勾配蓄積のための正規化 ===
                total_loss = total_loss / ACCUMULATION_STEPS 

            # backward
            if use_amp:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            # === 勾配蓄積とパラメータ更新 ===
            # ACCUMULATION_STEPSごとに勾配を更新
            is_update_step = (i + 1) % ACCUMULATION_STEPS == 0
            is_last_step = (i + 1) == len(train_loader)
            
            if is_update_step or is_last_step:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad() # 更新後に勾配をクリア

            # ロギング (正規化前の損失値をベースに計算)
            epoch_l1 += l1.item() 
            epoch_ssim += ssim_loss_val.item()
            epoch_feat += feat_loss.item()
            epoch_total += total_loss.item() * ACCUMULATION_STEPS # 非正規化に戻して加算
            global_step += 1

            if (i+1) % PRINT_FREQ == 0 or is_last_step:
                # 表示用に、現在のバッチの非正規化損失を表示
                print(f"  [{i+1}/{len(train_loader)}] total {total_loss.item() * ACCUMULATION_STEPS:.4f} l1 {l1.item():.4f} ssim {ssim_loss_val.item():.4f} feat {feat_loss.item():.4f}")

        # epoch logging
        n_batch = len(train_loader)
        writer.add_scalar("train/total_loss", epoch_total / n_batch, epoch)
        writer.add_scalar("train/l1", epoch_l1 / n_batch, epoch)
        writer.add_scalar("train/ssim", epoch_ssim / n_batch, epoch)
        writer.add_scalar("train/feat", epoch_feat / n_batch, epoch)
        writer.add_scalar("train/lambda_det", lambda_det, epoch)

        # 7. 検証 (Validation)
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
        
        # 8. ベストモデルの保存
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

        # 9. 定期チェックポイントの保存
        if epoch % SAVE_EVERY == 0:
            torch.save({
                "epoch": epoch,
                "restorer_state": restorer.state_dict(),
                "optimizer": optimizer.state_dict()
            }, CKPT_DIR / f"restorer_epoch{epoch:03d}.pth")

    writer.close()
    print("Training finished. Checkpoints and logs saved to", OUT_DIR)

if __name__ == '__main__':
    # 実行前に必要なファイル (models.py, data.py, config.py) が存在することを確認してください。
    main()