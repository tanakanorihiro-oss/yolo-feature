#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
from pathlib import Path

# === パス設定 ===
img_dir = Path("/home/usrs/tananori/Study/yolov8_end/original/112/DAWN/Fog")
label_dir = Path("/home/usrs/tananori/Study/yolov8_end/original/112/DAWN/Fog/labels")
out_dir = Path("/home/usrs/tananori/Study/yolov8_end/original/112/DAWN/visualized_labels/Fog")
out_dir.mkdir(parents=True, exist_ok=True)

img_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
img_files = [p for p in img_dir.iterdir() if p.suffix.lower() in img_extensions]

print(f"画像数: {len(img_files)}")

for img_path in img_files:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"読み込み失敗: {img_path}")
        continue

    h, w = img.shape[:2]

    label_path = label_dir / (img_path.stem + ".txt")
    if not label_path.exists():
        continue

    with open(label_path, "r") as f:
        lines = f.read().strip().splitlines()

    for line in lines:
        parts = line.strip().split()

        # ラベル形式：cls x y w h conf（6個）
        if len(parts) < 5:
            print(f"無効な行をスキップ: {line}")
            continue

        cls, x, y, bw, bh = parts[:5]  # confidence は無視
        cls = int(cls)
        x, y, bw, bh = map(float, [x, y, bw, bh])

        # YOLO → pixel
        px1 = int((x - bw / 2) * w)
        py1 = int((y - bh / 2) * h)
        px2 = int((x + bw / 2) * w)
        py2 = int((y + bh / 2) * h)

        cv2.rectangle(img, (px1, py1), (px2, py2), (0, 255, 0), 2)
        cv2.putText(img, str(cls), (px1, py1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    out_path = out_dir / img_path.name
    cv2.imwrite(str(out_path), img)

print("=== 完了 ===")
print(f"保存先: {out_dir}")