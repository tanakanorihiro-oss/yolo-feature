# yolov8xで???を検出

from ultralytics import YOLO
import os
from glob import glob

# === 設定 ===
input_dir = "/home/usrs/tananori/Study/yolov8_end/dataset/DAWN/Snow"
output_img_dir = "/home/usrs/tananori/Study/yolov8_end/original/000/DAWN/visualized_labels/Snow"
output_label_dir = "/home/usrs/tananori/Study/yolov8_end/original/000/DAWN/Snow/labels"
model_name = "yolov8n.pt"

# === あなたの定義するクラス順（0〜9）===
target_classes = [
    "person", "bicycle", "car", "motorcycle", "trailer",
    "bus", "train", "truck", "traffic light", "trafficsign"
]

# === COCO → あなたのクラスIDへの対応表 ===
id_map = {
    0: 0,   # person
    1: 1,   # bicycle
    2: 2,   # car
    3: 3,   # motorcycle
    5: 5,   # bus
    6: 6,   # train
    7: 7,   # truck
    9: 8,   # traffic light
    11: 9   # stop sign ≈ trafficsign
    # trailer はCOCOに存在しない
}

# === 検出対象クラスID（COCO基準）===
detect_ids = list(id_map.keys())

# === 出力フォルダ作成 ===
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# === モデル読み込み ===
model = YOLO(model_name)

# === 推論対象画像 ===
image_paths = glob(os.path.join(input_dir, "*.jpg")) + glob(os.path.join(input_dir, "*.png"))

# === 推論ループ ===
for img_path in image_paths:
    results = model.predict(
        source=img_path,
        conf=0.25,
        classes=detect_ids,  # ←★ 検出クラスを制限
        save=False
    )

    for r in results:
        # 可視化画像保存
        save_img_path = os.path.join(output_img_dir, os.path.basename(img_path))
        r.plot(save=True, filename=save_img_path)

        # YOLO形式ラベル保存
        label_path = os.path.join(output_label_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt")
        with open(label_path, "w") as f:
            for box in r.boxes:
                coco_id = int(box.cls[0])
                if coco_id in id_map:
                    new_id = id_map[coco_id]
                    x_center, y_center, w, h = box.xywhn[0]
                    conf = box.conf[0]
                    f.write(f"{new_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f} {conf:.6f}\n")

print("✅ YOLOv8x による指定クラスのみ検出完了！")
print(f"画像出力先: {output_img_dir}")
print(f"ラベル出力先: {output_label_dir}")