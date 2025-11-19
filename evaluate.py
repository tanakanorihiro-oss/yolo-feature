# TP, FP, FN を計算して Precision, Recallを求めるコード

import os
import glob

# === パスの設定（必要に応じて絶対パスに変更） ===
gt_dir = '/home/usrs/tananori/Study/yolov8_end/dataset/DAWN/Snow/GT_labels_yolo'
pred_dir = '/home/usrs/tananori/Study/yolov8_end/original/000/DAWN/Snow/labels'

# === IoU計算 ===
def compute_iou(box1, box2):
    x1_min = box1[0] - box1[2]/2
    y1_min = box1[1] - box1[3]/2
    x1_max = box1[0] + box1[2]/2
    y1_max = box1[1] + box1[3]/2

    x2_min = box2[0] - box2[2]/2
    y2_min = box2[1] - box2[3]/2
    x2_max = box2[0] + box2[2]/2
    y2_max = box2[1] + box2[3]/2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area != 0 else 0

# === ラベルファイルの読み込み ===
def load_labels(file_path, is_prediction=False):
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if is_prediction:
                cls, x, y, w, h, conf = parts
                labels.append((int(cls), float(x), float(y), float(w), float(h), float(conf)))
            else:
                cls, x, y, w, h = parts
                labels.append((int(cls), float(x), float(y), float(w), float(h)))
    return labels

# === 評価関数 ===
def evaluate(iou_thresh=0.5):
    all_files = sorted(glob.glob(os.path.join(gt_dir, '*.txt')))
    total_tp, total_fp, total_fn = 0, 0, 0

    for gt_file in all_files:
        filename = os.path.basename(gt_file)
        pred_file = os.path.join(pred_dir, filename)

        gts = load_labels(gt_file, is_prediction=False)
        preds = load_labels(pred_file, is_prediction=True) if os.path.exists(pred_file) else []

        matched_gt = set()
        tp = 0

        preds = sorted(preds, key=lambda x: x[5], reverse=True)

        for pred in preds:
            pred_cls, *pred_box, _ = pred
            best_iou = 0
            best_idx = -1
            for idx, gt in enumerate(gts):
                gt_cls, *gt_box = gt
                if idx in matched_gt or pred_cls != gt_cls:
                    continue
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_iou >= iou_thresh:
                tp += 1
                matched_gt.add(best_idx)
            else:
                total_fp += 1

        fn = len(gts) - len(matched_gt)

        total_tp += tp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

    print("====== 結果 ======")
    print(f"TP: {total_tp}")
    print(f"FP: {total_fp}")
    print(f"FN: {total_fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

# === 実行 ===
if __name__ == "__main__":
    evaluate(iou_thresh=0.5)