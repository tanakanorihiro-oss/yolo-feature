import os, glob, numpy as np

# --- 設定 ---
gt_dir = "/home/usrs/tananori/Study/yolov8_end/dataset/DAWN/Snow/GT_labels_yolo"
pred_dir = "/home/usrs/tananori/Study/yolov8_end/original/000/DAWN/Snow/labels"  # ←要修正
target_class_ids = {0, 1, 2, 3, 5, 6, 7}

def compute_iou(box1, box2):
    x1_min, y1_min = box1[0]-box1[2]/2, box1[1]-box1[3]/2
    x1_max, y1_max = box1[0]+box1[2]/2, box1[1]+box1[3]/2
    x2_min, y2_min = box2[0]-box2[2]/2, box2[1]-box2[3]/2
    x2_max, y2_max = box2[0]+box2[2]/2, box2[1]+box2[3]/2
    inter_xmin, inter_ymin = max(x1_min,x2_min), max(y1_min,y2_min)
    inter_xmax, inter_ymax = min(x1_max,x2_max), min(y1_max,y2_max)
    inter_area = max(0,inter_xmax-inter_xmin)*max(0,inter_ymax-inter_ymin)
    area1 = (x1_max-x1_min)*(y1_max-y1_min)
    area2 = (x2_max-x2_min)*(y2_max-y2_min)
    union = area1+area2-inter_area
    return inter_area/union if union>0 else 0

def load_labels(file_path, filter_classes):
    boxes=[]
    if not os.path.exists(file_path): return boxes
    with open(file_path,"r") as f:
        for line in f:
            elems=line.strip().split()
            cid=int(elems[0])
            if cid in filter_classes:
                box=list(map(float,elems[1:5]))
                boxes.append((cid, box))
    return boxes

def load_predictions(file_path, filter_classes):
    preds=[]
    if not os.path.exists(file_path): return preds
    with open(file_path,"r") as f:
        for line in f:
            elems=line.strip().split()
            cid=int(elems[0])
            if cid in filter_classes:
                box=list(map(float,elems[1:5]))
                score=float(elems[5]) if len(elems)>5 else 1.0
                preds.append((cid, box, score))
    return preds

def calculate_ap(tp, fp, total_gt):
    if total_gt == 0:
        return None
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / total_gt
    precision = tp_cum / (tp_cum + fp_cum + 1e-6)
    for i in range(len(precision)-1,0,-1):
        precision[i-1] = max(precision[i-1], precision[i])
    ap = np.trapz(precision, recall)
    return float(ap)

# --- GT読込キャッシュ ---
all_gt_boxes_by_file = {}
for txt_file in glob.glob(os.path.join(gt_dir,"*.txt")):
    filename = os.path.basename(txt_file)
    all_gt_boxes_by_file[filename] = load_labels(txt_file, target_class_ids)

# --- クラスごとに評価 ---
all_ap=[]
for cid in target_class_ids:
    total_gt=0
    preds_all=[]
    for filename, gt_data in all_gt_boxes_by_file.items():
        gt_boxes = [b for c,b in gt_data if c==cid]
        total_gt += len(gt_boxes)
        preds = load_predictions(os.path.join(pred_dir, filename), {cid})
        for _,box,score in preds:
            preds_all.append((filename, box, score))
    if total_gt==0:
        print(f"Class {cid} has no GT samples, skipping AP calc.")
        continue

    preds_all.sort(key=lambda x:x[2], reverse=True)
    tp_all=[]; fp_all=[]
    gt_matched={}

    for filename,pbox,score in preds_all:
        gt_boxes=[b for c,b in all_gt_boxes_by_file.get(filename, []) if c==cid]
        matched = gt_matched.setdefault(filename,set())
        max_iou,best_idx=0,-1
        for i,gt in enumerate(gt_boxes):
            iou=compute_iou(pbox, gt)
            if iou>max_iou:
                max_iou,best_idx=iou,i
        if max_iou>=0.5 and best_idx not in matched:
            tp_all.append(1); fp_all.append(0)
            matched.add(best_idx)
        else:
            tp_all.append(0); fp_all.append(1)

    ap=calculate_ap(tp_all, fp_all, total_gt)
    if ap is not None:
        all_ap.append(ap)
        print(f"Class {cid} AP={ap:.3f}")

if all_ap:
    print(f"\nFiltered mAP@0.5={np.mean(all_ap):.3f}")
else:
    print("\nNo valid classes with GT found.")