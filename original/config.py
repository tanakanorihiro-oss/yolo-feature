# config.py
# 設定ファイル(ハイパーパラメータ, パスなど)

from pathlib import Path

# ==========================================================
# === CONFIGURATION (設定) =================================
# ==========================================================

# Paths - edit if different on your machine
TRAIN_NOISE_DIR = "/home/usrs/tananori/Study/yolov8_end/dataset/bdd100k_clear/noise/train"
TRAIN_GT_DIR    = "/home/usrs/tananori/Study/yolov8_end/dataset/bdd100k_clear/images/train"
VAL_NOISE_DIR   = "/home/usrs/tananori/Study/yolov8_end/dataset/bdd100k_clear/noise/val"
VAL_GT_DIR      = "/home/usrs/tananori/Study/yolov8_end/dataset/bdd100k_clear/images/val"

YOLO_WEIGHTS = "yolov8n.pt"

OUT_DIR = Path("./output_unet_restorer_yolo") 
CKPT_DIR = OUT_DIR / "checkpoints"
LOG_DIR = OUT_DIR / "logs"

# Hardware and Memory
DEVICE = "cuda" # 'cuda' or 'cpu'

# --- 勾配蓄積のための設定 ---
# 論理バッチサイズ 32 を目標とする
PHYSICAL_BATCH_SIZE = 16   # GPUに実際にロードするバッチサイズ (変更可能)
ACCUMULATION_STEPS = 2     # 勾配を蓄積する回数 (PHYSICAL_BATCH_SIZE * ACCUMULATION_STEPS = 論理バッチサイズ)

# --- Training Hyperparameters ---
# データローダーに渡すバッチサイズは物理サイズを使用
BATCH_SIZE = PHYSICAL_BATCH_SIZE 
VAL_BATCH_SIZE = 32
IMG_SIZE = 320
EPOCHS = 100
LR = 5e-4
WEIGHT_DECAY = 1e-4

# Loss weights & schedule
LAMBDA_SSIM = 0.5
LAMBDA_DET_MAX = 2.0       
WARMUP_EPOCHS = 30       

# Misc
SAVE_EVERY = 1
PRINT_FREQ = 50
NUM_WORKERS = 8