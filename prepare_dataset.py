"""
Plant Disease Detection - Dataset Preparation
Compatible with: Python 3.11 / 3.12  (no TensorFlow needed for this script)

Organizes raw Kaggle images into train / val / test splits.

Expected input structure (after downloading from Kaggle):
    raw_dataset/
        Downy Mildew/
            img1.jpg ...
        Healthy/
            img1.jpg ...
        Iris Yellow Spot/
            img1.jpg ...
        Purple Blotch/
            img1.jpg ...

Output structure:
    dataset/
        train/  val/  test/
            <each class subfolder>
"""

import os
import shutil
import random
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────
RAW_DIR     = "raw_dataset"
OUTPUT_DIR  = "dataset"
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 0.15  (remainder)

SEED = 42
random.seed(SEED)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def prepare_dataset():
    raw_path = Path(RAW_DIR)
    if not raw_path.exists():
        print(f"'{RAW_DIR}' folder not found.")
        print("   Steps to fix:")
        print("   1. Download from: https://www.kaggle.com/datasets/tejasbargujechef/onion-diseases")
        print("   2. Extract the ZIP into a folder named 'raw_dataset' in this directory.")
        return

    class_dirs = [d for d in raw_path.iterdir() if d.is_dir()]
    if not class_dirs:
        print(f"No class subfolders found inside '{RAW_DIR}'.")
        return

    print(f"Found {len(class_dirs)} class(es): {[d.name for d in class_dirs]}\n")

    # Create output folders
    for split in ["train", "val", "test"]:
        for cls in class_dirs:
            Path(OUTPUT_DIR, split, cls.name).mkdir(parents=True, exist_ok=True)

    total_moved = 0

    for cls_dir in class_dirs:
        images = [f for f in cls_dir.iterdir()
                  if f.suffix.lower() in IMG_EXTS]
        random.shuffle(images)

        n       = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)

        split_map = {
            "train": images[:n_train],
            "val"  : images[n_train : n_train + n_val],
            "test" : images[n_train + n_val :],
        }

        print(f"  {cls_dir.name:<25} total={n:>5} | "
              f"train={len(split_map['train']):>4}, "
              f"val={len(split_map['val']):>4}, "
              f"test={len(split_map['test']):>4}")

        for split, files in split_map.items():
            dest = Path(OUTPUT_DIR, split, cls_dir.name)
            for f in files:
                shutil.copy2(f, dest / f.name)
                total_moved += 1

    print(f"\nDone! {total_moved} images organised into '{OUTPUT_DIR}/'")
    print("   Next step → run:  python train.py")

if __name__ == "__main__":
    prepare_dataset()