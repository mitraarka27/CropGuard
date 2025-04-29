import os
import shutil
import random
from src.utils.config import DATA_DIR, SPLIT_DIR, SEED

def split_clean_data(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=SEED):
    """
    Split cleaned data into train/val/test splits under data/split/.
    """
    random.seed(seed)

    clean_dir = os.path.join(DATA_DIR, "clean")

    # Remove previous split if exists
    if os.path.exists(SPLIT_DIR):
        shutil.rmtree(SPLIT_DIR)
    os.makedirs(SPLIT_DIR)

    for crop in os.listdir(clean_dir):
        crop_path = os.path.join(clean_dir, crop)

        for disease_folder in os.listdir(crop_path):
            disease_path = os.path.join(crop_path, disease_folder)
            images = os.listdir(disease_path)
            if len(images) == 0:
                print(f"[WARNING] No images found in {disease_path}, skipping.")
                continue

            random.shuffle(images)

            n_total = len(images)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)

            # Safety check: at least 1 sample in each split
            if n_train == 0 or n_val == 0 or (n_total - n_train - n_val) == 0:
                print(f"[WARNING] Not enough images to split {disease_path} properly, skipping.")
                continue

            splits = {
                "train": images[:n_train],
                "val": images[n_train:n_train+n_val],
                "test": images[n_train+n_val:]
            }

            for split_name, split_images in splits.items():
                target_dir = os.path.join(SPLIT_DIR, split_name, crop, disease_folder)
                os.makedirs(target_dir, exist_ok=True)

                for img_name in split_images:
                    src_img = os.path.join(disease_path, img_name)
                    dst_img = os.path.join(target_dir, img_name)
                    shutil.copy(src_img, dst_img)

    print("[INFO] Finished creating train/val/test split.")