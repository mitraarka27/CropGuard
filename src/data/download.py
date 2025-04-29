import os
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi
from src.utils.config import DATA_DIR, CLEAN_DIR, RAW_DIR, TARGET_CROPS  # ✅ Import cleanly

# -------------------------------
# Functions
# -------------------------------

def download_and_extract_dataset():
    """
    Download and extract the full PlantVillage dataset.
    """
    os.makedirs(RAW_DIR, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print("[INFO] Downloading PlantVillage dataset...")
    api.dataset_download_files('mohitsingh1804/plantvillage', path=RAW_DIR, unzip=True)
    print("[INFO] Download complete and extracted.")

def clean_and_organize_dataset():
    """
    Organize Potato, Tomato, Grape from train/ and val/ into clean/ directory.
    """
    extracted_dir = os.path.join(RAW_DIR, "plantvillage")

    train_dir = os.path.join(extracted_dir, "train")
    val_dir = os.path.join(extracted_dir, "val")

    if not os.path.exists(CLEAN_DIR):
        os.makedirs(CLEAN_DIR)
        print(f"[INFO] Created clean directory at: {CLEAN_DIR}")

    for split_dir in [train_dir, val_dir]:
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"[ERROR] {split_dir} not found.")

        for folder in os.listdir(split_dir):
            full_folder_path = os.path.join(split_dir, folder)
            if os.path.isdir(full_folder_path) and any(folder.startswith(crop) for crop in TARGET_CROPS):
                crop_name = folder.split("___")[0].lower()
                disease_folder = folder

                destination_crop_dir = os.path.join(CLEAN_DIR, crop_name)
                os.makedirs(destination_crop_dir, exist_ok=True)

                destination_disease_dir = os.path.join(destination_crop_dir, disease_folder)
                os.makedirs(destination_disease_dir, exist_ok=True)

                for img_file in os.listdir(full_folder_path):
                    src_img = os.path.join(full_folder_path, img_file)
                    dst_img = os.path.join(destination_disease_dir, img_file)
                    shutil.copy(src_img, dst_img)

    print("[INFO] Crops cleaned and organized into 'clean/' directory from train and val folders.")

    # -------------------------------
    # Remove plant_disease_raw after cleaning
    # -------------------------------
    if os.path.exists(RAW_DIR):
        shutil.rmtree(RAW_DIR)
        print(f"[INFO] Deleted raw data directory at {RAW_DIR} after cleaning.")

def check_data_integrity():
    """
    Quick check that clean/ has the crops properly.
    """
    if not os.path.exists(CLEAN_DIR):
        raise FileNotFoundError(f"[ERROR] Clean folder {CLEAN_DIR} not found!")

    for crop in os.listdir(CLEAN_DIR):
        crop_dir = os.path.join(CLEAN_DIR, crop)
        print(f"[INFO] {crop.capitalize()}: {len(os.listdir(crop_dir))} disease classes found.")