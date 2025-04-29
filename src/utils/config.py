import os

# -------------------------------
# Paths
# -------------------------------

# Dynamically find the real project root (CropGuard/)
CURRENT_FILE = os.path.abspath(__file__)
SRC_DIR = os.path.dirname(os.path.dirname(CURRENT_FILE))  # src/
PROJECT_ROOT = os.path.dirname(SRC_DIR)  # CropGuard/

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CLEAN_DIR = os.path.join(DATA_DIR, "clean")
SPLIT_DIR = os.path.join(DATA_DIR, "split")

TRAIN_DIR = os.path.join(SPLIT_DIR, "train")
VAL_DIR = os.path.join(SPLIT_DIR, "val")
TEST_DIR = os.path.join(SPLIT_DIR, "test")

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "cropguard_best.pt")
LABELS_MAP_PATH = os.path.join(MODEL_DIR, "labels.json")

# Target folders for download.py
RAW_DIR = os.path.join(DATA_DIR, "plant_disease_raw")
CLEAN_DIR = os.path.join(DATA_DIR, "clean")

# Target crops
TARGET_CROPS = ["Potato___", "Tomato___", "Grape___"]

# Binary classification mapping (0=healthy, 1=sick)
BINARY_CLASSES = {
    "healthy": 0,
    "sick": 1
}

# -------------------------------
# Random Seed
# -------------------------------

SEED = 42