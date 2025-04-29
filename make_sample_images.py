import os
import random
import shutil

# Source: CLEAN data
CLEAN_DIR = "data/clean"
SAMPLE_DIR = "sample_images"

# Create destination if doesn't exist
os.makedirs(SAMPLE_DIR, exist_ok=True)

# Crops we care about
crops = ["potato", "grape", "tomato"]

# Keyword to detect healthy leaves
healthy_keyword = "healthy"

# Clear old sample images first
for file in os.listdir(SAMPLE_DIR):
    os.remove(os.path.join(SAMPLE_DIR, file))

counter = 1

for crop in crops:
    crop_dir = os.path.join(CLEAN_DIR, crop)
    if not os.path.exists(crop_dir):
        print(f"[WARNING] {crop} directory not found.")
        continue

    healthy_images = []
    diseased_images = []

    for disease_folder in os.listdir(crop_dir):
        full_path = os.path.join(crop_dir, disease_folder)
        if not os.path.isdir(full_path):
            continue

        images = os.listdir(full_path)
        random.shuffle(images)

        if healthy_keyword in disease_folder.lower():
            healthy_images.extend([(full_path, img, disease_folder) for img in images])
        else:
            diseased_images.extend([(full_path, img, disease_folder) for img in images])

    # Pick samples
    selected_healthy = random.sample(healthy_images, min(2, len(healthy_images)))
    selected_diseased = random.sample(diseased_images, min(2, len(diseased_images)))

    # Copy selected healthy files
    for src_path, img_file, disease_folder in selected_healthy:
        new_filename = f"{crop}_healthy_sample_{counter}.jpg"
        shutil.copy(os.path.join(src_path, img_file), os.path.join(SAMPLE_DIR, new_filename))
        counter += 1

    # Copy selected diseased files
    for src_path, img_file, disease_folder in selected_diseased:
        clean_disease = disease_folder.replace("___", "_").replace("__", "_").replace(" ", "_").replace(".", "").lower()
        new_filename = f"{crop}_{clean_disease}_sample_{counter}.jpg"
        shutil.copy(os.path.join(src_path, img_file), os.path.join(SAMPLE_DIR, new_filename))
        counter += 1

print(f"[INFO] Sampling complete. Files saved to '{SAMPLE_DIR}'")