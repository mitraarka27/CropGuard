import os
from PIL import Image
from torch.utils.data import Dataset

class PlantVillageDataset(Dataset):
    """
    PyTorch-compatible dataset for the cleaned and split PlantVillage dataset.

    Directory structure should be:
        root/
            crop1/
                disease1/
                    img1.jpg
                    ...
                disease2/
                    ...
            crop2/
                ...
    """

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to split directory (e.g., data/split/train)
            transform (callable, optional): Transformations to apply to images
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self._prepare_dataset()

    def _prepare_dataset(self):
        """
        Scan directory and build (image_path, class_index) list
        """
        class_names = []
        for crop in sorted(os.listdir(self.root_dir)):
            crop_path = os.path.join(self.root_dir, crop)
            if not os.path.isdir(crop_path):
                continue

            for disease in sorted(os.listdir(crop_path)):
                disease_path = os.path.join(crop_path, disease)
                if not os.path.isdir(disease_path):
                    continue  # Safety check

                class_name = f"{crop}___{disease}"
                if class_name not in self.class_to_idx:
                    self.class_to_idx[class_name] = len(self.class_to_idx)
                    class_names.append(class_name)

                label = self.class_to_idx[class_name]

                for fname in os.listdir(disease_path):
                    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue
                    img_path = os.path.join(disease_path, fname)
                    self.samples.append((img_path, label))

        # print(f"[INFO] {len(self.samples)} images found across {len(self.class_to_idx)} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label