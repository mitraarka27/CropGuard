import torchvision.transforms as T

class AugmentationPipeline:
    """
    Data augmentation and preprocessing transformations for CropGuard.
    """

    def __init__(self):
        # Mean and Std from ImageNet (can be adjusted later if needed)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # Define transformations
        self.train_transforms = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=30),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std)
        ])

        self.val_transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std)
        ])

        self.test_transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std)
        ])

    def get_transforms(self, phase="train"):
        """
        Returns the appropriate transformation based on phase.
        """
        if phase == "train":
            return self.train_transforms
        elif phase == "val":
            return self.val_transforms
        elif phase == "test":
            return self.test_transforms
        else:
            raise ValueError(f"Unknown phase: {phase}. Use 'train', 'val', or 'test'.")