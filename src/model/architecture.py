import torch
import torch.nn as nn
from torchvision import models

def build_model(num_classes, freeze_backbone=True):
    """
    Build and return a MobileNetV2 model fine-tuned for our custom classes.

    Args:
        num_classes (int): Number of disease classes
        freeze_backbone (bool): If True, freeze feature extractor layers

    Returns:
        model (nn.Module)
    """
    model = models.mobilenet_v2(weights='IMAGENET1K_V1')

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace the classifier
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, num_classes)
    )

    return model