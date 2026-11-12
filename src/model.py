# src/model.py
"""Model definition for the Agri Quality Classifier.
We use torchvision's MobileNetV3 Small pretrained on ImageNet
and replace the classifier head to match the number of target classes.
"""
import torch
import torch.nn as nn
import torchvision.models as models


def build_mobilenet_v3_small(num_classes: int) -> nn.Module:
    """Create a MobileNetV3‑Small model with a custom output layer.

    Args:
        num_classes: Number of target classes.
    Returns:
        A torch.nn.Module ready for training/evaluation.
    """
    # Load pretrained MobileNetV3 Small
    model = models.mobilenet_v3_small(pretrained=True)
    # Freeze feature extractor initially (optional, caller can unfreeze later)
    for param in model.features.parameters():
        param.requires_grad = False
    # Replace the classifier (last linear layer)
    in_features = model.classifier[3].in_features  # The default classifier ends with a Linear
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model
