# src/inference.py
"""Inference utilities shared between CLI and Streamlit.
Provides functions to load a checkpoint, reconstruct the model,
and run a forward pass returning class names and confidence scores.
"""
import json
import pathlib
from typing import Tuple, List, Dict

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from config import BEST_MODEL_PATH, InferenceConfig
from src.model import build_mobilenet_v3_small

# Load checkpoint and return model + class mapping
def load_checkpoint(device: torch.device = None) -> Tuple[torch.nn.Module, Dict[int, str]]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not BEST_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model checkpoint not found at {BEST_MODEL_PATH}. Train the model first.")
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    num_classes = checkpoint["num_classes"]
    model = build_mobilenet_v3_small(num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    # class mapping saved as list of class names ordered by index
    class_idx_to_name = {int(k): v for k, v in checkpoint["class_to_idx"].items()}
    return model, class_idx_to_name

# Preprocess a PIL image for the model
def preprocess_image(img: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)  # add batch dim

def predict_image(image_path: pathlib.Path, device: torch.device = None, cfg: InferenceConfig = InferenceConfig()) -> Dict:
    """Run inference on a single image.
    Returns a dictionary with keys:
        - predicted_class (str)
        - confidence (float 0-1)
        - top3: List[Tuple[str, float]] sorted by confidence desc
        - warning (optional str) if confidence < threshold
    """
    model, class_idx_to_name = load_checkpoint(device)
    with Image.open(image_path) as img:
        img = img.convert("RGB")
    input_tensor = preprocess_image(img)
    input_tensor = input_tensor.to(next(model.parameters()).device)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)
    top_probs, top_idxs = torch.topk(probs, k=3)
    top3 = [(class_idx_to_name[int(idx)], float(prob)) for idx, prob in zip(top_idxs, top_probs)]
    pred_idx = top_idxs[0].item()
    confidence = top_probs[0].item()
    result = {
        "predicted_class": class_idx_to_name[pred_idx],
        "confidence": confidence,
        "top3": top3,
    }
    if confidence < cfg.confidence_threshold:
        result["warning"] = f"Low confidence ({confidence:.2%}); prediction may be unreliable."
    return result
