# src/data.py
"""Data handling for the Agri Quality Classifier.
- Discovers dataset structure under RAW_DATASET_DIR.
- Filters to the selected fruits (both Healthy and Rotten).
- Provides train/val/test splits as PyTorch Dataset objects.
"""
import pathlib
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from config import RAW_DATASET_DIR, SELECTED_FRUITS, TrainConfig

# Helper to map folder name to our canonical class label
def _canonical_label(folder_name: str) -> str:
    # Expected pattern: Fruit__Condition (e.g., Apple__Healthy)
    parts = folder_name.split("__")
    if len(parts) != 2:
        return folder_name
    fruit, condition = parts
    return f"{fruit}_{condition}"  # Apple_Healthy

@dataclass
class FilteredDataset(Dataset):
    """A dataset that only yields images belonging to the selected fruit classes.
    """
    image_paths: List[pathlib.Path]
    labels: List[int]
    transform: transforms.Compose

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        # Load image safely
        with Image.open(img_path) as img:
            img = img.convert("RGB")
        return self.transform(img), label

def _gather_all_images() -> Tuple[List[pathlib.Path], List[str]]:
    """Walk the raw dataset directory and collect image paths with their folder names.
    Returns two parallel lists: image_paths, folder_names.
    """
    image_paths = []
    folder_names = []
    for class_dir in RAW_DATASET_DIR.iterdir():
        if not class_dir.is_dir():
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"):
            for img_path in class_dir.glob(ext):
                image_paths.append(img_path)
                folder_names.append(class_dir.name)
    return image_paths, folder_names

def _filter_selected(image_paths: List[pathlib.Path], folder_names: List[str]) -> Tuple[List[pathlib.Path], List[str]]:
    """Keep only images whose folder corresponds to the selected fruits.
    Also ensure both Healthy and Rotten are present; missing ones are reported.
    """
    selected_paths = []
    selected_folders = []
    for img_path, folder in zip(image_paths, folder_names):
        fruit = folder.split("__")[0]
        if fruit in SELECTED_FRUITS:
            selected_paths.append(img_path)
            selected_folders.append(folder)
    return selected_paths, selected_folders

def build_class_mapping(selected_folders: List[str]) -> Dict[str, int]:
    """Create a deterministic mapping from canonical label to integer index.
    Missing Healthy/Rotten combos are warned but not added.
    """
    canonical_labels = sorted({_canonical_label(f) for f in selected_folders})
    mapping = {label: idx for idx, label in enumerate(canonical_labels)}
    # Verify that we have both conditions for each fruit
    missing = []
    for fruit in SELECTED_FRUITS:
        for cond in ["Healthy", "Rotten"]:
            label = f"{fruit}_{cond}"
            if label not in mapping:
                missing.append(label)
    if missing:
        print("[WARNING] The following expected classes are missing from the dataset:")
        for m in missing:
            print(f"  - {m}")
    return mapping

def create_datasets(config: TrainConfig = TrainConfig()):
    """Return train, val, test DataLoaders for the filtered dataset.
    If the raw dataset does not contain a predefined split, a stratified split is created.
    """
    # Gather all images
    all_paths, all_folders = _gather_all_images()
    # Filter to selected fruits
    sel_paths, sel_folders = _filter_selected(all_paths, all_folders)
    if not sel_paths:
        raise RuntimeError("No images found for the selected fruit classes.")
    # Build class mapping
    class_to_idx = build_class_mapping(sel_folders)
    # Convert folder names to canonical labels then to indices
    labels = [_canonical_label(f) for f in sel_folders]
    numeric_labels = [class_to_idx[l] for l in labels]
    # Shuffle with deterministic seed
    combined = list(zip(sel_paths, numeric_labels))
    random.Random(config.seed).shuffle(combined)
    paths, numeric_labels = zip(*combined)
    # Split (70/15/15) stratified using sklearn if available, otherwise simple split
    from sklearn.model_selection import train_test_split
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, numeric_labels, test_size=(config.val_split + config.test_split),
        stratify=numeric_labels, random_state=config.seed)
    val_ratio = config.val_split / (config.val_split + config.test_split)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=(1 - val_ratio),
        stratify=temp_labels, random_state=config.seed)
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = FilteredDataset(list(train_paths), list(train_labels), train_transform)
    val_dataset = FilteredDataset(list(val_paths), list(val_labels), eval_transform)
    test_dataset = FilteredDataset(list(test_paths), list(test_labels), eval_transform)
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader, class_to_idx
