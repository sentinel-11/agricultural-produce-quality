# config.py
"""Central configuration for the Agri Quality Classifier project.
All paths are defined relative to the repository root (agri-quality-classifier).
"""
import pathlib
from dataclasses import dataclass

# Repository root (the directory containing this file)
ROOT_DIR = pathlib.Path(__file__).resolve().parent

# Data directories (downloaded dataset resides in the parent project folder)
# The raw dataset is located at:
# <project_root>/../Fruit And Vegetable Diseases Dataset
RAW_DATASET_DIR = ROOT_DIR.parent / "Fruit And Vegetable Diseases Dataset"
# Extracted images will be copied/linked into this directory for processing
DATA_ROOT = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_ROOT / "raw"
DOWNLOAD_TMP_DIR = DATA_ROOT / "download_tmp"

# Model checkpoints and artifacts
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
BEST_MODEL_PATH = ARTIFACTS_DIR / "best_model.pth"

# Reports
REPORTS_DIR = ROOT_DIR / "reports"
METRICS_JSON = REPORTS_DIR / "metrics.json"
CONFUSION_MATRIX_PNG = REPORTS_DIR / "confusion_matrix.png"

# Selected fruits (both Healthy and Rotten) – must match dataset folder naming pattern
SELECTED_FRUITS = ["Apple", "Banana", "Grape", "Guava", "Orange"]

# Training hyper‑parameters (can be overridden via CLI if desired)
@dataclass
class TrainConfig:
    batch_size: int = 32
    num_epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    seed: int = 42
    val_split: float = 0.15
    test_split: float = 0.15
    early_stop_patience: int = 5

# Inference settings
@dataclass
class InferenceConfig:
    confidence_threshold: float = 0.5  # below this we warn about uncertainty

# Ensure required directories exist (created lazily where needed)
for d in [DATA_ROOT, RAW_DATA_DIR, DOWNLOAD_TMP_DIR, ARTIFACTS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
