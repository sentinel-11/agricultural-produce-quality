# Agricultural Produce Quality Classifier

## Project Overview

This repository contains a **fruit and vegetable quality classifier** that distinguishes between **healthy** and **rotten** images for a selected set of fruits. The model is built with PyTorch using a pretrained **MobileNetV3‑Small** backbone and can be run via a Streamlit dashboard or a simple command‑line utility.

## Selected Fruits

The classifier is limited to the following five fruits (both healthy and rotten variants):

| Fruit | Classes generated |
|-------|-------------------|
| Apple | `Apple_Healthy`, `Apple_Rotten` |
| Banana | `Banana_Healthy`, `Banana_Rotten` |
| Grape | `Grape_Healthy`, `Grape_Rotten` |
| Guava | `Guava_Healthy`, `Guava_Rotten` |
| Orange | `Orange_Healthy`, `Orange_Rotten` |

If any of these classes are missing from the original dataset, the script will report the omission and **will not** create a phantom class.

## Repository Structure

```
agri-quality-classifier/
│   README.md
│   requirements.txt
│   config.py
│   train.py
│   evaluate.py
│   predict.py
│   app.py
│
├───src/
│   │   __init__.py
│   │   data.py          # Dataset discovery, filtering, splits
│   │   model.py         # Model construction
│   │   inference.py     # Inference utilities used by predict.py & app.py
│   │   utils.py         # Helper functions (e.g., seed setting)
│
├───tests/
│   │   test_inference.py
│   │   test_app_utils.py
│
├───artifacts/
│   │   .gitkeep
│
├───reports/
│   │   .gitkeep
│
└───data/
    │   raw/                # <‑‑ populated by download_dataset.py (outside this repo)
    │   download_tmp/       # temporary download folder (ignored via .gitignore)
```

## Setup Instructions

### 1️⃣ Prerequisites

- **Python ≥ 3.11**
- **PowerShell** (Windows) or any terminal on Linux/macOS
- Optional: NVIDIA GPU with CUDA (the code will fall back to CPU automatically)

### 2️⃣ Create a virtual environment and install dependencies

```powershell
# From the repository root (agri-quality-classifier)
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # PowerShell
# or source .venv/bin/activate on Unix
pip install -r requirements.txt
```

### 3️⃣ Download the dataset

Follow the instructions in `download_dataset.py` (already provided in the parent folder) to download and extract the Kaggle dataset. After extraction you should have the folder:
```
data/raw/               # contains class sub‑folders like Apple__Healthy, …
```
The `.gitignore` file already excludes the raw data and Kaggle credentials from version control.

### 4️⃣ Train the model

```powershell
python train.py
```
Training outputs:
- `artifacts/best_model.pth` – the checkpoint containing weights and class mapping.
- `reports/metrics.json` – per‑epoch loss/accuracy.
- `reports/confusion_matrix.png` – confusion matrix on the test set.

### 5️⃣ Evaluate the model (optional)

```powershell
python evaluate.py
```
The script reads the checkpoint from `artifacts/best_model.pth` and writes evaluation metrics to `reports/metrics.json` and a confusion‑matrix image.

### 6️⃣ Run the dashboard

```powershell
streamlit run app.py
```
Upload an image, click **Analyze Image**, and view the prediction, confidence, and top‑3 probabilities.

### 7️⃣ Command‑line inference

```powershell
python predict.py --image path\to\image.jpg
```
The script prints the predicted class, confidence, and the top‑3 alternatives.

## Key Implementation Details

- **Dataset filtering** – `src/data.py` inspects the extracted directory, selects only the five fruits (both conditions) and builds a deterministic train/val/test split while preserving class ratios.
- **Class mapping** – The checkpoint stores a JSON mapping `{index: "Class_Name"}` ensuring the same ordering is used for training, evaluation, and inference.
- **Robustness** – Missing Kaggle credentials, network failures, or absent classes are reported with clear messages; the original dataset is never altered.
- **Evaluation** – Accuracy, macro/weighted precision, recall, F1‑score, and a confusion matrix are computed using `sklearn.metrics`.
- **Streamlit UI** – Shows a polished, minimal dashboard with top‑3 predictions, confidence bars, and helpful notes about model limitations.

## Limitations & Future Work

- The model only predicts the ten selected classes; any other fruit/vegetable will be classified as *unknown* (low confidence warning).
- No out‑of‑distribution detection beyond confidence threshold.
- No GPU‑specific optimisations beyond standard CUDA usage.
- Dataset size may still be modest; further data augmentation could improve performance.

---

*Happy coding and good luck with your college project!*
