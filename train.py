# train.py
"""Training script for the Agri Quality Classifier.
Usage:
    python train.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from config import TrainConfig, BEST_MODEL_PATH
from src.data import create_datasets
from src.model import build_mobilenet_v3_small
from src.utils import set_seed

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc

def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, _, class_to_idx = create_datasets(cfg)
    num_classes = len(class_to_idx)
    print(f"Number of classes: {num_classes}")
    model = build_mobilenet_v3_small(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    # Train only classifier parameters initially (feature extractor frozen in model definition)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    best_val_acc = 0.0
    epochs_no_improve = 0
    for epoch in range(1, cfg.num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{cfg.num_epochs} | Train loss: {train_loss:.4f} acc: {train_acc:.4f} | Val loss: {val_loss:.4f} acc: {val_acc:.4f}")
        # Early stopping based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            # Save checkpoint with index-to-label mapping
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_to_idx": {str(idx): label for label, idx in class_to_idx.items()},  # idx -> label
                "num_classes": num_classes,
                "epoch": epoch,
                "val_accuracy": val_acc,
            }
            BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, BEST_MODEL_PATH)
            print(f"Saved best model to {BEST_MODEL_PATH}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.early_stop_patience:
                print("Early stopping triggered.")
                break

if __name__ == "__main__":
    main()
