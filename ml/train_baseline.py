"""
Training script for the Baseline Cervical Cell Classification Model.

Features:
- Reproducible with fixed random seed (default: 42).
- Uses PyTorch CrossEntropyLoss and Adam optimizer.
- Evaluates on validation split after each epoch.
- Saves the best checkpoint to models/baseline_cnn.pth based on validation performance.
- Saves training history to outputs/training_history.json.
- Computes comprehensive test evaluation and 6x6 confusion matrix.
- Saves test evaluation report to outputs/test_evaluation.json.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.dataset import get_dataloaders, CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS
from ml.model import get_model

DATA_DIR = PROJECT_ROOT / "data" / "cells"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DEFAULT_SEED = 42

def set_seed(seed: int = DEFAULT_SEED):
    """Ensure fully deterministic runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    """Run one epoch of training with batch progress logging."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    total_batches = len(dataloader)

    for batch_idx, (images, labels) in enumerate(dataloader, 1):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if batch_idx % 40 == 0 or batch_idx == total_batches:
            batch_loss = running_loss / total
            batch_acc = correct / total
            print(f"  [Epoch {epoch:02d}/{total_epochs:02d} | Batch {batch_idx:03d}/{total_batches:03d}] "
                  f"Train Loss: {batch_loss:.4f} | Train Acc: {batch_acc*100:.2f}%")

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    """Evaluate model loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    split_loss = running_loss / total
    split_acc = correct / total
    return split_loss, split_acc

def full_test_evaluation(model, dataloader, criterion, device):
    """Compute detailed evaluation on test split including confusion matrix and per-class stats."""
    model.eval()
    running_loss = 0.0
    total = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(labels.cpu().tolist())
            total += labels.size(0)

    test_loss = running_loss / total
    test_acc = sum(p == t for p, t in zip(all_preds, all_targets)) / total

    # 6x6 Confusion Matrix: rows = Ground Truth, columns = Predicted
    num_classes = len(CLASS_NAMES)
    cm = [[0] * num_classes for _ in range(num_classes)]
    for target, pred in zip(all_targets, all_preds):
        cm[target][pred] += 1

    # Per-class metrics
    class_metrics = {}
    for i, class_name in enumerate(CLASS_NAMES):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(num_classes) if r != i)
        fn = sum(cm[i][c] for c in range(num_classes) if c != i)
        support = sum(cm[i])  # total ground truth for class i
        pred_count = sum(cm[r][i] for r in range(num_classes))  # total predicted as class i

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        class_metrics[class_name] = {
            "support_ground_truth": support,
            "predicted_count": pred_count,
            "true_positives": tp,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
        }

    return {
        "test_loss": round(test_loss, 4),
        "test_accuracy": round(test_acc, 4),
        "confusion_matrix": cm,
        "class_metrics": class_metrics,
        "class_names": CLASS_NAMES,
    }

def print_evaluation_report(eval_results):
    """Format and print test evaluation report and confusion matrix table."""
    cm = eval_results["confusion_matrix"]
    metrics = eval_results["class_metrics"]

    print("\n" + "=" * 70)
    print("FINAL TEST SET EVALUATION REPORT")
    print("=" * 70)
    print(f"Overall Test Loss:     {eval_results['test_loss']:.4f}")
    print(f"Overall Test Accuracy: {eval_results['test_accuracy']*100:.2f}%\n")

    print("-" * 70)
    print("Per-Class Metrics on Unseen Test Slides:")
    print("-" * 70)
    hdr = f"{'Class':10} | {'Support':>7} | {'Predicted':>9} | {'Precision':>9} | {'Recall':>8} | {'F1-Score':>8}"
    print(hdr)
    print("-" * len(hdr))
    for name in CLASS_NAMES:
        m = metrics[name]
        print(f"{name:10} | {m['support_ground_truth']:7d} | {m['predicted_count']:9d} | "
              f"{m['precision']*100:8.2f}% | {m['recall']*100:7.2f}% | {m['f1_score']:8.4f}")

    print("\n" + "-" * 70)
    print("Confusion Matrix (Rows = Ground Truth, Columns = Predicted):")
    print("-" * 70)
    header_cols = " | ".join(f"{c:>7}" for c in CLASS_NAMES)
    print(f"{'Actual':10} | {header_cols}")
    print("-" * (13 + len(header_cols)))
    for i, name in enumerate(CLASS_NAMES):
        row_str = " | ".join(f"{cm[i][j]:7d}" for j in range(len(CLASS_NAMES)))
        print(f"{name:10} | {row_str}")
    print("=" * 70 + "\n")

def run_sanity_check(device):
    """Run a quick sanity check: loads 1 batch, verifies forward & backward pass."""
    print("=" * 60)
    print("RUNNING SANITY CHECK (1 Batch Forward & Backward Pass)")
    print("=" * 60)

    dataloaders, dataset_sizes = get_dataloaders(DATA_DIR, batch_size=8, num_workers=0)
    train_loader = dataloaders["train"]

    images, labels = next(iter(train_loader))
    print(f"Batch shape (Images): {images.shape} (Expected: [8, 3, 224, 224])")
    print(f"Batch shape (Labels): {labels.shape} (Expected: [8])")
    print(f"Labels in batch: {labels.tolist()}")
    print(f"Classes represented: {[CLASS_NAMES[idx] for idx in labels.tolist()]}")

    model = get_model(num_classes=len(CLASS_NAMES)).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    print(f"Logits shape: {outputs.shape} (Expected: [8, 6])")

    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)
    print(f"Initial loss value: {loss.item():.4f}")

    optimizer = Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Backward pass & optimizer step: SUCCESS")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Train baseline CNN for cervical cell classification.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed (default: 42)")
    parser.add_argument("--sanity-check", action="store_true", help="Run sanity check on 1 batch and exit")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (threads={torch.get_num_threads()})")

    if args.sanity_check:
        run_sanity_check(device)
        return

    # Prepare directories
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    dataloaders, dataset_sizes = get_dataloaders(DATA_DIR, batch_size=args.batch_size, num_workers=0)
    print("Dataset sizes:", dataset_sizes)

    # Initialize model, loss, optimizer
    model = get_model(num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    history = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "random_seed": args.seed,
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_times": []
    }

    print("\n" + "=" * 60)
    print(f"Starting Training: {args.epochs} Epochs | Batch Size: {args.batch_size} | LR: {args.lr}")
    print("=" * 60)
    total_start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        print(f"\n--- Epoch {epoch:02d}/{args.epochs:02d} ---")
        train_loss, train_acc = train_epoch(model, dataloaders["train"], criterion, optimizer, device, epoch, args.epochs)
        val_loss, val_acc = evaluate(model, dataloaders["val"], criterion, device)
        epoch_duration = time.time() - epoch_start

        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["epoch_times"].append(round(epoch_duration, 1))

        print(
            f"--> Epoch [{epoch:02d}/{args.epochs:02d}] Summary: "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
            f"Time: {epoch_duration:.1f}s"
        )

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = MODELS_DIR / "baseline_cnn.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "class_names": CLASS_NAMES,
                "class_to_idx": CLASS_TO_IDX,
            }, save_path)
            print(f"    * New best validation accuracy: {val_acc*100:.2f}% -> Saved to {save_path.name}")

    total_duration = time.time() - total_start_time
    print("\n" + "=" * 60)
    print(f"Training Complete in {total_duration/60:.2f} minutes | Best Val Acc: {best_val_acc*100:.2f}%")
    print("=" * 60)

    # Save training history
    history["best_val_acc"] = round(best_val_acc, 4)
    history["total_training_time_seconds"] = round(total_duration, 1)

    # Evaluate best saved model on test set
    print("\nLoading best checkpoint for evaluation on unseen test split...")
    best_checkpoint = torch.load(MODELS_DIR / "baseline_cnn.pth", weights_only=True)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    test_results = full_test_evaluation(model, dataloaders["test"], criterion, device)
    print_evaluation_report(test_results)

    # Merge history and test results
    history["test_results"] = test_results

    with open(OUTPUTS_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {OUTPUTS_DIR / 'training_history.json'}")

    with open(OUTPUTS_DIR / "test_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)
    print(f"Saved test evaluation to {OUTPUTS_DIR / 'test_evaluation.json'}")

if __name__ == "__main__":
    main()
