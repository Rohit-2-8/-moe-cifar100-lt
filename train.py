"""
Training loop shared by both ResNet-20 and MoE-ResNet-20.
Supports standard CE and class-balanced CE losses.
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import config
from utils.metrics import compute_accuracy


def train_model(model, train_loader, test_loader, model_name="model",
                is_moe=False, device="cpu", epochs=None, class_counts=None):
    """
    Train a model and return training history.

    Args:
        model: nn.Module to train
        train_loader: training DataLoader
        test_loader: test DataLoader
        model_name: name for checkpoint saving
        is_moe: whether the model returns (output, aux_loss, gate_scores)
        device: 'cpu' or 'cuda'
        epochs: override config.EPOCHS
        class_counts: per-class sample counts (for future use with CB loss)

    Returns:
        history: dict with train_loss, train_acc, test_acc per epoch
    """
    epochs = epochs or config.EPOCHS
    model = model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
    )

    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # History
    history = {"train_loss": [], "train_acc": [], "test_acc": []}

    # Checkpoints directory
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    best_acc = 0.0

    print(f"\n{'='*60}")
    print(f"  Training {model_name}")
    print(f"  Epochs: {epochs} | Device: {device} | LR: {config.LEARNING_RATE}")
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for images, targets in pbar:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()

            if is_moe:
                outputs, aux_loss, _ = model(images)
                ce_loss = criterion(outputs, targets)
                loss = ce_loss + config.LOAD_BALANCE_COEFF * aux_loss
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

            pbar.set_postfix({
                "loss": f"{loss.item():.3f}",
                "acc": f"{correct/total*100:.1f}%",
            })

        scheduler.step()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total * 100.0
        test_acc = compute_accuracy(model, test_loader, device, is_moe=is_moe)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        # Save best checkpoint
        if test_acc > best_acc:
            best_acc = test_acc
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{model_name}_best.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_acc": test_acc,
            }, ckpt_path)

        print(f"  Epoch {epoch+1:3d}/{epochs} │ "
              f"Loss: {train_loss:.4f} │ "
              f"Train Acc: {train_acc:.2f}% │ "
              f"Test Acc: {test_acc:.2f}% │ "
              f"Best: {best_acc:.2f}% │ "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

    print(f"\n  ✓ {model_name} training complete. Best test accuracy: {best_acc:.2f}%\n")
    return history
