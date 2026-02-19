"""
Evaluation metrics for CIFAR-100-LT.
"""
import numpy as np
import torch
from collections import defaultdict


def compute_accuracy(model, dataloader, device, is_moe=False):
    """Compute overall top-1 accuracy."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)

            if is_moe:
                outputs, _, _ = model(images)
            else:
                outputs = model(images)

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    return correct / total * 100.0


def compute_per_class_accuracy(model, dataloader, device, num_classes=100, is_moe=False):
    """Compute per-class accuracy."""
    model.eval()
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)

            if is_moe:
                outputs, _, _ = model(images)
            else:
                outputs = model(images)

            _, predicted = outputs.max(1)

            for i in range(targets.size(0)):
                label = targets[i].item()
                class_total[label] += 1
                if predicted[i].item() == label:
                    class_correct[label] += 1

    per_class_acc = []
    for c in range(num_classes):
        if class_total[c] > 0:
            per_class_acc.append(class_correct[c] / class_total[c] * 100.0)
        else:
            per_class_acc.append(0.0)

    return per_class_acc


def compute_group_accuracy(per_class_acc, class_groups):
    """
    Compute accuracy for Head / Medium / Tail groups.

    Args:
        per_class_acc: list of per-class accuracies
        class_groups: dict with keys 'head', 'medium', 'tail' → list of class indices
    """
    group_acc = {}
    for group_name, class_indices in class_groups.items():
        accs = [per_class_acc[c] for c in class_indices]
        group_acc[group_name] = np.mean(accs) if accs else 0.0
    return group_acc


def compute_expert_utilization(model, dataloader, device):
    """Compute how often each expert is selected (for MoE models)."""
    model.eval()
    expert_counts = None

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            _, _, gate_scores = model(images)

            # Count top-k selections
            top_k_indices = torch.topk(gate_scores, k=2, dim=-1).indices  # (B, K)

            if expert_counts is None:
                expert_counts = torch.zeros(gate_scores.size(1), device=device)

            for k in range(top_k_indices.size(1)):
                for idx in top_k_indices[:, k]:
                    expert_counts[idx.item()] += 1

    if expert_counts is not None:
        expert_counts = expert_counts.cpu().numpy()
        expert_counts = expert_counts / expert_counts.sum() * 100  # percentages
    return expert_counts
