"""
Evaluation script: loads best checkpoints, computes metrics, generates comparison.
"""
import os
import json
import torch
import numpy as np

import config
from models.resnet import ResNet20
from models.moe_resnet import MoEResNet20
from datasets.cifar100_lt import get_dataloaders
from utils.metrics import (
    compute_accuracy,
    compute_per_class_accuracy,
    compute_group_accuracy,
    compute_expert_utilization,
)
from utils.visualize import (
    plot_class_distribution,
    plot_training_curves,
    plot_group_comparison,
    plot_per_class_accuracy,
    plot_expert_utilization,
    plot_summary_table,
)


def load_model(model, model_name, device):
    """Load best checkpoint for a model."""
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{model_name}_best.pth")
    if not os.path.exists(ckpt_path):
        print(f"  ⚠ Checkpoint not found: {ckpt_path}")
        return model, 0.0
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"  ✓ Loaded {model_name} (epoch {checkpoint['epoch']}, "
          f"test acc: {checkpoint['test_acc']:.2f}%)")
    return model, checkpoint["test_acc"]


def evaluate(device="cpu", histories=None):
    """Run full evaluation and generate all comparison plots."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("  EVALUATION & COMPARISON")
    print("=" * 60)

    # ─── Load data ────────────────────────────────────────────────────────────
    _, test_loader, train_dataset = get_dataloaders()
    class_counts = train_dataset.get_class_counts()
    class_groups = train_dataset.get_class_groups()

    # ─── Plot class distribution ──────────────────────────────────────────────
    print("\n📊 Generating class distribution plot...")
    plot_class_distribution(class_counts,
                            os.path.join(config.RESULTS_DIR, "class_distribution.png"))

    # ─── Load models ──────────────────────────────────────────────────────────
    print("\n📦 Loading model checkpoints...")
    resnet = ResNet20(num_classes=config.NUM_CLASSES)
    moe_resnet = MoEResNet20(num_classes=config.NUM_CLASSES)

    resnet, resnet_best_acc = load_model(resnet, "ResNet-20", device)
    moe_resnet, moe_best_acc = load_model(moe_resnet, "MoE-ResNet-20", device)

    # ─── Compute metrics ──────────────────────────────────────────────────────
    print("\n📐 Computing metrics...")

    resnet_overall = compute_accuracy(resnet, test_loader, device, is_moe=False)
    moe_overall = compute_accuracy(moe_resnet, test_loader, device, is_moe=True)

    resnet_per_class = compute_per_class_accuracy(resnet, test_loader, device,
                                                    config.NUM_CLASSES, is_moe=False)
    moe_per_class = compute_per_class_accuracy(moe_resnet, test_loader, device,
                                                 config.NUM_CLASSES, is_moe=True)

    resnet_groups = compute_group_accuracy(resnet_per_class, class_groups)
    moe_groups = compute_group_accuracy(moe_per_class, class_groups)

    # ─── Print results ────────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print(f"  {'Metric':<20} {'ResNet-20':>12} {'MoE-ResNet-20':>15}")
    print("─" * 50)
    print(f"  {'Overall Acc':<20} {resnet_overall:>11.2f}% {moe_overall:>14.2f}%")
    print(f"  {'Head Acc':<20} {resnet_groups['head']:>11.2f}% {moe_groups['head']:>14.2f}%")
    print(f"  {'Medium Acc':<20} {resnet_groups['medium']:>11.2f}% {moe_groups['medium']:>14.2f}%")
    print(f"  {'Tail Acc':<20} {resnet_groups['tail']:>11.2f}% {moe_groups['tail']:>14.2f}%")
    print("─" * 50)

    delta = moe_overall - resnet_overall
    sign = "+" if delta >= 0 else ""
    print(f"\n  Δ Overall (MoE − Baseline): {sign}{delta:.2f}%")

    # ─── Generate plots ───────────────────────────────────────────────────────
    print("\n📈 Generating comparison plots...")

    # Training curves
    if histories:
        plot_training_curves(histories,
                             os.path.join(config.RESULTS_DIR, "training_curves.png"))

    # Group accuracy comparison
    group_accs = {
        "ResNet-20": resnet_groups,
        "MoE-ResNet-20": moe_groups,
    }
    plot_group_comparison(group_accs,
                          os.path.join(config.RESULTS_DIR, "group_comparison.png"))

    # Per-class accuracy
    per_class_accs = {
        "ResNet-20": resnet_per_class,
        "MoE-ResNet-20": moe_per_class,
    }
    plot_per_class_accuracy(per_class_accs, class_counts,
                            os.path.join(config.RESULTS_DIR, "per_class_accuracy.png"))

    # Expert utilization
    print("  Computing expert utilization...")
    expert_pcts = compute_expert_utilization(moe_resnet, test_loader, device)
    if expert_pcts is not None:
        plot_expert_utilization(expert_pcts,
                                os.path.join(config.RESULTS_DIR, "expert_utilization.png"))

    # Summary table
    metrics = {
        "ResNet-20": {
            "Overall Acc": resnet_overall,
            "Head Acc": resnet_groups["head"],
            "Medium Acc": resnet_groups["medium"],
            "Tail Acc": resnet_groups["tail"],
        },
        "MoE-ResNet-20": {
            "Overall Acc": moe_overall,
            "Head Acc": moe_groups["head"],
            "Medium Acc": moe_groups["medium"],
            "Tail Acc": moe_groups["tail"],
        },
    }
    plot_summary_table(metrics, os.path.join(config.RESULTS_DIR, "summary_table.png"))

    # Save metrics as JSON
    with open(os.path.join(config.RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {os.path.join(config.RESULTS_DIR, 'metrics.json')}")

    print(f"\n✅ All results saved to {config.RESULTS_DIR}")
    return metrics
