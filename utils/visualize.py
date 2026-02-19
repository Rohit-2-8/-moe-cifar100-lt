"""
Visualization utilities for CIFAR-100-LT project.
Generates publication-quality comparison plots.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt


def set_style():
    """Set a clean, modern plot style."""
    plt.style.use("seaborn-v0_8-whitegrid") if "seaborn-v0_8-whitegrid" in plt.style.available else None
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.2,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def plot_class_distribution(class_counts, save_path):
    """Plot the long-tail class distribution."""
    set_style()
    fig, ax = plt.subplots(figsize=(12, 5))

    sorted_counts = sorted(class_counts, reverse=True)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(sorted_counts)))

    ax.bar(range(len(sorted_counts)), sorted_counts, color=colors, edgecolor="none")
    ax.set_xlabel("Class Index (sorted by frequency)")
    ax.set_ylabel("Number of Training Samples")
    ax.set_title("CIFAR-100-LT Class Distribution (Imbalance Ratio = 100)")
    ax.set_xlim(-1, len(sorted_counts))

    # Add annotations
    ax.annotate(f"Max: {max(sorted_counts)}", xy=(0, max(sorted_counts)),
                fontsize=10, color="darkgreen", fontweight="bold")
    ax.annotate(f"Min: {min(sorted_counts)}",
                xy=(len(sorted_counts) - 1, min(sorted_counts)),
                fontsize=10, color="darkred", fontweight="bold",
                xytext=(-40, 20), textcoords="offset points")

    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_training_curves(histories, save_path):
    """
    Plot training loss and accuracy curves for both models.

    Args:
        histories: dict of {model_name: {"train_loss": [...], "train_acc": [...],
                                         "test_acc": [...]}}
    """
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {"ResNet-20": "#2196F3", "MoE-ResNet-20": "#FF5722"}

    # Training Loss
    for name, hist in histories.items():
        axes[0].plot(hist["train_loss"], label=name, color=colors.get(name, None), linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()

    # Training Accuracy
    for name, hist in histories.items():
        axes[1].plot(hist["train_acc"], label=name, color=colors.get(name, None), linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Training Accuracy")
    axes[1].legend()

    # Test Accuracy
    for name, hist in histories.items():
        axes[2].plot(hist["test_acc"], label=name, color=colors.get(name, None), linewidth=2)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy (%)")
    axes[2].set_title("Test Accuracy")
    axes[2].legend()

    plt.suptitle("Training Curves: ResNet-20 vs MoE-ResNet-20 on CIFAR-100-LT", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_group_comparison(group_accs, save_path):
    """
    Bar chart comparing Head / Medium / Tail accuracy between models.

    Args:
        group_accs: dict of {model_name: {"head": acc, "medium": acc, "tail": acc}}
    """
    set_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    groups = ["head", "medium", "tail"]
    group_labels = ["Head\n(many-shot)", "Medium\n(medium-shot)", "Tail\n(few-shot)"]
    model_names = list(group_accs.keys())

    x = np.arange(len(groups))
    width = 0.30
    colors = ["#2196F3", "#FF5722"]

    for i, name in enumerate(model_names):
        values = [group_accs[name][g] for g in groups]
        bars = ax.bar(x + i * width, values, width, label=name, color=colors[i],
                      edgecolor="white", linewidth=1.5)
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlabel("Class Group")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Head / Medium / Tail Accuracy Comparison")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(group_labels)
    ax.legend()
    ax.set_ylim(0, max(max(v.values()) for v in group_accs.values()) + 10)

    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_per_class_accuracy(per_class_accs, class_counts, save_path):
    """
    Scatter/line plot of per-class accuracy vs class frequency.

    Args:
        per_class_accs: dict of {model_name: [100 accuracy values]}
        class_counts: list of per-class sample counts
    """
    set_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {"ResNet-20": "#2196F3", "MoE-ResNet-20": "#FF5722"}

    # Sort classes by count (descending)
    sorted_indices = np.argsort(class_counts)[::-1]
    sorted_counts = [class_counts[i] for i in sorted_indices]

    for name, accs in per_class_accs.items():
        sorted_accs = [accs[i] for i in sorted_indices]
        ax.plot(range(len(sorted_accs)), sorted_accs, label=name,
                color=colors.get(name, None), alpha=0.7, linewidth=1.5)

    ax.set_xlabel("Class Index (sorted by training frequency, descending)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Per-Class Test Accuracy vs Training Set Size")
    ax.legend()

    # Add a secondary x-axis for sample counts
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    tick_positions = [0, len(sorted_counts) // 4, len(sorted_counts) // 2,
                      3 * len(sorted_counts) // 4, len(sorted_counts) - 1]
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels([f"{sorted_counts[i]} samples" for i in tick_positions], fontsize=8)
    ax2.set_xlabel("Training Samples per Class", fontsize=9)

    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_expert_utilization(expert_pcts, save_path):
    """Bar chart of expert utilization percentages."""
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    n = len(expert_pcts)
    colors = plt.cm.Set2(np.linspace(0, 1, n))

    bars = ax.bar(range(n), expert_pcts, color=colors, edgecolor="white", linewidth=1.5)
    ax.axhline(y=100.0 / n, color="red", linestyle="--", linewidth=1.5,
               label=f"Ideal ({100.0/n:.1f}%)")

    for bar, val in zip(bars, expert_pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xlabel("Expert Index")
    ax.set_ylabel("Selection Rate (%)")
    ax.set_title("MoE Expert Utilization on Test Set")
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"Expert {i}" for i in range(n)])
    ax.legend()

    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_summary_table(metrics, save_path):
    """Create a summary comparison table as an image."""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    columns = ["Metric", "ResNet-20", "MoE-ResNet-20", "Δ (MoE − Baseline)"]
    rows = []

    for metric_name in ["Overall Acc", "Head Acc", "Medium Acc", "Tail Acc"]:
        r_val = metrics["ResNet-20"].get(metric_name, 0)
        m_val = metrics["MoE-ResNet-20"].get(metric_name, 0)
        delta = m_val - r_val
        sign = "+" if delta >= 0 else ""
        rows.append([metric_name, f"{r_val:.2f}%", f"{m_val:.2f}%", f"{sign}{delta:.2f}%"])

    table = ax.table(cellText=rows, colLabels=columns, loc="center",
                     cellLoc="center", edges="horizontal")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)

    # Style header
    for j in range(len(columns)):
        table[0, j].set_facecolor("#37474F")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(1, len(rows) + 1):
        for j in range(len(columns)):
            table[i, j].set_facecolor("#ECEFF1" if i % 2 == 0 else "white")

    ax.set_title("Performance Comparison Summary", fontsize=14, fontweight="bold", pad=20)
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")
