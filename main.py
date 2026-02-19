"""
Main entry point for the MoE vs Traditional Model comparison on CIFAR-100-LT.

Usage:
    python main.py                   # Train both models and evaluate
    python main.py --model resnet    # Train only ResNet-20
    python main.py --model moe       # Train only MoE-ResNet-20
    python main.py --evaluate-only   # Skip training, just evaluate from checkpoints
    python main.py --epochs 10       # Override number of epochs
"""
import argparse
import os
import sys
import json
import torch
import time

import config
from datasets.cifar100_lt import get_dataloaders
from models.resnet import ResNet20
from models.moe_resnet import MoEResNet20
from train import train_model
from evaluate import evaluate


def parse_args():
    parser = argparse.ArgumentParser(
        description="MoE vs Traditional Model on CIFAR-100-LT"
    )
    parser.add_argument("--model", type=str, default="both",
                        choices=["resnet", "moe", "both"],
                        help="Which model to train (default: both)")
    parser.add_argument("--epochs", type=int, default=None,
                        help=f"Number of training epochs (default: {config.EPOCHS})")
    parser.add_argument("--imbalance-ratio", type=int, default=None,
                        help=f"Imbalance ratio (default: {config.IMBALANCE_RATIO})")
    parser.add_argument("--batch-size", type=int, default=None,
                        help=f"Batch size (default: {config.BATCH_SIZE})")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Skip training, evaluate from existing checkpoints")
    parser.add_argument("--no-cuda", action="store_true",
                        help="Disable CUDA even if available")
    return parser.parse_args()


def main():
    args = parse_args()

    # ─── Device ───────────────────────────────────────────────────────────────
    if args.no_cuda:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n🖥  Device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print()

    # ─── Override config ──────────────────────────────────────────────────────
    epochs = args.epochs or config.EPOCHS
    ir = args.imbalance_ratio or config.IMBALANCE_RATIO
    bs = args.batch_size or config.BATCH_SIZE

    # ─── Data ─────────────────────────────────────────────────────────────────
    print("📂 Loading CIFAR-100-LT dataset...")
    train_loader, test_loader, train_dataset = get_dataloaders(
        imbalance_ratio=ir, batch_size=bs
    )
    class_counts = train_dataset.get_class_counts()
    total_train = len(train_dataset)
    print(f"   Training samples: {total_train:,} (balanced CIFAR-100 has 50,000)")
    print(f"   Max class: {max(class_counts)} samples | Min class: {min(class_counts)} samples")
    print(f"   Imbalance ratio: {max(class_counts) / max(min(class_counts), 1):.0f}x")
    print(f"   Test samples: 10,000 (balanced)")
    print()

    histories = {}

    if not args.evaluate_only:
        # ─── Train ResNet-20 ──────────────────────────────────────────────────
        if args.model in ("resnet", "both"):
            print("🏋️ Training ResNet-20 (baseline)...")
            resnet = ResNet20(num_classes=config.NUM_CLASSES)
            start = time.time()
            histories["ResNet-20"] = train_model(
                resnet, train_loader, test_loader,
                model_name="ResNet-20", is_moe=False,
                device=device, epochs=epochs, class_counts=class_counts,
            )
            resnet_time = time.time() - start
            print(f"  ⏱  ResNet-20 training took {resnet_time:.1f}s")

        # ─── Train MoE-ResNet-20 ─────────────────────────────────────────────
        if args.model in ("moe", "both"):
            print("🏋️ Training MoE-ResNet-20...")
            moe_resnet = MoEResNet20(num_classes=config.NUM_CLASSES)
            start = time.time()
            histories["MoE-ResNet-20"] = train_model(
                moe_resnet, train_loader, test_loader,
                model_name="MoE-ResNet-20", is_moe=True,
                device=device, epochs=epochs, class_counts=class_counts,
            )
            moe_time = time.time() - start
            print(f"  ⏱  MoE-ResNet-20 training took {moe_time:.1f}s")

        # Save histories
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        history_path = os.path.join(config.RESULTS_DIR, "histories.json")
        with open(history_path, "w") as f:
            json.dump(histories, f, indent=2)
        print(f"\n  Saved training histories to {history_path}")

    else:
        # Load existing histories if available
        history_path = os.path.join(config.RESULTS_DIR, "histories.json")
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                histories = json.load(f)

    # ─── Evaluate & Compare ───────────────────────────────────────────────────
    if args.model == "both" or args.evaluate_only:
        metrics = evaluate(device=device, histories=histories or None)
    else:
        print(f"\n  ℹ Skipping comparison (only trained {args.model}). "
              f"Run with --model both to compare.")

    print("\n🎉 Done!")


if __name__ == "__main__":
    main()
