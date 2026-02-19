"""
Centralized configuration for the MoE vs Traditional model comparison on CIFAR-100-LT.
Adjusted for a lightweight trial run.
"""
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# ─── Dataset ──────────────────────────────────────────────────────────────────
NUM_CLASSES = 100
IMBALANCE_RATIO = 100        # max_class / min_class sample ratio
MAX_SAMPLES_PER_CLASS = 500  # CIFAR-100 has 500 train samples per class

# ─── Training ─────────────────────────────────────────────────────────────────
EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
NUM_WORKERS = 2

# ─── MoE ──────────────────────────────────────────────────────────────────────
NUM_EXPERTS = 4
TOP_K = 2
EXPERT_HIDDEN_DIM = 128
LOAD_BALANCE_COEFF = 0.01   # weight for auxiliary load-balancing loss

# ─── ResNet ───────────────────────────────────────────────────────────────────
RESNET_DEPTH = 20            # ResNet-20 for fast training

# ─── Data Augmentation (CIFAR standard) ──────────────────────────────────────
CIFAR_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR_STD  = (0.2675, 0.2565, 0.2761)
