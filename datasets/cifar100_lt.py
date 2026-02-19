"""
CIFAR-100-LT: Long-Tailed version of CIFAR-100.

Creates an exponentially imbalanced training set while keeping the test set balanced.
"""
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from collections import Counter

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def get_long_tail_indices(targets, num_classes, imbalance_ratio, max_samples):
    """
    Compute per-class sample counts following an exponential decay:
        n_i = max_samples * (1/imbalance_ratio)^(i/(num_classes-1))

    Returns indices to keep and the per-class counts.
    """
    targets = np.array(targets)
    class_indices = {c: np.where(targets == c)[0] for c in range(num_classes)}

    # Sort classes by original count (all equal for CIFAR-100, but sort for generality)
    sorted_classes = sorted(class_indices.keys())

    per_class_counts = []
    selected_indices = []

    for i, cls in enumerate(sorted_classes):
        # Exponential decay
        n_i = int(max_samples * ((1.0 / imbalance_ratio) ** (i / (num_classes - 1))))
        n_i = max(n_i, 1)  # at least 1 sample
        per_class_counts.append(n_i)

        indices = class_indices[cls]
        np.random.seed(42 + cls)  # reproducible subsampling
        chosen = np.random.choice(indices, size=min(n_i, len(indices)), replace=False)
        selected_indices.extend(chosen.tolist())

    return selected_indices, per_class_counts


class CIFAR100LT(Dataset):
    """
    CIFAR-100 with Long-Tailed (exponential imbalance) training distribution.
    Test set remains balanced.
    """

    def __init__(self, root, train=True, imbalance_ratio=100, download=True, transform=None):
        self.train = train
        self.imbalance_ratio = imbalance_ratio

        # Load full CIFAR-100
        full_dataset = torchvision.datasets.CIFAR100(
            root=root, train=train, download=download, transform=None
        )

        if train:
            # Apply long-tail subsampling
            indices, self.class_counts = get_long_tail_indices(
                full_dataset.targets,
                num_classes=config.NUM_CLASSES,
                imbalance_ratio=imbalance_ratio,
                max_samples=config.MAX_SAMPLES_PER_CLASS,
            )
            self.data = [full_dataset.data[i] for i in indices]
            self.targets = [full_dataset.targets[i] for i in indices]
        else:
            self.data = list(full_dataset.data)
            self.targets = list(full_dataset.targets)
            self.class_counts = list(Counter(self.targets).values())

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]

        # Convert numpy array to PIL for transforms
        from PIL import Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def get_class_counts(self):
        """Return per-class sample counts (sorted by class index)."""
        counter = Counter(self.targets)
        return [counter.get(i, 0) for i in range(config.NUM_CLASSES)]

    def get_class_groups(self):
        """Split classes into Head / Medium / Tail groups."""
        counts = self.get_class_counts()
        sorted_classes = np.argsort(counts)[::-1]  # descending
        n = len(sorted_classes)
        head = sorted_classes[: n // 3].tolist()
        medium = sorted_classes[n // 3: 2 * n // 3].tolist()
        tail = sorted_classes[2 * n // 3:].tolist()
        return {"head": head, "medium": medium, "tail": tail}


def get_train_transform():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(config.CIFAR_MEAN, config.CIFAR_STD),
    ])


def get_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config.CIFAR_MEAN, config.CIFAR_STD),
    ])


def get_dataloaders(imbalance_ratio=None, batch_size=None):
    """Create train and test DataLoaders."""
    ir = imbalance_ratio or config.IMBALANCE_RATIO
    bs = batch_size or config.BATCH_SIZE

    train_dataset = CIFAR100LT(
        root=config.DATA_ROOT, train=True, imbalance_ratio=ir,
        download=True, transform=get_train_transform()
    )
    test_dataset = CIFAR100LT(
        root=config.DATA_ROOT, train=False,
        download=True, transform=get_test_transform()
    )

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False,
                             num_workers=config.NUM_WORKERS, pin_memory=True)

    return train_loader, test_loader, train_dataset
