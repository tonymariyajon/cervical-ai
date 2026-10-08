"""
CRIC Cervical Cell Dataset and DataLoader utilities.

Loads cell crops from data/cells/{train,val,test}/{class_name}/.
Uses standard normalization and deterministic label indexing.
"""

from pathlib import Path
from typing import Dict, Tuple
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Explicit 6-class Bethesda order
CLASS_NAMES = ["Negative", "ASC-US", "ASC-H", "LSIL", "HSIL", "SCC"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}

def get_transforms() -> transforms.Compose:
    """Basic preprocessing: convert to Tensor and normalize.
    No augmentation is applied at this baseline stage.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        # Standard normalization for RGB computer vision models
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

class FixedImageFolder(datasets.ImageFolder):
    """ImageFolder with an explicitly enforced class-to-index mapping."""
    def find_classes(self, directory: str) -> Tuple[list, Dict[str, int]]:
        # Enforce exact CLASS_TO_IDX mapping rather than alphabetical sorting
        classes = CLASS_NAMES
        class_to_idx = CLASS_TO_IDX
        return classes, class_to_idx

def get_dataloaders(
    data_dir: Path,
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[Dict[str, DataLoader], Dict[str, int]]:
    """Creates DataLoaders for train, val, and test splits.

    Args:
        data_dir: Path to data/cells/ directory.
        batch_size: Number of images per batch (default: 32).
        num_workers: PyTorch DataLoader workers (default: 0 for compatibility).

    Returns:
        dataloaders: Dict containing 'train', 'val', 'test' DataLoaders.
        dataset_sizes: Dict containing counts of images per split.
    """
    data_dir = Path(data_dir)
    transform = get_transforms()

    dataloaders = {}
    dataset_sizes = {}

    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        dataset = FixedImageFolder(root=str(split_dir), transform=transform)
        is_train = (split == "train")

        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        dataset_sizes[split] = len(dataset)

    return dataloaders, dataset_sizes
