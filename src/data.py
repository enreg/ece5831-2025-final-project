import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler, Subset
from torchvision import datasets, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

GTSRB_CLASSES = [
    "Speed limit (20km/h)",
    "Speed limit (30km/h)",
    "Speed limit (50km/h)",
    "Speed limit (60km/h)",
    "Speed limit (70km/h)",
    "Speed limit (80km/h)",
    "End of speed limit (80km/h)",
    "Speed limit (100km/h)",
    "Speed limit (120km/h)",
    "No passing",
    "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles over 3.5 metric tons prohibited",
    "No entry",
    "General caution",
    "Dangerous curve to the left",
    "Dangerous curve to the right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on the right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing by vehicles over 3.5 metric tons",
]


@dataclass
class DataConfig:
    data_root: Path
    img_size: int = 96
    batch_size: int = 64
    num_workers: int = 4
    val_split: float = 0.1
    seed: int = 42
    balanced_sampler: bool = False
    download: bool = True


def _normalize_transform():
    return transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def build_transforms(img_size: int, train: bool = True) -> transforms.Compose:
    base = [transforms.Resize((img_size, img_size))]
    if train:
        base = [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        ]
    base += [transforms.ToTensor(), _normalize_transform()]
    return transforms.Compose(base)


def _ensure_metadata(ds):
    # TorchVision's GTSRB dataset lacks `classes`/`targets`; add them so downstream code works.
    if not hasattr(ds, "classes"):
        ds.classes = GTSRB_CLASSES
    if not hasattr(ds, "targets"):
        ds.targets = [label for _, label in ds._samples]


def _compute_class_weights(targets, num_classes: int):
    counts = torch.zeros(num_classes)
    for t in targets:
        counts[t] += 1
    weights = 1.0 / torch.clamp(counts, min=1)
    return weights


class TransformSubset(Subset):
    def __init__(self, dataset, indices, transform):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
        img, target = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, target


def get_datasets(config: DataConfig):
    train_tfms = build_transforms(config.img_size, train=True)
    eval_tfms = build_transforms(config.img_size, train=False)

    base_train = datasets.GTSRB(
        root=config.data_root,
        split="train",
        download=config.download,
        transform=train_tfms,
    )
    _ensure_metadata(base_train)
    base_val = datasets.GTSRB(
        root=config.data_root,
        split="train",
        download=config.download,
        transform=eval_tfms,
    )
    _ensure_metadata(base_val)
    test_set = datasets.GTSRB(
        root=config.data_root,
        split="test",
        download=config.download,
        transform=eval_tfms,
    )
    _ensure_metadata(test_set)

    val_size = int(len(base_train) * config.val_split)
    train_size = len(base_train) - val_size
    generator = torch.Generator().manual_seed(config.seed)
    perm = torch.randperm(len(base_train), generator=generator).tolist()
    train_indices = perm[:train_size]
    val_indices = perm[train_size:]
    train_set = Subset(base_train, train_indices)
    val_set = Subset(base_val, val_indices)

    return train_set, val_set, test_set


def get_dataloaders(config: DataConfig):
    train_set, val_set, test_set = get_datasets(config)
    num_classes = len(train_set.dataset.classes)

    sampler = None
    if config.balanced_sampler:
        targets = getattr(train_set.dataset, "targets", None) or getattr(train_set.dataset, "labels", None)
        if targets is not None:
            weights = _compute_class_weights(targets, num_classes)
            sample_targets = torch.tensor(targets)[train_set.indices]
            sample_weights = weights[sample_targets]
            sampler = WeightedRandomSampler(sample_weights.double(), len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, num_classes


def sample_batch(loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    images, labels = next(iter(loader))
    return images, labels
