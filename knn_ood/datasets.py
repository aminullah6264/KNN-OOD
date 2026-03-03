from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CIFAR_STATS = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))


def _cifar_transform(train: bool):
    mean, std = CIFAR_STATS
    if train:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


def get_cifar10(root: str, train: bool):
    return datasets.CIFAR10(root=root, train=train, download=True, transform=_cifar_transform(train))


def get_ood_dataset(name: str, root: str):
    mean, std = CIFAR_STATS
    t = transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    root = Path(root)
    key = name.lower()
    if key == "svhn":
        return datasets.SVHN(root=str(root), split="test", download=True, transform=t)
    if key in {"lsun", "isun", "textures", "places365"}:
        return datasets.ImageFolder(root=str(root / key), transform=t)
    raise ValueError(f"Unsupported OOD dataset: {name}")


def make_loader(dataset, batch_size: int, num_workers: int, shuffle: bool = False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
