"""Augmentation pipelines for SimCLR, FixMatch weak, and FixMatch strong."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import RandAugment


class SimCLRAugmentation(nn.Module):
    """Produces two independently augmented views of the same image."""

    def __init__(self, input_size: int = 64) -> None:
        super().__init__()
        color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=int(0.1 * input_size) | 1, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # x is a PIL image or tensor; apply transform twice for two views
        return self.transform(x), self.transform(x)  # type: ignore[return-value]


class WeakAugmentation(nn.Module):
    """Weak augmentation used for FixMatch pseudo-label generation."""

    def __init__(self, input_size: int = 64) -> None:
        super().__init__()
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(input_size, scale=(0.875, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.transform(x)  # type: ignore[return-value]


class StrongAugmentation(nn.Module):
    """Strong augmentation (RandAugment) used for FixMatch consistency loss."""

    def __init__(self, input_size: int = 64, num_ops: int = 2, magnitude: int = 10) -> None:
        super().__init__()
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(input_size, scale=(0.875, 1.0)),
                transforms.RandomHorizontalFlip(),
                RandAugment(num_ops=num_ops, magnitude=magnitude),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.transform(x)  # type: ignore[return-value]


class FixMatchAugmentation(nn.Module):
    """Produces (weak_view, strong_view) pair for a single unlabelled image."""

    def __init__(self, input_size: int = 64) -> None:
        super().__init__()
        self.weak = WeakAugmentation(input_size)
        self.strong = StrongAugmentation(input_size)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.weak(x), self.strong(x)
