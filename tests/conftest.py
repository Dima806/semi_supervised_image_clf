"""Shared pytest fixtures using synthetic data (no real STL-10 download needed)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

NUM_CLASSES = 10
IMG_SIZE = 64
FEATURE_DIM = 512


# ---------------------------------------------------------------------------
# Tiny in-memory image datasets
# ---------------------------------------------------------------------------


class _FakePILDataset(torch.utils.data.Dataset):  # type: ignore[type-arg]
    """Returns (PIL Image, label) pairs — compatible with torchvision transforms."""

    def __init__(
        self, n: int = 32, img_size: int = IMG_SIZE, num_classes: int = NUM_CLASSES
    ) -> None:
        self.n = n
        self.img_size = img_size
        self.num_classes = num_classes
        rng = np.random.default_rng(0)
        self._imgs = rng.integers(0, 255, (n, img_size, img_size, 3), dtype=np.uint8)
        self._labels = rng.integers(0, num_classes, n, dtype=np.int64)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        return Image.fromarray(self._imgs[idx]), int(self._labels[idx])


class _FakeTensorDataset(torch.utils.data.Dataset):  # type: ignore[type-arg]
    """Returns (Tensor image [C,H,W], label) pairs."""

    def __init__(
        self, n: int = 32, img_size: int = IMG_SIZE, num_classes: int = NUM_CLASSES
    ) -> None:
        self.imgs = torch.randn(n, 3, img_size, img_size)
        self.labels = torch.randint(0, num_classes, (n,))

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.imgs[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_pil_dataset() -> _FakePILDataset:
    return _FakePILDataset(n=40)


@pytest.fixture()
def fake_tensor_dataset() -> _FakeTensorDataset:
    return _FakeTensorDataset(n=40)


@pytest.fixture()
def labelled_loader() -> DataLoader:  # type: ignore[type-arg]
    ds = _FakeTensorDataset(n=32)
    return DataLoader(ds, batch_size=8, shuffle=False)


@pytest.fixture()
def unlabelled_loader() -> DataLoader:  # type: ignore[type-arg]
    """Loader that returns (weak_view, strong_view) pairs of tensor images."""
    imgs_weak = torch.randn(64, 3, IMG_SIZE, IMG_SIZE)
    imgs_strong = torch.randn(64, 3, IMG_SIZE, IMG_SIZE)
    ds = TensorDataset(imgs_weak, imgs_strong)
    return DataLoader(ds, batch_size=16, shuffle=False)


@pytest.fixture()
def test_loader() -> DataLoader:  # type: ignore[type-arg]
    ds = _FakeTensorDataset(n=32)
    return DataLoader(ds, batch_size=16, shuffle=False)
