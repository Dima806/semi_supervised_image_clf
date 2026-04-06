"""Tests for augmentations.py."""

from __future__ import annotations

import torch
from PIL import Image

from semi_supervised_image_clf.augmentations import (
    FixMatchAugmentation,
    SimCLRAugmentation,
    StrongAugmentation,
    WeakAugmentation,
)


def _random_pil(size: int = 64) -> Image.Image:

    arr = (torch.randn(size, size, 3).numpy() * 127 + 128).clip(0, 255).astype("uint8")
    return Image.fromarray(arr)


def test_simclr_augmentation_returns_two_views() -> None:
    aug = SimCLRAugmentation(input_size=64)
    img = _random_pil()
    v1, v2 = aug(img)
    assert v1.shape == (3, 64, 64)
    assert v2.shape == (3, 64, 64)
    # Two views should generally differ due to random augmentation
    assert not torch.allclose(v1, v2)


def test_weak_augmentation_output_shape() -> None:
    aug = WeakAugmentation(input_size=64)
    img = _random_pil()
    out = aug(img)
    assert out.shape == (3, 64, 64)


def test_strong_augmentation_output_shape() -> None:
    aug = StrongAugmentation(input_size=64)
    img = _random_pil()
    out = aug(img)
    assert out.shape == (3, 64, 64)


def test_fixmatch_augmentation_returns_pair() -> None:
    aug = FixMatchAugmentation(input_size=64)
    img = _random_pil()
    weak, strong = aug(img)
    assert weak.shape == (3, 64, 64)
    assert strong.shape == (3, 64, 64)


def test_augmentations_work_for_different_sizes() -> None:
    for size in [32, 64]:
        aug = SimCLRAugmentation(input_size=size)
        img = _random_pil(size=96)  # larger source, resized by crop
        v1, v2 = aug(img)
        assert v1.shape == (3, size, size)
