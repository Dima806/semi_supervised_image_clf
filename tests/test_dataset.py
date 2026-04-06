"""Tests for dataset.py using fake in-memory data."""

from __future__ import annotations

import torch
from torch.utils.data import TensorDataset

from semi_supervised_image_clf.dataset import sample_label_fraction


def _make_labelled_dataset(n_per_class: int = 50, num_classes: int = 10) -> TensorDataset:
    """Create a fake labelled dataset with n_per_class images per class."""
    total = n_per_class * num_classes
    imgs = torch.randn(total, 3, 64, 64)
    labels = torch.arange(num_classes).repeat_interleave(n_per_class)
    return TensorDataset(imgs, labels)


def test_sample_label_fraction_sizes() -> None:
    ds = _make_labelled_dataset(50, 10)
    n_labels = 5
    labelled, unlabelled = sample_label_fraction(
        ds, labels_per_class=n_labels, num_classes=10, seed=42
    )
    assert len(labelled) == n_labels * 10
    assert len(unlabelled) == len(ds) - n_labels * 10


def test_sample_label_fraction_class_balance() -> None:
    ds = _make_labelled_dataset(50, 10)
    labelled, _ = sample_label_fraction(ds, labels_per_class=5, num_classes=10, seed=0)
    class_counts: dict[int, int] = {}
    for i in range(len(labelled)):
        _, lbl = labelled[i]
        lbl_int = int(lbl)
        class_counts[lbl_int] = class_counts.get(lbl_int, 0) + 1
    for cls, count in class_counts.items():
        assert count == 5, f"Class {cls} has {count} samples, expected 5"


def test_sample_label_fraction_reproducible() -> None:
    ds = _make_labelled_dataset(50, 10)
    lab1, _ = sample_label_fraction(ds, labels_per_class=5, num_classes=10, seed=99)
    lab2, _ = sample_label_fraction(ds, labels_per_class=5, num_classes=10, seed=99)
    indices1 = lab1.indices  # type: ignore[attr-defined]
    indices2 = lab2.indices  # type: ignore[attr-defined]
    assert indices1 == indices2


def test_sample_label_fraction_different_seeds() -> None:
    ds = _make_labelled_dataset(50, 10)
    lab1, _ = sample_label_fraction(ds, labels_per_class=10, num_classes=10, seed=0)
    lab2, _ = sample_label_fraction(ds, labels_per_class=10, num_classes=10, seed=1)
    assert lab1.indices != lab2.indices  # type: ignore[attr-defined]
