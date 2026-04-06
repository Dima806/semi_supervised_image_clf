"""Tests for fixmatch.py: PseudoLabelFilter, FixMatchUnlabelledDataset, and train_fixmatch."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from semi_supervised_image_clf.config import FixMatchConfig
from semi_supervised_image_clf.fixmatch import (
    FixMatchUnlabelledDataset,
    PseudoLabelFilter,
    train_fixmatch,
)
from semi_supervised_image_clf.model import ResNet18Classifier

# ---------------------------------------------------------------------------
# PseudoLabelFilter tests
# ---------------------------------------------------------------------------


def test_pseudo_label_filter_all_above_threshold() -> None:
    plf = PseudoLabelFilter(threshold=0.5)
    probs = torch.zeros(4, 10)
    probs[:, 3] = 0.9  # class 3 has probability 0.9
    mask, pseudo_labels = plf.filter(probs)
    assert mask.all()
    assert (pseudo_labels == 3).all()


def test_pseudo_label_filter_all_below_threshold() -> None:
    plf = PseudoLabelFilter(threshold=0.95)
    probs = torch.full((4, 10), 0.1)  # uniform; max = 0.1
    mask, _ = plf.filter(probs)
    assert not mask.any()


def test_pseudo_label_filter_partial() -> None:
    plf = PseudoLabelFilter(threshold=0.8)
    probs = torch.zeros(4, 10)
    probs[0, 2] = 0.9  # above threshold
    probs[1, 5] = 0.7  # below threshold
    probs[2, 1] = 0.85  # above threshold
    probs[3, 8] = 0.5  # below threshold
    mask, pseudo_labels = plf.filter(probs)
    assert mask.tolist() == [True, False, True, False]
    assert pseudo_labels[0].item() == 2
    assert pseudo_labels[2].item() == 1


def test_pseudo_label_filter_returns_correct_shapes() -> None:
    plf = PseudoLabelFilter(threshold=0.7)
    probs = torch.softmax(torch.randn(8, 10), dim=-1)
    mask, pseudo_labels = plf.filter(probs)
    assert mask.shape == (8,)
    assert pseudo_labels.shape == (8,)
    assert mask.dtype == torch.bool
    assert pseudo_labels.dtype == torch.int64


# ---------------------------------------------------------------------------
# FixMatchUnlabelledDataset tests
# ---------------------------------------------------------------------------


def test_fixmatch_dataset_returns_pair(fake_pil_dataset) -> None:  # type: ignore[no-untyped-def]
    ds = FixMatchUnlabelledDataset(fake_pil_dataset, input_size=64)
    weak, strong = ds[0]
    assert weak.shape == (3, 64, 64)
    assert strong.shape == (3, 64, 64)


def test_fixmatch_dataset_len(fake_pil_dataset) -> None:  # type: ignore[no-untyped-def]
    ds = FixMatchUnlabelledDataset(fake_pil_dataset, input_size=64)
    assert len(ds) == len(fake_pil_dataset)


def test_fixmatch_dataset_views_differ(fake_pil_dataset) -> None:  # type: ignore[no-untyped-def]
    ds = FixMatchUnlabelledDataset(fake_pil_dataset, input_size=64)
    weak, strong = ds[0]
    # Weak and strong augmentations should produce different outputs
    assert not torch.allclose(weak, strong)


# ---------------------------------------------------------------------------
# train_fixmatch
# ---------------------------------------------------------------------------


def _labelled_loader(n: int = 32) -> DataLoader:  # type: ignore[type-arg]
    imgs = torch.randn(n, 3, 64, 64)
    labels = torch.randint(0, 10, (n,))
    return DataLoader(TensorDataset(imgs, labels), batch_size=8, drop_last=True)


def _unlabelled_loader(n: int = 64) -> DataLoader:  # type: ignore[type-arg]
    weak = torch.randn(n, 3, 64, 64)
    strong = torch.randn(n, 3, 64, 64)
    return DataLoader(TensorDataset(weak, strong), batch_size=16, drop_last=True)


def _fixmatch_config() -> FixMatchConfig:
    cfg = FixMatchConfig()
    cfg.smoke_test.max_epochs = 1
    cfg.training.lambda_u = 1.0
    return cfg


@patch("semi_supervised_image_clf.fixmatch.mlflow")
def test_train_fixmatch_returns_model(mock_mlflow: MagicMock, tmp_path) -> None:  # type: ignore[no-untyped-def]
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    model = ResNet18Classifier(num_classes=10)
    result = train_fixmatch(
        model,
        _labelled_loader(),
        _unlabelled_loader(),
        _fixmatch_config(),
        str(tmp_path),
        smoke_test=True,
    )
    assert isinstance(result, ResNet18Classifier)


@patch("semi_supervised_image_clf.fixmatch.mlflow")
def test_train_fixmatch_saves_checkpoint(mock_mlflow: MagicMock, tmp_path) -> None:  # type: ignore[no-untyped-def]
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    model = ResNet18Classifier(num_classes=10)
    cfg = _fixmatch_config()
    cfg.data.labels_per_class = 100
    train_fixmatch(
        model,
        _labelled_loader(),
        _unlabelled_loader(),
        cfg,
        str(tmp_path),
        smoke_test=True,
    )
    assert (tmp_path / "fixmatch_n100.pt").exists()


@patch("semi_supervised_image_clf.fixmatch.mlflow")
def test_train_fixmatch_high_threshold_all_masked(mock_mlflow: MagicMock, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """With threshold=1.0, no pseudo-labels pass; unsupervised loss should be zero."""
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    cfg = _fixmatch_config()
    cfg.training.threshold = 1.0  # nothing will pass

    model = ResNet18Classifier(num_classes=10)
    result = train_fixmatch(
        model,
        _labelled_loader(),
        _unlabelled_loader(),
        cfg,
        str(tmp_path),
        smoke_test=True,
    )
    assert isinstance(result, ResNet18Classifier)
