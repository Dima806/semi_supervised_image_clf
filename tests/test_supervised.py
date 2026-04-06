"""Tests for supervised.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from semi_supervised_image_clf.config import SupervisedConfig
from semi_supervised_image_clf.model import ResNet18Classifier
from semi_supervised_image_clf.supervised import train_supervised

IMG_SIZE = 64
NUM_CLASSES = 10


def _loader(n: int = 32) -> DataLoader:  # type: ignore[type-arg]
    imgs = torch.randn(n, 3, IMG_SIZE, IMG_SIZE)
    labels = torch.randint(0, NUM_CLASSES, (n,))
    return DataLoader(TensorDataset(imgs, labels), batch_size=8)


def _config(epochs: int = 1) -> SupervisedConfig:
    cfg = SupervisedConfig()
    cfg.training.epochs = epochs
    cfg.smoke_test.max_epochs = 1
    return cfg


@patch("semi_supervised_image_clf.supervised.mlflow")
def test_train_supervised_returns_model(mock_mlflow: MagicMock, tmp_path) -> None:  # type: ignore[no-untyped-def]
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    model = ResNet18Classifier(num_classes=NUM_CLASSES)
    result = train_supervised(
        model, _loader(), _loader(), _config(), str(tmp_path), smoke_test=True
    )
    assert isinstance(result, ResNet18Classifier)


@patch("semi_supervised_image_clf.supervised.mlflow")
def test_train_supervised_saves_checkpoint(mock_mlflow: MagicMock, tmp_path) -> None:  # type: ignore[no-untyped-def]
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    model = ResNet18Classifier(num_classes=NUM_CLASSES)
    train_supervised(model, _loader(), _loader(), _config(), str(tmp_path), smoke_test=True)
    assert (tmp_path / "supervised_best.pt").exists()


@patch("semi_supervised_image_clf.supervised.mlflow")
def test_train_supervised_smoke_uses_reduced_epochs(mock_mlflow: MagicMock, tmp_path) -> None:  # type: ignore[no-untyped-def]
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    cfg = SupervisedConfig()
    cfg.training.epochs = 100  # would be slow; smoke_test overrides to 1
    cfg.smoke_test.max_epochs = 1

    model = ResNet18Classifier(num_classes=NUM_CLASSES)
    # Should complete quickly (1 epoch only)
    train_supervised(model, _loader(), _loader(), cfg, str(tmp_path), smoke_test=True)
