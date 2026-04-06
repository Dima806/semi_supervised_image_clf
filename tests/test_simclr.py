"""Tests for simclr.py: NTXentLoss, SimCLRDataset, and train_simclr."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from semi_supervised_image_clf.config import SimCLRConfig
from semi_supervised_image_clf.model import ResNet18WithProjection
from semi_supervised_image_clf.simclr import NTXentLoss, SimCLRDataset, train_simclr


def test_ntxent_loss_scalar() -> None:
    loss_fn = NTXentLoss(temperature=0.5, batch_size=4)
    z_i = torch.randn(4, 128)
    z_j = torch.randn(4, 128)
    loss = loss_fn(z_i, z_j)
    assert loss.ndim == 0  # scalar
    assert loss.item() > 0


def test_ntxent_loss_perfect_alignment() -> None:
    """Identical projections should yield lower loss than random."""
    loss_fn = NTXentLoss(temperature=0.5, batch_size=8)
    z = torch.randn(8, 128)
    loss_perfect = loss_fn(z, z.clone())
    loss_random = loss_fn(z, torch.randn(8, 128))
    assert loss_perfect < loss_random


def test_ntxent_loss_batch_size_invariant_shape() -> None:
    for bs in [4, 8, 16]:
        loss_fn = NTXentLoss(temperature=0.5, batch_size=bs)
        z_i = torch.randn(bs, 64)
        z_j = torch.randn(bs, 64)
        loss = loss_fn(z_i, z_j)
        assert loss.ndim == 0


def test_simclr_dataset_returns_two_views(fake_pil_dataset) -> None:  # type: ignore[no-untyped-def]
    ds = SimCLRDataset(fake_pil_dataset, input_size=64)
    v1, v2 = ds[0]
    assert v1.shape == (3, 64, 64)
    assert v2.shape == (3, 64, 64)


def test_simclr_dataset_len(fake_pil_dataset) -> None:  # type: ignore[no-untyped-def]
    ds = SimCLRDataset(fake_pil_dataset, input_size=64)
    assert len(ds) == len(fake_pil_dataset)


# ---------------------------------------------------------------------------
# train_simclr
# ---------------------------------------------------------------------------


def _simclr_loader(n: int = 32) -> DataLoader:  # type: ignore[type-arg]
    """DataLoader yielding (view1, view2) tensor pairs."""
    v1 = torch.randn(n, 3, 64, 64)
    v2 = torch.randn(n, 3, 64, 64)
    return DataLoader(TensorDataset(v1, v2), batch_size=8, drop_last=True)


def _simclr_config() -> SimCLRConfig:
    cfg = SimCLRConfig()
    cfg.smoke_test.max_epochs = 1
    cfg.training.warmup_epochs = 0
    return cfg


@patch("semi_supervised_image_clf.simclr.mlflow")
def test_train_simclr_returns_model(mock_mlflow: MagicMock, tmp_path) -> None:  # type: ignore[no-untyped-def]
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    model = ResNet18WithProjection(projection_dim=128)
    result = train_simclr(model, _simclr_loader(), _simclr_config(), str(tmp_path), smoke_test=True)
    assert isinstance(result, ResNet18WithProjection)


@patch("semi_supervised_image_clf.simclr.mlflow")
def test_train_simclr_saves_checkpoints(mock_mlflow: MagicMock, tmp_path) -> None:  # type: ignore[no-untyped-def]
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    model = ResNet18WithProjection(projection_dim=128)
    train_simclr(model, _simclr_loader(), _simclr_config(), str(tmp_path), smoke_test=True)
    assert (tmp_path / "simclr_full.pt").exists()
    assert (tmp_path / "simclr_encoder.pt").exists()
