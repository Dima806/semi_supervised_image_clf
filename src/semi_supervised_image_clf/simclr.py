"""SimCLR self-supervised pretraining with NT-Xent loss."""

from __future__ import annotations

import argparse
from collections.abc import Sized
from pathlib import Path
from typing import Any, cast

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from semi_supervised_image_clf.augmentations import SimCLRAugmentation
from semi_supervised_image_clf.config import SimCLRConfig, load_simclr_config
from semi_supervised_image_clf.model import ResNet18WithProjection

# ---------------------------------------------------------------------------
# NT-Xent loss
# ---------------------------------------------------------------------------


class NTXentLoss(nn.Module):
    """Normalised temperature-scaled cross-entropy loss for SimCLR.

    Args:
        temperature: softmax temperature (default 0.5).
        batch_size: number of images per batch (not counting the second view).
    """

    def __init__(self, temperature: float = 0.5, batch_size: int = 256) -> None:
        super().__init__()
        self.temperature = temperature
        self.batch_size = batch_size

    def forward(self, z_i: Tensor, z_j: Tensor) -> Tensor:
        """
        Args:
            z_i: projected embeddings for view 1, shape (N, D).
            z_j: projected embeddings for view 2, shape (N, D).
        Returns:
            Scalar loss.
        """
        n = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)  # (2N, D)
        z = F.normalize(z, dim=1)

        # Cosine similarity matrix (2N, 2N)
        sim = torch.mm(z, z.T) / self.temperature

        # Mask out self-similarity on the diagonal
        mask = torch.eye(2 * n, device=z.device, dtype=torch.bool)
        sim.masked_fill_(mask, float("-inf"))

        # Positive pairs: (i, i+N) and (i+N, i)
        labels = torch.cat([torch.arange(n, 2 * n), torch.arange(n)], dim=0).to(z.device)

        loss = F.cross_entropy(sim, labels)
        return loss


# ---------------------------------------------------------------------------
# SimCLR dataset wrapper: applies SimCLR augmentations at load time
# ---------------------------------------------------------------------------


class SimCLRDataset(torch.utils.data.Dataset[tuple[Tensor, Tensor]]):
    """Wraps an existing dataset and returns two augmented views per sample."""

    def __init__(self, base_dataset: torch.utils.data.Dataset[Any], input_size: int = 64) -> None:
        self.base = base_dataset
        self.augment = SimCLRAugmentation(input_size)

    def __len__(self) -> int:
        return len(cast(Sized, self.base))

    def __getitem__(self, index: Any) -> tuple[Tensor, Tensor]:
        img, _ = self.base[index]
        view1, view2 = self.augment(img)
        return view1, view2


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_simclr(
    model: ResNet18WithProjection,
    loader: DataLoader,  # type: ignore[type-arg]
    config: SimCLRConfig,
    checkpoint_dir: str = "checkpoints/",
    smoke_test: bool = False,
) -> ResNet18WithProjection:
    """Run SimCLR pretraining.

    Returns the model with the projection head *removed* (encoder only).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    epochs = config.smoke_test.max_epochs if smoke_test else config.training.epochs
    bs = config.training.batch_size
    criterion = NTXentLoss(temperature=config.training.temperature, batch_size=bs)
    optimizer = Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    warmup_epochs = config.training.warmup_epochs
    warmup_scheduler = LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment("simclr_pretrain")
    with mlflow.start_run():
        mlflow.log_params(
            {
                "epochs": epochs,
                "batch_size": bs,
                "temperature": config.training.temperature,
                "lr": config.training.learning_rate,
            }
        )

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            for view1, view2 in loader:
                view1, view2 = view1.to(device), view2.to(device)
                z_i = model(view1)
                z_j = model(view2)
                loss = criterion(z_i, z_j)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            avg_loss = total_loss / len(loader)
            logger.info(f"SimCLR epoch {epoch}/{epochs} — loss: {avg_loss:.4f}")
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

        torch.save(model.state_dict(), ckpt_dir / "simclr_full.pt")
        encoder = model.get_encoder()
        torch.save(encoder.state_dict(), ckpt_dir / "simclr_encoder.pt")
        logger.info(f"SimCLR encoder saved to {ckpt_dir / 'simclr_encoder.pt'}")

    # Return model with projection head detached
    return model


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="SimCLR pretraining")
    parser.add_argument("--config", default="config/simclr.yaml")
    parser.add_argument("--checkpoint-dir", default="checkpoints/")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    config = load_simclr_config(args.config)
    smoke = args.smoke_test or config.smoke_test.enabled

    from semi_supervised_image_clf.dataset import get_unlabelled_loader

    loader = get_unlabelled_loader(
        data_dir=config.data.data_dir,
        input_size=config.model.input_size,
        batch_size=config.smoke_test.max_unlabelled // 4 if smoke else config.training.batch_size,
        num_workers=config.data.num_workers,
        smoke_test=smoke,
        max_unlabelled=config.smoke_test.max_unlabelled,
    )

    # Wrap loader dataset with SimCLR augmentations
    simclr_dataset = SimCLRDataset(loader.dataset, input_size=config.model.input_size)
    simclr_loader = DataLoader(
        simclr_dataset,
        batch_size=config.training.batch_size if not smoke else 32,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    model = ResNet18WithProjection(projection_dim=config.model.projection_dim)
    train_simclr(model, simclr_loader, config, args.checkpoint_dir, smoke_test=smoke)


if __name__ == "__main__":
    main()
