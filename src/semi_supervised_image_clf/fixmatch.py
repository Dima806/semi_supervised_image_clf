"""FixMatch semi-supervised training."""

from __future__ import annotations

import argparse
from collections.abc import Sized
from pathlib import Path
from typing import Any, cast

import mlflow
import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from semi_supervised_image_clf.config import FixMatchConfig, load_fixmatch_config
from semi_supervised_image_clf.model import EMAModel, ResNet18Classifier

# ---------------------------------------------------------------------------
# Pseudo-label filter
# ---------------------------------------------------------------------------


class PseudoLabelFilter:
    """Filters pseudo-labels by confidence threshold.

    Args:
        threshold: minimum max-probability to accept a pseudo-label.
    """

    def __init__(self, threshold: float = 0.95) -> None:
        self.threshold = threshold

    def filter(self, probs: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            probs: softmax probabilities, shape (N, C).
        Returns:
            mask: bool tensor (N,) — True where pseudo-label is accepted.
            pseudo_labels: int tensor (N,) with argmax predictions.
        """
        max_probs, pseudo_labels = probs.max(dim=-1)
        mask = max_probs >= self.threshold
        return mask, pseudo_labels


# ---------------------------------------------------------------------------
# FixMatch dataset wrapper
# ---------------------------------------------------------------------------


class FixMatchUnlabelledDataset(torch.utils.data.Dataset[tuple[Tensor, Tensor]]):
    """Wraps an unlabelled dataset and returns (weak_view, strong_view) per sample."""

    def __init__(
        self,
        base_dataset: torch.utils.data.Dataset[Any],
        input_size: int = 64,
    ) -> None:
        from semi_supervised_image_clf.augmentations import FixMatchAugmentation

        self.base = base_dataset
        self.augment = FixMatchAugmentation(input_size)

    def __len__(self) -> int:
        return len(cast(Sized, self.base))

    def __getitem__(self, index: Any) -> tuple[Tensor, Tensor]:
        img, _ = self.base[index]
        weak, strong = self.augment(img)
        return weak, strong


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_fixmatch(
    model: ResNet18Classifier,
    labelled_loader: DataLoader,  # type: ignore[type-arg]
    unlabelled_loader: DataLoader,  # type: ignore[type-arg]
    config: FixMatchConfig,
    checkpoint_dir: str = "checkpoints/",
    smoke_test: bool = False,
    run_name: str = "fixmatch",
) -> ResNet18Classifier:
    """FixMatch training loop.

    Args:
        model: classifier (randomly initialised or with SimCLR encoder).
        labelled_loader: loader yielding (images, labels).
        unlabelled_loader: loader yielding (weak_view, strong_view).
        config: FixMatch config.
        checkpoint_dir: where to save checkpoints.
        smoke_test: if True, use reduced epochs from smoke_test config.
        run_name: MLflow run name.
    Returns:
        Trained model (EMA weights applied).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    ema = EMAModel(model, decay=config.training.ema_decay)
    plf = PseudoLabelFilter(threshold=config.training.threshold)

    epochs = config.smoke_test.max_epochs if smoke_test else config.training.epochs
    optimizer = SGD(
        model.parameters(),
        lr=config.training.learning_rate,
        momentum=config.training.momentum,
        weight_decay=config.training.weight_decay,
        nesterov=True,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment("fixmatch")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "epochs": epochs,
                "labels_per_class": config.data.labels_per_class,
                "threshold": config.training.threshold,
                "lambda_u": config.training.lambda_u,
                "pretrained_simclr": config.model.pretrained_simclr,
            }
        )

        unlabelled_iter = iter(unlabelled_loader)
        best_loss = float("inf")

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = total_sup = total_unsup = 0.0
            mask_ratio_sum = 0.0
            n_batches = 0

            for imgs_l, labels_l in labelled_loader:
                # Get unlabelled batch (cycle if exhausted)
                try:
                    imgs_weak, imgs_strong = next(unlabelled_iter)
                except StopIteration:
                    unlabelled_iter = iter(unlabelled_loader)
                    imgs_weak, imgs_strong = next(unlabelled_iter)

                imgs_l = imgs_l.to(device)
                labels_l = labels_l.to(device)
                imgs_weak = imgs_weak.to(device)
                imgs_strong = imgs_strong.to(device)

                # Supervised loss on labelled data
                logits_l = model(imgs_l)
                loss_sup = F.cross_entropy(logits_l, labels_l)

                # Pseudo-labels from weak view (no gradient)
                with torch.no_grad():
                    probs_weak = F.softmax(model(imgs_weak), dim=-1)
                mask, pseudo_labels = plf.filter(probs_weak)

                # Unsupervised loss on strong view (only where mask is True)
                if mask.any():
                    logits_strong = model(imgs_strong)
                    loss_unsup = (
                        F.cross_entropy(logits_strong, pseudo_labels, reduction="none")
                        * mask.float()
                    ).mean()
                else:
                    loss_unsup = torch.tensor(0.0, device=device)

                loss = loss_sup + config.training.lambda_u * loss_unsup

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ema.update(model)

                total_loss += loss.item()
                total_sup += loss_sup.item()
                total_unsup += loss_unsup.item()
                mask_ratio_sum += mask.float().mean().item()
                n_batches += 1

            scheduler.step()
            avg_loss = total_loss / n_batches
            avg_mask = mask_ratio_sum / n_batches
            logger.info(
                f"FixMatch epoch {epoch}/{epochs} — loss: {avg_loss:.4f} "
                f"sup: {total_sup / n_batches:.4f} "
                f"unsup: {total_unsup / n_batches:.4f} "
                f"mask_ratio: {avg_mask:.3f}"
            )
            mlflow.log_metrics(
                {
                    "train_loss": avg_loss,
                    "loss_sup": total_sup / n_batches,
                    "loss_unsup": total_unsup / n_batches,
                    "mask_ratio": avg_mask,
                },
                step=epoch,
            )

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), ckpt_dir / "fixmatch_best.pt")

        # Apply EMA weights to model before returning
        ema.apply_shadow(model)
        tag = f"n{config.data.labels_per_class}"
        torch.save(model.state_dict(), ckpt_dir / f"fixmatch_{tag}.pt")
        logger.info(f"FixMatch training done. Saved to {ckpt_dir / f'fixmatch_{tag}.pt'}")

    return model


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="FixMatch semi-supervised training")
    parser.add_argument("--config", default="config/fixmatch.yaml")
    parser.add_argument("--checkpoint-dir", default="checkpoints/")
    parser.add_argument("--pretrained-simclr", default=None)
    parser.add_argument("--labels-per-class", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    config = load_fixmatch_config(args.config)
    if args.pretrained_simclr:
        config.model.pretrained_simclr = args.pretrained_simclr
    if args.labels_per_class is not None:
        config.data.labels_per_class = args.labels_per_class

    smoke = args.smoke_test or config.smoke_test.enabled

    from semi_supervised_image_clf.dataset import get_stl10_splits

    labelled_loader, unlabelled_base_loader, _ = get_stl10_splits(
        config=config.data,
        labels_per_class=config.data.labels_per_class,
        seed=config.data.random_seed,
        input_size=config.model.input_size,
        smoke_test=smoke,
        max_labelled=config.smoke_test.max_labelled,
        max_unlabelled=config.smoke_test.max_unlabelled,
    )

    # Re-wrap unlabelled dataset with FixMatch augmentations
    unlabelled_dataset = FixMatchUnlabelledDataset(
        unlabelled_base_loader.dataset, input_size=config.model.input_size
    )
    unlabelled_loader = DataLoader(
        unlabelled_dataset,
        batch_size=config.training.batch_size_unlabelled if not smoke else 32,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    model = ResNet18Classifier(num_classes=config.model.num_classes)

    if config.model.pretrained_simclr:
        # Load only encoder weights
        full_state = torch.load(config.model.pretrained_simclr, map_location="cpu")
        # If saved as full SimCLR model, extract encoder sub-keys
        encoder_state = {
            k.removeprefix("encoder."): v for k, v in full_state.items() if k.startswith("encoder.")
        }
        if encoder_state:
            model.encoder.load_state_dict(encoder_state)
        else:
            # Assume file is already just the encoder state dict
            model.encoder.load_state_dict(full_state)
        logger.info(f"Loaded SimCLR encoder from {config.model.pretrained_simclr}")

    run_name = f"fixmatch_n{config.data.labels_per_class}"
    if config.model.pretrained_simclr:
        run_name = f"simclr+fixmatch_n{config.data.labels_per_class}"

    train_fixmatch(
        model, labelled_loader, unlabelled_loader, config, args.checkpoint_dir, smoke, run_name
    )


if __name__ == "__main__":
    main()
