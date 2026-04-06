"""Supervised baseline training."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import torch
import torch.nn.functional as F
from loguru import logger
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms

from semi_supervised_image_clf.config import SupervisedConfig, load_supervised_config
from semi_supervised_image_clf.model import ResNet18Classifier


def _augmented_transform(input_size: int = 64) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(input_size, scale=(0.875, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def train_supervised(
    model: ResNet18Classifier,
    train_loader: DataLoader,  # type: ignore[type-arg]
    test_loader: DataLoader,  # type: ignore[type-arg]
    config: SupervisedConfig,
    checkpoint_dir: str = "checkpoints/",
    smoke_test: bool = False,
) -> ResNet18Classifier:
    """Supervised baseline training loop.

    Args:
        model: classifier with random init.
        train_loader: labelled training loader.
        test_loader: test set loader for validation.
        config: supervised config.
        checkpoint_dir: where to save checkpoints.
        smoke_test: use reduced epochs.
    Returns:
        Trained model at best validation accuracy.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

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

    mlflow.set_experiment("supervised_baseline")
    with mlflow.start_run():
        mlflow.log_params(
            {
                "epochs": epochs,
                "batch_size": config.training.batch_size,
                "lr": config.training.learning_rate,
                "labels_per_class": config.data.labels_per_class,
            }
        )

        best_acc = 0.0

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss = F.cross_entropy(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()

            # Validation
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    preds = model(imgs).argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            acc = correct / total
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Supervised epoch {epoch}/{epochs} — loss: {avg_loss:.4f}  acc: {acc:.4f}")
            mlflow.log_metrics({"train_loss": avg_loss, "val_acc": acc}, step=epoch)

            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), ckpt_dir / "supervised_best.pt")

        logger.info(f"Supervised training done. Best acc: {best_acc:.4f}")

    return model


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised baseline training")
    parser.add_argument("--config", default="config/supervised.yaml")
    parser.add_argument("--checkpoint-dir", default="checkpoints/")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    config = load_supervised_config(args.config)
    smoke = args.smoke_test or config.smoke_test.enabled

    from semi_supervised_image_clf.dataset import get_stl10_splits

    labelled_loader, _, test_loader = get_stl10_splits(
        config=config.data,
        labels_per_class=config.data.labels_per_class,
        seed=config.data.random_seed,
        input_size=config.model.input_size,
        smoke_test=smoke,
        max_labelled=config.smoke_test.max_labelled,
    )

    model = ResNet18Classifier(num_classes=config.model.num_classes)
    train_supervised(model, labelled_loader, test_loader, config, args.checkpoint_dir, smoke)


if __name__ == "__main__":
    main()
