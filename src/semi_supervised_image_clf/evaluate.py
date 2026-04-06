"""Evaluation utilities: accuracy, confusion matrix, t-SNE embeddings."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from semi_supervised_image_clf.model import ResNet18Classifier

STL10_CLASSES = [
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    accuracy: float
    top5_accuracy: float
    confusion_matrix: np.ndarray
    per_class_accuracy: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"Accuracy:      {self.accuracy:.4f}",
            f"Top-5 Accuracy:{self.top5_accuracy:.4f}",
            "Per-class accuracy:",
        ]
        for cls, acc in self.per_class_accuracy.items():
            lines.append(f"  {cls:<10}: {acc:.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------------------------


def evaluate(
    model: ResNet18Classifier,
    test_loader: DataLoader,  # type: ignore[type-arg]
    class_names: list[str] | None = None,
) -> EvalResult:
    """Evaluate model on a test loader.

    Args:
        model: trained classifier.
        test_loader: test set dataloader.
        class_names: optional list of class names (length = num_classes).
    Returns:
        EvalResult with accuracy, top-5 accuracy, confusion matrix,
        and per-class accuracy.
    """
    if class_names is None:
        class_names = STL10_CLASSES

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    num_classes = len(class_names)
    all_preds: list[int] = []
    all_labels: list[int] = []
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)

            # Top-1
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            # Top-5
            top5 = logits.topk(min(5, num_classes), dim=1).indices
            top5_correct += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
            total += labels.size(0)

    preds_arr = np.array(all_preds)
    labels_arr = np.array(all_labels)

    accuracy = float((preds_arr == labels_arr).mean())
    top5_accuracy = float(top5_correct / total)

    # Confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labels_arr, preds_arr, strict=True):
        cm[t][p] += 1

    # Per-class accuracy
    per_class: dict[str, float] = {}
    for i, name in enumerate(class_names):
        n = cm[i].sum()
        per_class[name] = float(cm[i, i] / n) if n > 0 else 0.0

    return EvalResult(
        accuracy=accuracy,
        top5_accuracy=top5_accuracy,
        confusion_matrix=cm,
        per_class_accuracy=per_class,
    )


# ---------------------------------------------------------------------------
# t-SNE visualisation
# ---------------------------------------------------------------------------


def plot_tsne_embeddings(
    model: ResNet18Classifier,
    loader: DataLoader,  # type: ignore[type-arg]
    save_path: str,
    max_samples: int = 2000,
    class_names: list[str] | None = None,
) -> None:
    """Compute and save a 2-D t-SNE plot of encoder embeddings.

    Args:
        model: classifier whose ``encode`` method is used.
        loader: data loader (images, labels).
        save_path: path to save the PNG figure.
        max_samples: cap on number of samples to embed (for speed).
        class_names: optional list of class names.
    """
    import matplotlib.pyplot as plt

    if class_names is None:
        class_names = STL10_CLASSES

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    feats: list[np.ndarray] = []
    lbls: list[int] = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            h = model.encode(imgs).cpu().numpy()
            feats.append(h)
            lbls.extend(labels.tolist())
            if len(lbls) >= max_samples:
                break

    feats_arr = np.concatenate(feats, axis=0)[:max_samples]
    lbls_arr = np.array(lbls[:max_samples])

    logger.info(f"Running t-SNE on {len(lbls_arr)} samples ...")
    emb = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(feats_arr)

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, name in enumerate(class_names):
        mask = lbls_arr == i
        ax.scatter(emb[mask, 0], emb[mask, 1], label=name, alpha=0.5, s=10)
    ax.legend(markerscale=3, fontsize=8)
    ax.set_title("t-SNE of encoder embeddings")
    ax.axis("off")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"t-SNE saved to {save_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoints")
    parser.add_argument("--checkpoint-dir", default="checkpoints/")
    parser.add_argument("--config", default="config/fixmatch.yaml")
    parser.add_argument("--tsne", action="store_true", help="Also produce t-SNE plots")
    args = parser.parse_args()

    from semi_supervised_image_clf.config import load_fixmatch_config
    from semi_supervised_image_clf.dataset import get_stl10_splits

    config = load_fixmatch_config(args.config)
    _, _, test_loader = get_stl10_splits(
        config=config.data,
        labels_per_class=config.data.labels_per_class,
        seed=config.data.random_seed,
        input_size=config.model.input_size,
    )

    ckpt_dir = Path(args.checkpoint_dir)
    for ckpt in sorted(ckpt_dir.glob("*.pt")):
        model = ResNet18Classifier(num_classes=config.model.num_classes)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        result = evaluate(model, test_loader)
        logger.info(f"\n=== {ckpt.name} ===\n{result}")
        if args.tsne:
            plot_tsne_embeddings(model, test_loader, f"plots/tsne_{ckpt.stem}.png")


if __name__ == "__main__":
    main()
