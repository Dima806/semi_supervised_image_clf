"""Tests for evaluate.py."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from semi_supervised_image_clf.evaluate import EvalResult, evaluate, plot_tsne_embeddings
from semi_supervised_image_clf.model import ResNet18Classifier

IMG_SIZE = 64
NUM_CLASSES = 10


def _loader(n: int = 40, always_class: int | None = None) -> DataLoader:  # type: ignore[type-arg]
    imgs = torch.randn(n, 3, IMG_SIZE, IMG_SIZE)
    if always_class is not None:
        labels = torch.full((n,), always_class, dtype=torch.long)
    else:
        labels = torch.randint(0, NUM_CLASSES, (n,))
    return DataLoader(TensorDataset(imgs, labels), batch_size=16)


def _perfect_model() -> ResNet18Classifier:
    """A model whose classifier bias is set to guarantee class-0 predictions."""
    model = ResNet18Classifier(num_classes=NUM_CLASSES)
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.zero_()
        model.classifier.bias[0] = 100.0  # always predicts class 0
    return model


# ---------------------------------------------------------------------------
# EvalResult
# ---------------------------------------------------------------------------


def test_eval_result_str_contains_accuracy() -> None:
    cm = np.zeros((10, 10), dtype=np.int64)
    result = EvalResult(
        accuracy=0.75,
        top5_accuracy=0.95,
        confusion_matrix=cm,
        per_class_accuracy={"airplane": 0.8, "bird": 0.7},
    )
    s = str(result)
    assert "0.7500" in s
    assert "0.9500" in s
    assert "airplane" in s


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------


def test_evaluate_perfect_model_on_class0_data() -> None:
    model = _perfect_model()
    loader = _loader(n=40, always_class=0)
    result = evaluate(model, loader)
    assert result.accuracy == pytest.approx(1.0)


def test_evaluate_returns_eval_result() -> None:
    model = ResNet18Classifier(num_classes=NUM_CLASSES)
    result = evaluate(model, _loader())
    assert isinstance(result, EvalResult)
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.top5_accuracy <= 1.0


def test_evaluate_confusion_matrix_shape() -> None:
    model = ResNet18Classifier(num_classes=NUM_CLASSES)
    result = evaluate(model, _loader())
    assert result.confusion_matrix.shape == (NUM_CLASSES, NUM_CLASSES)


def test_evaluate_confusion_matrix_sums_to_n() -> None:
    model = ResNet18Classifier(num_classes=NUM_CLASSES)
    loader = _loader(n=40)
    result = evaluate(model, loader)
    assert result.confusion_matrix.sum() == 40


def test_evaluate_per_class_accuracy_keys() -> None:
    model = ResNet18Classifier(num_classes=NUM_CLASSES)
    class_names = [f"cls{i}" for i in range(NUM_CLASSES)]
    result = evaluate(model, _loader(), class_names=class_names)
    assert set(result.per_class_accuracy.keys()) == set(class_names)


def test_evaluate_top5_accuracy_geq_top1() -> None:
    model = ResNet18Classifier(num_classes=NUM_CLASSES)
    result = evaluate(model, _loader())
    assert result.top5_accuracy >= result.accuracy


def test_evaluate_uses_default_stl10_class_names() -> None:
    model = ResNet18Classifier(num_classes=NUM_CLASSES)
    result = evaluate(model, _loader())
    assert "airplane" in result.per_class_accuracy


# ---------------------------------------------------------------------------
# plot_tsne_embeddings()
# ---------------------------------------------------------------------------


def test_plot_tsne_creates_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    model = ResNet18Classifier(num_classes=NUM_CLASSES)
    # Need enough samples for t-SNE perplexity=30 (requires > 90 points)
    imgs = torch.randn(100, 3, IMG_SIZE, IMG_SIZE)
    labels = torch.arange(NUM_CLASSES).repeat(10)
    loader = DataLoader(TensorDataset(imgs, labels), batch_size=32)
    save_path = str(tmp_path / "tsne.png")
    plot_tsne_embeddings(model, loader, save_path, max_samples=100)
    assert (tmp_path / "tsne.png").exists()
