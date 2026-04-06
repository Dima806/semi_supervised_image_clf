"""Tests for export.py."""

from __future__ import annotations

import torch

from semi_supervised_image_clf.export import export_onnx
from semi_supervised_image_clf.model import ResNet18Classifier


def test_export_onnx_creates_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # Save a fresh model checkpoint
    model = ResNet18Classifier(num_classes=10)
    ckpt = tmp_path / "model.pt"
    torch.save(model.state_dict(), ckpt)

    out = tmp_path / "model.onnx"
    export_onnx(str(ckpt), str(out), num_classes=10, input_size=64)

    assert out.exists()


def test_export_onnx_validates_numerics(tmp_path) -> None:  # type: ignore[no-untyped-def]
    model = ResNet18Classifier(num_classes=10)
    ckpt = tmp_path / "model.pt"
    torch.save(model.state_dict(), ckpt)

    out = tmp_path / "model.onnx"
    # Should not raise — numeric diff must be < 1e-4
    export_onnx(str(ckpt), str(out), num_classes=10, input_size=64)


def test_export_onnx_small_input_size(tmp_path) -> None:  # type: ignore[no-untyped-def]
    model = ResNet18Classifier(num_classes=5)
    ckpt = tmp_path / "model.pt"
    torch.save(model.state_dict(), ckpt)

    out = tmp_path / "model.onnx"
    export_onnx(str(ckpt), str(out), num_classes=5, input_size=32)
    assert out.exists()
