"""ONNX export and validation for the best trained model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from loguru import logger

from semi_supervised_image_clf.model import ResNet18Classifier


def export_onnx(
    checkpoint: str,
    output: str,
    num_classes: int = 10,
    input_size: int = 64,
) -> None:
    """Load a checkpoint and export to ONNX, then validate with onnxruntime.

    Args:
        checkpoint: path to a ``ResNet18Classifier`` state-dict ``.pt`` file.
        output: destination path for the ``.onnx`` file.
        num_classes: number of output classes.
        input_size: spatial resolution the model was trained on.
    """
    model = ResNet18Classifier(num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 3, input_size, input_size)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_path),
        opset_version=17,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )
    logger.info(f"ONNX model exported to {output_path}")

    # Validate with ONNX checker
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model check passed.")

    # Validate numerical consistency with onnxruntime
    sess = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    with torch.no_grad():
        torch_out = model(dummy_input).numpy()
    ort_out = sess.run(None, {input_name: dummy_input.numpy()})[0]

    max_diff = float(np.abs(torch_out - ort_out).max())
    logger.info(f"Max output difference (PyTorch vs ONNX): {max_diff:.6f}")
    assert max_diff < 1e-4, f"ONNX output diverged: max diff = {max_diff}"
    logger.info("ONNX validation passed.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output", default="checkpoints/model.onnx")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--input-size", type=int, default=64)
    args = parser.parse_args()

    export_onnx(args.checkpoint, args.output, args.num_classes, args.input_size)


if __name__ == "__main__":
    main()
