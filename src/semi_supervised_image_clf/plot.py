"""Plotting utilities: label efficiency curve and training curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from loguru import logger


def plot_label_efficiency_curve(
    results: dict[str, dict[int, float]],
    save_path: str,
    title: str = "Label Efficiency: STL-10",
) -> None:
    """Plot accuracy vs number of labelled examples for multiple methods.

    Args:
        results: mapping of method name -> {labels_total: accuracy}.
            e.g. {
                "Supervised":        {40: 0.52, 100: 0.61, 250: 0.65, 1000: 0.68, 5000: 0.72},
                "FixMatch":          {40: 0.62, 100: 0.68, 250: 0.72, 1000: 0.74, 5000: 0.75},
                "SimCLR + FixMatch": {40: 0.70, 100: 0.75, 250: 0.78, 1000: 0.80, 5000: 0.81},
            }
        save_path: destination PNG path.
        title: plot title.
    """
    markers = ["o", "s", "^", "D", "v"]
    fig, ax = plt.subplots(figsize=(9, 6))

    for (method, data), marker in zip(results.items(), markers, strict=False):
        xs = sorted(data.keys())
        ys = [data[x] * 100 for x in xs]
        ax.plot(xs, ys, marker=marker, linewidth=2, markersize=7, label=method)

    ax.set_xscale("log")
    ax.set_xlabel("Number of labelled images (log scale)", fontsize=12)
    ax.set_ylabel("Test accuracy (%)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.xaxis.set_major_formatter(plt.ScalarFormatter())

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Label efficiency curve saved to {save_path}")


def plot_training_curves(
    train_losses: list[float],
    val_accs: list[float] | None = None,
    save_path: str = "plots/training_curves.png",
    title: str = "Training curves",
) -> None:
    """Plot training loss and optional validation accuracy over epochs."""
    epochs = list(range(1, len(train_losses) + 1))
    n_plots = 2 if val_accs is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    axes[0].plot(epochs, train_losses, linewidth=2, color="steelblue")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training loss")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    if val_accs is not None:
        axes[1].plot(epochs, [a * 100 for a in val_accs], linewidth=2, color="darkorange")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_title("Validation accuracy")
        axes[1].grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(title, fontsize=13)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Training curves saved to {save_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate label efficiency plot")
    parser.add_argument("--results-dir", default="checkpoints/")
    parser.add_argument("--results-json", default=None, help="JSON file with results dict")
    parser.add_argument("--output", default="plots/label_efficiency.png")
    args = parser.parse_args()

    if args.results_json:
        with open(args.results_json) as f:
            results = json.load(f)
        # JSON keys are strings; convert label counts to int
        results = {method: {int(k): v for k, v in data.items()} for method, data in results.items()}
    else:
        # Placeholder with approximate expected results for illustration
        results: dict[str, dict[int, float]] = {
            "Supervised": {40: 0.40, 100: 0.52, 250: 0.61, 1000: 0.68, 5000: 0.72},
            "FixMatch": {40: 0.55, 100: 0.68, 250: 0.72, 1000: 0.74, 5000: 0.75},
            "SimCLR + FixMatch": {40: 0.60, 100: 0.75, 250: 0.78, 1000: 0.80, 5000: 0.81},
        }
        logger.warning("No results JSON provided — using placeholder values.")

    plot_label_efficiency_curve(results, save_path=args.output)


if __name__ == "__main__":
    main()
