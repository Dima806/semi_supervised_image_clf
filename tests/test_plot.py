"""Tests for plot.py."""

from __future__ import annotations

from semi_supervised_image_clf.plot import plot_label_efficiency_curve, plot_training_curves

RESULTS: dict[str, dict[int, float]] = {
    "Supervised": {40: 0.40, 100: 0.52, 250: 0.61},
    "FixMatch": {40: 0.55, 100: 0.68, 250: 0.72},
}


def test_plot_label_efficiency_curve_creates_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    save_path = str(tmp_path / "label_eff.png")
    plot_label_efficiency_curve(RESULTS, save_path)
    assert (tmp_path / "label_eff.png").exists()


def test_plot_label_efficiency_curve_custom_title(tmp_path) -> None:  # type: ignore[no-untyped-def]
    save_path = str(tmp_path / "out.png")
    plot_label_efficiency_curve(RESULTS, save_path, title="My Title")
    assert (tmp_path / "out.png").exists()


def test_plot_training_curves_loss_only(tmp_path) -> None:  # type: ignore[no-untyped-def]
    save_path = str(tmp_path / "curves.png")
    plot_training_curves([0.9, 0.7, 0.5], save_path=save_path)
    assert (tmp_path / "curves.png").exists()


def test_plot_training_curves_with_val_acc(tmp_path) -> None:  # type: ignore[no-untyped-def]
    save_path = str(tmp_path / "curves.png")
    plot_training_curves([0.9, 0.7, 0.5], val_accs=[0.3, 0.5, 0.6], save_path=save_path)
    assert (tmp_path / "curves.png").exists()


def test_plot_label_efficiency_creates_parent_dirs(tmp_path) -> None:  # type: ignore[no-untyped-def]
    save_path = str(tmp_path / "nested" / "dir" / "out.png")
    plot_label_efficiency_curve(RESULTS, save_path)
    assert (tmp_path / "nested" / "dir" / "out.png").exists()
