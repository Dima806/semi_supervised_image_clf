"""Tests for config.py loader functions."""

from __future__ import annotations

from semi_supervised_image_clf.config import (
    FixMatchConfig,
    SimCLRConfig,
    SupervisedConfig,
    load_fixmatch_config,
    load_simclr_config,
    load_supervised_config,
)

CONFIG_DIR = "config/"


def test_load_simclr_config() -> None:
    cfg = load_simclr_config(CONFIG_DIR + "simclr.yaml")
    assert isinstance(cfg, SimCLRConfig)
    assert cfg.model.backbone == "resnet18"
    assert cfg.training.temperature == 0.5
    assert cfg.training.epochs == 100


def test_load_fixmatch_config() -> None:
    cfg = load_fixmatch_config(CONFIG_DIR + "fixmatch.yaml")
    assert isinstance(cfg, FixMatchConfig)
    assert cfg.model.num_classes == 10
    assert cfg.training.threshold == 0.95
    assert cfg.data.labels_per_class == 100


def test_load_supervised_config() -> None:
    cfg = load_supervised_config(CONFIG_DIR + "supervised.yaml")
    assert isinstance(cfg, SupervisedConfig)
    assert cfg.data.labels_per_class == 500
    assert cfg.model.num_classes == 10


def test_fixmatch_config_pretrained_simclr_defaults_none() -> None:
    cfg = load_fixmatch_config(CONFIG_DIR + "fixmatch.yaml")
    assert cfg.model.pretrained_simclr is None


def test_smoke_test_disabled_by_default() -> None:
    for loader, path in [
        (load_simclr_config, CONFIG_DIR + "simclr.yaml"),
        (load_fixmatch_config, CONFIG_DIR + "fixmatch.yaml"),
        (load_supervised_config, CONFIG_DIR + "supervised.yaml"),
    ]:
        cfg = loader(path)  # type: ignore[operator]
        assert cfg.smoke_test.enabled is False
