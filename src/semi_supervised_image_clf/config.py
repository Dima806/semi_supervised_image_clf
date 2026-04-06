"""Pydantic v2 config schemas for all training modes."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Shared sub-schemas
# ---------------------------------------------------------------------------


class SmokeTestConfig(BaseModel):
    enabled: bool = False
    max_labelled: int = 50
    max_unlabelled: int = 1000
    max_epochs: int = 2


class DataConfig(BaseModel):
    dataset: str = "stl10"
    data_dir: str = "data/"
    num_workers: int = 2


# ---------------------------------------------------------------------------
# SimCLR
# ---------------------------------------------------------------------------


class SimCLRModelConfig(BaseModel):
    backbone: str = "resnet18"
    projection_dim: int = 128
    input_size: int = 64


class SimCLRTrainingConfig(BaseModel):
    epochs: int = 100
    batch_size: int = 256
    temperature: float = 0.5
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 10


class SimCLRConfig(BaseModel):
    model: SimCLRModelConfig = Field(default_factory=SimCLRModelConfig)
    training: SimCLRTrainingConfig = Field(default_factory=SimCLRTrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    smoke_test: SmokeTestConfig = Field(default_factory=SmokeTestConfig)


# ---------------------------------------------------------------------------
# FixMatch
# ---------------------------------------------------------------------------


class FixMatchModelConfig(BaseModel):
    backbone: str = "resnet18"
    pretrained_simclr: str | None = None
    input_size: int = 64
    num_classes: int = 10


class FixMatchTrainingConfig(BaseModel):
    epochs: int = 200
    batch_size_labelled: int = 64
    batch_size_unlabelled: int = 192
    learning_rate: float = 3e-2
    weight_decay: float = 5e-4
    momentum: float = 0.9
    ema_decay: float = 0.999
    threshold: float = 0.95
    lambda_u: float = 1.0
    warmup_epochs: int = 0


class FixMatchDataConfig(DataConfig):
    labels_per_class: int = 100
    random_seed: int = 42


class FixMatchConfig(BaseModel):
    model: FixMatchModelConfig = Field(default_factory=FixMatchModelConfig)
    training: FixMatchTrainingConfig = Field(default_factory=FixMatchTrainingConfig)
    data: FixMatchDataConfig = Field(default_factory=FixMatchDataConfig)
    smoke_test: SmokeTestConfig = Field(default_factory=SmokeTestConfig)


# ---------------------------------------------------------------------------
# Supervised
# ---------------------------------------------------------------------------


class SupervisedModelConfig(BaseModel):
    backbone: str = "resnet18"
    input_size: int = 64
    num_classes: int = 10


class SupervisedTrainingConfig(BaseModel):
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-2
    weight_decay: float = 5e-4
    momentum: float = 0.9
    lr_scheduler: str = "cosine"


class SupervisedDataConfig(DataConfig):
    labels_per_class: int = 500
    random_seed: int = 42


class SupervisedConfig(BaseModel):
    model: SupervisedModelConfig = Field(default_factory=SupervisedModelConfig)
    training: SupervisedTrainingConfig = Field(default_factory=SupervisedTrainingConfig)
    data: SupervisedDataConfig = Field(default_factory=SupervisedDataConfig)
    smoke_test: SmokeTestConfig = Field(default_factory=SmokeTestConfig)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_simclr_config(path: str | Path) -> SimCLRConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return SimCLRConfig.model_validate(raw)


def load_fixmatch_config(path: str | Path) -> FixMatchConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return FixMatchConfig.model_validate(raw)


def load_supervised_config(path: str | Path) -> SupervisedConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return SupervisedConfig.model_validate(raw)
