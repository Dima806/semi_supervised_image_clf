"""ResNet-18 backbone, SimCLR projection head, and FixMatch classifier."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import ResNet18_Weights, resnet18

# ---------------------------------------------------------------------------
# Base encoder
# ---------------------------------------------------------------------------


def _build_encoder(pretrained_imagenet: bool = False) -> tuple[nn.Module, int]:
    """Build ResNet-18 and return (encoder_without_fc, feature_dim)."""
    weights = ResNet18_Weights.DEFAULT if pretrained_imagenet else None
    backbone = resnet18(weights=weights)
    feature_dim: int = backbone.fc.in_features  # 512
    # Remove the classification head; keep everything up to the avg-pool
    encoder = nn.Sequential(*list(backbone.children())[:-1])  # output: (B, 512, 1, 1)
    return encoder, feature_dim


# ---------------------------------------------------------------------------
# SimCLR model
# ---------------------------------------------------------------------------


class ResNet18WithProjection(nn.Module):
    """ResNet-18 encoder + MLP projection head for SimCLR pretraining."""

    def __init__(
        self,
        projection_dim: int = 128,
        pretrained_imagenet: bool = False,
    ) -> None:
        super().__init__()
        self.encoder, feature_dim = _build_encoder(pretrained_imagenet)
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.encoder(x)  # (B, 512, 1, 1)
        h = h.flatten(start_dim=1)  # (B, 512)
        z = self.projector(h)  # (B, projection_dim)
        return z

    def get_encoder(self) -> nn.Module:
        """Return the bare encoder (projection head discarded)."""
        return self.encoder


# ---------------------------------------------------------------------------
# FixMatch / Supervised classifier
# ---------------------------------------------------------------------------


class ResNet18Classifier(nn.Module):
    """ResNet-18 encoder + linear classification head."""

    def __init__(
        self,
        num_classes: int = 10,
        pretrained_imagenet: bool = False,
    ) -> None:
        super().__init__()
        self.encoder, feature_dim = _build_encoder(pretrained_imagenet)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        h = self.encoder(x)  # (B, 512, 1, 1)
        h = h.flatten(start_dim=1)  # (B, 512)
        return self.classifier(h)  # (B, num_classes)

    def encode(self, x: Tensor) -> Tensor:
        """Return feature vectors without the classification head."""
        h = self.encoder(x)
        return h.flatten(start_dim=1)

    def load_simclr_encoder(self, encoder: nn.Module) -> None:
        """Copy weights from a SimCLR encoder into this model's encoder."""
        self.encoder.load_state_dict(encoder.state_dict())


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average) model wrapper
# ---------------------------------------------------------------------------


class EMAModel:
    """Maintains an EMA copy of a model's parameters for stable pseudo-labels."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: dict[str, Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data

    def apply_shadow(self, model: nn.Module) -> None:
        """Load EMA weights into *model* in-place."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module, backup: dict[str, Tensor]) -> None:
        """Restore original weights after temporarily applying shadow."""
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])

    def state_dict(self) -> dict[str, Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        self.shadow = {k: v.clone() for k, v in state.items()}
