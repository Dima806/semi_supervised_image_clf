"""Tests for model.py: ResNet18WithProjection, ResNet18Classifier, EMAModel."""

from __future__ import annotations

import torch

from semi_supervised_image_clf.model import EMAModel, ResNet18Classifier, ResNet18WithProjection


def test_simclr_model_output_shape() -> None:
    model = ResNet18WithProjection(projection_dim=128)
    model.eval()
    x = torch.randn(4, 3, 64, 64)
    z = model(x)
    assert z.shape == (4, 128)


def test_simclr_projection_dim_configurable() -> None:
    model = ResNet18WithProjection(projection_dim=64)
    x = torch.randn(2, 3, 64, 64)
    assert model(x).shape == (2, 64)


def test_classifier_output_shape() -> None:
    model = ResNet18Classifier(num_classes=10)
    model.eval()
    x = torch.randn(8, 3, 64, 64)
    logits = model(x)
    assert logits.shape == (8, 10)


def test_classifier_encode_shape() -> None:
    model = ResNet18Classifier(num_classes=10)
    model.eval()
    x = torch.randn(4, 3, 64, 64)
    feats = model.encode(x)
    assert feats.shape == (4, 512)


def test_load_simclr_encoder() -> None:
    simclr = ResNet18WithProjection(projection_dim=128)
    clf = ResNet18Classifier(num_classes=10)
    clf.load_simclr_encoder(simclr.get_encoder())
    # Check that the first conv weight is shared (equal values)
    for (n1, p1), (_n2, p2) in zip(
        simclr.encoder.named_parameters(), clf.encoder.named_parameters(), strict=True
    ):
        assert torch.allclose(p1, p2), f"Encoder weight mismatch at {n1}"


def test_ema_model_update() -> None:
    model = ResNet18Classifier(num_classes=10)
    ema = EMAModel(model, decay=0.9)
    # Capture initial shadow values
    initial_shadow = {k: v.clone() for k, v in ema.shadow.items()}
    # Add 1 to all parameters and update EMA
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.ones_like(p))
    ema.update(model)
    # shadow = 0.9 * old + 0.1 * (old + 1) = old + 0.1
    # So each shadow value should have increased by approximately 0.1
    for name, shadow_val in ema.shadow.items():
        delta = shadow_val - initial_shadow[name]
        assert torch.allclose(delta, torch.full_like(delta, 0.1), atol=1e-5), (
            f"EMA shadow for {name} did not update correctly"
        )


def test_ema_apply_shadow() -> None:
    model = ResNet18Classifier(num_classes=10)
    ema = EMAModel(model, decay=0.0)  # decay=0 means shadow = current param immediately
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(2.0)
    ema.update(model)
    # Now apply shadow to a fresh model
    fresh = ResNet18Classifier(num_classes=10)
    ema.apply_shadow(fresh)
    for p in fresh.parameters():
        assert torch.allclose(p, torch.full_like(p, 2.0))
