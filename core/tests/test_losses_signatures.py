"""Executable documented contracts."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from nta_zerodce.data.noise_synth import sample_noise
from nta_zerodce.losses.adaptive_exposure import adaptive_targets
from nta_zerodce.losses.registry import build_losses
from nta_zerodce.losses.temporal import temporal_loss
from nta_zerodce.training.config import FeatureFlags
from tests.support import small_config


def test_loss_gradients_and_targets(tmp_path: Path) -> None:
    """Every registered loss is scalar, with detached targets and active student gradients."""
    config = small_config(tmp_path, True)
    image = torch.full((2, 3, 32, 32), 0.1)
    image[..., 16:] = 0.4
    output = (image + 0.04).clone().requires_grad_()
    sigma = torch.full((2, 1, 32, 32), 0.02, requires_grad=True)
    targets = adaptive_targets(image, sigma)
    assert not targets.requires_grad and 0.45 <= targets.min() <= targets.max() <= 1
    context = {
        "image": image,
        "enhanced": output,
        "sigma": sigma,
        "gain": torch.ones_like(sigma),
        "A": torch.randn(2, 24, 32, 32, requires_grad=True),
        "target_sigma": torch.zeros_like(sigma),
        "noisy_enhanced": output,
        "teacher_enhanced": image,
        "teacher_raw": image,
        "targets": targets,
        "previous_output": image,
        "current_output": output,
        "backward_flow": torch.zeros(2, 2, 32, 32),
        "mask": torch.ones_like(sigma),
    }
    for spec in build_losses(asdict(config.flags), asdict(config.weights)).values():
        loss = spec.callable(*(context[key] for key in spec.required_inputs))
        assert loss.ndim == 0 and torch.isfinite(loss) and loss.device == image.device
        torch.autograd.backward(loss, retain_graph=True)
    assert (
        output.grad is not None
        and torch.isfinite(output.grad).all()
        and output.grad.abs().sum() > 0
    )
    zero = temporal_loss(image, output, context["backward_flow"], torch.zeros_like(sigma))
    assert zero == 0 and zero.requires_grad
    # Baseline diagnostics have exact mathematical expectations.
    baseline = build_losses(asdict(FeatureFlags()), asdict(config.weights))
    gray = torch.full_like(output, 0.1, requires_grad=True)
    assert baseline["spa"].callable(gray, gray) == 0
    assert baseline["col"].callable(gray) == 0
    assert baseline["tv"].callable(torch.zeros(2, 24, 32, 32)) == 0
    assert torch.allclose(baseline["exp"].callable(gray), torch.tensor(0.25))
    torch.autograd.backward(baseline["col"].callable(gray))
    assert gray.grad is not None and torch.isfinite(gray.grad).all()


def test_noise_calibration() -> None:
    """Known pre-clipping variance, per-image coefficients and zero-noise identity."""
    image = torch.full((2, 3, 192, 192), 0.5)
    noisy, sigma = sample_noise(image, torch.tensor([0.0, 0.002]), torch.tensor([0.02, 0.03]))
    assert torch.allclose(sigma[:, 0, 0, 0], torch.tensor([0.02, (0.002 * 0.5 + 0.03**2) ** 0.5]))
    assert torch.allclose((noisy - image).std((1, 2, 3)), sigma[:, 0, 0, 0], rtol=0.03)
    unchanged, zero = sample_noise(image, torch.tensor(0.0), torch.tensor(0.0))
    assert torch.equal(unchanged, image) and torch.count_nonzero(zero) == 0
