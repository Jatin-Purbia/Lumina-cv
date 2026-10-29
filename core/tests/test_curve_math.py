"""Executable documented contracts."""

from __future__ import annotations

import torch

from nta_zerodce.models.dce_net import apply_curves


def test_curve_math() -> None:
    """Canonical sign, endpoints, iteration order, monotonicity and finite gradients."""
    x = torch.linspace(0, 1, 81).reshape(1, 1, 9, 9).expand(1, 3, 9, 9).clone().requires_grad_()
    maps = torch.zeros(1, 24, 9, 9, requires_grad=True)
    assert torch.equal(apply_curves(x, maps), x)
    for coefficient in (-1.0, -0.5, 0.5, 1.0):
        result = apply_curves(x, maps + coefficient)
        assert 0 <= result.min() <= result.max() <= 1
        assert (result.flatten(2).diff() >= -1e-6).all()
        assert result[0, 0, 0, 0] == 0 and result[0, 0, -1, -1] == 1
    pixel = torch.full((1, 3, 1, 1), 0.25)
    assert torch.allclose(
        apply_curves(pixel, torch.full_like(pixel, 0.5), 1), torch.full_like(pixel, 0.34375)
    )
    assert torch.allclose(
        apply_curves(pixel, torch.full_like(pixel, -0.5), 1), torch.full_like(pixel, 0.15625)
    )
    torch.autograd.backward(apply_curves(x, maps).sum())
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert maps.grad is not None and maps.grad.abs().sum() > 0
