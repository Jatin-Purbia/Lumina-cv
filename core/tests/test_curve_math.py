"""Curve-math assertions for apply_curves; TESTING strategy, METHOD baseline curve.

All inputs are seeded synthetic CPU tensors. See docs/METHOD.md for the equations.
"""

from __future__ import annotations

import pytest
import torch

from nta_zerodce.models.dce_net import apply_curves


def _constant(value: float, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.full(shape, value, dtype=torch.float32)


def test_zero_curves_are_identity_and_endpoints_are_fixed() -> None:
    """A=0 leaves x unchanged after eight updates; x=0 and x=1 stay fixed for any A in [-1,1]."""
    generator = torch.Generator().manual_seed(0)
    x = torch.rand(2, 3, 8, 8, generator=generator)
    zeros = torch.zeros(2, 24, 8, 8)
    assert torch.allclose(apply_curves(x, zeros), x, atol=1e-7)

    for a in (-1.0, -0.37, 0.0, 0.6, 1.0):
        curves = _constant(a, (2, 24, 8, 8))
        assert torch.equal(apply_curves(torch.zeros(2, 3, 8, 8), curves), torch.zeros(2, 3, 8, 8))
        ones = torch.ones(2, 3, 8, 8)
        assert torch.allclose(apply_curves(ones, curves), ones, atol=1e-7)

    random_curves = torch.rand(2, 24, 8, 8, generator=generator) * 2 - 1
    assert torch.equal(
        apply_curves(torch.zeros(2, 3, 8, 8), random_curves), torch.zeros(2, 3, 8, 8)
    )
    assert torch.allclose(
        apply_curves(torch.ones(2, 3, 8, 8), random_curves), torch.ones(2, 3, 8, 8), atol=1e-6
    )


def test_single_step_numeric_examples_fix_sign_convention() -> None:
    """LE(.25,.5)=.34375 and LE(.25,-.5)=.15625 pin the plus-sign convention (one iteration)."""
    x = _constant(0.25, (1, 3, 2, 2))
    up = apply_curves(x, _constant(0.5, (1, 3, 2, 2)), iterations=1)
    down = apply_curves(x, _constant(-0.5, (1, 3, 2, 2)), iterations=1)
    assert torch.allclose(up, _constant(0.34375, (1, 3, 2, 2)), atol=1e-7)
    assert torch.allclose(down, _constant(0.15625, (1, 3, 2, 2)), atol=1e-7)


def test_iteration_major_rgb_channel_ordering() -> None:
    """Channel 3*i+c drives iteration i and colour c; unused iterations leave the value alone."""
    x = _constant(0.25, (1, 3, 2, 2))
    for iteration in range(8):
        for colour in range(3):
            curves = torch.zeros(1, 24, 2, 2)
            curves[:, 3 * iteration + colour] = 0.5
            out = apply_curves(x, curves)
            for channel in range(3):
                expected = 0.34375 if channel == colour else 0.25
                assert torch.allclose(out[:, channel], _constant(expected, (1, 2, 2)), atol=1e-7)

    # Order matters: iteration 0 is applied first. Applying +.5 at i=0 and -.5 at i=1 differs
    # from the swapped assignment because the two updates do not commute.
    first = torch.zeros(1, 24, 1, 1)
    first[:, 0] = 0.5
    first[:, 3] = -0.5
    swapped = torch.zeros(1, 24, 1, 1)
    swapped[:, 0] = -0.5
    swapped[:, 3] = 0.5
    x1 = _constant(0.25, (1, 3, 1, 1))
    step1 = 0.25 + 0.5 * 0.25 * 0.75
    expected_first = step1 + (-0.5) * step1 * (1 - step1)
    assert torch.allclose(apply_curves(x1, first)[0, 0, 0, 0], torch.tensor(expected_first))
    assert not torch.allclose(apply_curves(x1, first), apply_curves(x1, swapped))


def test_outputs_bounded_and_monotone_for_fixed_curves() -> None:
    """Eight updates keep [0,1] and are non-decreasing in x for fixed A in [-1,1]."""
    generator = torch.Generator().manual_seed(1)
    ramp = torch.linspace(0.0, 1.0, 257).view(1, 1, 1, 257).expand(1, 3, 1, 257).contiguous()
    for _ in range(20):
        a = torch.rand(1, 24, 1, 1, generator=generator) * 2 - 1
        curves = a.expand(1, 24, 1, 257).contiguous()
        out = apply_curves(ramp, curves)
        assert out.min() >= 0.0 and out.max() <= 1.0
        assert (out[..., 1:] - out[..., :-1]).min() >= -1e-6
    for a_value in (-1.0, 1.0):
        out = apply_curves(ramp, _constant(a_value, (1, 24, 1, 257)))
        assert out.min() >= 0.0 and out.max() <= 1.0
        assert (out[..., 1:] - out[..., :-1]).min() >= -1e-6


def test_positive_curves_brighten_and_negative_curves_darken_interior_pixels() -> None:
    """For x in (0,1) the sign of A fixes the direction of change."""
    x = torch.linspace(0.05, 0.95, 19).view(1, 1, 1, 19).expand(1, 3, 1, 19).contiguous()
    assert (apply_curves(x, _constant(0.4, (1, 24, 1, 19))) > x).all()
    assert (apply_curves(x, _constant(-0.4, (1, 24, 1, 19))) < x).all()


def test_gradients_are_finite_including_boundaries() -> None:
    """Autograd derivatives wrt image and curves are finite at x=0, x=1 and interior points."""
    x = torch.tensor([0.0, 1.0, 0.25, 0.5, 0.9]).view(1, 1, 1, 5).expand(1, 3, 1, 5)
    x = x.clone().requires_grad_(True)
    curves = (
        torch.rand(1, 24, 1, 5, generator=torch.Generator().manual_seed(2)) * 2 - 1
    ).requires_grad_(True)
    apply_curves(x, curves).sum().backward()
    assert x.grad is not None and curves.grad is not None
    assert torch.isfinite(x.grad).all() and torch.isfinite(curves.grad).all()
    # Boundary pixels have zero sensitivity to the curve parameters.
    assert torch.equal(curves.grad[..., 0], torch.zeros_like(curves.grad[..., 0]))
    assert torch.allclose(curves.grad[..., 1], torch.zeros_like(curves.grad[..., 1]), atol=1e-6)


def test_input_validation_rejects_bad_shapes() -> None:
    """Mismatched channel counts, spatial sizes and iteration counts raise ValueError."""
    x = torch.rand(1, 3, 4, 4)
    with pytest.raises(ValueError):
        apply_curves(x, torch.zeros(1, 23, 4, 4))
    with pytest.raises(ValueError):
        apply_curves(x, torch.zeros(1, 24, 4, 5))
    with pytest.raises(ValueError):
        apply_curves(x, torch.zeros(1, 24, 4, 4), iterations=0)
    with pytest.raises(ValueError):
        apply_curves(torch.rand(1, 4, 4, 4), torch.zeros(1, 24, 4, 4))
