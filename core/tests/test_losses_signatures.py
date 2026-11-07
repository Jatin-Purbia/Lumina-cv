"""Loss contract assertions: scalar/non-negative/differentiable, analytic values, masks, gates.

Seeded synthetic CPU tensors only. Equations: docs/METHOD.md; strategy: docs/TESTING.md.
The temporal loss and metrics are owned and tested elsewhere (flow/metrics tests).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest
import torch

from nta_zerodce.losses import _ops
from nta_zerodce.losses.adaptive_exposure import adaptive_exposure_loss, adaptive_targets
from nta_zerodce.losses.chroma import chroma_loss
from nta_zerodce.losses.color import color_loss
from nta_zerodce.losses.edge import edge_loss
from nta_zerodce.losses.exposure import exposure_loss
from nta_zerodce.losses.highlight import highlight_loss
from nta_zerodce.losses.noise import fidelity_loss, invariance_loss, noise_loss
from nta_zerodce.losses.registry import (
    LOSS_CONTRACTS,
    LOSS_REGISTRY,
    LossSpec,
    build_losses,
    register_loss,
)
from nta_zerodce.losses.smoothness import smoothness_loss
from nta_zerodce.losses.spatial import spatial_loss
from nta_zerodce.models.factory import build_model
from nta_zerodce.models.nta_zerodce import compute_gain
from nta_zerodce.training.config import Config, FeatureFlags, LossWeights

WEIGHTS = {
    "spa": 1.0, "exp": 10.0, "col": 5.0, "tv": 200.0, "edge": 20.0, "chroma": 20.0,
    "highlight": 20.0, "noise": 20.0, "invariance": 10.0, "fidelity": 1.0, "temporal": 10.0,
}  # fmt: skip
ALL_FLAGS = {
    "noise_aware": True, "denoise": True, "edge": True, "chroma": True,
    "adaptive_exp": True, "temporal": True, "light": False,
}  # fmt: skip


def _gray(value: float, size: int = 32, batch: int = 2) -> torch.Tensor:
    return torch.full((batch, 3, size, size), value, dtype=torch.float32)


def _textured(seed: int, size: int = 32, batch: int = 2) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(batch, 3, size, size, generator=generator)


def _step_edge(size: int = 32, low: float = 0.2, high: float = 0.6) -> torch.Tensor:
    image = torch.full((1, 3, size, size), low)
    image[..., size // 2 :] = high
    return image


def _context() -> dict[str, torch.Tensor]:
    """Every tensor name a registered loss may request, all requiring grad to expose leaks."""
    image = _textured(0) * 0.4 + 0.3
    context = {
        "image": image,
        "enhanced": _textured(1) * 0.5 + 0.25,
        "A": _textured(2, batch=2).repeat(1, 8, 1, 1)[:, :24] * 2 - 1,
        "sigma": _textured(3)[:, :1] * 0.05,
        "gain": 1.0 + 3.0 * _textured(4)[:, :1],
        "target_sigma": _textured(5)[:, :1] * 0.05,
        "noisy_enhanced": _textured(6) * 0.5 + 0.25,
        "teacher_enhanced": _textured(7) * 0.5 + 0.25,
        "teacher_raw": _textured(8) * 0.5 + 0.25,
    }
    # A bright strip in the input that the output brightens further covers the highlight mask.
    context["image"][..., :8] = 0.82 + 0.05 * context["image"][..., :8]
    context["enhanced"][..., :8] = 0.97
    context["targets"] = adaptive_targets(context["image"], context["sigma"])
    return {name: value.clone().requires_grad_(True) for name, value in context.items()}


STUDENT_INPUTS = {"enhanced", "A", "noisy_enhanced"}
DETACHED_INPUTS = {"gain", "target_sigma", "teacher_enhanced", "teacher_raw", "targets"}


def test_every_active_loss_is_scalar_nonnegative_finite_and_differentiable() -> None:
    """Each registered loss (except temporal) consumes its named tensors and trains students."""
    specs = build_losses(ALL_FLAGS, WEIGHTS)
    assert set(specs) == {
        "spa", "adaptive_exp", "col", "tv", "edge", "chroma", "highlight", "noise",
        "invariance", "fidelity", "temporal",
    }  # fmt: skip
    for name, spec in specs.items():
        if name == "temporal":
            continue  # owned by the flow/temporal tests
        context = _context()
        value = spec.callable(*[context[key] for key in spec.required_inputs])
        assert value.dim() == 0 and value.dtype == torch.float32, name
        assert torch.isfinite(value) and value >= 0, name
        assert value.requires_grad, name
        value.backward()
        grads = {key: context[key].grad for key in spec.required_inputs}
        student = [k for k in spec.required_inputs if k in STUDENT_INPUTS or k == "sigma"]
        if name in {"edge", "chroma"}:
            student = [k for k in student if k != "sigma"]  # sigma is detached there
        for key in student:
            grad = grads[key]
            assert grad is not None and torch.isfinite(grad).all(), (name, key)
        magnitudes = [float(g.abs().sum()) for k in student if (g := grads[k]) is not None]
        assert any(m > 0 for m in magnitudes), name
        for key in spec.required_inputs:
            if key in DETACHED_INPUTS or (key == "sigma" and name in {"edge", "chroma"}):
                grad = grads[key]
                assert grad is None or float(grad.abs().sum()) == 0.0, (name, key)


def test_losses_follow_the_input_dtype_and_stay_on_device() -> None:
    """Losses return a scalar on the input device (meta run proves no hidden CPU tensors)."""
    image = torch.empty(2, 3, 32, 32, device="meta")
    sigma = torch.empty(2, 1, 32, 32, device="meta")
    gain = torch.empty(2, 1, 32, 32, device="meta")
    curves = torch.empty(2, 24, 32, 32, device="meta")
    results = [
        spatial_loss(image, image),
        exposure_loss(image),
        color_loss(image),
        smoothness_loss(curves),
        highlight_loss(image, image),
        chroma_loss(image, image, sigma),
        edge_loss(image, image, sigma, gain),
        adaptive_exposure_loss(image, adaptive_targets(image, sigma)),
        noise_loss(sigma, sigma),
        invariance_loss(image, image),
        fidelity_loss(image, image),
    ]
    assert all(r.device.type == "meta" and r.dim() == 0 for r in results)


def test_constant_gray_identities_and_initial_scale_table() -> None:
    """METHOD's analytic constant-gray values."""
    gray_target = _gray(0.6, 32)
    assert float(exposure_loss(gray_target)) == pytest.approx(0.0, abs=1e-7)
    assert float(exposure_loss(_gray(0.1))) == pytest.approx(0.25, abs=1e-6)  # (.1-.6)^2
    assert float(exposure_loss(_gray(0.1), target=0.5, patch_size=8)) == pytest.approx(
        0.16, abs=1e-6
    )
    assert float(color_loss(_gray(0.3))) == 0.0
    assert float(smoothness_loss(torch.full((2, 24, 8, 8), 0.4))) == 0.0
    textured = _textured(0)
    assert float(spatial_loss(textured, textured)) == pytest.approx(0.0, abs=1e-12)
    assert float(spatial_loss(_gray(0.1), _gray(0.6))) == pytest.approx(0.0, abs=1e-12)
    # Noise: sigmoid-near-zero-logit prediction .1 against target .02 gives .08.
    predicted = torch.full((2, 1, 8, 8), 0.1)
    assert float(noise_loss(predicted, torch.full((2, 1, 8, 8), 0.02))) == pytest.approx(
        0.08, abs=1e-7
    )
    # Adaptive exposure at saturated sigma: target .45, input/output .1 -> (.1-.45)^2.
    image = _gray(0.1)
    targets = adaptive_targets(image, torch.full((2, 1, 32, 32), 0.2))
    assert targets.shape == (2, 1, 2, 2)
    assert torch.allclose(targets, torch.full_like(targets, 0.45), atol=1e-6)
    assert float(adaptive_exposure_loss(image, targets)) == pytest.approx(0.1225, abs=1e-6)
    # With sigma = 0 the noise penalty vanishes and a dark image targets exactly .6.
    assert torch.allclose(
        adaptive_targets(image, torch.zeros(2, 1, 32, 32)), torch.full_like(targets, 0.6)
    )
    # Highlight: .1 brightening of covered pixels gives .01 regardless of coverage.
    bright = _gray(0.85)
    assert float(highlight_loss(bright, bright + 0.1)) == pytest.approx(0.01, abs=1e-6)
    mixed = bright.clone()
    mixed[..., 16:] = 0.2  # uncovered half gets a huge increase that must not count
    out = mixed + 0.1
    out[..., 16:] = 0.9
    assert float(highlight_loss(mixed, out)) == pytest.approx(0.01, abs=1e-6)
    # Darkening bright pixels is never penalized.
    assert float(highlight_loss(bright, bright - 0.3)) == 0.0
    # Chroma: a reliable uniform gray keeps its chromaticity whatever its brightness.
    assert float(chroma_loss(_gray(0.5), _gray(0.3), torch.zeros(2, 1, 32, 32))) == pytest.approx(
        0.0, abs=1e-6
    )
    # Edge: flat identity gives zero.
    flat_sigma = torch.zeros(2, 1, 32, 32)
    flat_gain = torch.ones(2, 1, 32, 32)
    assert float(edge_loss(_gray(0.3), _gray(0.3), flat_sigma, flat_gain)) == pytest.approx(
        0.0, abs=1e-6
    )


def test_loss_values_for_hand_computable_inputs() -> None:
    """Color, spatial and smoothness reductions against explicit arithmetic."""
    # Colour: per-image channel means (.5,.3,.1): d = .2, .4, .2 -> sqrt(.2^4+.4^4+.2^4).
    out = torch.zeros(1, 3, 4, 4)
    out[:, 0], out[:, 1], out[:, 2] = 0.5, 0.3, 0.1
    expected = (0.2**4 + 0.4**4 + 0.2**4) ** 0.5
    assert float(color_loss(out)) == pytest.approx(expected, rel=1e-5)
    # Smoothness: a horizontal ramp of slope s per pixel has dx^2=s^2, dy=0 -> 2*s^2.
    ramp = torch.arange(8, dtype=torch.float32).view(1, 1, 1, 8).expand(1, 24, 8, 8) * 0.1
    assert float(smoothness_loss(ramp)) == pytest.approx(2 * 0.1**2, rel=1e-5)
    # Spatial: input regions flat, output has one region 0.2 above its neighbour.
    x = torch.zeros(1, 3, 8, 4)  # 2x1 regions with patch 4
    y = x.clone()
    y[:, :, 4:, :] = 0.2
    # Two regions, K=1*2*1=2; the single pair is visited twice, each (0.2-0)^2 = .04.
    assert float(spatial_loss(x, y)) == pytest.approx((0.04 + 0.04) / 2, rel=1e-5)
    # Exposure: only the pooled mean matters, not the arrangement inside a patch.
    patch = torch.zeros(1, 3, 16, 16)
    patch[..., :8, :] = 1.2  # mean 0.6 although each half differs
    assert float(exposure_loss(patch)) == pytest.approx(0.0, abs=1e-6)


def test_color_loss_is_zero_safe() -> None:
    """Exact-zero channel differences give zero loss and finite (zero) gradients."""
    output = _gray(0.4).requires_grad_(True)
    loss = color_loss(output)
    loss.backward()
    assert loss.item() == 0.0
    assert output.grad is not None and torch.isfinite(output.grad).all()
    assert float(output.grad.abs().sum()) == 0.0
    # Slightly off-gray inputs still get a finite, nonzero gradient.
    off = _gray(0.4)
    off[:, 0] += 1e-4
    off.requires_grad_(True)
    color_loss(off).backward()
    assert off.grad is not None and torch.isfinite(off.grad).all() and off.grad.abs().sum() > 0


def test_empty_masks_return_graph_connected_zero_without_nan() -> None:
    """Empty highlight/chroma/edge masks give exact zero tied to the output graph."""
    sigma_big = torch.full((1, 1, 32, 32), 0.2)
    cases: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
        "highlight": lambda out: highlight_loss(_gray(0.5, batch=1), out),
        "chroma": lambda out: chroma_loss(_gray(0.02, batch=1), out, sigma_big * 0),
        "edge": lambda out: edge_loss(_step_edge(), out, sigma_big, torch.ones_like(sigma_big)),
    }
    for name, make in cases.items():
        output = torch.full((1, 3, 32, 32), 0.5, requires_grad=True)
        loss = make(output)
        assert loss.requires_grad and loss.grad_fn is not None, name
        assert loss.item() == 0.0, name
        loss.backward()
        assert output.grad is not None and torch.isfinite(output.grad).all(), name
        assert float(output.grad.abs().sum()) == 0.0, name


def test_masked_mean_normalizes_by_covered_pixels() -> None:
    """masked_mean(v, m, C) = sum(m*v)/(C*sum(m)+eps); empty mask yields zero."""
    value = torch.arange(24, dtype=torch.float32).view(2, 3, 2, 2)
    mask = torch.zeros(2, 1, 2, 2)
    mask[0, 0, 0, 0] = 1.0
    mask[1, 0, 1, 1] = 0.5
    expected = (value[0, :, 0, 0].sum() + 0.5 * value[1, :, 1, 1].sum()) / (3 * 1.5 + 1e-6)
    assert float(_ops.masked_mean(value, mask, 3)) == pytest.approx(float(expected), rel=1e-6)
    assert float(_ops.masked_mean(value, torch.zeros_like(mask), 3)) == 0.0


def test_edge_operators_and_loss() -> None:
    """Sobel/8 returns the ramp slope, flat regions give zero, gain scales the reference."""
    slope = 0.01
    ramp = (torch.arange(32, dtype=torch.float32) * slope).view(1, 1, 1, 32).expand(1, 1, 32, 32)
    magnitude = _ops.sobel_magnitude(ramp.contiguous())
    assert torch.allclose(
        magnitude[..., 2:-2], torch.full_like(magnitude[..., 2:-2], slope), atol=1e-6
    )
    assert float(_ops.sobel_magnitude(torch.zeros(1, 1, 8, 8)).abs().max()) < 1e-9
    # Gaussian blur preserves constants and the sum of an impulse; kernel is normalized.
    constant = torch.full((1, 1, 9, 9), 0.7)
    assert torch.allclose(_ops.gaussian_blur_reflect(constant, 0.8, 5), constant, atol=1e-6)
    taps = _ops.gaussian_kernel_1d(0.8, 5, constant)
    assert float(taps.sum()) == pytest.approx(1.0, abs=1e-6) and taps[2] == taps.max()

    image = _step_edge()
    sigma = torch.zeros(1, 1, 32, 32)
    flat_output = torch.full((1, 3, 32, 32), 0.4)
    loss_one = edge_loss(image, flat_output, sigma, torch.ones(1, 1, 32, 32))
    loss_two = edge_loss(image, flat_output, sigma, torch.full((1, 1, 32, 32), 2.0))
    assert float(loss_one) > 0.0
    # With a flat output d_out = 0, so the loss is the weighted mean of gain*d_in: linear in gain.
    assert float(loss_two) == pytest.approx(2 * float(loss_one), rel=1e-4)
    # A noise-dominated edge (large sigma) is excluded entirely.
    assert (
        float(
            edge_loss(image, flat_output, torch.full((1, 1, 32, 32), 0.2), torch.ones(1, 1, 32, 32))
        )
        == 0.0
    )


def test_chroma_uses_local_average_after_normalization_and_reliability() -> None:
    """Reference is avgpool(c(x)) (not c(avgpool x)); unreliable dark pixels are ignored."""
    image = _textured(3, size=16, batch=1) * 0.5 + 0.3
    sigma = torch.zeros(1, 1, 16, 16)
    # Output equal to the input: only the local-average mismatch remains, which is small but > 0.
    identity = float(chroma_loss(image, image, sigma))
    assert 0.0 < identity < 0.2
    # Output carrying the local reference chromaticity scaled arbitrarily gives zero loss.
    c_in = image / (image.sum(dim=1, keepdim=True) + 1e-6)
    c_ref = _ops.avg_pool_reflect(c_in, 5)
    matched = c_ref * 3.0 * 0.5
    assert float(chroma_loss(image, matched, sigma)) == pytest.approx(0.0, abs=1e-5)
    # Dark pixels (Y(x) < .05) get zero weight even when the output colour is wrong.
    dark = torch.full((1, 3, 16, 16), 0.02)
    wrong = torch.zeros(1, 3, 16, 16)
    wrong[:, 0] = 0.5
    assert float(chroma_loss(dark, wrong, sigma)) == 0.0
    # Inflating sigma lowers the weights, but sigma receives no gradient.
    sigma_req = torch.full((1, 1, 16, 16), 0.01, requires_grad=True)
    chroma_loss(image, (wrong + 0.2).requires_grad_(True), sigma_req).backward()
    assert sigma_req.grad is None


def test_adaptive_targets_are_detached_and_follow_the_formula() -> None:
    """Loss cannot lower targets by inflating sigma; target range and smoothstep branch."""
    image = _textured(0, size=32).requires_grad_(True)
    sigma = torch.full((2, 1, 32, 32), 0.03, requires_grad=True)
    targets = adaptive_targets(image, sigma)
    assert not targets.requires_grad and targets.grad_fn is None
    output = (_textured(1) * 0.5).requires_grad_(True)
    adaptive_exposure_loss(output, targets).backward()
    assert sigma.grad is None and image.grad is None and output.grad is not None
    # Formula check for dark and bright patches: bright input is pulled towards itself.
    patch = torch.cat([torch.full((1, 3, 16, 16), 0.2), torch.full((1, 3, 16, 16), 0.9)], dim=3)
    zero_sigma = torch.zeros(1, 1, 16, 32)
    t = adaptive_targets(patch, zero_sigma)
    assert t.shape == (1, 1, 1, 2)
    assert float(t[0, 0, 0, 0]) == pytest.approx(0.6, abs=1e-6)  # smoothstep(.2)=0 -> target
    assert float(t[0, 0, 0, 1]) == pytest.approx(0.9, abs=1e-5)  # smoothstep(.9)=1 -> luminance
    # Half-saturated noise penalty: sigma .025 -> 0.5*0.15.
    half = adaptive_targets(_gray(0.2, 16, 1), torch.full((1, 1, 16, 16), 0.025))
    assert float(half) == pytest.approx(0.6 - 0.075, abs=1e-6)
    # All targets stay in [.45, 1] for in-range inputs.
    sweep = adaptive_targets(_textured(5), _textured(6)[:, :1] * 0.4)
    assert float(sweep.min()) >= 0.45 - 1e-6 and float(sweep.max()) <= 1.0 + 1e-6
    with pytest.raises(ValueError):
        adaptive_exposure_loss(output, torch.zeros(2, 1, 3, 3))


def test_teacher_targets_receive_no_gradient() -> None:
    """Noise, invariance and fidelity losses detach their targets/teachers."""
    pred = torch.rand(1, 1, 4, 4, requires_grad=True)
    target = torch.rand(1, 1, 4, 4, requires_grad=True)
    noise_loss(pred, target).backward()
    assert pred.grad is not None and target.grad is None
    student = torch.rand(1, 3, 4, 4, requires_grad=True)
    teacher = torch.rand(1, 3, 4, 4, requires_grad=True)
    for loss in (invariance_loss, fidelity_loss):
        student.grad = None
        loss(student, teacher).backward()
        assert student.grad is not None and teacher.grad is None
    assert invariance_loss(student, student.detach()).item() == 0.0
    with pytest.raises(ValueError):
        noise_loss(pred, torch.rand(1, 1, 5, 4))


def test_edge_and_denoise_gain_use_detached_gain_from_model_helper() -> None:
    """compute_gain feeds edge_loss; the loss never back-propagates into the gain or sigma."""
    image = _step_edge()
    raw = (image * 2.0).clamp(0, 1)
    gain = compute_gain(image, raw)
    sigma = torch.zeros(1, 1, 32, 32, requires_grad=True)
    output = raw.clone().requires_grad_(True)
    loss = edge_loss(image, output, sigma, gain)
    loss.backward()
    assert output.grad is not None and torch.isfinite(output.grad).all()
    assert sigma.grad is None


def test_registry_metadata_and_mutual_exposure_gate() -> None:
    """build_losses applies METHOD's total-objective gates, weight inheritance and dependency."""
    assert LOSS_REGISTRY.keys() == LOSS_CONTRACTS.keys()
    baseline = build_losses({}, WEIGHTS)
    assert list(baseline) == ["spa", "exp", "col", "tv"]
    assert [baseline[k].default_weight for k in baseline] == [1.0, 10.0, 5.0, 200.0]
    assert baseline["exp"].callable is exposure_loss

    adaptive = build_losses({"adaptive_exp": True}, {**WEIGHTS, "exp": 7.0, "highlight": 3.0})
    assert "exp" not in adaptive and "adaptive_exp" in adaptive and "highlight" in adaptive
    assert adaptive["adaptive_exp"].default_weight == 7.0  # inherits the exp weight
    assert adaptive["highlight"].default_weight == 3.0  # own weight
    assert adaptive["adaptive_exp"].callable is adaptive_exposure_loss

    for flags in ({}, {"adaptive_exp": True}, ALL_FLAGS):
        active = build_losses(flags, WEIGHTS)
        assert ("exp" in active) != ("adaptive_exp" in active)

    noisy = build_losses({"noise_aware": True}, WEIGHTS)
    assert {"noise", "invariance"} <= set(noisy) and "fidelity" not in noisy
    assert "fidelity" in build_losses({"noise_aware": True, "denoise": True}, WEIGHTS)
    assert set(build_losses({"edge": True, "chroma": True, "temporal": True}, WEIGHTS)) == {
        "spa", "exp", "col", "tv", "edge", "chroma", "temporal",
    }  # fmt: skip
    # Built-in metadata is untouched (the gate returns copies carrying resolved weights).
    assert LOSS_REGISTRY["exp"].default_weight == 10.0
    assert isinstance(baseline["spa"], LossSpec)

    with pytest.raises(ValueError):
        build_losses({"denoise": True}, WEIGHTS)
    with pytest.raises(ValueError):
        build_losses({"bogus": True}, WEIGHTS)
    with pytest.raises(ValueError):
        build_losses({}, {k: v for k, v in WEIGHTS.items() if k != "tv"})
    with pytest.raises(ValueError):
        build_losses({}, {**WEIGHTS, "tv": -1.0})
    with pytest.raises(ValueError):
        build_losses({}, {**WEIGHTS, "spa": float("nan")})
    # The default LossWeights record plugs straight in.
    defaults = {name: getattr(LossWeights(), name) for name in WEIGHTS}
    assert defaults == WEIGHTS


def test_register_loss_decorator_rejects_duplicates_and_unknown_flags() -> None:
    """Dynamic registration uses a separate registry and validates names and gates."""
    from nta_zerodce.losses import registry

    before = dict(LOSS_REGISTRY)
    try:

        @register_loss("unit_test_loss", "edge", 2.5, ("enhanced",))
        def toy(enhanced: torch.Tensor) -> torch.Tensor:
            return enhanced.mean()

        assert LOSS_REGISTRY == before  # built-in metadata untouched
        active = build_losses({"edge": True}, {**WEIGHTS, "unit_test_loss": 0.5})
        assert active["unit_test_loss"].callable is toy
        assert active["unit_test_loss"].default_weight == 0.5
        assert "unit_test_loss" not in build_losses({}, {**WEIGHTS, "unit_test_loss": 0.5})
        with pytest.raises(ValueError):
            register_loss("unit_test_loss", None, 1.0, ())
        with pytest.raises(ValueError):
            register_loss("spa", None, 1.0, ())
        with pytest.raises(ValueError):
            register_loss("another", "nonexistent_flag", 1.0, ())
        with pytest.raises(ValueError):
            register_loss("another", "not nonexistent_flag", 1.0, ())
        with pytest.raises(ValueError):
            register_loss("another", None, -1.0, ())
        # The literal gate used by the baseline exposure term is accepted.
        register_loss("unit_test_not", "not adaptive_exp", 1.0, ())(toy)
        assert "unit_test_not" in build_losses({}, {**WEIGHTS, "unit_test_not": 1.0})
    finally:
        registry._DYNAMIC_REGISTRY.pop("unit_test_loss", None)
        registry._DYNAMIC_REGISTRY.pop("unit_test_not", None)
    assert LOSS_REGISTRY == before


def test_shape_validation_errors() -> None:
    """Clear ValueErrors for malformed inputs."""
    with pytest.raises(ValueError):
        spatial_loss(_gray(0.1, 32), _gray(0.1, 16))
    with pytest.raises(ValueError):
        exposure_loss(_gray(0.1, 8))  # smaller than one 16x16 patch
    with pytest.raises(ValueError):
        smoothness_loss(torch.zeros(1, 24, 1, 8))
    with pytest.raises(ValueError):
        color_loss(torch.zeros(1, 1, 8, 8))


def test_tiny_overfit_improves_exposure_error() -> None:
    """Two 32x32 low-light images, 30 Adam steps: finite losses, exposure error at least halves.

    Declared tolerance: final exposure loss < 0.5 * initial (the initial value is ~0.25 for
    mean ~0.1 inputs; with lr 2e-3 the observed ratio is below 0.01). A sanity check, not a
    training-quality experiment, so no per-step monotonicity is required.
    """
    torch.manual_seed(0)
    config = replace(Config(), flags=FeatureFlags())
    model = build_model(config)
    generator = torch.Generator().manual_seed(0)
    images = torch.rand(2, 3, 32, 32, generator=generator) * 0.15 + 0.03
    specs = build_losses({}, WEIGHTS)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    exposure_history: list[float] = []
    for _ in range(30):
        out = model(images)
        context = {"image": images, "enhanced": out["enhanced"], "A": out["A"]}
        terms = {
            name: spec.default_weight * spec.callable(*[context[k] for k in spec.required_inputs])
            for name, spec in specs.items()
        }
        total = sum(terms.values())
        assert torch.isfinite(total)
        optimizer.zero_grad()
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        assert torch.isfinite(grad_norm)
        optimizer.step()
        exposure_history.append(float(exposure_loss(out["enhanced"]).detach()))
    # The head gets a nonzero gradient (the curve estimator is actually learning).
    assert model.dce.head.weight.grad is not None and model.dce.head.weight.grad.abs().sum() > 0
    final = float(exposure_loss(model(images)["enhanced"]).detach())
    assert exposure_history[0] > 0.15
    assert final < 0.5 * exposure_history[0]
