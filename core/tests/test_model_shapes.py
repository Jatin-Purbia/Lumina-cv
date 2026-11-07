"""Model shape, range, gradient and composition assertions; TESTING strategy.

Seeded synthetic CPU tensors only; images are at most 64x80.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from nta_zerodce.models.dce_net import DCENet, LightDCENet
from nta_zerodce.models.denoiser import ResidualDenoiser
from nta_zerodce.models.factory import build_model
from nta_zerodce.models.noise_branch import NoiseBranch
from nta_zerodce.models.nta_zerodce import NTAZeroDCE, compute_gain
from nta_zerodce.training.config import Config, FeatureFlags

OUTPUT_KEYS = {"enhanced", "enhanced_raw", "A", "sigma"}


def _config(**flags: bool) -> Config:
    return replace(Config(), flags=FeatureFlags(**flags))


def _model(**flags: bool) -> NTAZeroDCE:
    torch.manual_seed(0)
    return build_model(_config(**flags))


def _image(batch: int = 2, height: int = 64, width: int = 80, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(batch, 3, height, width, generator=generator) * 0.3


def _count(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def test_baseline_forward_shapes_ranges_and_zero_sigma() -> None:
    """Baseline B=2,H=64,W=80: RGB outputs, A=(2,24,64,80), exact-zero sigma, same device."""
    model = _model()
    x = _image()
    out = model(x)
    assert set(out) == OUTPUT_KEYS
    assert out["enhanced"].shape == out["enhanced_raw"].shape == (2, 3, 64, 80)
    assert out["A"].shape == (2, 24, 64, 80)
    assert out["sigma"].shape == (2, 1, 64, 80)
    assert torch.equal(out["sigma"], torch.zeros(2, 1, 64, 80))
    assert torch.equal(out["enhanced"], out["enhanced_raw"])
    for key in ("enhanced", "enhanced_raw"):
        assert out[key].dtype == torch.float32 and out[key].device == x.device
        assert torch.isfinite(out[key]).all()
        assert out[key].min() >= 0.0 and out[key].max() <= 1.0
    assert torch.isfinite(out["A"]).all() and out["A"].abs().max() <= 1.0
    assert out["A"].dtype == torch.float32
    assert not hasattr(model, "noise_branch") or model.noise_branch is None
    assert model.denoiser is None


@pytest.mark.parametrize("light", [False, True])
def test_full_variant_shapes_and_gradients_reach_every_enabled_head(light: bool) -> None:
    """Noise+denoise accepts RGB only; gradients reach DCE, noise head and denoiser."""
    model = _model(noise_aware=True, denoise=True, light=light)
    assert isinstance(model.dce, LightDCENet if light else DCENet)
    assert model.dce.in_channels == 4
    x = _image()
    out = model(x)
    assert set(out) == OUTPUT_KEYS
    assert out["enhanced"].shape == (2, 3, 64, 80)
    assert out["sigma"].shape == (2, 1, 64, 80)
    assert out["sigma"].min() >= 0.0 and out["sigma"].max() <= 0.2
    assert out["enhanced"].min() >= 0.0 and out["enhanced"].max() <= 1.0
    assert not torch.equal(out["enhanced"], out["enhanced_raw"])

    # Use an objective that cannot be saturated away by the clamp: every term touches the heads.
    loss = out["enhanced"].mean() + out["enhanced_raw"].mean() + out["A"].mean()
    loss.backward()
    assert model.noise_branch is not None and model.denoiser is not None
    for module in (model.dce, model.noise_branch, model.denoiser):
        grads = [p.grad for p in module.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_denoiser_alone_trains_the_denoiser_and_noise_head_through_sigma() -> None:
    """A loss on enhanced only reaches the denoiser, the noise head (sigma*gain path) and DCE."""
    model = _model(noise_aware=True, denoise=True)
    out = model(_image(batch=1, height=32, width=32))
    (out["enhanced"] - out["enhanced_raw"].detach()).abs().mean().backward()
    assert model.denoiser is not None and model.noise_branch is not None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.denoiser.parameters())
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.noise_branch.parameters()
    )


def test_noise_head_gradient_paths() -> None:
    """Noise head gets gradient from sigma losses and via the DCE input channel, nothing else."""
    model = _model(noise_aware=True)
    assert model.noise_branch is not None
    x = _image(batch=1, height=32, width=32)

    # (1) a loss on sigma alone touches only the noise head.
    out = model(x)
    out["sigma"].mean().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.noise_branch.parameters()
    )
    assert all(p.grad is None for p in model.dce.parameters())

    # (2) a loss on the curve maps reaches the noise head through the fourth DCE input only.
    model.zero_grad(set_to_none=True)
    out = model(x)
    out["A"].square().mean().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.noise_branch.parameters()
    )

    # (3) without denoise there is no other path: detaching sigma before the DCE removes it.
    model.zero_grad(set_to_none=True)
    sigma = model.noise_branch(x).detach()
    model.dce(torch.cat([x, sigma], dim=1)).square().mean().backward()
    assert all(p.grad is None for p in model.noise_branch.parameters())


def test_flag_composition_and_parameter_counts() -> None:
    """Module presence and parameter counts follow the flags and METHOD widths."""
    c = 32
    baseline = _model()
    assert _count(baseline) == _count(baseline.dce) == 79416  # 7 convs, 3->32->...->24
    noise = _model(noise_aware=True)
    assert noise.dce.in_channels == 4 and noise.denoiser is None
    assert _count(noise.dce) == 79416 + c * 9  # one extra input channel in conv1
    assert noise.noise_branch is not None
    assert _count(noise.noise_branch) == (3 * 16 * 9 + 16) + (16 * 16 * 9 + 16) + (16 * 9 + 1)
    full = _model(noise_aware=True, denoise=True)
    assert full.denoiser is not None
    assert _count(full.denoiser) == (4 * 9 + 4) + (4 * 16 + 16) + (16 * 9 + 16) + (16 * 16 + 16) + (
        16 * 3 * 9 + 3
    )
    light = _model(light=True)
    expected_light = (
        (3 * 9 + 3)
        + (3 * c + c)  # conv1: depthwise on 3 channels then pointwise
        + 3 * ((c * 9 + c) + (c * c + c))  # conv2..conv4
        + 2 * ((2 * c * 9 + 2 * c) + (2 * c * c + c))  # conv5, conv6 on concatenations
        + (2 * c * 24 * 9 + 24)  # regular final head
    )
    assert _count(light.dce) == expected_light
    assert _count(light.dce) < 0.35 * _count(baseline.dce)
    # Hidden convolutions are depthwise-separable; the head is a regular 3x3.
    assert isinstance(light.dce.head, torch.nn.Conv2d) and light.dce.head.kernel_size == (3, 3)
    assert light.dce.conv2.depthwise.groups == c and light.dce.conv2.pointwise.kernel_size == (1, 1)


def test_denoise_without_noise_aware_is_rejected() -> None:
    """The model refuses the invalid combination instead of silently disabling the denoiser."""
    with pytest.raises(ValueError):
        build_model(_config(denoise=True))


def test_skip_connections_follow_method() -> None:
    """h5 consumes concat(h3,h4), h6 concat(h2,h5) and the head concat(h1,h6): widths 64."""
    net = DCENet()
    assert net.conv5.in_channels == net.conv6.in_channels == net.head.in_channels == 64
    assert net.head.out_channels == 24
    assert [net.conv1.in_channels, net.conv2.in_channels] == [3, 32]
    out = net(torch.rand(1, 3, 16, 16))
    assert out.shape == (1, 24, 16, 16) and out.abs().max() <= 1.0
    # Perturbing conv2's weights must change the output through both the chain and the skip.
    reference = net(torch.ones(1, 3, 16, 16))
    with torch.no_grad():
        net.conv2.weight.add_(0.5)
    assert not torch.allclose(net(torch.ones(1, 3, 16, 16)), reference)


def test_noise_branch_and_denoiser_ranges() -> None:
    """NoiseBranch is bounded by sigma_max; ResidualDenoiser output is clamped and bounded."""
    torch.manual_seed(0)
    branch = NoiseBranch(16, 0.2)
    x = torch.rand(2, 3, 16, 16)
    sigma = branch(x)
    assert sigma.shape == (2, 1, 16, 16) and sigma.min() >= 0 and sigma.max() <= 0.2
    big = NoiseBranch(16, 0.5)
    assert big(x).max() <= 0.5
    denoiser = ResidualDenoiser(16, 0.1)
    z = torch.rand(2, 3, 16, 16)
    gain = torch.full((2, 1, 16, 16), 4.0)
    out = denoiser(z, sigma, gain)
    assert out.shape == z.shape and out.min() >= 0 and out.max() <= 1
    assert (out - z).abs().max() <= 0.1 + 1e-6  # |residual| <= residual_scale
    with pytest.raises(ValueError):
        NoiseBranch(16, 0.0)
    with pytest.raises(ValueError):
        ResidualDenoiser(16, 0.0)


def test_estimate_and_apply_maps_support_curve_scale_below_one() -> None:
    """Maps estimated on a half-size copy, upsampled and applied give full-resolution outputs."""
    model = _model(noise_aware=True, denoise=True)
    x = _image(batch=1, height=64, width=80)
    small = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
    with torch.no_grad():
        a_small, sigma_small = model.estimate_maps(small)
        assert a_small.shape == (1, 24, 32, 40) and sigma_small.shape == (1, 1, 32, 40)
        a_full = F.interpolate(a_small, size=(64, 80), mode="bilinear", align_corners=False)
        sigma_full = F.interpolate(sigma_small, size=(64, 80), mode="bilinear", align_corners=False)
        out = model.apply_maps(x, a_full, sigma_full)
    assert out["enhanced"].shape == out["enhanced_raw"].shape == (1, 3, 64, 80)
    assert out["A"].shape == (1, 24, 64, 80) and out["sigma"].shape == (1, 1, 64, 80)
    assert out["A"].abs().max() <= 1.0 and out["sigma"].max() <= 0.2


@pytest.mark.parametrize(
    "flags", [{}, {"noise_aware": True}, {"noise_aware": True, "denoise": True}]
)
def test_forward_equals_apply_of_estimate_and_override_semantics(flags: dict[str, bool]) -> None:
    """forward == apply_maps(estimate_maps); A_override bypasses curve estimation only."""
    model = _model(**flags).eval()
    x = _image(batch=1, height=32, width=48)
    with torch.no_grad():
        direct = model(x)
        composed = model.apply_maps(x, *model.estimate_maps(x))
        for key in OUTPUT_KEYS:
            assert torch.equal(direct[key], composed[key])

        override = torch.full((1, 24, 32, 48), 0.5)
        out = model(x, A_override=override)
        assert torch.equal(out["A"], override)
        expected_raw = model.apply_maps(x, override, direct["sigma"])["enhanced_raw"]
        assert torch.equal(out["enhanced_raw"], expected_raw)
        # sigma is still computed when noise_aware; zeros otherwise.
        assert torch.equal(out["sigma"], direct["sigma"])
        if not model.noise_aware:
            assert torch.equal(out["sigma"], torch.zeros(1, 1, 32, 48))
        if not model.denoise:
            assert torch.equal(out["enhanced"], out["enhanced_raw"])
        # +0.5 curves brighten interior pixels of a dark image.
        assert out["enhanced_raw"].mean() > x.mean()
        with pytest.raises(ValueError):
            model(x, A_override=torch.zeros(1, 23, 32, 48))


def test_compute_gain_is_detached_clamped_and_matches_pooling() -> None:
    """G = clamp(avgpool5(Y(z))/(avgpool5(Y(x))+eps), 1, 8) with no autograd graph."""
    x = torch.full((1, 3, 12, 12), 0.1, requires_grad=True)
    z = torch.full((1, 3, 12, 12), 0.3, requires_grad=True)
    gain = compute_gain(x, z, 5, 1.0, 8.0, 1e-6)
    assert gain.shape == (1, 1, 12, 12) and not gain.requires_grad
    assert torch.allclose(gain, torch.full_like(gain, 3.0), atol=1e-4)
    # Clamps: darkening gives 1, a huge ratio is capped at 8.
    assert torch.equal(compute_gain(z, x), torch.ones(1, 1, 12, 12))
    assert torch.equal(
        compute_gain(torch.full((1, 3, 12, 12), 0.01), z), torch.full((1, 1, 12, 12), 8.0)
    )
    # A single bright input pixel is averaged over the 5x5 window (interior check).
    xs = torch.full((1, 3, 12, 12), 0.2)
    zs = torch.full((1, 3, 12, 12), 0.2)
    zs[:, :, 6, 6] = 1.0
    g = compute_gain(xs, zs, 5, 1.0, 8.0, 0.0)
    assert g[0, 0, 6, 6].item() == pytest.approx((0.2 * 24 + 1.0) / 25 / 0.2, rel=1e-5)
    assert g[0, 0, 0, 0].item() == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(ValueError):
        compute_gain(x, z, window=4)


def test_determinism_and_state_dict_roundtrip() -> None:
    """Same seed gives identical weights/outputs; state_dict reload reproduces outputs."""
    x = _image(batch=1, height=32, width=32)
    first = _model(noise_aware=True, denoise=True, light=True)
    second = _model(noise_aware=True, denoise=True, light=True)
    with torch.no_grad():
        a, b = first(x), second(x)
    for key in OUTPUT_KEYS:
        assert torch.equal(a[key], b[key])
    other = build_model(_config(noise_aware=True, denoise=True, light=True))
    other.load_state_dict(first.state_dict())
    with torch.no_grad():
        c = other(x)
    assert torch.equal(c["enhanced"], a["enhanced"])


def test_no_implicit_device_transfer() -> None:
    """Every model path runs on the meta device (nothing creates CPU tensors) and stays there."""
    model = _model(noise_aware=True, denoise=True).to("meta")
    x = torch.empty(1, 3, 32, 32, device="meta")
    out = model(x)
    assert all(tensor.device.type == "meta" for tensor in out.values())
    baseline = _model().to("meta")
    assert all(t.device.type == "meta" for t in baseline(x).values())
    override = baseline(x, A_override=torch.empty(1, 24, 32, 32, device="meta"))
    assert override["sigma"].device.type == "meta"
    if torch.cuda.is_available():  # optional device check, skipped on CPU-only machines
        cuda_out = _model(noise_aware=True).cuda()(x.new_zeros(1, 3, 32, 32, device="cuda"))
        assert all(t.device.type == "cuda" for t in cuda_out.values())


def test_arbitrary_inference_size_runs() -> None:
    """Sizes that are not multiples of 16 (but >= 16) run end to end."""
    model = _model(noise_aware=True, denoise=True)
    with torch.no_grad():
        out = model(_image(batch=1, height=23, width=37))
    assert out["enhanced"].shape == (1, 3, 23, 37) and out["A"].shape == (1, 24, 23, 37)
