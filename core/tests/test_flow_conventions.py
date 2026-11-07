"""Flow, warp, reliability-mask, scene-cut and temporal-loss conventions (TESTING.md recipe).

Analytic affine flows are used for the deterministic assertions; the OpenCV DIS adapter
is covered by separate tolerance-based tests on a visible synthetic texture.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest
import torch
from torch import Tensor

from nta_zerodce.flow import estimator
from nta_zerodce.flow.estimator import estimate_bidirectional, estimate_flow
from nta_zerodce.flow.masks import is_scene_cut, reliability_mask
from nta_zerodce.flow.warp import warp
from nta_zerodce.losses.temporal import temporal_loss
from nta_zerodce.training.config import FlowConfig

H, W = 64, 80


# ----------------------------------------------------------------------------- fixtures


def _coords(height: int = H, width: int = W) -> tuple[Tensor, Tensor]:
    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=torch.float64),
        torch.arange(width, dtype=torch.float64),
        indexing="ij",
    )
    return xs, ys


def _texture(x: Tensor, y: Tensor) -> Tensor:
    """Smooth analytic RGB texture of continuous pixel coordinates, values in [0.05, 0.95]."""
    two_pi = 2.0 * math.pi
    r = 0.5 + 0.25 * torch.sin(two_pi * x / 17.0 + 0.5) + 0.2 * torch.cos(two_pi * y / 13.0)
    g = 0.5 + 0.45 * torch.sin(two_pi * (x / 23.0 + y / 19.0))
    b = (
        0.5
        + 0.25 * torch.cos(two_pi * x / 11.0) * torch.sin(two_pi * y / 29.0)
        + 0.2 * (torch.sin(two_pi * (x - y) / 31.0))
    )
    return torch.stack((r, g, b)).to(torch.float32).unsqueeze(0)  # (1,3,H,W)


def _affine(angle_deg: float, tx: float, ty: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return (first, second, forward_flow, backward_flow) for second(q)=first(T^-1(q)).

    T(p) = R(p - c) + c + t about the image centre c; f(p)=T(p)-p; b(q)=T^-1(q)-q.
    """
    xs, ys = _coords()
    theta = math.radians(angle_deg)
    cos, sin = math.cos(theta), math.sin(theta)
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    # forward map T
    tx_p = cos * (xs - cx) - sin * (ys - cy) + cx + tx
    ty_p = sin * (xs - cx) + cos * (ys - cy) + cy + ty
    forward = torch.stack((tx_p - xs, ty_p - ys)).to(torch.float32).unsqueeze(0)
    # inverse map T^-1 (rotate back by -theta after removing the translation)
    ux, uy = xs - cx - tx, ys - cy - ty
    ix = cos * ux + sin * uy + cx
    iy = -sin * ux + cos * uy + cy
    backward = torch.stack((ix - xs, iy - ys)).to(torch.float32).unsqueeze(0)
    first = _texture(xs, ys)
    second = _texture(ix, iy)
    return first, second, forward, backward


def _rgb_from_gray(gray: np.ndarray) -> Tensor:
    tensor = torch.from_numpy(gray.astype(np.float32) / 255.0)
    return tensor.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).contiguous()


def _dis_pair(tx: int, ty: int) -> tuple[Tensor, Tensor]:
    """Two crops of one blurred-noise canvas; second(q)=first(q-t), so flow = (tx,ty)."""
    rng = np.random.default_rng(0)
    canvas = cv2.GaussianBlur(rng.random((200, 240)).astype(np.float32), (0, 0), 2.0)
    canvas = ((canvas - canvas.min()) / (canvas.max() - canvas.min()) * 255).astype(np.uint8)
    y0, x0 = 60, 60
    first = canvas[y0 : y0 + H, x0 : x0 + W]
    second = canvas[y0 - ty : y0 - ty + H, x0 - tx : x0 - tx + W]
    return _rgb_from_gray(first), _rgb_from_gray(second)


def _config(**changes: object) -> FlowConfig:
    """Valid FlowConfig, then bypass the frozen dataclass to inject deliberately bad values."""
    cfg = FlowConfig()
    for key, value in changes.items():
        object.__setattr__(cfg, key, value)
    return cfg


# ----------------------------------------------------------------------------- warp


def test_translation_flows_and_warp_reproduce_first_frame() -> None:
    """Translation (+3,-2): forward (+3,-2), backward (-3,+2), warp(second,forward)=first."""
    first, second, forward, backward = _affine(0.0, 3.0, -2.0)
    assert torch.allclose(forward, torch.tensor([3.0, -2.0]).view(1, 2, 1, 1).expand_as(forward))
    assert torch.allclose(backward, torch.tensor([-3.0, 2.0]).view(1, 2, 1, 1).expand_as(backward))
    warped, valid = warp(second, forward)
    expected_valid = torch.zeros(1, 1, H, W)
    expected_valid[:, :, 2:, : W - 3] = 1.0  # x+3<=W-1 and y-2>=0
    assert torch.equal(valid, expected_valid)
    inside = valid.bool().expand_as(first)
    assert (warped - first).abs()[inside].max() < 1e-5
    # backward direction: warp(first, backward) reproduces second in current coordinates
    warped_back, valid_back = warp(first, backward)
    assert (warped_back - second).abs()[valid_back.bool().expand_as(second)].max() < 1e-5


def test_rotation_translation_analytic_flows_compose_to_zero() -> None:
    """warp(forward, backward) + backward ~ 0 on valid samples; negation is not the inverse."""
    first, second, forward, backward = _affine(8.0, 2.5, -1.5)
    warped_forward, valid = warp(forward, backward)
    residual = (warped_forward + backward).abs()
    inside = valid.bool().expand_as(residual)
    assert inside.float().mean() > 0.7
    assert residual[inside].max() < 1e-3  # affine fields are reproduced exactly by bilinear
    # negating the forward flow does not give the backward flow for a rotation
    assert (-forward - backward).abs().max() > 1.0
    negated_warped, negated_valid = warp(forward, -forward)
    negated_residual = (negated_warped - forward).abs()[negated_valid.bool().expand(1, 2, H, W)]
    assert negated_residual.max() > 0.5
    # the image itself matches up to bilinear interpolation error of the smooth texture
    warped, valid_img = warp(second, forward)
    assert (warped - first).abs()[valid_img.bool().expand_as(first)].max() < 0.03


def test_zero_flow_is_identity_and_out_of_bounds_mask_is_zero() -> None:
    gen = torch.Generator().manual_seed(0)
    source = torch.rand(2, 3, 16, 24, generator=gen)
    warped, valid = warp(source, torch.zeros(2, 2, 16, 24))
    assert torch.allclose(warped, source, atol=1e-6)
    assert torch.equal(valid, torch.ones(2, 1, 16, 24))
    # everything sampled outside the image: zero padding and zero mask
    far = torch.zeros(2, 2, 16, 24)
    far[:, 0] = 100.0
    warped_far, valid_far = warp(source, far)
    assert float(valid_far.sum()) == 0.0
    assert float(warped_far.abs().max()) == 0.0
    # exact border rules: a +1 px shift in x invalidates only the last column
    shift = torch.zeros(1, 2, 16, 24)
    shift[:, 0] = 1.0
    _, valid_shift = warp(source[:1], shift)
    assert torch.equal(valid_shift[0, 0, :, :-1], torch.ones(16, 23))
    assert float(valid_shift[0, 0, :, -1].sum()) == 0.0
    # y-shift of -1 invalidates the first row
    up = torch.zeros(1, 2, 16, 24)
    up[:, 1] = -1.0
    _, valid_up = warp(source[:1], up)
    assert float(valid_up[0, 0, 0].sum()) == 0.0 and float(valid_up[0, 0, 1:].min()) == 1.0


def test_warp_subpixel_bilinear_and_align_corners_modes() -> None:
    source = torch.arange(5, dtype=torch.float32).view(1, 1, 1, 5).repeat(1, 1, 4, 1)
    flow = torch.zeros(1, 2, 4, 5)
    flow[:, 0] = 0.5
    for align in (True, False):
        warped, valid = warp(source, flow, align_corners=align)
        assert torch.allclose(warped[0, 0, :, :4], source[0, 0, :, :4] + 0.5, atol=1e-5)
        assert float(valid[0, 0, :, 4].sum()) == 0.0  # 4.5 > W-1


def test_warp_gradients_reach_source_and_flow() -> None:
    first, second, forward, _ = _affine(0.0, 3.0, -2.0)
    source = second.clone().requires_grad_(True)
    warped, valid = warp(source, forward)
    warped.sum().backward()
    assert source.grad is not None and torch.isfinite(source.grad).all()
    # bilinear weights sum to one: total gradient mass equals valid samples * channels
    assert float(source.grad.sum()) == pytest.approx(float(valid.sum()) * 3, rel=1e-4)
    assert not valid.requires_grad
    # gradient w.r.t. the flow exists and is finite for a smooth texture at a half pixel
    half = torch.full((1, 2, H, W), 0.5, requires_grad=True)
    warped_half, _ = warp(first, half)
    warped_half.sum().backward()
    assert half.grad is not None and torch.isfinite(half.grad).all()
    assert float(half.grad.abs().sum()) > 0.0


def test_warp_rejects_invalid_inputs() -> None:
    src = torch.zeros(1, 3, 8, 8)
    flow = torch.zeros(1, 2, 8, 8)
    with pytest.raises(ValueError, match="H>=2"):
        warp(torch.zeros(1, 3, 1, 8), torch.zeros(1, 2, 1, 8))
    with pytest.raises(ValueError, match="H>=2"):
        warp(torch.zeros(1, 3, 8, 1), torch.zeros(1, 2, 8, 1))
    with pytest.raises(ValueError, match="B, H and W"):
        warp(src, torch.zeros(1, 2, 8, 9))
    with pytest.raises(ValueError, match="B, H and W"):
        warp(src, torch.zeros(2, 2, 8, 8))
    with pytest.raises(ValueError, match="flow must be"):
        warp(src, torch.zeros(1, 3, 8, 8))
    with pytest.raises(ValueError, match="source must be"):
        warp(torch.zeros(3, 8, 8), flow)
    with pytest.raises(ValueError, match="same device"):
        warp(src, torch.zeros(1, 2, 8, 8, device="meta"))
    with pytest.raises(ValueError, match="dtypes"):
        warp(src, flow.double())
    with pytest.raises(ValueError, match="floating"):
        warp(src.long(), flow.long())


# ----------------------------------------------------------------------------- estimator


def test_estimator_restores_x_and_y_scale_per_axis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Downscaled estimates are multiplied by W/small_W (dx) and H/small_H (dy)."""
    seen: list[tuple[int, int]] = []

    def fake_dis(first: np.ndarray, second: np.ndarray, preset: str) -> np.ndarray:
        seen.append((first.shape[1], first.shape[0]))
        assert first.dtype == np.uint8 and first.shape == second.shape
        flow = np.zeros((*first.shape, 2), dtype=np.float32)
        flow[..., 0] = 1.0  # dx = 1 small-image pixel
        flow[..., 1] = 2.0  # dy = 2 small-image pixels
        return flow

    monkeypatch.setattr(estimator, "_dis_flow_gray", fake_dis)
    width, height = 100, 64
    frames = torch.zeros(2, 3, height, width)
    out = estimate_flow(frames, frames, _config(scale=0.3))
    small_w, small_h = 30, 19  # round(30.0), round(19.2)
    assert seen == [(small_w, small_h)] * 2
    assert out.shape == (2, 2, height, width) and out.dtype == torch.float32
    ratio_x, ratio_y = width / small_w, height / small_h
    assert ratio_x != pytest.approx(ratio_y)  # genuinely non-square factors
    assert torch.allclose(out[:, 0], torch.full((2, height, width), 1.0 * ratio_x), atol=1e-4)
    assert torch.allclose(out[:, 1], torch.full((2, height, width), 2.0 * ratio_y), atol=1e-4)
    # scale == 1 runs at full size and applies no rescaling
    seen.clear()
    full = estimate_flow(frames[:1], frames[:1], _config(scale=1.0))
    assert seen == [(width, height)]
    assert torch.allclose(full[:, 0], torch.ones(1, height, width))
    assert torch.allclose(full[:, 1], torch.full((1, height, width), 2.0))
    assert not full.requires_grad


def test_estimator_input_validation() -> None:
    ok = torch.zeros(1, 3, 32, 32)
    cfg = FlowConfig()
    with pytest.raises(ValueError, match="CPU"):
        estimate_flow(torch.zeros(1, 3, 32, 32, device="meta"), ok, cfg)
    with pytest.raises(ValueError, match="float32"):
        estimate_flow(ok.double(), ok, cfg)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        estimate_flow(ok + 2.0, ok, cfg)
    with pytest.raises(ValueError, match=r"\(B,3,H,W\)"):
        estimate_flow(torch.zeros(1, 1, 32, 32), torch.zeros(1, 1, 32, 32), cfg)
    with pytest.raises(ValueError, match="equal shape"):
        estimate_flow(ok, torch.zeros(1, 3, 32, 40), cfg)
    with pytest.raises(ValueError, match="'dis'"):
        estimate_flow(ok, ok, _config(method="farneback"))
    with pytest.raises(ValueError, match="preset"):
        estimate_flow(ok, ok, _config(preset="slow"))
    for bad_scale in (0.0, 1.5, -0.5, float("nan")):
        with pytest.raises(ValueError, match="scale"):
            estimate_flow(ok, ok, _config(scale=bad_scale))
    with pytest.raises(ValueError, match="at least"):
        estimate_flow(torch.zeros(1, 3, 32, 32), ok, _config(scale=0.25))  # 8x8 < 16


def test_opencv_dis_adapter_creates_flow_object() -> None:
    for preset in ("ultrafast", "fast", "medium"):
        assert estimator._create_dis(preset) is not None


@pytest.mark.parametrize("preset", ["ultrafast", "fast", "medium"])
def test_dis_finds_known_translation_in_both_directions(preset: str) -> None:
    """Integer translation (+3,-2) on a blurred-noise texture: median error < 0.25 px."""
    first, second = _dis_pair(3, -2)
    forward, backward = estimate_bidirectional(first, second, _config(preset=preset, scale=1.0))
    assert forward.shape == backward.shape == (1, 2, H, W)
    assert forward.dtype == torch.float32 and forward.device.type == "cpu"
    centre = (slice(None), slice(None), slice(12, H - 12), slice(12, W - 12))
    for flow, expected in ((forward, (3.0, -2.0)), (backward, (-3.0, 2.0))):
        target = torch.tensor(expected).view(1, 2, 1, 1)
        error = (flow[centre] - target).abs()
        assert float(error.median()) < 0.25
    # computed independently, never obtained by negation
    assert not torch.equal(backward, -forward)
    assert torch.equal(backward, estimate_flow(second, first, _config(preset=preset, scale=1.0)))


def test_dis_at_reduced_scale_recovers_full_resolution_units() -> None:
    """At scale 0.5 a (+4,-2) full-resolution shift is still reported as ~(+4,-2)."""
    first, second = _dis_pair(4, -2)
    forward, backward = estimate_bidirectional(first, second, _config(preset="fast", scale=0.5))
    centre = (slice(None), slice(None), slice(12, H - 12), slice(12, W - 12))
    for flow, expected in ((forward, (4.0, -2.0)), (backward, (-4.0, 2.0))):
        error = (flow[centre] - torch.tensor(expected).view(1, 2, 1, 1)).abs()
        assert float(error.median()) < 0.5


def test_dis_flow_warps_second_onto_first() -> None:
    """Estimated flow follows first(p) ~ second(p + flow(p)): warping reproduces first."""
    first, second = _dis_pair(3, -2)
    forward = estimate_flow(first, second, _config(preset="medium", scale=1.0))
    warped, valid = warp(second, forward)
    region = valid.bool().expand_as(first).clone()
    region[:, :, :8] = False
    region[:, :, -8:] = False
    region[..., :8] = False
    region[..., -8:] = False
    assert float((warped - first).abs()[region].mean()) < 0.02


# ----------------------------------------------------------------------------- masks


def _expected_in_bounds(backward: Tensor) -> Tensor:
    return warp(torch.ones(1, 1, H, W), backward)[1]


def test_reliability_mask_is_one_on_consistent_analytic_flows() -> None:
    first, second, forward, backward = _affine(8.0, 2.5, -1.5)
    mask = reliability_mask(forward, backward, first, second)
    assert mask.shape == (1, 1, H, W) and mask.dtype == torch.float32
    assert not mask.requires_grad
    assert set(mask.unique().tolist()) <= {0.0, 1.0}
    expected = _expected_in_bounds(backward)
    assert torch.equal(mask, expected)  # FB and photometric pass everywhere in bounds
    assert 0.5 < float(mask.mean()) < 1.0


def test_reliability_mask_zero_where_fb_check_fails() -> None:
    first, second, forward, backward = _affine(0.0, 3.0, -2.0)
    bad_forward = forward.clone()
    bad_forward[:, :, 20:40, 30:50] += 5.0  # inconsistent region in *previous* coordinates
    mask = reliability_mask(bad_forward, backward, first, second)
    # current coordinates q solve q + b(q) = p with b=(-3,+2): q = p + (3,-2), so the
    # previous rows 20:40, cols 30:50 appear at current rows 18:38, cols 33:53
    assert float(mask[:, :, 18:38, 33:53].sum()) == 0.0
    clean = reliability_mask(forward, backward, first, second)
    assert float(clean[:, :, 18:38, 33:53].min()) == 1.0
    changed = (mask != clean).float()
    assert float(changed[:, :, 18:38, 33:53].min()) == 1.0
    assert float(changed.sum()) == 20 * 20  # nothing else changes
    assert float(changed[:, :, :, 60:].sum()) == 0.0  # far from the region nothing changes
    # a looser beta accepts the 5 px inconsistency (r^2=50 <= beta)
    loose = reliability_mask(bad_forward, backward, first, second, beta=100.0)
    assert torch.equal(loose, clean)


def test_reliability_mask_zero_where_photometry_differs() -> None:
    first, second, forward, backward = _affine(0.0, 3.0, -2.0)
    changed = second.clone()
    block = changed[:, :, 20:40, 30:50]
    offset = torch.where(block.mean(dim=1, keepdim=True) < 0.5, 0.3, -0.3)
    changed[:, :, 20:40, 30:50] = (block + offset).clamp(0, 1)
    mask = reliability_mask(forward, backward, first, changed)
    clean = reliability_mask(forward, backward, first, second)
    assert float(mask[:, :, 24:36, 34:46].sum()) == 0.0
    assert float((mask - clean)[:, :, :, 55:].abs().sum()) == 0.0
    # threshold 1.0 disables the photometric test (strict inequality, |diff| < 1)
    assert torch.equal(reliability_mask(forward, backward, first, changed, 0.01, 0.5, 1.0), clean)


def test_reliability_mask_is_detached_even_with_grad_inputs() -> None:
    first, second, forward, backward = _affine(4.0, 1.0, 1.0)
    inputs = [t.clone().requires_grad_(True) for t in (forward, backward, first, second)]
    mask = reliability_mask(*inputs)
    assert not mask.requires_grad and mask.grad_fn is None


def test_reliability_mask_validation() -> None:
    f = torch.zeros(1, 2, 8, 8)
    img = torch.zeros(1, 3, 8, 8)
    with pytest.raises(ValueError, match="alpha"):
        reliability_mask(f, f, img, img, alpha=-1.0)
    with pytest.raises(ValueError, match="photometric_threshold"):
        reliability_mask(f, f, img, img, photometric_threshold=2.0)
    with pytest.raises(ValueError, match="does not match"):
        reliability_mask(torch.zeros(1, 2, 8, 9), f, img, img)
    with pytest.raises(ValueError, match="must match"):
        reliability_mask(f, f, img, torch.zeros(1, 3, 8, 9))


def test_scene_cut_true_for_brightness_swap_false_for_noisy_near_identical() -> None:
    xs, ys = _coords(32, 32)
    base = _texture(xs, ys)
    dark = 0.1 + 0.2 * base  # luminance in ~[0.11, 0.29]
    bright = 0.7 + 0.2 * base  # luminance in ~[0.71, 0.89]
    assert is_scene_cut(dark, bright) is True
    gen = torch.Generator().manual_seed(1)
    noisy = (base + 0.01 * torch.randn(base.shape, generator=gen)).clamp(0, 1)
    assert is_scene_cut(base, noisy) is False
    assert is_scene_cut(base, base) is False


def test_scene_cut_bins_threshold_and_validation() -> None:
    ones = torch.ones(1, 3, 8, 8)
    almost = torch.full((1, 3, 8, 8), 0.97)  # 0.97*32 = 31.04 -> last bin, like 1.0
    assert is_scene_cut(ones, almost) is False
    assert is_scene_cut(ones, torch.full((1, 3, 8, 8), 0.5)) is True  # histograms disjoint
    # half-L1 distance of two disjoint point masses is exactly 1: only threshold 1 never cuts
    assert is_scene_cut(ones, torch.zeros(1, 3, 8, 8), threshold=1.0) is False
    with pytest.raises(ValueError, match="bins"):
        is_scene_cut(ones, ones, bins=1)
    with pytest.raises(ValueError, match="bins"):
        is_scene_cut(ones, ones, bins=2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="single frame"):
        is_scene_cut(torch.ones(2, 3, 8, 8), torch.ones(2, 3, 8, 8))
    with pytest.raises(ValueError, match="must match"):
        is_scene_cut(ones, torch.ones(1, 3, 8, 9))
    with pytest.raises(ValueError, match="RGB"):
        is_scene_cut(torch.ones(1, 1, 8, 8), torch.ones(1, 1, 8, 8))


# ----------------------------------------------------------------------------- temporal loss


def _pair() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    first, second, _, backward = _affine(0.0, 3.0, -2.0)
    return first, second, backward, torch.ones(1, 1, H, W)


def test_temporal_loss_zero_for_correctly_warped_identical_pair() -> None:
    previous, current, backward, mask = _pair()
    loss = temporal_loss(previous, current, backward, mask)
    assert loss.dim() == 0 and loss.dtype == torch.float32
    assert float(loss) < 1e-6


def test_temporal_loss_positive_and_matches_closed_form() -> None:
    previous, current, backward, mask = _pair()
    loss = temporal_loss(previous, current + 0.1, backward, mask)
    # exactly 0.1 over in-bounds, masked pixels; eps shifts the value by ~1e-6 relative
    assert float(loss) == pytest.approx(0.1, abs=1e-5)
    # the mask restricts the average: perturbing outside the mask does not matter
    half = torch.zeros(1, 1, H, W)
    half[..., :40] = 1.0
    perturbed = current.clone()
    perturbed[..., 40:] += 0.3
    assert float(temporal_loss(previous, perturbed, backward, half)) < 1e-6
    # in-bounds is applied on top of the mask: wrapped border pixels are excluded
    wrong_current = torch.roll(previous, shifts=(-2, 3), dims=(2, 3))
    assert float(temporal_loss(previous, wrong_current, backward, mask)) < 1e-6


def test_temporal_loss_empty_mask_is_graph_connected_zero() -> None:
    previous, current, backward, _ = _pair()
    prev = previous.clone().requires_grad_(True)
    cur = (current + 0.2).requires_grad_(True)
    loss = temporal_loss(prev, cur, backward, torch.zeros(1, 1, H, W))
    assert float(loss.detach()) == 0.0 and loss.requires_grad and loss.grad_fn is not None
    loss.backward()
    assert prev.grad is not None and cur.grad is not None
    assert torch.isfinite(prev.grad).all() and torch.isfinite(cur.grad).all()
    assert float(prev.grad.abs().max()) == 0.0 and float(cur.grad.abs().max()) == 0.0


def test_temporal_loss_gradients_reach_both_outputs_and_not_flow_or_mask() -> None:
    previous, current, backward, mask = _pair()
    prev = previous.clone().requires_grad_(True)
    cur = (current + 0.1).requires_grad_(True)
    flow = backward.clone().requires_grad_(True)
    weights = mask.clone().requires_grad_(True)
    temporal_loss(prev, cur, flow, weights).backward()
    assert prev.grad is not None and float(prev.grad.abs().sum()) > 0.0
    assert cur.grad is not None and float(cur.grad.abs().sum()) > 0.0
    assert flow.grad is None and weights.grad is None


def test_temporal_loss_batch_and_validation() -> None:
    previous, current, backward, mask = _pair()

    def b2(t: Tensor) -> Tensor:
        return torch.cat([t, t])

    loss = temporal_loss(b2(previous), b2(current + 0.1), b2(backward), b2(mask))
    assert float(loss) == pytest.approx(0.1, abs=1e-5)
    with pytest.raises(ValueError, match="mask must be"):
        temporal_loss(previous, current, backward, torch.ones(1, 3, H, W))
    with pytest.raises(ValueError, match="backward_flow must be"):
        temporal_loss(previous, current, torch.zeros(1, 2, H, 7), mask)
    with pytest.raises(ValueError, match="equal shape"):
        temporal_loss(previous, current[..., :70], backward, mask)
    with pytest.raises(ValueError, match="one device"):
        temporal_loss(previous, current, backward, torch.ones(1, 1, H, W, device="meta"))
