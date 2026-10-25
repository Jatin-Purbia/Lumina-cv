"""Metric definitions from EXPERIMENTS: fidelity, noise, colour, clipping and warping error."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch import Tensor

from nta_zerodce.flow.warp import warp
from nta_zerodce.losses.temporal import temporal_loss
from nta_zerodce.metrics.colour import angular_error
from nta_zerodce.metrics.exposure import clipping_fraction
from nta_zerodce.metrics.fidelity import psnr, ssim
from nta_zerodce.metrics.noise import noise_estimate
from nta_zerodce.metrics.temporal import warping_error


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _smooth_images(batch: int, size: int = 32, seed: int = 0) -> Tensor:
    """Seeded smooth-ish RGB images in [0,1]: low-frequency random field plus a ramp."""
    gen = _gen(seed)
    low = torch.rand(batch, 3, 4, 4, generator=gen)
    up = torch.nn.functional.interpolate(low, size=(size, size), mode="bilinear")
    ramp = torch.linspace(0, 1, size).view(1, 1, 1, size)
    return (0.7 * up + 0.3 * ramp).clamp(0, 1)


def _noisy(image: Tensor, sigma: float, seed: int = 1) -> Tensor:
    return (image + sigma * torch.randn(image.shape, generator=_gen(seed))).clamp(0, 1)


def _np(image: Tensor) -> np.ndarray:
    return np.asarray(image.permute(0, 2, 3, 1).double().numpy(), dtype=np.float64)  # (B,H,W,3)


# ----------------------------------------------------------------------------- PSNR


def test_psnr_identity_is_infinite_and_known_value() -> None:
    img = _smooth_images(2)
    value = psnr(img, img.clone())
    assert value.shape == (2,) and torch.isinf(value).all() and (value > 0).all()
    ref = torch.full((1, 3, 8, 8), 0.4)
    assert float(psnr(ref + 0.1, ref)) == pytest.approx(20.0, abs=1e-4)  # MSE = 0.01


def test_psnr_matches_skimage_per_image() -> None:
    ref = _smooth_images(3, seed=3)
    out = _noisy(ref, 0.05, seed=4)
    ours = psnr(out, ref)
    for i in range(3):
        expected = peak_signal_noise_ratio(  # type: ignore[no-untyped-call]
            _np(ref)[i], _np(out)[i], data_range=1.0
        )
        assert float(ours[i]) == pytest.approx(expected, abs=1e-4)


def test_psnr_is_monotone_in_noise_level_and_validates() -> None:
    ref = _smooth_images(1, seed=5)
    values = [float(psnr(_noisy(ref, s), ref)) for s in (0.01, 0.03, 0.1)]
    assert values[0] > values[1] > values[2]
    with pytest.raises(ValueError, match="equal shape"):
        psnr(ref, ref[..., :16])
    with pytest.raises(ValueError, match="share a device"):
        psnr(ref, torch.zeros(1, 3, 32, 32, device="meta"))
    with pytest.raises(ValueError, match=r"\(B,3,H,W\)"):
        psnr(ref[0], ref[0])


# ----------------------------------------------------------------------------- SSIM


def test_ssim_identity_and_symmetry() -> None:
    img = _smooth_images(2, seed=6)
    assert torch.allclose(ssim(img, img.clone()), torch.ones(2), atol=1e-6)
    other = _noisy(img, 0.08)
    assert torch.allclose(ssim(img, other), ssim(other, img), atol=1e-6)
    assert (ssim(img, other) < 1.0).all()


def test_ssim_constant_images_closed_form() -> None:
    # zero variance and covariance: SSIM = (2*a*b + C1) / (a^2 + b^2 + C1)
    a, b = 0.2, 0.7
    c1 = 0.01**2
    value = ssim(torch.full((1, 3, 16, 16), a), torch.full((1, 3, 16, 16), b))
    assert float(value) == pytest.approx((2 * a * b + c1) / (a * a + b * b + c1), abs=1e-6)


def test_ssim_matches_skimage_on_the_same_valid_window() -> None:
    ref = _smooth_images(2, size=40, seed=7)
    out = _noisy(ref, 0.06, seed=8)
    ours = ssim(out, ref)
    for i in range(2):
        mean_score, ssim_map = structural_similarity(  # type: ignore[no-untyped-call]
            _np(ref)[i],
            _np(out)[i],
            data_range=1.0,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
            win_size=11,
            channel_axis=-1,
            full=True,
        )
        # skimage reflects at the border; the valid window drops pad = (11-1)//2 = 5 pixels
        valid = ssim_map[5:-5, 5:-5].mean()
        assert float(ours[i]) == pytest.approx(float(valid), abs=1e-3)
        assert float(ours[i]) == pytest.approx(float(mean_score), abs=1e-3)


def test_ssim_decreases_with_noise_and_rejects_small_images() -> None:
    ref = _smooth_images(1, size=32, seed=9)
    values = [float(ssim(_noisy(ref, s), ref)) for s in (0.02, 0.06, 0.15)]
    assert values[0] > values[1] > values[2]
    assert ssim(torch.rand(1, 3, 11, 11), torch.rand(1, 3, 11, 11)).shape == (1,)
    with pytest.raises(ValueError, match="at least 11x11"):
        ssim(torch.rand(1, 3, 10, 32), torch.rand(1, 3, 10, 32))
    with pytest.raises(ValueError, match="at least 11x11"):
        ssim(torch.rand(1, 3, 32, 10), torch.rand(1, 3, 32, 10))


# ----------------------------------------------------------------------------- noise


def test_noise_estimate_recovers_gray_gaussian_sigma() -> None:
    sigma = 0.05
    noise = torch.randn(2, 1, 64, 64, generator=_gen(10)) * sigma
    image = (0.5 + noise).repeat(1, 3, 1, 1)  # identical noise in R, G, B: Y noise = sigma
    estimate = noise_estimate(image)
    assert estimate.shape == (2,)
    assert torch.allclose(estimate, torch.full((2,), sigma), rtol=0.15)


def test_noise_estimate_is_on_mean_rgb_luminance() -> None:
    sigma = 0.06
    image = 0.5 + sigma * torch.randn(1, 3, 64, 64, generator=_gen(11))  # independent channels
    # Y = mean of three independent channels has noise sigma/sqrt(3)
    assert float(noise_estimate(image)) == pytest.approx(sigma / math.sqrt(3), rel=0.15)


def test_noise_estimate_flat_mask_behaviour() -> None:
    sigma = 0.04
    gen = _gen(12)
    noise = sigma * torch.randn(2, 1, 64, 64, generator=gen)
    # 2-D texture: the Immerkaer kernel is the outer product of [1,-2,1] with itself, so a
    # pattern that is constant along either axis would produce no response at all
    axis = torch.arange(64).float()
    stripes = (
        0.3 * torch.sin(axis * 2.1).view(1, 1, 64, 1) * torch.sin(axis * 1.7).view(1, 1, 1, 64)
    )
    image = (0.5 + noise).repeat(1, 3, 1, 1).clone()
    image[:, :, :, 32:] += stripes[..., 32:]  # strong texture in the right half only
    left_only = torch.zeros(2, 1, 64, 64)
    left_only[..., :32] = 1.0
    masked = noise_estimate(image, left_only)
    assert torch.allclose(masked, torch.full((2,), sigma), rtol=0.15)
    assert (noise_estimate(image) > 2 * sigma).all()  # texture biases the unmasked proxy up
    # an empty mask, or one whose 3x3 supports never fit inside, gives NaN
    assert torch.isnan(noise_estimate(image, torch.zeros(2, 1, 64, 64))).all()
    checker = torch.zeros(2, 1, 64, 64)
    checker[..., ::2, ::2] = 1.0
    assert torch.isnan(noise_estimate(image, checker)).all()
    # per-sample NaN: only the second image has an empty mask
    mixed = torch.stack([left_only[0], torch.zeros(1, 64, 64)])
    result = noise_estimate(image, mixed)
    assert not torch.isnan(result[0]) and torch.isnan(result[1])
    # constant image has zero estimated noise; invalid inputs are rejected
    assert float(noise_estimate(torch.full((1, 3, 8, 8), 0.3))) == pytest.approx(0.0, abs=1e-7)
    with pytest.raises(ValueError, match="binary"):
        noise_estimate(image, torch.full((2, 1, 64, 64), 0.5))
    with pytest.raises(ValueError, match="flat_mask must be"):
        noise_estimate(image, torch.ones(2, 1, 32, 32))
    with pytest.raises(ValueError, match="3x3"):
        noise_estimate(torch.rand(1, 3, 2, 8))


def test_noise_estimate_orders_noise_levels() -> None:
    base = torch.full((1, 3, 48, 48), 0.5)
    values = [float(noise_estimate(_noisy(base, s, seed=13))) for s in (0.01, 0.03, 0.08)]
    assert values[0] < values[1] < values[2]


# ----------------------------------------------------------------------------- colour


def test_angular_error_zero_for_identical_and_scaled_colours() -> None:
    img = _smooth_images(2, seed=14).clamp(min=0.05)
    assert torch.allclose(angular_error(img, img.clone()), torch.zeros(2), atol=1e-3)
    assert torch.allclose(angular_error(0.4 * img, img), torch.zeros(2), atol=1e-3)


def test_angular_error_known_angles() -> None:
    red = torch.zeros(1, 3, 4, 4)
    red[:, 0] = 1.0
    green = torch.zeros(1, 3, 4, 4)
    green[:, 1] = 1.0
    yellowish = 0.3 * (red + green)  # direction (1,1,0)/sqrt(2); magnitude is irrelevant
    assert float(angular_error(red, green)) == pytest.approx(90.0, abs=1e-4)
    assert float(angular_error(red, yellowish)) == pytest.approx(45.0, abs=1e-4)
    assert float(angular_error(red, red)) == pytest.approx(0.0, abs=1e-3)
    # constructed 30 degree pair in the R-G plane
    theta = math.radians(30.0)
    rotated = torch.zeros(1, 3, 4, 4)
    rotated[:, 0] = math.cos(theta)
    rotated[:, 1] = math.sin(theta)
    assert float(angular_error(rotated, red)) == pytest.approx(30.0, abs=1e-3)


def test_angular_error_excludes_black_vectors_and_empty_is_nan() -> None:
    reference = torch.zeros(2, 3, 4, 4)
    reference[0, 0] = 1.0  # image 0: red everywhere
    reference[1, 0, :, :2] = 1.0  # image 1: red on the left half, black on the right
    output = torch.zeros(2, 3, 4, 4)
    output[:, 1] = 1.0  # green everywhere: 90 degrees to red
    result = angular_error(output, reference)
    assert torch.allclose(result, torch.tensor([90.0, 90.0]), atol=1e-4)
    # a black output pixel is excluded as well; fully black reference gives NaN
    assert torch.isnan(angular_error(output, torch.zeros(2, 3, 4, 4))).all()
    assert torch.isnan(angular_error(torch.zeros(1, 3, 4, 4), reference[:1])).all()
    with pytest.raises(ValueError, match="eps"):
        angular_error(output, reference, eps=0.0)


# ----------------------------------------------------------------------------- clipping


def test_clipping_fraction_exact_on_constructed_image() -> None:
    img = torch.full((2, 3, 10, 10), 0.5)
    img[0, :, :2, :] = 1.0  # 20 pixels clipped
    img[0, :, 5, :3] = 0.98  # below the 0.99 threshold: not counted
    img[1, 0, :, :] = 1.0  # only R is saturated: Y = (1 + .5 + .5)/3 < .99
    result = clipping_fraction(img)
    assert result.shape == (2,)
    assert torch.allclose(result, torch.tensor([0.2, 0.0]))
    assert float(clipping_fraction(img, threshold=0.5)[1]) == 1.0  # Y(1.0, .5, .5) = .667 >= .5
    with pytest.raises(ValueError, match="threshold"):
        clipping_fraction(img, threshold=1.5)


# ----------------------------------------------------------------------------- warping error


def _translated_pair() -> tuple[Tensor, Tensor, Tensor]:
    """current(q) = previous(q + b(q)) with b = (-3,+2): current = roll of previous."""
    previous = _smooth_images(2, size=32, seed=15)
    current = torch.roll(previous, shifts=(-2, 3), dims=(2, 3))
    backward = torch.zeros(2, 2, 32, 32)
    backward[:, 0] = -3.0
    backward[:, 1] = 2.0
    return previous, current, backward


def test_warping_error_zero_for_consistent_pair_and_matches_loss() -> None:
    previous, current, backward = _translated_pair()
    mask = torch.ones(2, 1, 32, 32)
    assert torch.allclose(
        warping_error(previous, current, backward, mask), torch.zeros(2), atol=1e-6
    )
    shifted = current + 0.1
    err = warping_error(previous, shifted, backward, mask)
    assert err.shape == (2,) and torch.allclose(err, torch.full((2,), 0.1), atol=1e-5)
    # same normalisation as the training loss (single-pair batches coincide exactly)
    loss = temporal_loss(previous[:1], shifted[:1], backward[:1], mask[:1])
    assert float(err[0]) == pytest.approx(float(loss), abs=1e-7)


def test_warping_error_respects_mask_and_in_bounds() -> None:
    previous, current, backward = _translated_pair()
    perturbed = current.clone()
    perturbed[..., 5:15] += 0.2  # inside the valid region (x>=3, y<=29)
    region_a = torch.zeros(2, 1, 32, 32)
    region_a[..., 5:15] = 1.0
    region_b = torch.zeros(2, 1, 32, 32)
    region_b[..., 15:25] = 1.0
    assert torch.allclose(
        warping_error(previous, perturbed, backward, region_a), torch.full((2,), 0.2), atol=1e-5
    )
    assert torch.allclose(
        warping_error(previous, perturbed, backward, region_b), torch.zeros(2), atol=1e-6
    )
    # wrapped (out-of-bounds) pixels carry large error but are excluded by in_bounds
    ones = torch.ones(2, 1, 32, 32)
    assert float(warping_error(previous, current, backward, ones).max()) < 1e-6
    _, valid = warp(previous, backward)
    assert 0.5 < float(valid.mean()) < 1.0


def test_warping_error_empty_mask_is_nan_per_pair_and_validates() -> None:
    previous, current, backward = _translated_pair()
    mask = torch.ones(2, 1, 32, 32)
    mask[1] = 0.0
    result = warping_error(previous, current + 0.1, backward, mask)
    assert not torch.isnan(result[0]) and torch.isnan(result[1])
    assert torch.isnan(warping_error(previous, current, backward, torch.zeros(2, 1, 32, 32))).all()
    # a mask that only covers out-of-bounds pixels is also empty
    corner = torch.zeros(2, 1, 32, 32)
    corner[..., :, 0:3] = 1.0  # x < 3 samples x-3 < 0: out of bounds
    assert torch.isnan(warping_error(previous, current, backward, corner)).all()
    with pytest.raises(ValueError, match="mask must be"):
        warping_error(previous, current, backward, torch.ones(2, 3, 32, 32))
    with pytest.raises(ValueError, match="backward_flow must be"):
        warping_error(previous, current, backward[:, :1], mask)
    with pytest.raises(ValueError, match="same device"):
        warping_error(previous, current, backward, torch.ones(2, 1, 32, 32, device="meta"))
