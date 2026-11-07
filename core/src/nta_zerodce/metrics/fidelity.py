"""Paired full-reference metrics for LOL; EXPERIMENTS definitions.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from torch import Tensor

from nta_zerodce.metrics._util import accumulation_dtype, check_pair

SSIM_WINDOW = 11
SSIM_SIGMA = 1.5
SSIM_K1 = 0.01
SSIM_K2 = 0.03


def psnr(output: Tensor, reference: Tensor) -> Tensor:
    """Measure peak signal-to-noise ratio.

    Args:
        output: RGB float32 tensor (B,3,H,W) in [0,1], on the input device.
        reference: Registered RGB (B,3,H,W) in [0,1], same device.

    Returns:
        Per-image (B,) dB values; +infinity on exact equality.

    Method:
        EXPERIMENTS: 10*log10(1/MSE); higher is better. MSE is the mean over RGB, H and W
        of each image with data_range 1; values are not clamped or range-checked here.

    Raises:
        ValueError: Not (B,3,H,W) floating tensors, or shape/device/dtype mismatch.
    """
    check_pair(output, reference, ("output", "reference"))
    with torch.no_grad():
        dtype = accumulation_dtype(output)
        diff = output.detach().to(dtype) - reference.detach().to(dtype)
        mse = (diff * diff).mean(dim=(1, 2, 3))
        value = -10.0 * torch.log10(mse)
        value = torch.where(mse == 0, torch.full_like(value, math.inf), value)
    return value.to(output.dtype)


def _gaussian_kernel(device: torch.device, dtype: torch.dtype) -> Tensor:
    """Normalised 1-D Gaussian (11 taps, sigma 1.5)."""
    coords = torch.arange(SSIM_WINDOW, device=device, dtype=dtype) - (SSIM_WINDOW - 1) / 2.0
    kernel = torch.exp(-(coords**2) / (2.0 * SSIM_SIGMA**2))
    return kernel / kernel.sum()


def ssim(output: Tensor, reference: Tensor) -> Tensor:
    """Measure structural similarity.

    Args:
        output: RGB float32 tensor (B,3,H,W) in [0,1], on the input device.
        reference: Registered RGB (B,3,H,W) in [0,1].

    Returns:
        Per-image (B,) SSIM, usually [-1,1]; fixed RGB averaging policy.

    Method:
        EXPERIMENTS: Gaussian 11x11 sigma=1.5, K1=.01 K2=.03, data_range=1; [R10].
        Local moments come from a normalised separable Gaussian window evaluated on the
        valid region only (no padding), so the map has (H-10)x(W-10) positions; local
        covariances use the biased (population) normalisation. The score is the mean of
        the SSIM map over that valid window, per channel, then averaged over R, G, B.

    Raises:
        ValueError: Invalid tensors, shape/device/dtype mismatch, or H<11 or W<11.
    """
    check_pair(output, reference, ("output", "reference"))
    height, width = output.shape[2], output.shape[3]
    if height < SSIM_WINDOW or width < SSIM_WINDOW:
        raise ValueError(
            f"ssim needs images of at least {SSIM_WINDOW}x{SSIM_WINDOW} pixels, "
            f"got {height}x{width}"
        )
    with torch.no_grad():
        dtype = accumulation_dtype(output)
        x = output.detach().to(dtype)
        y = reference.detach().to(dtype)
        channels = x.shape[1]
        k1d = _gaussian_kernel(x.device, dtype)
        kernel_h = k1d.view(1, 1, 1, SSIM_WINDOW).repeat(channels, 1, 1, 1)
        kernel_v = k1d.view(1, 1, SSIM_WINDOW, 1).repeat(channels, 1, 1, 1)

        def blur(t: Tensor) -> Tensor:
            return functional.conv2d(
                functional.conv2d(t, kernel_h, groups=channels), kernel_v, groups=channels
            )

        mu_x, mu_y = blur(x), blur(y)
        var_x = blur(x * x) - mu_x * mu_x
        var_y = blur(y * y) - mu_y * mu_y
        cov = blur(x * y) - mu_x * mu_y
        c1 = SSIM_K1**2
        c2 = SSIM_K2**2
        ssim_map = ((2.0 * mu_x * mu_y + c1) * (2.0 * cov + c2)) / (
            (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
        )
        value = ssim_map.mean(dim=(1, 2, 3))
    return value.to(output.dtype)
