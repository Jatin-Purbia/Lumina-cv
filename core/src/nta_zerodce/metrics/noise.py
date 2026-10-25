"""Flat-region noise proxy, not a perceptual quality guarantee.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from torch import Tensor

from nta_zerodce.metrics._util import accumulation_dtype, check_rgb, intensity


def noise_estimate(image: Tensor, flat_mask: Tensor | None = None) -> Tensor:
    """Estimate flat-region Gaussian noise standard deviation.

    Args:
        image: RGB float32 tensor (B,3,H,W) in [0,1]; no implicit device transfer.
        flat_mask: Optional (B,1,H,W) binary valid region; no reliable pixels => NaN.

    Returns:
        Per-image (B,) normalized RGB-unit sigma estimates.

    Method:
        EXPERIMENTS: Immerkaer kernel and calibration [R8]; texture causes bias.
        K=[[1,-2,1],[-2,4,-2],[1,-2,1]] is applied to Y = mean RGB with valid 3x3
        responses only. With a flat_mask, a response counts only if all nine pixels of its
        3x3 support are inside the mask. sigma = sqrt(pi/2) * sum|response| /
        (6 * valid_count); NaN when valid_count is zero.

    Raises:
        ValueError: Invalid image, image smaller than 3x3, flat_mask not (B,1,H,W) binary
            (0/1 or bool) on the same device.
    """
    check_rgb("image", image)
    batch, _, height, width = image.shape
    if height < 3 or width < 3:
        raise ValueError(f"noise_estimate needs at least a 3x3 image, got {height}x{width}")
    if flat_mask is not None:
        if tuple(flat_mask.shape) != (batch, 1, height, width):
            raise ValueError(
                f"flat_mask must be {(batch, 1, height, width)}, got {tuple(flat_mask.shape)}"
            )
        if flat_mask.device != image.device:
            raise ValueError("flat_mask must be on the same device as image")
        if flat_mask.dtype != torch.bool and not bool(((flat_mask == 0) | (flat_mask == 1)).all()):
            raise ValueError("flat_mask must be binary (0/1 or bool)")
    with torch.no_grad():
        dtype = accumulation_dtype(image)
        luma = intensity(image.detach().to(dtype))
        kernel = torch.tensor(
            [[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]],
            device=image.device,
            dtype=dtype,
        ).view(1, 1, 3, 3)
        response = functional.conv2d(luma, kernel).abs()
        if flat_mask is None:
            valid = torch.ones_like(response)
        else:
            inside = flat_mask.detach().to(dtype)
            ones = torch.ones(1, 1, 3, 3, device=image.device, dtype=dtype)
            valid = (functional.conv2d(inside, ones) > 8.5).to(dtype)
        count = valid.sum(dim=(1, 2, 3))
        total = (response * valid).sum(dim=(1, 2, 3))
        sigma = math.sqrt(math.pi / 2.0) * total / (6.0 * count.clamp(min=1.0))
        sigma = torch.where(count > 0, sigma, torch.full_like(sigma, math.nan))
    return sigma.to(image.dtype)
