"""Flat-region noise proxy, not a perceptual quality guarantee.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def noise_estimate(image: Tensor, flat_mask: Tensor | None = None) -> Tensor:
    """Estimate flat-region Gaussian noise standard deviation.

    Args:
        image: RGB float32 tensor (B,3,H,W) in [0,1]; no implicit device transfer.
        flat_mask: Optional (B,1,H,W) binary valid region; no reliable pixels => NaN.

    Returns:
        Per-image (B,) normalized RGB-unit sigma estimates.

    Method:
        EXPERIMENTS: Immerkaer kernel and calibration [R8]; texture causes bias.

    Raises:
        ValueError: Invalid inputs or incompatible configuration.
    """
    kernel = image.new_tensor([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])[None, None]
    response = F.conv2d(image.mean(1, keepdim=True), kernel).abs()
    mask = (
        torch.ones_like(response[:, :1])
        if flat_mask is None
        else (F.avg_pool2d(flat_mask.detach(), 3, stride=1) >= 1).to(image.dtype)
    )
    count = mask.sum((1, 2, 3))
    estimate = (response * mask).sum((1, 2, 3)) / count.clamp_min(1) * math.sqrt(math.pi / 2) / 6
    return torch.where(count > 0, estimate, torch.full_like(estimate, float("nan")))
