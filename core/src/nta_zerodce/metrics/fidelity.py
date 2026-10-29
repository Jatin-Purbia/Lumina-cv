"""Paired full-reference metrics for LOL; EXPERIMENTS definitions.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def psnr(output: Tensor, reference: Tensor) -> Tensor:
    """Measure peak signal-to-noise ratio.

    Args:
        output: RGB float32 tensor (B,3,H,W) in [0,1], on the input device.
        reference: Registered RGB (B,3,H,W) in [0,1], same device.

    Returns:
        Per-image (B,) dB values; +infinity on exact equality.

    Method:
        EXPERIMENTS: 10*log10(1/MSE); higher is better.

    Raises:
        ValueError: Invalid inputs or incompatible configuration.
    """
    if output.shape != reference.shape:
        raise ValueError("PSNR inputs must have identical shapes")
    return -10 * (output - reference).square().mean((1, 2, 3)).log10()


def ssim(output: Tensor, reference: Tensor) -> Tensor:
    """Measure structural similarity.

    Args:
        output: RGB float32 tensor (B,3,H,W) in [0,1], on the input device.
        reference: Registered RGB (B,3,H,W) in [0,1].

    Returns:
        Per-image (B,) SSIM, usually [-1,1]; fixed RGB averaging policy.

    Method:
        EXPERIMENTS: Gaussian 11x11 sigma=1.5, K1=.01 K2=.03, data_range=1; [R10].

    Raises:
        ValueError: Invalid inputs or incompatible configuration.
    """
    if output.shape != reference.shape or min(output.shape[-2:]) < 11:
        raise ValueError("SSIM requires matching RGB images at least 11x11")
    coords = torch.arange(11, device=output.device, dtype=output.dtype) - 5
    g = torch.exp(-coords.square() / (2 * 1.5**2))
    g = g / g.sum()
    kernel = (g[:, None] * g[None, :])[None, None].expand(3, 1, 11, 11)
    mx = F.conv2d(output, kernel, groups=3)
    my = F.conv2d(reference, kernel, groups=3)
    vx = F.conv2d(output.square(), kernel, groups=3) - mx.square()
    vy = F.conv2d(reference.square(), kernel, groups=3) - my.square()
    cov = F.conv2d(output * reference, kernel, groups=3) - mx * my
    score = ((2 * mx * my + 0.01**2) * (2 * cov + 0.03**2)) / (
        (mx.square() + my.square() + 0.01**2) * (vx + vy + 0.03**2)
    )
    return score.mean((1, 2, 3))
