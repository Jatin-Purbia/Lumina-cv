"""Pure tensor losses/targets; no IO, OpenCV, optimizer or registry side effects.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

import torch
from torch import Tensor

from nta_zerodce.losses._ops import luminance, pool_mean, smoothstep


def adaptive_targets(
    image: Tensor,
    sigma: Tensor,
    target: float = 0.6,
    low: float = 0.55,
    high: float = 0.85,
    noise_scale: float = 0.05,
    noise_penalty: float = 0.15,
    patch_size: int = 16,
) -> Tensor:
    """Compute detached noise-aware patch exposure targets.

    Args:
        image: RGB float32 tensor (B,3,H,W) in [0,1]; no implicit device transfer.
        sigma: Nonnegative float32 tensor (B,1,H,W), standard deviation in RGB units.
        target/low/high: [0,1] constants with low<high.
        noise_scale: Positive normalization, default 0.05.
        noise_penalty: Target reduction, default 0.15.
        patch_size: Pool size, default 16.

    Returns:
        Detached patch targets (B,1,H//patch_size,W//patch_size) in [0,1].

    Method:
        Extension 4: with Yp=P_k(Y(x)) and sigma_p=P_k(sigma),
        T = target + (Yp-target)*smoothstep(low,high,Yp) - penalty*clamp(sigma_p/noise_scale,0,1).
        Neither the image nor sigma receives gradient, so the loss cannot lower targets by
        inflating sigma. See docs/METHOD.md; extensions are project proposals.

    Raises:
        ValueError: If shapes disagree, noise_scale is not positive or low >= high.
    """
    if noise_scale <= 0:
        raise ValueError(f"noise_scale must be positive, got {noise_scale}")
    if image.shape[0] != sigma.shape[0] or image.shape[-2:] != sigma.shape[-2:]:
        raise ValueError(f"image {tuple(image.shape)} and sigma {tuple(sigma.shape)} disagree")
    with torch.no_grad():
        yp = pool_mean(luminance(image.detach()), patch_size)
        sigma_p = pool_mean(sigma.detach(), patch_size)
        raised = target + (yp - target) * smoothstep(low, high, yp)
        penalty = noise_penalty * (sigma_p / noise_scale).clamp(0.0, 1.0)
        return (raised - penalty).detach()


def adaptive_exposure_loss(output: Tensor, targets: Tensor, patch_size: int = 16) -> Tensor:
    """Match adaptive patch exposure targets.

    Args:
        output: RGB float32 tensor (B,3,H,W) in [0,1], on the input device.
        targets: Detached (B,1,H//patch_size,W//patch_size) in [0,1].
        patch_size: Non-overlapping pool size, default 16.

    Returns:
        Scalar float32 nonnegative differentiable Tensor on the input device.

    Method:
        Extension 4; mean squared pooled luminance error mean (P_k(Y(o))-T)^2, replaces L_exp.
        Targets are detached defensively. See docs/METHOD.md; extensions are project proposals.

    Raises:
        ValueError: If the targets do not match the pooled output shape.
    """
    pooled = pool_mean(luminance(output), patch_size)
    if pooled.shape != targets.shape:
        raise ValueError(f"targets {tuple(targets.shape)} must match pooled {tuple(pooled.shape)}")
    diff = pooled - targets.detach()
    return (diff * diff).mean()
