"""Pure tensor losses/targets; no IO, OpenCV, optimizer or registry side effects.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from nta_zerodce.tensor_ops import smoothstep


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
        Extension 4 target formula and smoothstep in METHOD. See docs/METHOD.md; baseline attribution [R1,R2], extensions are project proposals.

    Raises:
        ValueError: Invalid inputs or incompatible configuration.
    """
    yp = F.avg_pool2d(image.mean(1, keepdim=True), patch_size)
    sp = F.avg_pool2d(sigma.detach(), patch_size)
    return (
        target
        + (yp - target) * smoothstep(yp, low, high)
        - noise_penalty * (sp / noise_scale).clamp(0, 1)
    ).detach()


def adaptive_exposure_loss(output: Tensor, targets: Tensor, patch_size: int = 16) -> Tensor:
    """Match adaptive patch exposure targets.

    Args:
        output: RGB float32 tensor (B,3,H,W) in [0,1], on the input device.
        targets: Detached (B,1,H//patch_size,W//patch_size) in [0,1].
        patch_size: Non-overlapping pool size, default 16.

    Returns:
        Scalar float32 nonnegative differentiable Tensor on the input device.

    Method:
        Extension 4; mean squared pooled luminance error, replaces L_exp. See docs/METHOD.md; baseline attribution [R1,R2], extensions are project proposals.

    Raises:
        ValueError: Invalid inputs or incompatible configuration.
    """
    return (
        (F.avg_pool2d(output.mean(1, keepdim=True), patch_size) - targets.detach()).square().mean()
    )
