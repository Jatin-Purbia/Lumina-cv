"""Pure tensor losses/targets; no IO, OpenCV, optimizer or registry side effects.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def spatial_loss(image: Tensor, output: Tensor, patch_size: int = 4) -> Tensor:
    """Preserve local intensity contrast.

    Args:
        image: RGB float32 tensor (B,3,H,W) in [0,1]; no implicit device transfer.
        output: RGB float32 tensor (B,3,H,W) in [0,1], on the input device.
        patch_size: Non-overlapping region size, default 4.

    Returns:
        Scalar float32 nonnegative differentiable Tensor on the input device.

    Method:
        Baseline L_spa; valid four-neighbour region pairs, squared absolute differences. See docs/METHOD.md; baseline attribution [R1,R2], extensions are project proposals.

    Raises:
        ValueError: Invalid inputs or incompatible configuration.
    """
    u = F.avg_pool2d(image.mean(1, keepdim=True), patch_size)
    v = F.avg_pool2d(output.mean(1, keepdim=True), patch_size)
    dx = ((v[..., 1:] - v[..., :-1]).abs() - (u[..., 1:] - u[..., :-1]).abs()).square().sum()
    dy = (
        ((v[..., 1:, :] - v[..., :-1, :]).abs() - (u[..., 1:, :] - u[..., :-1, :]).abs())
        .square()
        .sum()
    )
    return 2 * (dx + dy) / u.numel()
