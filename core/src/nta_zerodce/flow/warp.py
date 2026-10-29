"""Differentiable tensor sampling; no OpenCV dependency.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def warp(source: Tensor, flow: Tensor, align_corners: bool = True) -> tuple[Tensor, Tensor]:
    """Sample a target-domain source into the flow origin coordinates.

    Args:
        source: Float32 (B,C,H,W), arbitrary C, same device as flow.
        flow: Float32 (B,2,H,W), origin->source displacement dx,dy in pixels.
        align_corners: True by default; fixed in training and inference.

    Returns:
        (warped source (B,C,H,W), valid mask (B,1,H,W)); bilinear, zero padding.

    Method:
        INTERFACES: warp(b,f(a,b))(p)=b(p+f(p)); METHOD normalized grid formula.

    Raises:
        ValueError: Invalid inputs or incompatible configuration.
    """
    if (
        not align_corners
        or source.ndim != 4
        or flow.shape != (source.shape[0], 2, *source.shape[-2:])
        or min(source.shape[-2:]) < 2
    ):
        raise ValueError("Warp requires matching BCHW/ B2HW, H,W>1 and align_corners=True")
    if source.device != flow.device or source.dtype != flow.dtype:
        raise ValueError("Warp device/dtype mismatch")
    h, w = source.shape[-2:]
    y, x = torch.meshgrid(
        torch.arange(h, device=source.device, dtype=source.dtype),
        torch.arange(w, device=source.device, dtype=source.dtype),
        indexing="ij",
    )
    px, py = x + flow[:, 0], y + flow[:, 1]
    bounds = ((px >= 0) & (px <= w - 1) & (py >= 0) & (py <= h - 1))[:, None].to(source.dtype)
    grid = torch.stack((2 * px / (w - 1) - 1, 2 * py / (h - 1) - 1), -1)
    return F.grid_sample(
        source, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    ), bounds
