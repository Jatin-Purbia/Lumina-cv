"""Motion-compensated flicker metric; flow reliability required.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from nta_zerodce.flow.warp import warp
from nta_zerodce.metrics._util import check_pair

_EPS = 1e-6


def warping_error(previous: Tensor, current: Tensor, backward_flow: Tensor, mask: Tensor) -> Tensor:
    """Measure per-pair flow-aligned temporal L1 error.

    Args:
        previous/current: RGB (B,3,H,W) in [0,1].
        backward_flow: Current->previous (B,2,H,W) pixel units.
        mask: Detached (B,1,H,W) in [0,1].

    Returns:
        Per-pair (B,) normalized L1; empty masks => NaN for metrics, lower better.

    Method:
        EXPERIMENTS temporal metric; also report mask coverage to expose flow rejection.
        error_b = sum(w*|current - warp(previous, backward_flow)|) / (3*sum(w) + 1e-6)
        per pair with w = mask * in_bounds(backward_flow), i.e. the same normalisation as
        the temporal loss. NaN when sum(w) == 0.

    Raises:
        ValueError: Invalid RGB tensors, flow/mask shape mismatch, or device mismatch.
    """
    check_pair(previous, current, ("previous", "current"))
    batch, _, height, width = current.shape
    if tuple(backward_flow.shape) != (batch, 2, height, width):
        raise ValueError(
            f"backward_flow must be {(batch, 2, height, width)}, got {tuple(backward_flow.shape)}"
        )
    if tuple(mask.shape) != (batch, 1, height, width):
        raise ValueError(f"mask must be {(batch, 1, height, width)}, got {tuple(mask.shape)}")
    if backward_flow.device != current.device or mask.device != current.device:
        raise ValueError("flow and mask must be on the same device as the frames")
    with torch.no_grad():
        warped, in_bounds = warp(previous.detach(), backward_flow.detach())
        weight = mask.detach().to(current.dtype) * in_bounds
        residual = (current.detach() - warped).abs()
        total = (weight * residual).sum(dim=(1, 2, 3))
        mass = weight.sum(dim=(1, 2, 3))
        value = total / (3.0 * mass + _EPS)
        return torch.where(mass > 0, value, torch.full_like(value, math.nan))
