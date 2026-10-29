"""Motion-compensated flicker metric; flow reliability required.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

import torch
from torch import Tensor

from nta_zerodce.flow.warp import warp


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

    Raises:
        ValueError: Invalid inputs or incompatible configuration.
    """
    aligned, bounds = warp(previous, backward_flow)
    valid = mask * bounds
    count = valid.sum((1, 2, 3)) * current.shape[1]
    value = ((current - aligned).abs() * valid).sum((1, 2, 3)) / count.clamp_min(1)
    return torch.where(count > 0, value, torch.full_like(value, float("nan")))
