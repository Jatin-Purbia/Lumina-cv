"""Reference chromaticity/angular error and colour proxy reporting.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

import torch
from torch import Tensor


def angular_error(output: Tensor, reference: Tensor, eps: float = 1e-6) -> Tensor:
    """Measure RGB angular error on reliable reference pixels.

    Args:
        output: RGB float32 tensor (B,3,H,W) in [0,1], on the input device.
        reference: RGB (B,3,H,W) in [0,1], aligned.
        eps: Positive norm guard; exclude near-black vectors.

    Returns:
        Per-image (B,) mean degrees, lower better; empty reliable set => NaN.

    Method:
        EXPERIMENTS acos(clamp(dot/(norms),-1,1)); not perceptual Delta-E.

    Raises:
        ValueError: Invalid inputs or incompatible configuration.
    """
    n1, n2 = output.norm(dim=1), reference.norm(dim=1)
    mask = (n1 > eps) & (n2 > eps)
    cosine = ((output * reference).sum(1) / (n1 * n2).clamp_min(eps)).clamp(-1, 1)
    angles = torch.rad2deg(cosine.acos())
    count = mask.sum((1, 2))
    return torch.where(
        count > 0,
        (angles * mask).sum((1, 2)) / count.clamp_min(1),
        torch.full_like(count, float("nan"), dtype=output.dtype),
    )
