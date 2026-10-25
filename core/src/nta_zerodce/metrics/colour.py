"""Reference chromaticity/angular error and colour proxy reporting.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from nta_zerodce.metrics._util import accumulation_dtype, check_pair


def angular_error(output: Tensor, reference: Tensor, eps: float = 1e-6) -> Tensor:
    """Measure RGB angular error on reliable reference pixels.

    Args:
        output: RGB float32 tensor (B,3,H,W) in [0,1], on the input device.
        reference: RGB (B,3,H,W) in [0,1], aligned.
        eps: Positive norm guard; exclude near-black vectors.

    Returns:
        Per-image (B,) mean degrees, lower better; empty reliable set => NaN.

    Method:
        EXPERIMENTS acos(clamp(dot/(norms),-1,1)); not perceptual Delta-E. Pixels whose
        RGB vector norm is <= eps in either image are excluded; for the remaining pixels
        both norms exceed eps, so the normalised dot product is divided by the plain norm
        product (no additive eps, which would bias tiny vectors). Computed in float64
        where available so identical vectors give 0 degrees.

    Raises:
        ValueError: Invalid tensors, shape/device/dtype mismatch, or eps not positive.
    """
    check_pair(output, reference, ("output", "reference"))
    if not eps > 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    with torch.no_grad():
        dtype = accumulation_dtype(output)
        x = output.detach().to(dtype)
        y = reference.detach().to(dtype)
        norm_x = x.norm(dim=1)
        norm_y = y.norm(dim=1)
        reliable = (norm_x > eps) & (norm_y > eps)  # (B,H,W)
        denom = torch.where(reliable, norm_x * norm_y, torch.ones_like(norm_x))
        cosine = ((x * y).sum(dim=1) / denom).clamp(-1.0, 1.0)
        degrees = torch.acos(cosine) * (180.0 / math.pi)
        weight = reliable.to(dtype)
        count = weight.sum(dim=(1, 2))
        mean = (degrees * weight).sum(dim=(1, 2)) / count.clamp(min=1.0)
        mean = torch.where(count > 0, mean, torch.full_like(mean, math.nan))
    return mean.to(output.dtype)
