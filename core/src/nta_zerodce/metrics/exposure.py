"""Luminance clipping fraction; EXPERIMENTS highlight cautions.

See docs/ARCHITECTURE.md for the normative contract.
"""

from __future__ import annotations

import torch
from torch import Tensor

from nta_zerodce.metrics._util import check_rgb, intensity


def clipping_fraction(output: Tensor, threshold: float = 0.99) -> Tensor:
    """Measure near-saturated luminance fraction.

    Args:
        output: RGB float32 tensor (B,3,H,W) in [0,1], on the input device.
        threshold: In [0,1], default .99.

    Returns:
        Per-image (B,) fraction [0,1], lower conditionally better.

    Method:
        EXPERIMENTS: mean(Y(output)>=threshold); report input clipping alongside output.
        Y is the mean of RGB (compatibility intensity, not linear luminance).

    Raises:
        ValueError: Invalid tensor or threshold outside [0,1].
    """
    check_rgb("output", output)
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0,1], got {threshold}")
    with torch.no_grad():
        flagged = intensity(output.detach()) >= threshold
        return flagged.to(output.dtype).mean(dim=(1, 2, 3))
