"""Pure tensor losses/targets; no IO, OpenCV, optimizer or registry side effects.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

import torch
from torch import Tensor


def color_loss(output: Tensor) -> Tensor:
    """Penalize unequal global RGB channel means.

    Args:
        output: RGB float32 tensor (B,3,H,W) in [0,1], on the input device.

    Returns:
        Scalar float32 nonnegative differentiable Tensor on the input device.

    Method:
        Baseline L_col = mean_batch sqrt(d_RG^4+d_RB^4+d_GB^4) with d_cd the difference of the
        per-image channel means. The square root uses a zero-safe derivative: where the radicand
        is exactly zero the value is zero and the gradient is zero, never NaN or infinity.
        See docs/METHOD.md; baseline attribution [R1,R2].

    Raises:
        ValueError: If the output is not an RGB tensor (B,3,H,W).
    """
    if output.dim() != 4 or output.shape[1] != 3:
        raise ValueError(f"expected an RGB tensor (B,3,H,W), got {tuple(output.shape)}")
    mean = output.mean(dim=(2, 3))  # (B,3)
    d_rg = mean[:, 0] - mean[:, 1]
    d_rb = mean[:, 0] - mean[:, 2]
    d_gb = mean[:, 1] - mean[:, 2]
    radicand = d_rg**4 + d_rb**4 + d_gb**4
    positive = radicand > 0
    safe = torch.where(positive, radicand, torch.ones_like(radicand))
    root = torch.where(positive, torch.sqrt(safe), torch.zeros_like(radicand))
    return root.mean()
