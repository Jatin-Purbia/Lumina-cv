"""Pure tensor losses/targets; no IO, OpenCV, optimizer or registry side effects.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from torch import Tensor

from nta_zerodce.losses._ops import luminance, pool_mean


def exposure_loss(output: Tensor, target: float = 0.6, patch_size: int = 16) -> Tensor:
    """Control baseline patch exposure.

    Args:
        output: RGB float32 tensor (B,3,H,W) in [0,1], on the input device.
        target: Desired mean RGB intensity in [0,1], default 0.6.
        patch_size: Non-overlapping region size, default 16.

    Returns:
        Scalar float32 nonnegative differentiable Tensor on the input device.

    Method:
        Baseline L_exp = mean (P_k(Y(o)) - target)^2, the reference-profile squared error (the
        paper's absolute variant mean|q-target| is documented in METHOD but not used here).
        See docs/METHOD.md; baseline attribution [R1,R2].

    Raises:
        ValueError: If the output is not RGB or is smaller than one patch.
    """
    pooled = pool_mean(luminance(output), patch_size)
    diff = pooled - target
    return (diff * diff).mean()
