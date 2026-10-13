"""Pure tensor losses/targets; no IO, OpenCV, optimizer or registry side effects.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from torch import Tensor


def exposure_loss(output: Tensor, target: float = 0.6, patch_size: int = 16) -> Tensor:
    """Control baseline patch exposure.

    Args:
        output: RGB float32 tensor (B,3,H,W) in [0,1], on the input device.
        target: Desired mean RGB intensity in [0,1], default 0.6.
        patch_size: Non-overlapping region size, default 16.

    Returns:
        Scalar float32 nonnegative differentiable Tensor on the input device.

    Method:
        Baseline L_exp; reference-profile squared error; paper absolute variant documented. See docs/METHOD.md; baseline attribution [R1,R2], extensions are project proposals.

    Raises:
        NotImplementedError: Scaffold only; implement this contract before use.
    """
    raise NotImplementedError("TODO: Control baseline patch exposure")
