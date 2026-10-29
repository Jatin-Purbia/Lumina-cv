"""Pure tensor losses/targets; no IO, OpenCV, optimizer or registry side effects.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from torch import Tensor


def smoothness_loss(A: Tensor) -> Tensor:
    """Regularize curve maps spatially.

    Args:
        A: (B,24,H,W) in [-1,1], H,W>=2.

    Returns:
        Scalar float32 nonnegative differentiable Tensor on the input device.

    Method:
        Baseline L_tv = 2*(mean(dx(A)^2)+mean(dy(A)^2)). See docs/METHOD.md; baseline attribution [R1,R2], extensions are project proposals.

    Raises:
        ValueError: Invalid inputs or incompatible configuration.
    """
    zero = A.sum() * 0
    dx = (A[..., 1:] - A[..., :-1]).square().mean() if A.shape[-1] > 1 else zero
    dy = (A[..., 1:, :] - A[..., :-1, :]).square().mean() if A.shape[-2] > 1 else zero
    return 2 * (dx + dy)
