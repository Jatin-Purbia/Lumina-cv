"""Pure tensor losses/targets; no IO, OpenCV, optimizer or registry side effects.

See docs/ARCHITECTURE.md for the normative contract.
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
        Baseline L_tv = 2*(mean(dx(A)^2)+mean(dy(A)^2)); the two directional means are taken
        independently over their own element counts. See docs/ARCHITECTURE.md; baseline attribution
        [R1,R2].

    Raises:
        ValueError: If A is not 4-D or has a spatial side smaller than two.
    """
    if A.dim() != 4 or A.shape[-2] < 2 or A.shape[-1] < 2:
        raise ValueError(f"expected A of shape (B,C,H,W) with H,W>=2, got {tuple(A.shape)}")
    dx = A[..., :, 1:] - A[..., :, :-1]
    dy = A[..., 1:, :] - A[..., :-1, :]
    return 2.0 * ((dx * dx).mean() + (dy * dy).mean())
