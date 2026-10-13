"""Differentiable tensor sampling; no OpenCV dependency.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from torch import Tensor


def warp(source: Tensor, flow: Tensor, align_corners: bool = True) -> tuple[Tensor, Tensor]:
    """Sample a target-domain source into the flow origin coordinates.

    Args:
        source: Float32 (B,C,H,W), arbitrary C, same device as flow.
        flow: Float32 (B,2,H,W), origin->source displacement dx,dy in pixels.
        align_corners: True by default; fixed in training and inference.

    Returns:
        (warped source (B,C,H,W), valid mask (B,1,H,W)); bilinear, zero padding.

    Method:
        INTERFACES: warp(b,f(a,b))(p)=b(p+f(p)); METHOD normalized grid formula.

    Raises:
        NotImplementedError: Scaffold only; implement this contract before use.
    """
    raise NotImplementedError(
        "TODO: Sample a target-domain source into the flow origin coordinates"
    )
