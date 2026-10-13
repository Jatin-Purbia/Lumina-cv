"""Motion-compensated flicker metric; flow reliability required.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from torch import Tensor


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
        NotImplementedError: Scaffold only; implement this contract before use.
    """
    raise NotImplementedError("TODO: Measure per-pair flow-aligned temporal L1 error")
