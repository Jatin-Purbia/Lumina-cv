"""Only owner of OpenCV optical flow; CPU boundary and displacement rescaling.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from torch import Tensor

from nta_zerodce.training.config import FlowConfig


def estimate_flow(first: Tensor, second: Tensor, config: FlowConfig) -> Tensor:
    """Estimate source-to-target pixel displacement.

    Args:
        first/second: CPU RGB float32 (B,3,H,W) in [0,1], matching size.
        config: DIS method/preset, flow scale in (0,1].

    Returns:
        CPU float32 flow (B,2,H,W), dx,dy in original pixels; detached.

    Method:
        INTERFACES: first(p) ~ second(p+flow(p)); DIS [R6]. Resize vectors by full/small dimensions.

    Raises:
        NotImplementedError: Scaffold only; implement this contract before use.
    """
    raise NotImplementedError("TODO: Estimate source-to-target pixel displacement")


def estimate_bidirectional(
    first: Tensor, second: Tensor, config: FlowConfig
) -> tuple[Tensor, Tensor]:
    """Estimate forward and backward flow.

    Args:
        first/second: CPU RGB (B,3,H,W) in [0,1].
        config: Validated DIS settings.

    Returns:
        (first->second, second->first), each CPU (B,2,H,W) in original pixel units.

    Method:
        METHOD extension 5; compute both independently, never negate forward flow.

    Raises:
        NotImplementedError: Scaffold only; implement this contract before use.
    """
    raise NotImplementedError("TODO: Estimate forward and backward flow")
