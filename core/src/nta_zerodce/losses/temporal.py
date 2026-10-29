"""Pure tensor losses/targets; no IO, OpenCV, optimizer or registry side effects.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from torch import Tensor

from nta_zerodce.flow.warp import warp
from nta_zerodce.tensor_ops import masked_mean


def temporal_loss(
    previous_output: Tensor, current_output: Tensor, backward_flow: Tensor, mask: Tensor
) -> Tensor:
    """Penalize reliable flow-aligned frame differences.

    Args:
        previous_output/current_output: RGB (B,3,H,W) in [0,1].
        backward_flow: (B,2,H,W), pixel displacement current->previous, x then y.
        mask: Detached reliable weights (B,1,H,W) in [0,1].

    Returns:
        Scalar float32 nonnegative differentiable Tensor on the input device.

    Method:
        Extension 5; masked L1 current vs warp(previous,backward_flow); no detach on outputs. See docs/METHOD.md; baseline attribution [R1,R2], extensions are project proposals.

    Raises:
        ValueError: Invalid inputs or incompatible configuration.
    """
    aligned, bounds = warp(previous_output, backward_flow.detach())
    return masked_mean((current_output - aligned).abs(), mask.detach() * bounds)
