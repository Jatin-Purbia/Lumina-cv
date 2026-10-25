"""Pure tensor losses/targets; no IO, OpenCV, optimizer or registry side effects.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from torch import Tensor

from nta_zerodce.flow.warp import warp

_EPS = 1e-6


def _masked_mean_rgb(values: Tensor, weight: Tensor) -> Tensor:
    """masked_mean_C(v,m) = sum(m*v) / (C*sum(m) + eps), C = values.shape[1] (local helper)."""
    channels = values.shape[1]
    return (weight * values).sum() / (channels * weight.sum() + _EPS)


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
        Extension 5; masked L1 current vs warp(previous,backward_flow); no detach on outputs.
        L = sum(w*|o_t - warp(o_{t-1}, b)|) / (3*sum(w) + 1e-6) with
        w = mask * in_bounds(b). Gradients reach both outputs; the flow and the mask are
        constants (detached inside). An empty mask (sum(w)=0) gives exactly 0 that is still
        connected to the autograd graph of the outputs (zero gradients, no NaN). See
        docs/METHOD.md; baseline attribution [R1,R2], extensions are project proposals.

    Raises:
        ValueError: Outputs are not same-shape (B,3,H,W) tensors, flow/mask shapes do not
            match, or tensors live on different devices or dtypes.
    """
    for name, frame in (("previous_output", previous_output), ("current_output", current_output)):
        if frame.dim() != 4 or frame.shape[1] != 3:
            raise ValueError(f"{name} must be (B,3,H,W) RGB, got shape {tuple(frame.shape)}")
    if previous_output.shape != current_output.shape:
        raise ValueError(
            f"previous_output {tuple(previous_output.shape)} and current_output "
            f"{tuple(current_output.shape)} must have equal shape"
        )
    batch, _, height, width = current_output.shape
    if tuple(backward_flow.shape) != (batch, 2, height, width):
        raise ValueError(
            f"backward_flow must be {(batch, 2, height, width)}, got {tuple(backward_flow.shape)}"
        )
    if tuple(mask.shape) != (batch, 1, height, width):
        raise ValueError(f"mask must be {(batch, 1, height, width)}, got {tuple(mask.shape)}")
    devices = {t.device for t in (previous_output, current_output, backward_flow, mask)}
    if len(devices) != 1:
        raise ValueError(f"all tensors must share one device, got {sorted(map(str, devices))}")
    if len({t.dtype for t in (previous_output, current_output, backward_flow, mask)}) != 1:
        raise ValueError("all tensors must share one dtype")

    warped_previous, in_bounds = warp(previous_output, backward_flow.detach())
    weight = mask.detach() * in_bounds
    residual = (current_output - warped_previous).abs()
    return _masked_mean_rgb(residual, weight)
