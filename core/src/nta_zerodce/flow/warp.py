"""Differentiable tensor sampling; no OpenCV dependency.

See docs/ARCHITECTURE.md for the normative contract.
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor


def _validate_warp_inputs(source: Tensor, flow: Tensor) -> None:
    """Reject inputs that violate the warp contract (shape, device, dtype, size)."""
    if source.dim() != 4:
        raise ValueError(f"source must be (B,C,H,W), got shape {tuple(source.shape)}")
    if flow.dim() != 4 or flow.shape[1] != 2:
        raise ValueError(f"flow must be (B,2,H,W), got shape {tuple(flow.shape)}")
    if source.shape[0] != flow.shape[0] or source.shape[2:] != flow.shape[2:]:
        raise ValueError(
            f"source {tuple(source.shape)} and flow {tuple(flow.shape)} must share B, H and W"
        )
    if source.device != flow.device:
        raise ValueError(
            f"source ({source.device}) and flow ({flow.device}) must be on the same device; "
            "warp never transfers tensors implicitly"
        )
    if not (source.is_floating_point() and flow.is_floating_point()):
        raise ValueError("source and flow must be floating point tensors")
    if source.dtype != flow.dtype:
        raise ValueError(f"source ({source.dtype}) and flow ({flow.dtype}) dtypes must match")
    if source.shape[2] < 2 or source.shape[3] < 2:
        raise ValueError(
            "warp requires H>=2 and W>=2 (the normalised grid divides by size-1); "
            f"got H={source.shape[2]}, W={source.shape[3]}"
        )


def warp(source: Tensor, flow: Tensor, align_corners: bool = True) -> tuple[Tensor, Tensor]:
    """Sample a target-domain source into the flow origin coordinates.

    Args:
        source: Float32 (B,C,H,W), arbitrary C, same device as flow.
        flow: Float32 (B,2,H,W), origin->source displacement dx,dy in pixels.
        align_corners: True by default; fixed in training and inference.

    Returns:
        (warped source (B,C,H,W), valid mask (B,1,H,W)); bilinear, zero padding. The mask
        is 1 where the sampled location p+flow(p) lies inside [0,W-1]x[0,H-1] (pixel-centre
        coordinates), else 0; it is binary, non-differentiable and has the source dtype.
        The warped tensor is differentiable with respect to both source and flow.

    Method:
        INTERFACES: warp(b,f(a,b))(p)=b(p+f(p)); METHOD normalized grid formula
        grid_x=2*(p_x+dx)/(W-1)-1, grid_y=2*(p_y+dy)/(H-1)-1 for align_corners=True.
        With align_corners=False the equivalent pixel-centre mapping (2*x+1)/size-1 is
        used, so the same pixel displacement is meant in both modes.

    Raises:
        ValueError: Shapes/batch/spatial sizes differ, flow does not have 2 channels,
            devices or dtypes differ, tensors are not floating point, or H<2 or W<2.
    """
    _validate_warp_inputs(source, flow)
    height, width = int(source.shape[2]), int(source.shape[3])
    ys = torch.arange(height, device=flow.device, dtype=flow.dtype)
    xs = torch.arange(width, device=flow.device, dtype=flow.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    sample_x = grid_x.unsqueeze(0) + flow[:, 0]  # (B,H,W) sampled pixel x
    sample_y = grid_y.unsqueeze(0) + flow[:, 1]
    if align_corners:
        norm_x = 2.0 * sample_x / (width - 1) - 1.0
        norm_y = 2.0 * sample_y / (height - 1) - 1.0
    else:
        norm_x = (2.0 * sample_x + 1.0) / width - 1.0
        norm_y = (2.0 * sample_y + 1.0) / height - 1.0
    grid = torch.stack((norm_x, norm_y), dim=-1)
    warped = functional.grid_sample(
        source, grid, mode="bilinear", padding_mode="zeros", align_corners=align_corners
    )
    with torch.no_grad():
        inside = (
            (sample_x >= 0) & (sample_x <= width - 1) & (sample_y >= 0) & (sample_y <= height - 1)
        )
        valid = inside.unsqueeze(1).to(source.dtype)
    return warped, valid
