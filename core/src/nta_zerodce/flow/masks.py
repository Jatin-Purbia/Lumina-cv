"""Detached temporal reliability and scene-cut tests; pure tensors.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

import torch
from torch import Tensor

from nta_zerodce.flow.warp import warp


def reliability_mask(
    forward_flow: Tensor,
    backward_flow: Tensor,
    previous: Tensor,
    current: Tensor,
    alpha: float = 0.01,
    beta: float = 0.5,
    photometric_threshold: float = 0.1,
) -> Tensor:
    """Compute a current-coordinate reliable-pixel mask.

    Args:
        forward_flow/backward_flow: (B,2,H,W), prev->current and current->prev.
        previous/current: RGB (B,3,H,W) in [0,1].
        alpha: Nonnegative relative FB threshold.
        beta: Nonnegative threshold in squared pixels.
        photometric_threshold: Mean-RGB residual threshold in [0,1].

    Returns:
        Detached mask (B,1,H,W), float32 0 or 1; combines FB, bounds and photometric tests.

    Method:
        METHOD extension 5; compare backward + warp(forward,backward), in current coordinates.

    Raises:
        ValueError: Invalid inputs or incompatible configuration.
    """
    with torch.no_grad():
        warped_forward, bounds = warp(forward_flow.detach(), backward_flow.detach())
        aligned, _ = warp(previous.detach(), backward_flow.detach())
        error = (backward_flow + warped_forward).square().sum(1, keepdim=True)
        limit = (
            alpha
            * (
                backward_flow.square().sum(1, keepdim=True)
                + warped_forward.square().sum(1, keepdim=True)
            )
            + beta
        )
        photo = (
            current.mean(1, keepdim=True) - aligned.mean(1, keepdim=True)
        ).abs() < photometric_threshold
        return ((error <= limit) & photo).to(current.dtype) * bounds


def is_scene_cut(previous: Tensor, current: Tensor, threshold: float = 0.3, bins: int = 32) -> bool:
    """Detect a scene cut before reusing temporal state.

    Args:
        previous/current: One RGB frame (1,3,H,W) in [0,1], same size.
        threshold: Half-L1 normalized histogram threshold in [0,1].
        bins: Number of mean-RGB histogram bins, integer >=2, default 32.

    Returns:
        True if normalized mean-RGB bins-bin histogram distance exceeds threshold.

    Method:
        METHOD extension 5; conservative project heuristic, not a learned cut detector.

    Raises:
        ValueError: Invalid inputs or incompatible configuration.
    """
    if previous.shape != current.shape:
        return True
    if previous.shape[0] != 1 or bins < 2 or not 0 <= threshold <= 1:
        raise ValueError("Scene-cut detection requires one frame and valid histogram settings")
    a = torch.histc(previous.detach().mean(1), bins=bins, min=0, max=1)
    b = torch.histc(current.detach().mean(1), bins=bins, min=0, max=1)
    return bool((0.5 * (a / a.sum() - b / b.sum()).abs().sum() > threshold).item())
