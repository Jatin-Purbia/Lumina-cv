"""Detached temporal reliability and scene-cut tests; pure tensors.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

import torch
from torch import Tensor

from nta_zerodce.flow.warp import warp


def _check_rgb_pair(previous: Tensor, current: Tensor, batch_one: bool = False) -> None:
    """Validate two RGB frames of identical shape on one device."""
    for name, frame in (("previous", previous), ("current", current)):
        if frame.dim() != 4 or frame.shape[1] != 3:
            raise ValueError(f"{name} must be (B,3,H,W) RGB, got shape {tuple(frame.shape)}")
    if previous.shape != current.shape:
        raise ValueError(
            f"previous {tuple(previous.shape)} and current {tuple(current.shape)} must match"
        )
    if previous.device != current.device:
        raise ValueError("previous and current must be on the same device")
    if batch_one and previous.shape[0] != 1:
        raise ValueError(f"expected a single frame (1,3,H,W), got batch {previous.shape[0]}")


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
        With b=backward_flow, f=forward_flow and warp(v,b)(p)=v(p+b(p)):
        r_fb = b + warp(f,b);
        m_fb = 1[ sum_xy(r_fb^2) <= alpha*(sum_xy(b^2) + sum_xy(warp(f,b)^2)) + beta ]
        (sum_xy is over the two flow components of each pixel);
        m_photo = 1[ |Y(current) - Y(warp(previous,b))| < photometric_threshold ],
        Y = mean of RGB; m = m_fb * m_photo * in_bounds(b). Bilinear flow sampling is
        used for warp(f,b); out-of-bounds samples are zero-padded and then rejected by
        in_bounds. Computed under no_grad.

    Raises:
        ValueError: Inconsistent shapes/devices, wrong channel counts, negative alpha or
            beta, or photometric_threshold outside [0,1].
    """
    _check_rgb_pair(previous, current)
    for name, flow in (("forward_flow", forward_flow), ("backward_flow", backward_flow)):
        if flow.dim() != 4 or flow.shape[1] != 2:
            raise ValueError(f"{name} must be (B,2,H,W), got shape {tuple(flow.shape)}")
        if flow.shape[0] != current.shape[0] or flow.shape[2:] != current.shape[2:]:
            raise ValueError(f"{name} {tuple(flow.shape)} does not match frames")
        if flow.device != current.device:
            raise ValueError(f"{name} must be on the same device as the frames")
    if not (alpha >= 0.0 and beta >= 0.0):
        raise ValueError(f"alpha and beta must be nonnegative, got {alpha}, {beta}")
    if not (0.0 <= photometric_threshold <= 1.0):
        raise ValueError(f"photometric_threshold must be in [0,1], got {photometric_threshold}")
    with torch.no_grad():
        fwd = forward_flow.detach()
        bwd = backward_flow.detach()
        warped_forward, in_bounds = warp(fwd, bwd)
        residual = bwd + warped_forward
        lhs = (residual * residual).sum(dim=1, keepdim=True)
        energy = (bwd * bwd).sum(dim=1, keepdim=True)
        energy = energy + (warped_forward * warped_forward).sum(dim=1, keepdim=True)
        m_fb = (lhs <= alpha * energy + beta).to(current.dtype)
        warped_prev, _ = warp(previous.detach(), bwd)
        diff = current.detach().mean(dim=1, keepdim=True) - warped_prev.mean(dim=1, keepdim=True)
        m_photo = (diff.abs() < photometric_threshold).to(current.dtype)
        mask = m_fb * m_photo * in_bounds.to(current.dtype)
    return mask.detach()


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
        distance = 0.5*sum|hist(Y(current)) - hist(Y(previous))| with Y the per-pixel mean
        RGB and each histogram normalised to sum one; bin = floor(Y*bins), so values equal
        to 1.0 fall into the last bin (values are clamped into [0,1] first). Returns a
        Python bool; the comparison is strict (distance > threshold).

    Raises:
        ValueError: bins is not an integer >=2, threshold outside [0,1], or frames are
            not same-shape single (1,3,H,W) tensors on one device.
    """
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
        raise ValueError(f"bins must be an integer >= 2, got {bins!r}")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0,1], got {threshold}")
    _check_rgb_pair(previous, current, batch_one=True)

    def histogram(frame: Tensor) -> Tensor:
        luma = frame.detach().to(torch.float64).mean(dim=1).reshape(-1).clamp(0.0, 1.0)
        index = torch.clamp((luma * bins).floor().to(torch.int64), 0, bins - 1)
        counts = torch.bincount(index, minlength=bins).to(torch.float64)
        return counts / counts.sum()

    distance = 0.5 * float((histogram(current) - histogram(previous)).abs().sum())
    return bool(distance > threshold)
