"""Detached temporal reliability and scene-cut tests; pure tensors.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from torch import Tensor


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
        NotImplementedError: Scaffold only; implement this contract before use.
    """
    raise NotImplementedError("TODO: Compute a current-coordinate reliable-pixel mask")


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
        NotImplementedError: Scaffold only; implement this contract before use.
    """
    raise NotImplementedError("TODO: Detect a scene cut before reusing temporal state")
