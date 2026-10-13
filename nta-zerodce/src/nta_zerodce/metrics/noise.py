"""Flat-region noise proxy, not a perceptual quality guarantee.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from torch import Tensor


def noise_estimate(image: Tensor, flat_mask: Tensor | None = None) -> Tensor:
    """Estimate flat-region Gaussian noise standard deviation.

    Args:
        image: RGB float32 tensor (B,3,H,W) in [0,1]; no implicit device transfer.
        flat_mask: Optional (B,1,H,W) binary valid region; no reliable pixels => NaN.

    Returns:
        Per-image (B,) normalized RGB-unit sigma estimates.

    Method:
        EXPERIMENTS: Immerkaer kernel and calibration [R8]; texture causes bias.

    Raises:
        NotImplementedError: Scaffold only; implement this contract before use.
    """
    raise NotImplementedError("TODO: Estimate flat-region Gaussian noise standard deviation")
