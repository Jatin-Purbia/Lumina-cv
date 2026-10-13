"""Reference chromaticity/angular error and colour proxy reporting.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from torch import Tensor


def angular_error(output: Tensor, reference: Tensor, eps: float = 1e-6) -> Tensor:
    """Measure RGB angular error on reliable reference pixels.

    Args:
        output: RGB float32 tensor (B,3,H,W) in [0,1], on the input device.
        reference: RGB (B,3,H,W) in [0,1], aligned.
        eps: Positive norm guard; exclude near-black vectors.

    Returns:
        Per-image (B,) mean degrees, lower better; empty reliable set => NaN.

    Method:
        EXPERIMENTS acos(clamp(dot/(norms),-1,1)); not perceptual Delta-E.

    Raises:
        NotImplementedError: Scaffold only; implement this contract before use.
    """
    raise NotImplementedError("TODO: Measure RGB angular error on reliable reference pixels")
