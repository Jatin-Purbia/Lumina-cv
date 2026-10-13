"""Pure tensor losses/targets; no IO, OpenCV, optimizer or registry side effects.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from torch import Tensor


def chroma_loss(
    image: Tensor,
    output: Tensor,
    sigma: Tensor,
    window: int = 5,
    eps: float = 1e-6,
    reliability_low: float = 0.05,
    reliability_high: float = 0.2,
    reliability_noise: float = 2.0,
) -> Tensor:
    """Preserve reliable local chromaticity.

    Args:
        image: RGB float32 tensor (B,3,H,W) in [0,1]; no implicit device transfer.
        output: RGB float32 tensor (B,3,H,W) in [0,1], on the input device.
        sigma: Nonnegative float32 tensor (B,1,H,W), standard deviation in RGB units.
        window: Odd local averaging width, default 5.
        eps: Positive division guard.
        reliability_low/high: Luminance ramp endpoints, low<high.
        reliability_noise: Sigma multiplier subtracted from input luminance.

    Returns:
        Scalar float32 nonnegative differentiable Tensor on the input device.

    Method:
        Extension 3; masked L1 distance of RGB/sum to detached local input chromaticity. See docs/METHOD.md; baseline attribution [R1,R2], extensions are project proposals.

    Raises:
        NotImplementedError: Scaffold only; implement this contract before use.
    """
    raise NotImplementedError("TODO: Preserve reliable local chromaticity")
