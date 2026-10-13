"""Pure tensor losses/targets; no IO, OpenCV, optimizer or registry side effects.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from torch import Tensor


def edge_loss(
    image: Tensor,
    output: Tensor,
    sigma: Tensor,
    gain: Tensor,
    smooth_sigma: float = 0.8,
    kernel_size: int = 5,
    noise_multiplier: float = 2.0,
    eps: float = 1e-6,
) -> Tensor:
    """Match noise-reliable gain-scaled input edges.

    Args:
        image: RGB float32 tensor (B,3,H,W) in [0,1]; no implicit device transfer.
        output: RGB float32 tensor (B,3,H,W) in [0,1], on the input device.
        sigma: Nonnegative float32 tensor (B,1,H,W), standard deviation in RGB units.
        gain: Detached (B,1,H,W) in [1,8].
        smooth_sigma: Positive Gaussian standard deviation in pixels.
        kernel_size: Odd Gaussian width, default 5.
        noise_multiplier: Positive edge reliability threshold, default 2.
        eps: Positive gradient/division guard.

    Returns:
        Scalar float32 nonnegative differentiable Tensor on the input device.

    Method:
        Extension 2; normalized Sobel magnitude of output against gain*smoothed input gradient. See docs/METHOD.md; baseline attribution [R1,R2], extensions are project proposals.

    Raises:
        NotImplementedError: Scaffold only; implement this contract before use.
    """
    raise NotImplementedError("TODO: Match noise-reliable gain-scaled input edges")
