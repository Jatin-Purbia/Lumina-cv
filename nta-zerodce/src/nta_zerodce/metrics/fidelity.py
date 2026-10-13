"""Paired full-reference metrics for LOL; EXPERIMENTS definitions.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from torch import Tensor


def psnr(output: Tensor, reference: Tensor) -> Tensor:
    """Measure peak signal-to-noise ratio.

    Args:
        output: RGB float32 tensor (B,3,H,W) in [0,1], on the input device.
        reference: Registered RGB (B,3,H,W) in [0,1], same device.

    Returns:
        Per-image (B,) dB values; +infinity on exact equality.

    Method:
        EXPERIMENTS: 10*log10(1/MSE); higher is better.

    Raises:
        NotImplementedError: Scaffold only; implement this contract before use.
    """
    raise NotImplementedError("TODO: Measure peak signal-to-noise ratio")


def ssim(output: Tensor, reference: Tensor) -> Tensor:
    """Measure structural similarity.

    Args:
        output: RGB float32 tensor (B,3,H,W) in [0,1], on the input device.
        reference: Registered RGB (B,3,H,W) in [0,1].

    Returns:
        Per-image (B,) SSIM, usually [-1,1]; fixed RGB averaging policy.

    Method:
        EXPERIMENTS: Gaussian 11x11 sigma=1.5, K1=.01 K2=.03, data_range=1; [R10].

    Raises:
        NotImplementedError: Scaffold only; implement this contract before use.
    """
    raise NotImplementedError("TODO: Measure structural similarity")
