"""Pure tensor losses/targets; no IO, OpenCV, optimizer or registry side effects.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from nta_zerodce.tensor_ops import masked_mean


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
        ValueError: Invalid inputs or incompatible configuration.
    """
    coords = torch.arange(kernel_size, device=image.device, dtype=image.dtype) - kernel_size // 2
    kernel = torch.exp(-coords.square() / (2 * smooth_sigma**2))
    kernel = kernel / kernel.sum()
    kernel = (kernel[:, None] * kernel[None, :])[None, None]
    yin = image.mean(1, keepdim=True)
    blurred = F.conv2d(F.pad(yin, (kernel_size // 2,) * 4, mode="reflect"), kernel)
    sx = image.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    sobel = torch.stack((sx, sx.T))[:, None]

    def magnitude(value: Tensor) -> Tensor:
        gradients = F.conv2d(F.pad(value, (1,) * 4, mode="reflect"), sobel)
        return (gradients.square().sum(1, keepdim=True) + eps**2).sqrt() - eps

    reference = magnitude(blurred).detach()
    mask = (
        ((reference - noise_multiplier * sigma.detach()) / (reference + eps)).clamp(0, 1).detach()
    )
    return masked_mean(
        (magnitude(output.mean(1, keepdim=True)) - gain.detach() * reference).abs(), mask, eps
    )
