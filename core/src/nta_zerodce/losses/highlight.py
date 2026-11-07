"""Pure tensor losses/targets; no IO, OpenCV, optimizer or registry side effects.

See docs/ARCHITECTURE.md for the normative contract.
"""

from __future__ import annotations

import torch
from torch import Tensor

from nta_zerodce.losses._ops import luminance, masked_mean


def highlight_loss(image: Tensor, output: Tensor, threshold: float = 0.8) -> Tensor:
    """Protect already bright input pixels.

    Args:
        image: RGB float32 tensor (B,3,H,W) in [0,1]; no implicit device transfer.
        output: RGB float32 tensor (B,3,H,W) in [0,1], on the input device.
        threshold: Input luminance threshold in [0,1], default 0.8.

    Returns:
        Scalar float32 nonnegative differentiable Tensor on the input device.

    Method:
        Extension 4; masked_mean_1(relu(Y(o)-Y(x))^2, m_hi) with the detached mask
        m_hi=1[Y(x)>threshold]. An empty mask yields exactly zero that is still connected to the
        output graph (numerator sum(m*v) with m=0), so backward gives finite zero gradients.
        See docs/ARCHITECTURE.md; extensions are project proposals.

    Raises:
        ValueError: If image and output shapes differ.
    """
    if image.shape != output.shape:
        raise ValueError(f"image {tuple(image.shape)} and output {tuple(output.shape)} differ")
    y_in = luminance(image).detach()
    y_out = luminance(output)
    mask = (y_in > threshold).to(y_out.dtype)
    increase = torch.relu(y_out - y_in)
    return masked_mean(increase * increase, mask, channels=1)
