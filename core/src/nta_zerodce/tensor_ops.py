"""Shared tensor reductions, preserving input device and student gradients."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from nta_zerodce.training.config import ModelConfig


def check_image(image: Tensor) -> None:
    """Validate the public normalized RGB inference contract."""
    if image.ndim != 4 or image.shape[1] != 3 or min(image.shape[-2:]) < 16:
        raise ValueError("Expected RGB BCHW with H,W >= 16")
    if image.dtype != torch.float32 or not torch.isfinite(image).all():
        raise ValueError("Expected finite float32 RGB")
    if image.min() < 0 or image.max() > 1:
        raise ValueError("RGB must lie in [0,1]")


def local_mean(value: Tensor, window: int) -> Tensor:
    """Stride-one pooling with reflection padding."""
    return F.avg_pool2d(F.pad(value, (window // 2,) * 4, mode="reflect"), window, 1)


def local_gain(image: Tensor, raw: Tensor, config: ModelConfig) -> Tensor:
    """Detached bounded brightness ratio used by edge and denoise branches."""
    return (
        (
            local_mean(raw.mean(1, keepdim=True), config.gain_window)
            / (local_mean(image.mean(1, keepdim=True), config.gain_window) + config.eps)
        )
        .clamp(config.gain_min, config.gain_max)
        .detach()
    )


def masked_mean(value: Tensor, mask: Tensor, eps: float = 1e-6) -> Tensor:
    """Covered-pixel reduction; empty masks give graph-connected zero."""
    mask = mask.detach()
    return (value * mask).sum() / (value.shape[1] * mask.sum() + eps)


def smoothstep(value: Tensor, low: float, high: float) -> Tensor:
    """Cubic transition between two ordered thresholds."""
    t = ((value - low) / (high - low)).clamp(0, 1)
    return cast(Tensor, t.square() * (3 - 2 * t))
