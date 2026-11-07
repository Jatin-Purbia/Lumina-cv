"""Shared pure tensor helpers for the loss modules.

Every function here is stateless, device-agnostic and free of IO: it never moves tensors,
reads configuration or touches a registry. Helpers follow the notation block of
docs/ARCHITECTURE.md (``Y``, ``P_k``, ``masked_mean_C``, ``smoothstep``, stride-1 average pooling
with reflection padding). Written from the published mathematics only.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

EPS = 1e-6


def luminance(image: Tensor) -> Tensor:
    """Compatibility intensity Y = (R+G+B)/3 as a (B,1,H,W) tensor.

    Args:
        image: RGB tensor (B,3,H,W).

    Returns:
        Channel mean with the channel axis kept.

    Method:
        METHOD notation, Y(x) = (x_R+x_G+x_B)/3; not linear luminance.
    """
    if image.dim() != 4 or image.shape[1] != 3:
        raise ValueError(f"expected an RGB tensor (B,3,H,W), got {tuple(image.shape)}")
    return image.mean(dim=1, keepdim=True)


def pool_mean(value: Tensor, size: int) -> Tensor:
    """Non-overlapping size*size mean pool, stride size, no padding.

    Args:
        value: Tensor (B,C,H,W) with H,W >= size.
        size: Positive pooling width.

    Returns:
        (B,C,H//size,W//size); incomplete right/bottom regions are dropped, never padded.

    Method:
        METHOD notation P_k.
    """
    if size < 1:
        raise ValueError(f"pool size must be positive, got {size}")
    if value.shape[-2] < size or value.shape[-1] < size:
        raise ValueError(f"spatial size {tuple(value.shape[-2:])} is smaller than pool {size}")
    return F.avg_pool2d(value, kernel_size=size, stride=size)


def avg_pool_reflect(value: Tensor, window: int) -> Tensor:
    """Stride-1 average pooling with reflection padding (same output size).

    Args:
        value: Tensor (B,C,H,W) with H,W > window//2.
        window: Odd positive window width.

    Returns:
        Tensor with the same shape as ``value``.

    Method:
        METHOD avgpool_w: stride 1, reflection padding.
    """
    if window < 1 or window % 2 == 0:
        raise ValueError(f"window must be odd and positive, got {window}")
    pad = window // 2
    if pad == 0:
        return value
    padded = F.pad(value, (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(padded, kernel_size=window, stride=1)


def masked_mean(value: Tensor, mask: Tensor, channels: int, eps: float = EPS) -> Tensor:
    """masked_mean_C(v, m) = sum(m*v) / (C*sum(m) + eps).

    Args:
        value: Non-negative-or-signed tensor (B,C,H,W) (C may equal ``channels``).
        mask: Weights (B,1,H,W) broadcastable against ``value``.
        channels: Number of channels C covered by each mask element.
        eps: Positive denominator guard.

    Returns:
        Scalar. An empty mask gives exactly zero, and the result stays connected to ``value``'s
        graph (the numerator is ``sum(m*v)``), so backward yields finite zero gradients.

    Method:
        METHOD notation block.
    """
    return (mask * value).sum() / (channels * mask.sum() + eps)


def smoothstep(low: float, high: float, value: Tensor) -> Tensor:
    """Cubic smoothstep: t = clamp((v-low)/(high-low), 0, 1); t^2 * (3 - 2t).

    Args:
        low: Lower edge.
        high: Upper edge, strictly greater than ``low``.
        value: Input tensor.

    Returns:
        Tensor in [0,1] with the shape of ``value``.

    Method:
        METHOD extension 4.
    """
    if not high > low:
        raise ValueError(f"smoothstep needs low < high, got {low}, {high}")
    t = ((value - low) / (high - low)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def gaussian_kernel_1d(sigma: float, kernel_size: int, like: Tensor) -> Tensor:
    """Normalized 1-D Gaussian taps created on ``like``'s device and dtype.

    Args:
        sigma: Positive standard deviation in pixels.
        kernel_size: Odd positive width.
        like: Tensor whose device/dtype the taps follow (no transfer).

    Returns:
        Tensor (kernel_size,) summing to one.
    """
    if sigma <= 0 or not math.isfinite(sigma):
        raise ValueError(f"Gaussian sigma must be positive and finite, got {sigma}")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(f"kernel size must be odd and positive, got {kernel_size}")
    half = kernel_size // 2
    coords = torch.arange(-half, half + 1, device=like.device, dtype=like.dtype)
    taps = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    return taps / taps.sum()


def gaussian_blur_reflect(value: Tensor, sigma: float, kernel_size: int) -> Tensor:
    """Separable Gaussian blur with reflection padding (same output size).

    Args:
        value: Tensor (B,C,H,W) with H,W > kernel_size//2.
        sigma: Positive standard deviation in pixels.
        kernel_size: Odd width.

    Returns:
        Blurred tensor, same shape.

    Method:
        METHOD extension 2, GaussianBlur(sigma=.8, kernel=5, reflect padding).
    """
    taps = gaussian_kernel_1d(sigma, kernel_size, value)
    channels = value.shape[1]
    pad = kernel_size // 2
    padded = F.pad(value, (pad, pad, pad, pad), mode="reflect")
    horizontal = taps.view(1, 1, 1, kernel_size).expand(channels, 1, 1, kernel_size)
    vertical = taps.view(1, 1, kernel_size, 1).expand(channels, 1, kernel_size, 1)
    out = F.conv2d(padded, horizontal, groups=channels)
    return F.conv2d(out, vertical, groups=channels)


def sobel_magnitude(value: Tensor, eps: float = EPS) -> Tensor:
    """Eps-stabilized normalized Sobel magnitude D(v) = sqrt(Sx^2+Sy^2+eps^2) - eps.

    Args:
        value: Single-channel tensor (B,1,H,W) with H,W >= 2.
        eps: Positive stabilizer; keeps the derivative finite on flat regions.

    Returns:
        (B,1,H,W) magnitude; Sobel kernels are divided by 8 so a unit-slope ramp gives 1.
        Reflection padding is used so borders do not create artificial edges.

    Method:
        METHOD extension 2.
    """
    kx = value.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
    ky = kx.t()
    kernels = torch.stack([kx, ky]).unsqueeze(1)  # (2,1,3,3)
    padded = F.pad(value, (1, 1, 1, 1), mode="reflect")
    grads = F.conv2d(padded, kernels)
    gx = grads[:, 0:1]
    gy = grads[:, 1:2]
    return torch.sqrt(gx * gx + gy * gy + eps * eps) - eps
