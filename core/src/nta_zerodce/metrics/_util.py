"""Shared validation helpers for metrics; no implicit device transfer."""

from __future__ import annotations

import torch
from torch import Tensor


def check_rgb(name: str, image: Tensor) -> None:
    """Require a floating point (B,3,H,W) tensor with nonzero size."""
    if image.dim() != 4 or image.shape[1] != 3:
        raise ValueError(f"{name} must be (B,3,H,W) RGB, got shape {tuple(image.shape)}")
    if not image.is_floating_point():
        raise ValueError(f"{name} must be floating point, got {image.dtype}")
    if image.numel() == 0:
        raise ValueError(f"{name} is empty")


def check_pair(output: Tensor, reference: Tensor, names: tuple[str, str]) -> None:
    """Require two RGB tensors with equal shape, device and dtype."""
    check_rgb(names[0], output)
    check_rgb(names[1], reference)
    if output.shape != reference.shape:
        raise ValueError(
            f"{names[0]} {tuple(output.shape)} and {names[1]} {tuple(reference.shape)} "
            "must have equal shape"
        )
    if output.device != reference.device:
        raise ValueError(
            f"{names[0]} ({output.device}) and {names[1]} ({reference.device}) must share a "
            "device; metrics never transfer tensors implicitly"
        )
    if output.dtype != reference.dtype:
        raise ValueError(f"{names[0]} and {names[1]} dtypes differ")


def intensity(image: Tensor) -> Tensor:
    """Compatibility intensity Y = mean of RGB, (B,3,H,W) -> (B,1,H,W)."""
    return image.mean(dim=1, keepdim=True)


def accumulation_dtype(image: Tensor) -> torch.dtype:
    """float64 for accuracy where supported (not MPS), otherwise float32."""
    return torch.float32 if image.device.type == "mps" else torch.float64
