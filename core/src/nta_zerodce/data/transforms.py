"""Shared geometric augmentation and RGB normalization; DATA preprocessing.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

import torch
from torch import Generator, Tensor


def normalize_rgb(image: Tensor) -> Tensor:
    """Convert decoded RGB bytes to normalized float32.

    Args:
        image: RGB uint8 (3,H,W) in [0,255]; BGR conversion must happen at decoding.

    Returns:
        RGB float32 (3,H,W) in [0,1], same device.

    Method:
        INTERFACES IO range contract; no gamma or white-balance correction.

    Raises:
        ValueError: Invalid inputs or incompatible configuration.
    """
    if image.dtype == torch.uint8:
        return image.to(torch.float32) / 255
    if (
        image.dtype != torch.float32
        or not torch.isfinite(image).all()
        or image.min() < 0
        or image.max() > 1
    ):
        raise ValueError("Expected uint8 RGB or normalized float32")
    return image


def paired_crop(
    first: Tensor, second: Tensor, crop_size: int = 256, generator: Generator | None = None
) -> tuple[Tensor, Tensor]:
    """Crop both frames using one sampled origin.

    Args:
        first/second: Matching RGB (3,H,W) in [0,1].
        crop_size: Positive multiple of 16, <=min(H,W).
        generator: Device-matched RNG.

    Returns:
        Two aligned RGB (3,crop_size,crop_size) tensors.

    Method:
        DATA shared geometry; crop before computing flow, or transform cached flow consistently.

    Raises:
        ValueError: Invalid inputs or incompatible configuration.
    """
    if first.shape != second.shape or crop_size < 1 or min(first.shape[-2:]) < crop_size:
        raise ValueError("Matching images must be at least crop_size in both dimensions")
    h, w = first.shape[-2:]
    top = int(torch.randint(h - crop_size + 1, (), generator=generator))
    left = int(torch.randint(w - crop_size + 1, (), generator=generator))
    return first[..., top : top + crop_size, left : left + crop_size], second[
        ..., top : top + crop_size, left : left + crop_size
    ]
