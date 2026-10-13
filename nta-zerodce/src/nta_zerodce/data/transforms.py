"""Shared geometric augmentation and RGB normalization; DATA preprocessing.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

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
        NotImplementedError: Scaffold only; implement this contract before use.
    """
    raise NotImplementedError("TODO: Convert decoded RGB bytes to normalized float32")


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
        NotImplementedError: Scaffold only; implement this contract before use.
    """
    raise NotImplementedError("TODO: Crop both frames using one sampled origin")
