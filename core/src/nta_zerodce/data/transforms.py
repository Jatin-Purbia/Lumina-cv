"""Shared geometric augmentation and RGB normalization; DATA preprocessing.

See docs/ARCHITECTURE.md for the normative contract.

Public additions beyond INTERFACES (additive helpers shared by the datasets):
``random_crop`` (single-tensor crop with the same origin rule as ``paired_crop``) and
``item_generator`` (per-item CPU generator seeded from ``(seed, epoch, index)``).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Generator, Tensor


def normalize_rgb(image: Tensor) -> Tensor:
    """Convert decoded RGB bytes to normalized float32.

    Args:
        image: RGB uint8 (3,H,W) in [0,255]; BGR conversion must happen at decoding.

    Returns:
        RGB float32 (3,H,W) in [0,1], same device.

    Method:
        INTERFACES IO range contract; no gamma or white-balance correction. The value is
        ``byte / 255`` exactly, so 0 maps to 0.0 and 255 maps to 1.0.

    Raises:
        TypeError: If ``image`` is not a uint8 tensor.
        ValueError: If the shape is not (3,H,W) with positive H and W.
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"image must be a torch.Tensor, got {type(image).__name__}")
    if image.dtype != torch.uint8:
        raise TypeError(f"image must be uint8, got {image.dtype}")
    if image.ndim != 3 or image.shape[0] != 3 or image.shape[1] < 1 or image.shape[2] < 1:
        raise ValueError(f"image must have shape (3,H,W), got {tuple(image.shape)}")
    return image.to(torch.float32) / 255.0


def _validate_crop(shape: tuple[int, int], crop_size: int) -> None:
    """Validate a crop size against an image (H,W) shape."""
    if isinstance(crop_size, bool) or not isinstance(crop_size, int):
        raise TypeError(f"crop_size must be an int, got {type(crop_size).__name__}")
    if crop_size <= 0 or crop_size % 16 != 0:
        raise ValueError(f"crop_size must be a positive multiple of 16, got {crop_size}")
    if crop_size > min(shape):
        raise ValueError(f"crop_size {crop_size} exceeds image size {shape[0]}x{shape[1]} (HxW)")


def _sample_origin(
    shape: tuple[int, int], crop_size: int, generator: Generator | None
) -> tuple[int, int]:
    """Draw the (top, left) origin uniformly over all valid crop positions."""
    height, width = shape
    high = torch.tensor([height - crop_size + 1, width - crop_size + 1])
    # Draw on the generator's device when given, then move the two integers to the host.
    device = generator.device if generator is not None else torch.device("cpu")
    draw = torch.rand(2, generator=generator, device=device).cpu()
    origin = torch.minimum((draw * high).to(torch.int64), high - 1)
    return int(origin[0]), int(origin[1])


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
        One origin (top,left) is drawn uniformly from all valid positions; the result is
        deterministic for a given generator state. The global RNG is never reseeded.

    Raises:
        ValueError: If shapes differ, are not (3,H,W), or ``crop_size`` is invalid / too large.
    """
    if first.ndim != 3 or first.shape[0] != 3:
        raise ValueError(f"first must have shape (3,H,W), got {tuple(first.shape)}")
    if first.shape != second.shape:
        raise ValueError(
            f"frames must have equal shapes, got {tuple(first.shape)} and {tuple(second.shape)}"
        )
    shape = (int(first.shape[1]), int(first.shape[2]))
    _validate_crop(shape, crop_size)
    top, left = _sample_origin(shape, crop_size, generator)
    window = (slice(None), slice(top, top + crop_size), slice(left, left + crop_size))
    return first[window].contiguous(), second[window].contiguous()


def random_crop(image: Tensor, crop_size: int = 256, generator: Generator | None = None) -> Tensor:
    """Crop one RGB (3,H,W) tensor to (3,crop_size,crop_size) at a random origin.

    Args:
        image: RGB (3,H,W).
        crop_size: Positive multiple of 16, <=min(H,W).
        generator: RNG; the same generator state gives the same origin as ``paired_crop``.

    Returns:
        Contiguous cropped tensor.

    Raises:
        ValueError: If the shape or ``crop_size`` is invalid.
    """
    cropped, _ = paired_crop(image, image, crop_size, generator)
    return cropped


def item_generator(seed: int, epoch: int, index: int) -> Generator:
    """Create a CPU generator whose state depends only on ``(seed, epoch, index)``.

    The three integers are mixed through ``numpy.random.SeedSequence`` so that nearby
    tuples give unrelated streams. The global RNG is never touched.

    Args:
        seed: Base seed (non-negative).
        epoch: Epoch counter (non-negative).
        index: Sample index (non-negative).

    Returns:
        Freshly seeded ``torch.Generator`` on CPU.

    Raises:
        ValueError: If any argument is negative.
    """
    if seed < 0 or epoch < 0 or index < 0:
        raise ValueError(f"seed, epoch and index must be non-negative, got {(seed, epoch, index)}")
    state = np.random.SeedSequence([int(seed), int(epoch), int(index)]).generate_state(
        1, dtype=np.uint64
    )
    generator = Generator()
    generator.manual_seed(int(state[0]))
    return generator
