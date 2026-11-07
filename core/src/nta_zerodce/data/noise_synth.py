"""Heteroscedastic synthetic corruption; DATA generation protocol.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

import torch
from torch import Generator, Tensor


def check_generator(generator: Generator | None, device: torch.device) -> None:
    """Require a generator to live on the same device type as the tensors it will fill.

    Args:
        generator: RNG or None (None means the global RNG, which is always valid).
        device: Device of the tensors to be generated.

    Raises:
        ValueError: If the generator's device type differs from ``device``.
    """
    if generator is not None and generator.device.type != device.type:
        raise ValueError(
            f"generator is on {generator.device.type} but tensors are on {device.type}; "
            "create the generator on the same device"
        )


def _check_coefficient(name: str, value: Tensor, image: Tensor) -> None:
    """Validate one noise coefficient tensor (B,1,1,1): finite, nonnegative, same device."""
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    expected = (image.shape[0], 1, 1, 1)
    if tuple(value.shape) != expected:
        raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}")
    if value.device != image.device:
        raise ValueError(f"{name} is on {value.device} but image is on {image.device}")
    if not torch.isfinite(value).all() or bool((value < 0).any()):
        raise ValueError(f"{name} must be finite and nonnegative")


def sample_noise(
    image: Tensor, a: Tensor, b: Tensor, generator: Generator | None = None
) -> tuple[Tensor, Tensor]:
    """Synthesize Gaussian-approximated Poisson-Gaussian noise.

    Args:
        image: RGB float32 tensor (B,3,H,W) in [0,1]; no implicit device transfer.
        a/b: Nonnegative (B,1,1,1); b is read-noise std, not variance.
        generator: Device-matched RNG or None; do not reset global seed.

    Returns:
        (clipped noisy RGB, sigma_target (B,1,H,W)); targets are pre-clipping.

    Method:
        METHOD extension 1: n_c=sqrt(a*x_c+b^2)*eps_c with independent standard normal eps_c per
        RGB pixel; noisy=clamp(x+n,0,1); sigma_target=sqrt(a*Y(x)+b^2) with Y the mean of
        RGB, computed from the clean image and never clipped. a=b=0 returns the input values
        and zero sigma. The generator is the only source of randomness; the global seed is never
        reset (with ``generator=None`` the global RNG is consumed, not reseeded).

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: On wrong shapes/dtype, values outside [0,1], negative or non-finite a/b,
            or a generator on a different device type than the image.
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"image must be a torch.Tensor, got {type(image).__name__}")
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"image must have shape (B,3,H,W), got {tuple(image.shape)}")
    if image.dtype != torch.float32:
        raise ValueError(f"image must be float32, got {image.dtype}")
    if image.numel() == 0:
        raise ValueError("image must be non-empty")
    if not torch.isfinite(image).all() or bool((image < 0).any()) or bool((image > 1).any()):
        raise ValueError("image values must be finite and within [0,1]")
    _check_coefficient("a", a, image)
    _check_coefficient("b", b, image)
    check_generator(generator, image.device)

    a = a.to(image.dtype)
    b = b.to(image.dtype)
    epsilon = torch.randn(image.shape, generator=generator, device=image.device, dtype=image.dtype)
    std = torch.sqrt(a * image + b * b)
    noisy = torch.clamp(image + std * epsilon, 0.0, 1.0)
    luma = image.mean(dim=1, keepdim=True)
    sigma_target = torch.sqrt(a * luma + b * b)
    return noisy, sigma_target
