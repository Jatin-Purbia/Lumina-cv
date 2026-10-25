"""Still-image enhancement with optional downscaled curve estimation.

See docs/REAL_TIME.md: only sigma and the curve maps are estimated at reduced size. They are
upsampled bilinearly (``align_corners=False``) and the eight curves are applied to the original
full-resolution RGB, so the final image is never a resized result.
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor

from nta_zerodce.models.nta_zerodce import ModelOutput, NTAZeroDCE

_MIN_SIDE = 16


def _validate(image: Tensor, curve_scale: float) -> None:
    """Check the image tensor contract and the curve scale range."""
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"image must have shape (B,3,H,W); got {tuple(image.shape)}")
    if not image.is_floating_point():
        raise ValueError(f"image must be floating point; got {image.dtype}")
    if image.shape[2] < _MIN_SIDE or image.shape[3] < _MIN_SIDE:
        raise ValueError(f"image sides must be >= {_MIN_SIDE}; got {tuple(image.shape[2:])}")
    if isinstance(curve_scale, bool) or not 0.0 < float(curve_scale) <= 1.0:
        raise ValueError(f"curve_scale must be in (0, 1]; got {curve_scale!r}")


def estimate_full_res_maps(
    model: NTAZeroDCE, image: Tensor, curve_scale: float = 1.0
) -> tuple[Tensor, Tensor]:
    """Estimate (A, sigma) at ``curve_scale`` and return them at the image's resolution.

    Args:
        model: Eval-mode model on the image device.
        image: RGB float32 (B,3,H,W) in [0,1] with H,W >= 16.
        curve_scale: In (0,1]; the estimator sees a bilinear (antialiased) resize whose sides
            are ``max(16, round(side * curve_scale))``.

    Returns:
        ``(A, sigma)`` of shapes (B,24,H,W) and (B,1,H,W) at full resolution, no gradients.
    """
    _validate(image, curve_scale)
    height, width = int(image.shape[2]), int(image.shape[3])
    with torch.no_grad():
        if curve_scale == 1.0:
            return model.estimate_maps(image)
        small_size = (
            max(_MIN_SIDE, round(height * curve_scale)),
            max(_MIN_SIDE, round(width * curve_scale)),
        )
        small = functional.interpolate(
            image, size=small_size, mode="bilinear", align_corners=False, antialias=True
        ).clamp(0.0, 1.0)
        a_small, sigma_small = model.estimate_maps(small)
        a_full = functional.interpolate(
            a_small, size=(height, width), mode="bilinear", align_corners=False
        ).clamp(-1.0, 1.0)
        sigma_full = functional.interpolate(
            sigma_small, size=(height, width), mode="bilinear", align_corners=False
        ).clamp(min=0.0)
        return a_full, sigma_full


def enhance_image(model: NTAZeroDCE, image: Tensor, curve_scale: float = 1.0) -> ModelOutput:
    """Enhance RGB using inference-only model execution.

    Args:
        model: Eval-mode module on the image device.
        image: RGB float32 tensor (B,3,H,W) in [0,1]; no implicit device transfer.
        curve_scale: In (0,1]; estimate sigma/A at reduced size and resize the maps.

    Returns:
        Full-resolution ModelOutput; no gradients, RGB in [0,1].

    Raises:
        ValueError: Bad image shape/dtype/size or curve_scale outside (0,1].
    """
    _validate(image, curve_scale)
    a_full, sigma_full = estimate_full_res_maps(model, image, curve_scale)
    with torch.no_grad():
        return model.apply_maps(image, a_full, sigma_full)
