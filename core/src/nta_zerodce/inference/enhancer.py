"""Still-image enhancement with optional downscaled curve estimation.

See docs/ARCHITECTURE.md: only sigma and the curve maps are estimated at reduced size. They are
upsampled bilinearly (``align_corners=False``) and the eight curves are applied to the original
full-resolution RGB, so the final image is never a resized result.
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor

from nta_zerodce.models.nta_zerodce import ModelOutput, NTAZeroDCE

MIN_SIDE = 16
"""Smallest image side the estimator accepts (REAL_TIME: rounded estimator dimensions >= 16)."""


def _validate(model: NTAZeroDCE, image: Tensor, curve_scale: float) -> None:
    """Check the image tensor contract, the model/image placement and the curve scale range."""
    if not isinstance(image, Tensor):
        raise ValueError(f"image must be a torch.Tensor; got {type(image).__name__}")
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"image must have shape (B,3,H,W); got {tuple(image.shape)}")
    if image.shape[0] < 1:
        raise ValueError("image batch is empty")
    if not image.is_floating_point():
        raise ValueError(f"image must be floating point; got {image.dtype}")
    if image.shape[2] < MIN_SIDE or image.shape[3] < MIN_SIDE:
        raise ValueError(f"image sides must be >= {MIN_SIDE}; got {tuple(image.shape[2:])}")
    try:
        scale_ok = (not isinstance(curve_scale, bool)) and 0.0 < float(curve_scale) <= 1.0
    except (TypeError, ValueError):
        scale_ok = False
    if not scale_ok:
        raise ValueError(f"curve_scale must be in (0, 1]; got {curve_scale!r}")
    reference = next(model.parameters(), None)
    if reference is not None:
        if image.device != reference.device:
            raise ValueError(
                f"image is on {image.device} but the model is on {reference.device}; "
                "move the image explicitly (no implicit transfer)"
            )
        if image.dtype != reference.dtype:
            raise ValueError(
                f"image dtype {image.dtype} differs from model dtype {reference.dtype}"
            )


def estimate_full_res_maps(
    model: NTAZeroDCE, image: Tensor, curve_scale: float = 1.0
) -> tuple[Tensor, Tensor]:
    """Estimate (A, sigma) at ``curve_scale`` and return them at the image's resolution.

    Args:
        model: Eval-mode model on the image device.
        image: RGB float (B,3,H,W) in [0,1] with H,W >= 16, same device and dtype as the model.
            Non-contiguous tensors are accepted.
        curve_scale: In (0,1]; the estimator sees a bilinear (antialiased) resize whose sides
            are ``max(16, round(side * curve_scale))``. At 1.0 no resize happens.

    Returns:
        ``(A, sigma)`` of shapes (B,24,H,W) and (B,1,H,W) at full resolution, no gradients.
        A lies in [-1,1] and sigma is nonnegative.

    Raises:
        ValueError: Bad image shape/dtype/size/device or ``curve_scale`` outside (0,1].
    """
    _validate(model, image, curve_scale)
    height, width = int(image.shape[2]), int(image.shape[3])
    with torch.no_grad():
        if float(curve_scale) == 1.0:
            return model.estimate_maps(image)
        small_size = (
            max(MIN_SIDE, round(height * float(curve_scale))),
            max(MIN_SIDE, round(width * float(curve_scale))),
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
        Full-resolution ModelOutput; no gradients, RGB in [0,1]. At ``curve_scale == 1`` this
        equals ``model.forward(image)`` under ``torch.no_grad``.

    Raises:
        ValueError: Bad image shape/dtype/size/device or curve_scale outside (0,1].
    """
    a_full, sigma_full = estimate_full_res_maps(model, image, curve_scale)
    with torch.no_grad():
        return model.apply_maps(image, a_full, sigma_full)
