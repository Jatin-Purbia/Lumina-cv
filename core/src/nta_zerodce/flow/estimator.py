"""Only owner of OpenCV optical flow; CPU boundary and displacement rescaling.

See docs/ARCHITECTURE.md for the normative contract.

OpenCV compatibility: the DIS (Dense Inverse Search) optical flow object is created by
``cv2.DISOpticalFlow.create(preset)`` (class API, OpenCV 4.5+ and 5.x) or, on older 4.x
builds, by the factory function ``cv2.DISOpticalFlow_create(preset)``. The preset
constants are ``cv2.DISOPTICAL_FLOW_PRESET_{ULTRAFAST,FAST,MEDIUM}`` (4.x and 5.x) or the
class-style ``cv2.DISOpticalFlow_PRESET_*`` aliases. The adapter probes these names with
``hasattr``/``getattr`` and raises a clear ``RuntimeError`` when none exists. Observed
OpenCV 5.0.0 behaviour: ``calc(I0, I1, None)`` returns an (H,W,2) float32 field with
I0(p) ~ I1(p + flow(p)), and requires C-contiguous uint8 inputs.
"""

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np
import torch
from torch import Tensor

from nta_zerodce.training.config import FlowConfig


class _DisLike(Protocol):
    """The part of cv2.DISOpticalFlow used here."""

    def calc(self, i0: np.ndarray, i1: np.ndarray, flow: None) -> np.ndarray:
        """Return the (H,W,2) float32 flow from i0 to i1."""
        ...


#: Smallest (downscaled) side DIS is asked to process. OpenCV 5.0.0 itself fails below 12.
MIN_FLOW_SIDE = 16

_PRESETS = ("ultrafast", "fast", "medium")


def _preset_constant(preset: str) -> int:
    """Look up the OpenCV DIS preset constant across 4.x and 5.x spellings."""
    suffix = preset.upper()
    for name in (f"DISOPTICAL_FLOW_PRESET_{suffix}", f"DISOpticalFlow_PRESET_{suffix}"):
        if hasattr(cv2, name):
            return int(getattr(cv2, name))
    # Documented numeric values of the DIS presets (ULTRAFAST=0, FAST=1, MEDIUM=2).
    return {"ultrafast": 0, "fast": 1, "medium": 2}[preset]


def _create_dis(preset: str) -> _DisLike:
    """Create an OpenCV DIS optical-flow object, supporting the 4.x and 5.x APIs.

    Raises:
        RuntimeError: This OpenCV build exposes no DIS optical flow.
    """
    constant = _preset_constant(preset)
    cls = getattr(cv2, "DISOpticalFlow", None)
    if cls is not None and hasattr(cls, "create"):
        return cls.create(constant)  # type: ignore[no-any-return]
    factory = getattr(cv2, "DISOpticalFlow_create", None)
    if factory is not None:
        return factory(constant)  # type: ignore[no-any-return]
    available = sorted(name for name in dir(cv2) if "DIS" in name and "Optical" in name)
    raise RuntimeError(
        f"OpenCV {cv2.__version__} does not provide DIS optical flow "
        f"(looked for cv2.DISOpticalFlow.create and cv2.DISOpticalFlow_create; "
        f"related names: {available}). Install opencv-python(-headless) >= 4.8."
    )


def _dis_flow_gray(first: np.ndarray, second: np.ndarray, preset: str) -> np.ndarray:
    """Run DIS on two uint8 grayscale images; returns (H,W,2) float32, first(p)~second(p+f)."""
    dis = _create_dis(preset)
    flow = dis.calc(np.ascontiguousarray(first), np.ascontiguousarray(second), None)
    return np.asarray(flow, dtype=np.float32)


def _validate_pair(first: Tensor, second: Tensor, config: FlowConfig) -> None:
    """Validate frames and DIS settings with explicit, actionable messages."""
    if config.method != "dis":
        raise ValueError(f"flow method must be 'dis', got {config.method!r}")
    if config.preset not in _PRESETS:
        raise ValueError(f"flow preset must be one of {_PRESETS}, got {config.preset!r}")
    scale = config.scale
    if not (0.0 < scale <= 1.0):  # also rejects NaN
        raise ValueError(f"flow scale must be in (0, 1], got {scale!r}")
    for name, frame in (("first", first), ("second", second)):
        if frame.dim() != 4 or frame.shape[1] != 3:
            raise ValueError(f"{name} must be (B,3,H,W) RGB, got shape {tuple(frame.shape)}")
        if frame.device.type != "cpu":
            raise ValueError(
                f"{name} must be on the CPU (OpenCV boundary; move it explicitly), "
                f"got device {frame.device}"
            )
        if frame.dtype != torch.float32:
            raise ValueError(f"{name} must be float32, got {frame.dtype}")
        if frame.numel() == 0:
            raise ValueError(f"{name} is empty")
        low, high = float(frame.min()), float(frame.max())
        if not (low >= 0.0 and high <= 1.0):  # also rejects NaN
            raise ValueError(f"{name} must lie in [0,1], got range [{low}, {high}]")
    if first.shape != second.shape:
        raise ValueError(
            f"first {tuple(first.shape)} and second {tuple(second.shape)} must have equal shape"
        )


def _to_gray_uint8(frame: Tensor) -> np.ndarray:
    """(3,H,W) float RGB in [0,1] -> (H,W) uint8 grayscale via OpenCV RGB2GRAY."""
    rgb = (frame.detach().permute(1, 2, 0).numpy() * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY))


def estimate_flow(first: Tensor, second: Tensor, config: FlowConfig) -> Tensor:
    """Estimate source-to-target pixel displacement.

    Args:
        first/second: CPU RGB float32 (B,3,H,W) in [0,1], matching size.
        config: DIS method/preset, flow scale in (0,1].

    Returns:
        CPU float32 flow (B,2,H,W), dx,dy in original pixels; detached.

    Method:
        INTERFACES: first(p) ~ second(p+flow(p)); DIS [R6]. Each batch item is converted
        to 8-bit grayscale (RGB2GRAY on uint8). When config.scale < 1 both frames are
        resized (INTER_AREA) to small_W=round(W*scale), small_H=round(H*scale); DIS runs
        there; the field is resized back to (H,W) (INTER_LINEAR) and the vectors are
        multiplied by W/small_W (dx) and H/small_H (dy), which are the exact, generally
        non-equal, per-axis size ratios actually used.

    Raises:
        ValueError: Wrong device/dtype/shape/range, method not 'dis', unknown preset,
            scale outside (0,1], or a downscaled side below MIN_FLOW_SIDE (16).
        RuntimeError: This OpenCV build has no DIS optical flow.
    """
    _validate_pair(first, second, config)
    batch, _, height, width = first.shape
    scale = float(config.scale)
    if scale < 1.0:
        small_w = max(1, int(round(width * scale)))
        small_h = max(1, int(round(height * scale)))
    else:
        small_w, small_h = width, height
    if min(small_w, small_h) < MIN_FLOW_SIDE:
        raise ValueError(
            f"DIS needs at least {MIN_FLOW_SIDE} px per side after downscaling, got "
            f"{small_w}x{small_h} from {width}x{height} at scale {scale}"
        )
    ratio_x = width / small_w
    ratio_y = height / small_h
    out = np.empty((batch, 2, height, width), dtype=np.float32)
    for index in range(batch):
        gray_a = _to_gray_uint8(first[index])
        gray_b = _to_gray_uint8(second[index])
        if (small_w, small_h) != (width, height):
            gray_a = cv2.resize(gray_a, (small_w, small_h), interpolation=cv2.INTER_AREA)
            gray_b = cv2.resize(gray_b, (small_w, small_h), interpolation=cv2.INTER_AREA)
        flow = _dis_flow_gray(gray_a, gray_b, config.preset)
        if flow.shape != (small_h, small_w, 2):
            raise RuntimeError(f"DIS returned shape {flow.shape}, expected {(small_h, small_w, 2)}")
        if (small_w, small_h) != (width, height):
            flow = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
            flow = flow * np.array([ratio_x, ratio_y], dtype=np.float32)
        out[index] = flow.transpose(2, 0, 1)
    return torch.from_numpy(out)


def estimate_bidirectional(
    first: Tensor, second: Tensor, config: FlowConfig
) -> tuple[Tensor, Tensor]:
    """Estimate forward and backward flow.

    Args:
        first/second: CPU RGB (B,3,H,W) in [0,1].
        config: Validated DIS settings.

    Returns:
        (first->second, second->first), each CPU (B,2,H,W) in original pixel units.

    Method:
        METHOD extension 5; compute both independently, never negate forward flow.

    Raises:
        ValueError: As estimate_flow.
        RuntimeError: This OpenCV build has no DIS optical flow.
    """
    forward = estimate_flow(first, second, config)
    backward = estimate_flow(second, first, config)
    return forward, backward
