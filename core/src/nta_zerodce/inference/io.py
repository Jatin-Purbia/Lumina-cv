"""Image/video decoding and saving boundaries; no optical-flow computation.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor


def read_image(path: Path) -> Tensor:
    """Decode one image into RGB.

    Args:
        path: Existing PNG/JPEG path; reject grayscale/alpha ambiguity or explicitly convert to RGB.

    Returns:
        CPU RGB float32 (1,3,H,W) in [0,1].

    Method:
        INTERFACES IO; ignore EXIF orientation only if documented; planned policy applies orientation.

    Raises:
        ValueError: Invalid inputs or incompatible configuration.
    """
    with Image.open(path) as decoded:
        oriented = ImageOps.exif_transpose(decoded)
        assert oriented is not None
        array = np.asarray(oriented.convert("RGB")).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div(255).unsqueeze(0)


def write_image(image: Tensor, path: Path) -> None:
    """Encode one normalized RGB result.

    Args:
        image: RGB float32 (1,3,H,W) in [0,1]; may be on GPU.
        path: PNG/JPEG output path; create parent only in CLI runner.

    Returns:
        None; detach, move to CPU and quantize once at save boundary.

    Method:
        INTERFACES IO; round(255*x), PNG preferred for evaluation.

    Raises:
        ValueError: Invalid inputs or incompatible configuration.
    """
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError("write_image accepts one RGB image")
        image = image[0]
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Expected RGB CHW")
    path.parent.mkdir(parents=True, exist_ok=True)
    array = image.detach().cpu().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()
    Image.fromarray(array).save(path)  # type: ignore[no-untyped-call]


def read_video(path: Path) -> Iterator[Tensor]:
    """Stream RGB frames in presentation order.

    Args:
        path: Existing video path; no frame resampling unless requested.

    Returns:
        Iterator of CPU RGB (1,3,H,W) in [0,1]; close decoder on exhaustion/error.

    Method:
        DATA video protocol; preserve original timestamps in downstream metadata.

    Raises:
        ValueError: Invalid inputs or incompatible configuration.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield (
                torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                .permute(2, 0, 1)
                .float()
                .div(255)
                .unsqueeze(0)
            )
    finally:
        capture.release()


def write_video(frames: Iterator[Tensor], path: Path, fps: float) -> None:
    """Encode sequential enhanced RGB frames.

    Args:
        frames: RGB (1,3,H,W) in [0,1], constant dimensions.
        path: Output video path.
        fps: Positive source frame rate.

    Returns:
        None; audio is omitted; release encoder on error.

    Method:
        INTERFACES IO; timing and video-metric quantization caveats.

    Raises:
        ValueError: Invalid inputs or incompatible configuration.
    """
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    size = None
    try:
        for frame in frames:
            if frame.ndim != 4 or frame.shape[:2] != (1, 3):
                raise ValueError("Expected video frames (1,3,H,W)")
            array = (
                frame[0].detach().cpu().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()
            )
            current_size = (array.shape[1], array.shape[0])
            if writer is None:
                size = current_size
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, size)
                if not writer.isOpened():
                    raise ValueError(f"Cannot create video: {path}")
            if current_size != size:
                raise ValueError("Video frame size changed")
            writer.write(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
        if writer is None:
            raise ValueError("Cannot encode an empty video")
    finally:
        if writer is not None:
            writer.release()
