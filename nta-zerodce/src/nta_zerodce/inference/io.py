"""Image/video decoding and saving boundaries; no optical-flow computation.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

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
        NotImplementedError: Scaffold only; implement this contract before use.
    """
    raise NotImplementedError("TODO: Decode one image into RGB")


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
        NotImplementedError: Scaffold only; implement this contract before use.
    """
    raise NotImplementedError("TODO: Encode one normalized RGB result")


def read_video(path: Path) -> Iterator[Tensor]:
    """Stream RGB frames in presentation order.

    Args:
        path: Existing video path; no frame resampling unless requested.

    Returns:
        Iterator of CPU RGB (1,3,H,W) in [0,1]; close decoder on exhaustion/error.

    Method:
        DATA video protocol; preserve original timestamps in downstream metadata.

    Raises:
        NotImplementedError: Scaffold only; implement this contract before use.
    """
    raise NotImplementedError("TODO: Stream RGB frames in presentation order")


def write_video(frames: Iterator[Tensor], path: Path, fps: float) -> None:
    """Encode sequential enhanced RGB frames.

    Args:
        frames: RGB (1,3,H,W) in [0,1], constant dimensions.
        path: Output video path.
        fps: Positive source frame rate.

    Returns:
        None; no audio copying promised at M0; release encoder on error.

    Method:
        INTERFACES IO; timing and video-metric quantization caveats.

    Raises:
        NotImplementedError: Scaffold only; implement this contract before use.
    """
    raise NotImplementedError("TODO: Encode sequential enhanced RGB frames")
