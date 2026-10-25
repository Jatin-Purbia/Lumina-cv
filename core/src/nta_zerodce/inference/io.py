"""Image/video decoding and saving boundaries; no optical-flow computation.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch import Tensor

from nta_zerodce.data.transforms import normalize_rgb

_IMAGE_WRITE_FORMATS = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG"}
_JPEG_QUALITY = 95
_VIDEO_FOURCC = {".mp4": "mp4v", ".avi": "MJPG"}
_UNSUPPORTED_MODES = ("I;16", "I", "F")
# Bound at import so the helper stays available when tests replace ``cv2.VideoWriter``.
_fourcc = cv2.VideoWriter.fourcc


def _to_rgb(image: Image.Image) -> Image.Image:
    """Explicitly convert any supported PIL mode to 8-bit RGB.

    Grayscale ("1", "L"), CMYK and palette images are converted by PIL. Images with an alpha
    channel (RGBA, LA, PA, or palette with transparency) are composited on a black background.
    16-bit integer and float modes are rejected because their scaling is ambiguous.
    """
    if image.mode.startswith(_UNSUPPORTED_MODES):
        raise ValueError(f"unsupported image mode {image.mode!r}; convert to 8-bit first")
    has_alpha = (
        image.mode in ("RGBA", "LA", "PA", "RGBa", "La")
        or "transparency" in image.info
        or "A" in image.getbands()
    )
    if has_alpha:
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (0, 0, 0))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return image.convert("RGB")


def read_image(path: Path) -> Tensor:
    """Decode one image into RGB.

    Args:
        path: Existing PNG/JPEG path; reject grayscale/alpha ambiguity or explicitly convert to RGB.

    Returns:
        CPU RGB float32 (1,3,H,W) in [0,1].

    Method:
        INTERFACES IO. EXIF orientation is applied (``ImageOps.exif_transpose``) so pixels
        are upright. Grayscale, palette and CMYK inputs are converted to RGB explicitly;
        images with alpha are composited onto black. 16-bit/float modes are rejected. Values are
        ``byte / 255`` with no gamma, white-balance or resizing.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If ``path`` is a directory, not a decodable image, truncated, or in an
            unsupported mode.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"image not found: {path}")
    if not path.is_file():
        raise ValueError(f"not a file: {path}")
    try:
        with Image.open(path) as opened:
            opened.load()
            upright = ImageOps.exif_transpose(opened)
            rgb = _to_rgb(upright)
            array = np.array(rgb, dtype=np.uint8)
    except (UnidentifiedImageError, OSError, SyntaxError) as error:
        raise ValueError(f"cannot decode image {path}: {error}") from error
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return normalize_rgb(tensor).unsqueeze(0).contiguous()


def _quantize(image: Tensor) -> np.ndarray:
    """Validate a (1,3,H,W) image and quantize once to a uint8 HxWx3 RGB array.

    Policy: non-finite values are rejected; finite values outside [0,1] are clamped to
    [0,1] (small excursions from the curve arithmetic are expected); then ``round(255*x)``.
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"image must be a torch.Tensor, got {type(image).__name__}")
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError(f"image must have shape (1,3,H,W), got {tuple(image.shape)}")
    if not image.is_floating_point():
        raise TypeError(f"image must be floating point, got {image.dtype}")
    if image.shape[2] < 1 or image.shape[3] < 1:
        raise ValueError(f"image must be non-empty, got {tuple(image.shape)}")
    cpu = image.detach().to("cpu", torch.float32)
    if not bool(torch.isfinite(cpu).all()):
        raise ValueError("image contains NaN or infinite values")
    quantized = torch.round(255.0 * cpu.clamp(0.0, 1.0)).to(torch.uint8)
    return np.ascontiguousarray(quantized[0].permute(1, 2, 0).numpy())


def write_image(image: Tensor, path: Path) -> None:
    """Encode one normalized RGB result.

    Args:
        image: RGB float32 (1,3,H,W) in [0,1]; may be on GPU.
        path: PNG/JPEG output path; create parent only in CLI runner.

    Returns:
        None; detach, move to CPU and quantize once at save boundary.

    Method:
        INTERFACES IO; round(255*x), PNG preferred for evaluation (lossless). The format is
        chosen by extension (.png, .jpg, .jpeg; JPEG quality 95). Finite values outside [0,1]
        are clamped before quantization; NaN/inf is rejected. Parent directories are not
        created.

    Raises:
        TypeError: If ``image`` is not a floating-point tensor.
        ValueError: On wrong shape, non-finite values or an unsupported extension.
        FileNotFoundError: If the parent directory does not exist.
    """
    path = Path(path)
    format_name = _IMAGE_WRITE_FORMATS.get(path.suffix.lower())
    if format_name is None:
        raise ValueError(f"unsupported image extension {path.suffix!r}; use .png, .jpg or .jpeg")
    array = _quantize(image)
    if not path.parent.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {path.parent}")
    pil_image = Image.fromarray(array, mode="RGB")
    if format_name == "JPEG":
        pil_image.save(path, format=format_name, quality=_JPEG_QUALITY)
    else:
        pil_image.save(path, format=format_name)


def _video_frames(path: Path) -> Iterator[Tensor]:
    """Generator body of ``read_video``; owns the capture and always releases it."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open video: {path}")
        count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            count += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
            yield normalize_rgb(tensor).unsqueeze(0).contiguous()
        if count == 0:
            raise ValueError(f"video contains no decodable frames: {path}")
    finally:
        capture.release()


def read_video(path: Path) -> Iterator[Tensor]:
    """Stream RGB frames in presentation order.

    Args:
        path: Existing video path; no frame resampling unless requested.

    Returns:
        Iterator of CPU RGB (1,3,H,W) in [0,1]; close decoder on exhaustion/error.

    Method:
        DATA video protocol; preserve original timestamps in downstream metadata. OpenCV
        decodes BGR; frames are converted to RGB here. The capture is opened when the
        iterator is first advanced and is released in a ``finally`` block on exhaustion, on
        error, and when the generator is closed early (``close()`` or garbage collection after
        at least one frame was requested). Audio is ignored.

    Raises:
        FileNotFoundError: Immediately, if ``path`` does not exist.
        ValueError: On first advance, if the file cannot be opened or yields no frames.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"video not found: {path}")
    if not path.is_file():
        raise ValueError(f"not a file: {path}")
    return _video_frames(path)


def write_video(frames: Iterator[Tensor], path: Path, fps: float) -> None:
    """Encode sequential enhanced RGB frames.

    Args:
        frames: RGB (1,3,H,W) in [0,1], constant dimensions.
        path: Output video path.
        fps: Positive source frame rate.

    Returns:
        None; no audio copying promised at M0; release encoder on error.

    Method:
        INTERFACES IO; timing and video-metric quantization caveats. Each frame is quantized
        exactly as in ``write_image`` and written through ``cv2.VideoWriter`` (fourcc mp4v for
        .mp4, MJPG for .avi). The writer is created from the first frame's size and released
        in a ``finally`` block; a partially written output file is removed on error.

    Raises:
        ValueError: If ``fps`` is not a positive finite number, the extension is unsupported,
            there are no frames, or a frame's dimensions differ from the first frame.
        FileNotFoundError: If the parent directory does not exist.
        RuntimeError: If OpenCV cannot open the writer (e.g. codec unavailable).
    """
    path = Path(path)
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be positive and finite, got {fps}")
    codec = _VIDEO_FOURCC.get(path.suffix.lower())
    if codec is None:
        raise ValueError(f"unsupported video extension {path.suffix!r}; use .mp4 or .avi")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {path.parent}")

    writer: cv2.VideoWriter | None = None
    size: tuple[int, int] | None = None
    written = 0
    succeeded = False
    try:
        for frame in frames:
            array = _quantize(frame)
            height, width = array.shape[:2]
            if writer is None:
                size = (width, height)
                writer = cv2.VideoWriter(str(path), _fourcc(*codec), fps, size)
                if not writer.isOpened():
                    raise RuntimeError(
                        f"cannot open video writer for {path} (codec {codec}, size {size})"
                    )
            elif size != (width, height):
                raise ValueError(
                    f"frame {written} has size {(width, height)} but expected {size} (WxH)"
                )
            writer.write(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
            written += 1
        if written == 0:
            raise ValueError("no frames to write")
        succeeded = True
    finally:
        if writer is not None:
            writer.release()
            if not succeeded:
                path.unlink(missing_ok=True)
