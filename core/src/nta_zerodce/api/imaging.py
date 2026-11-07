"""Upload validation, decoding and PNG encoding for the HTTP API.

Uploads live only in memory. Decoding is restricted to PNG and JPEG (Pillow is never asked to
identify arbitrary formats), the image header is checked against the size limits before any
pixel data is decoded, and decoding failures are mapped to the contract's JSON errors.
"""

from __future__ import annotations

import io
import re

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch import Tensor

from nta_zerodce.api.errors import (
    INVALID_IMAGE,
    TOO_LARGE,
    UNSUPPORTED_TYPE,
    ApiError,
)
from nta_zerodce.data.transforms import normalize_rgb

ACCEPTED_TYPES: tuple[str, ...] = ("image/png", "image/jpeg")
MIN_SIDE = 16

_MEDIA_TYPE_TO_FORMAT = {"image/png": "png", "image/jpeg": "jpeg"}
_FORMAT_NAMES = {"png": "PNG", "jpeg": "JPEG"}
_PILLOW_FORMATS = ["PNG", "JPEG"]
# Real image formats that are recognised only to answer 415 instead of 400.
_UNSUPPORTED_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"GIF87a", "GIF"),
    (b"GIF89a", "GIF"),
    (b"II*\x00", "TIFF"),
    (b"MM\x00*", "TIFF"),
    (b"BM", "BMP"),
)
_UNSUPPORTED_MODES = ("I;16", "I", "F")
_ALPHA_MODES = ("RGBA", "LA", "PA", "RGBa", "La")
_PARAMS = re.compile(r"\s*;.*$", re.DOTALL)


def normalize_media_type(content_type: str | None) -> str:
    """Return the lower-case media type without parameters (``''`` when absent)."""
    if not content_type:
        return ""
    return _PARAMS.sub("", content_type).strip().lower()


def sniff_format(data: bytes) -> str | None:
    """Identify the container by magic bytes.

    Returns:
        ``'png'`` or ``'jpeg'`` for supported images, an upper-case name such as ``'GIF'`` for
        a recognised but unsupported image format, or ``None`` when the bytes look like neither.
    """
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WebP"
    for signature, name in _UNSUPPORTED_SIGNATURES:
        if data.startswith(signature):
            return name
    return None


def check_upload_type(declared_content_type: str | None, data: bytes) -> str:
    """Validate the declared media type against the content's magic bytes.

    Returns:
        ``'png'`` or ``'jpeg'``.

    Raises:
        ApiError: 415 for an undeclared/unsupported media type or a real but unsupported image
            format; 400 when the bytes are not an image or contradict the declared type.
    """
    declared = normalize_media_type(declared_content_type)
    if declared not in _MEDIA_TYPE_TO_FORMAT:
        raise ApiError(415, UNSUPPORTED_TYPE, "Only image/png and image/jpeg uploads are accepted.")
    actual = sniff_format(data)
    if actual is None:
        raise ApiError(400, INVALID_IMAGE, "The upload is not a PNG or JPEG image.")
    if actual not in _FORMAT_NAMES:
        raise ApiError(
            415, UNSUPPORTED_TYPE, f"{actual} images are not supported; upload PNG or JPEG."
        )
    if actual != _MEDIA_TYPE_TO_FORMAT[declared]:
        raise ApiError(
            400,
            INVALID_IMAGE,
            f"The declared type {declared} does not match the {_FORMAT_NAMES[actual]} content.",
        )
    return actual


def _too_large_side(max_side: int) -> ApiError:
    return ApiError(
        413, TOO_LARGE, f"The image is larger than the {max_side} pixel limit on width or height."
    )


def _check_size(width: int, height: int, max_side: int, max_pixels: int | None = None) -> None:
    if width > max_side or height > max_side:
        raise _too_large_side(max_side)
    if max_pixels is not None and width * height > max_pixels:
        raise ApiError(
            413,
            TOO_LARGE,
            f"The image has {width * height} pixels; the limit is {max_pixels} "
            f"(for example {int(max_pixels**0.5)}x{int(max_pixels**0.5)}).",
        )
    if width < MIN_SIDE or height < MIN_SIDE:
        raise ApiError(
            400, INVALID_IMAGE, f"The image is too small; width and height must be >= {MIN_SIDE}."
        )


def _to_rgb(image: Image.Image) -> Image.Image:
    """Convert any supported mode to 8-bit RGB; alpha is composited on black."""
    if image.mode.startswith(_UNSUPPORTED_MODES):
        raise ApiError(
            400, INVALID_IMAGE, "16-bit and floating-point images are not supported; use 8-bit."
        )
    has_alpha = (
        image.mode in _ALPHA_MODES or "transparency" in image.info or "A" in image.getbands()
    )
    if has_alpha:
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (0, 0, 0))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return image.convert("RGB")


def decode_image(
    data: bytes, image_format: str, max_side: int, max_pixels: int | None = None
) -> Tensor:
    """Decode a PNG/JPEG upload into a CPU RGB float32 ``(1,3,H,W)`` tensor in [0,1].

    The header size is checked before pixels are decoded (decompression-bomb guard), integrity
    is verified, EXIF orientation is applied and the result is converted to RGB (alpha over
    black). Values are ``byte / 255`` like ``nta_zerodce.inference.io.read_image``.

    Raises:
        ApiError: 413 for oversized dimensions or pixel count or a decompression bomb, 400 for corrupt,
            truncated, too small or unsupported-mode images.
    """
    expected = _FORMAT_NAMES[image_format]
    try:
        with Image.open(io.BytesIO(data), formats=_PILLOW_FORMATS) as probe:
            if probe.format != expected:
                raise ApiError(400, INVALID_IMAGE, "The image content is inconsistent.")
            width, height = probe.size
            _check_size(width, height, max_side, max_pixels)
            probe.verify()
        with Image.open(io.BytesIO(data), formats=_PILLOW_FORMATS) as opened:
            opened.load()
            upright = ImageOps.exif_transpose(opened)
            rgb = _to_rgb(upright)
            _check_size(rgb.width, rgb.height, max_side, max_pixels)
            array = np.array(rgb, dtype=np.uint8)
    except ApiError:
        raise
    except Image.DecompressionBombError as error:
        raise _too_large_side(max_side) from error
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, EOFError, TypeError) as error:
        raise ApiError(
            400, INVALID_IMAGE, "The image is corrupt, truncated or could not be decoded."
        ) from error
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return normalize_rgb(tensor).unsqueeze(0).contiguous()


def encode_png(image: Tensor) -> bytes:
    """Quantize a ``(1,3,H,W)`` float image once (``round(255*x)``, clamped) and encode PNG.

    Raises:
        ValueError: Wrong shape or non-finite values.
    """
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError(f"image must have shape (1,3,H,W); got {tuple(image.shape)}")
    cpu = image.detach().to("cpu", torch.float32)
    if not bool(torch.isfinite(cpu).all()):
        raise ValueError("image contains NaN or infinite values")
    quantized = torch.round(255.0 * cpu.clamp(0.0, 1.0)).to(torch.uint8)
    array = np.ascontiguousarray(quantized[0].permute(1, 2, 0).numpy())
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue()
