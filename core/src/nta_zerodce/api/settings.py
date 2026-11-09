"""Server settings read from environment variables.

Invalid values fail at startup with a message that names the variable. See docs/ARCHITECTURE.md for the
full table.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

DEFAULT_MAX_UPLOAD_BYTES = 20 * 1024 * 1024
DEFAULT_MAX_SIDE = 4096
# Full-resolution inference on CPU peaked at about 1.4 KB per pixel when measured (1024 x 1024
# needs about 2 GB, 1448 x 1448 about 3.4 GB), so the area is bounded separately from the side
# length. 2 Mi pixels fits a 1920 x 1080 frame.
DEFAULT_MAX_PIXELS = 2 * 1024 * 1024
MIN_SIDE = 16
# Allowance on top of ``max_upload_bytes`` for multipart framing and the small form fields.
BODY_OVERHEAD_BYTES = 64 * 1024

_DEVICE = re.compile(r"^(auto|cpu|cuda(:[0-9]+)?)$")
_ORIGIN = re.compile(r"^https?://[^/\s,]+$")
_UINT = re.compile(r"^[0-9]+$")
_FLOAT = re.compile(r"^(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$")


class SettingsError(ValueError):
    """An environment variable (or settings field) has an invalid value."""


@dataclass(frozen=True)
class Settings:
    """Immutable server configuration; build from the environment with ``from_env``.

    Attributes:
        checkpoint: ``LUMINA_CHECKPOINT``; trusted local checkpoint. ``None`` starts the server
            without a model (``/enhance`` answers 503).
        config_path: ``LUMINA_CONFIG``; optional experiment YAML (must match the checkpoint).
        device: ``LUMINA_DEVICE``; ``auto``, ``cpu``, ``cuda`` or ``cuda:N``.
        max_upload_bytes: ``LUMINA_MAX_UPLOAD_BYTES``; largest accepted image file.
        max_side: ``LUMINA_MAX_SIDE``; largest accepted width or height in pixels.
        max_pixels: ``LUMINA_MAX_PIXELS``; largest accepted width x height (bounds memory).
        default_curve_scale: ``LUMINA_DEFAULT_CURVE_SCALE``; ``None`` uses the model config's
            ``model.curve_scale``.
        max_concurrency: ``LUMINA_MAX_CONCURRENCY``; simultaneous upload+inference jobs.
            Additional enhancement requests receive 503 ``server_busy`` without reading a body.
        cors_origins: ``LUMINA_CORS_ORIGINS``; allowed browser origins (empty means none).
        host: ``LUMINA_HOST``; bind address used by ``python -m nta_zerodce.api``.
        port: ``LUMINA_PORT``; bind port used by ``python -m nta_zerodce.api``.
    """

    checkpoint: Path | None = None
    config_path: Path | None = None
    device: str = "auto"
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES
    max_side: int = DEFAULT_MAX_SIDE
    max_pixels: int = DEFAULT_MAX_PIXELS
    default_curve_scale: float | None = None
    max_concurrency: int = 1
    cors_origins: tuple[str, ...] = ()
    host: str = "127.0.0.1"
    port: int = 8000

    def __post_init__(self) -> None:
        if not _DEVICE.match(self.device):
            raise SettingsError(
                f"LUMINA_DEVICE must be 'auto', 'cpu', 'cuda' or 'cuda:N'; got {self.device!r}"
            )
        if self.max_upload_bytes < 1:
            raise SettingsError(
                f"LUMINA_MAX_UPLOAD_BYTES must be a positive integer; got {self.max_upload_bytes}"
            )
        if self.max_side < MIN_SIDE:
            raise SettingsError(
                f"LUMINA_MAX_SIDE must be an integer >= {MIN_SIDE}; got {self.max_side}"
            )
        pixel_limit = Image.MAX_IMAGE_PIXELS
        if pixel_limit is not None and self.max_side * self.max_side > pixel_limit:
            raise SettingsError(
                f"LUMINA_MAX_SIDE={self.max_side} allows more than Pillow's decompression-bomb "
                f"limit of {pixel_limit} pixels; use at most {int(pixel_limit**0.5)}"
            )
        scale = self.default_curve_scale
        if scale is not None and not 0.0 < scale <= 1.0:
            raise SettingsError(f"LUMINA_DEFAULT_CURVE_SCALE must be in (0, 1]; got {scale}")
        if self.max_pixels < MIN_SIDE * MIN_SIDE:
            raise SettingsError(
                f"LUMINA_MAX_PIXELS must be an integer >= {MIN_SIDE * MIN_SIDE}; got {self.max_pixels}"
            )
        if self.max_concurrency < 1:
            raise SettingsError(
                f"LUMINA_MAX_CONCURRENCY must be an integer >= 1; got {self.max_concurrency}"
            )
        for origin in self.cors_origins:
            if origin != "*" and not _ORIGIN.match(origin):
                raise SettingsError(
                    "LUMINA_CORS_ORIGINS must be comma-separated origins such as "
                    f"'http://localhost:3000' (no path) or '*'; got {origin!r}"
                )
        if not self.host.strip():
            raise SettingsError("LUMINA_HOST must not be empty")
        if not 1 <= self.port <= 65535:
            raise SettingsError(f"LUMINA_PORT must be an integer in 1..65535; got {self.port}")

    @property
    def max_body_bytes(self) -> int:
        """Largest accepted request body: the upload limit plus multipart framing allowance."""
        return self.max_upload_bytes + BODY_OVERHEAD_BYTES

    @classmethod
    def from_env(cls: type[Settings], environ: Mapping[str, str] | None = None) -> Settings:
        """Read settings from ``environ`` (default ``os.environ``).

        Empty or whitespace-only values count as unset.

        Raises:
            SettingsError: A variable is malformed or out of range; the message names it.
        """
        env = os.environ if environ is None else environ

        def get(name: str) -> str | None:
            value = env.get(name)
            if value is None or not value.strip():
                return None
            return value.strip()

        def integer(name: str, default: int) -> int:
            text = get(name)
            if text is None:
                return default
            if not _UINT.match(text):
                raise SettingsError(f"{name} must be a non-negative integer; got {text!r}")
            return int(text)

        scale_text = get("LUMINA_DEFAULT_CURVE_SCALE")
        scale: float | None = None
        if scale_text is not None:
            if not _FLOAT.match(scale_text):
                raise SettingsError(
                    f"LUMINA_DEFAULT_CURVE_SCALE must be a number in (0, 1]; got {scale_text!r}"
                )
            scale = float(scale_text)
        checkpoint = get("LUMINA_CHECKPOINT")
        config_path = get("LUMINA_CONFIG")
        origins = tuple(
            part.strip() for part in (get("LUMINA_CORS_ORIGINS") or "").split(",") if part.strip()
        )
        return cls(
            checkpoint=Path(checkpoint) if checkpoint else None,
            config_path=Path(config_path) if config_path else None,
            device=get("LUMINA_DEVICE") or "auto",
            max_upload_bytes=integer("LUMINA_MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES),
            max_side=integer("LUMINA_MAX_SIDE", DEFAULT_MAX_SIDE),
            max_pixels=integer("LUMINA_MAX_PIXELS", DEFAULT_MAX_PIXELS),
            default_curve_scale=scale,
            max_concurrency=integer("LUMINA_MAX_CONCURRENCY", 1),
            cors_origins=origins,
            host=get("LUMINA_HOST") or "127.0.0.1",
            port=integer("LUMINA_PORT", 8000),
        )
