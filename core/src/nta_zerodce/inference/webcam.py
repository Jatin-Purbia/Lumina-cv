"""Capture loop contracts; decode/capture may use OpenCV but not flow APIs.

See docs/INTERFACES.md, docs/REAL_TIME.md and docs/METHOD.md for the normative contract.

The loop is fully testable without a camera or a display: pass ``capture_factory`` (any object
with ``read()``, ``release()`` and ``isOpened()``) and ``show=False``.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from typing import Protocol

import cv2
import numpy as np
import torch
from torch import Tensor

from nta_zerodce.inference.video_enhancer import VideoEnhancer
from nta_zerodce.models.nta_zerodce import NTAZeroDCE
from nta_zerodce.training.config import Config

FPS_WINDOW = 30
"""Number of recent frame times in the moving-average FPS overlay."""

MAX_CONSECUTIVE_READ_FAILURES = 3
"""Failed ``read()`` calls in a row that end the loop (a camera that stopped delivering)."""

_ESCAPE = 27


class CaptureLike(Protocol):
    """The subset of ``cv2.VideoCapture`` used by the loop."""

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Return ``(ok, BGR uint8 frame)``."""
        ...

    def release(self) -> None:
        """Free the device."""
        ...

    def isOpened(self) -> bool:
        """Whether the device is usable."""
        ...


FrameCallback = Callable[[int, np.ndarray, VideoEnhancer], None]
"""``on_frame(frame_index, displayed_bgr_uint8, enhancer)``, called after each processed frame."""


def _default_capture(camera_id: int) -> CaptureLike:
    """Open a real camera with OpenCV."""
    return cv2.VideoCapture(camera_id)


def _as_bgr(frame: object) -> np.ndarray:
    """Validate a captured frame and return a contiguous HxWx3 uint8 BGR array."""
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise ValueError("captured frames must be uint8 numpy arrays")
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"unsupported captured frame shape {frame.shape}")
    if frame.shape[0] < 1 or frame.shape[1] < 1:
        raise ValueError("captured frame is empty")
    return np.ascontiguousarray(frame)


def bgr_to_tensor(frame: np.ndarray, device: torch.device) -> Tensor:
    """BGR uint8 HxWx3 to RGB float32 (1,3,H,W) in [0,1] on ``device`` (explicit transfer)."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div(255.0)
    return tensor.unsqueeze(0).contiguous().to(device)


def tensor_to_bgr(image: Tensor) -> np.ndarray:
    """RGB float (1,3,H,W) to BGR uint8 HxWx3; values are clamped then rounded once."""
    cpu = image.detach().to("cpu", torch.float32)[0].clamp(0.0, 1.0)
    rgb = torch.round(255.0 * cpu).to(torch.uint8).permute(1, 2, 0).numpy()
    return np.ascontiguousarray(cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR))


def _overlay(canvas: np.ndarray, lines: list[str]) -> None:
    """Draw outlined text lines in the top-left corner (in place)."""
    for row, text in enumerate(lines):
        origin = (8, 20 + 20 * row)
        cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(
            canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )


def run_webcam(
    model: NTAZeroDCE,
    config: Config,
    camera_id: int = 0,
    *,
    max_frames: int | None = None,
    capture_factory: Callable[[int], CaptureLike] | None = None,
    show: bool = True,
    window_name: str = "Lumina CV - original | enhanced",
    on_frame: FrameCallback | None = None,
    split: bool = True,
) -> None:
    """Run a real-time webcam demonstration.

    Args:
        model: Loaded eval-mode model on config device.
        config: Validated inference/flow settings.
        camera_id: Nonnegative OpenCV capture index.
        max_frames: Stop after this many processed frames (None runs until the stream ends or
            the user quits).
        capture_factory: ``camera_id -> capture`` returning an object with ``read()``,
            ``release()`` and ``isOpened()``; defaults to ``cv2.VideoCapture``.
        show: Open an OpenCV window. Keys: ``q`` or ESC quit, ``s`` toggles side-by-side
            (original | enhanced) versus enhanced-only. With False the loop is headless.
        window_name: Window title.
        on_frame: Optional ``(index, displayed_bgr_uint8, enhancer)`` callback per frame.
        split: Initial display mode (True shows original and enhanced side by side).

    Returns:
        None; the capture is released and windows destroyed on normal exit, error and
        KeyboardInterrupt.

    Method:
        REAL_TIME budget/profiling; sequential VideoEnhancer state and RGB/BGR boundary. One
        VideoEnhancer lives exactly as long as this call (so reopening the camera starts clean)
        and is reset when the frame size changes. The overlay shows a moving-average FPS over
        the last 30 loop iterations (wall time, including capture and display) and the mask
        coverage of the temporal blend (``off``/``reset`` when no history was used).

    Raises:
        ValueError: Negative ``camera_id``, non-positive ``max_frames`` or an invalid frame.
        RuntimeError: The camera cannot be opened, delivered no frame, or ``show=True`` but
            OpenCV has no GUI support (for example opencv-python-headless).
    """
    if isinstance(camera_id, bool) or not isinstance(camera_id, int) or camera_id < 0:
        raise ValueError(f"camera_id must be a nonnegative integer; got {camera_id!r}")
    if max_frames is not None and max_frames < 1:
        raise ValueError(f"max_frames must be positive; got {max_frames}")
    factory = capture_factory if capture_factory is not None else _default_capture
    device = next(model.parameters()).device
    capture = factory(camera_id)
    try:
        if not capture.isOpened():
            raise RuntimeError(
                f"cannot open camera {camera_id}; check that it is connected, not in use by "
                "another program, and that this user may access it"
            )
        enhancer = VideoEnhancer(model, config)
        frame_times: deque[float] = deque(maxlen=FPS_WINDOW)
        last_size: tuple[int, int] | None = None
        processed = 0
        failures = 0
        side_by_side = split
        previous_tick = time.perf_counter()
        while max_frames is None or processed < max_frames:
            ok, raw = capture.read()
            if not ok or raw is None:
                failures += 1
                if failures >= MAX_CONSECUTIVE_READ_FAILURES:
                    if processed == 0:
                        raise RuntimeError(f"camera {camera_id} opened but delivered no frames")
                    break
                continue
            failures = 0
            frame = _as_bgr(raw)
            size = (int(frame.shape[0]), int(frame.shape[1]))
            if last_size is not None and size != last_size:
                enhancer.reset()
            last_size = size
            output = enhancer.enhance_frame(bgr_to_tensor(frame, device))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            enhanced = tensor_to_bgr(output["enhanced"])
            shown = np.concatenate([frame, enhanced], axis=1) if side_by_side else enhanced
            now = time.perf_counter()
            frame_times.append(now - previous_tick)
            previous_tick = now
            fps = len(frame_times) / max(sum(frame_times), 1e-9)
            if not config.flags.temporal:
                coverage = "temporal off"
            elif enhancer.last_mask_coverage is None:
                coverage = "mask: reset"
            else:
                coverage = f"mask: {100.0 * enhancer.last_mask_coverage:.0f}%"
            _overlay(shown, [f"{fps:.1f} FPS", coverage])
            if on_frame is not None:
                on_frame(processed, shown, enhancer)
            processed += 1
            if show:
                key = _show(window_name, shown)
                if key in (ord("q"), ord("Q"), _ESCAPE):
                    break
                if key in (ord("s"), ord("S")):
                    side_by_side = not side_by_side
    finally:
        capture.release()
        if show:
            _close_windows()


def _show(window_name: str, canvas: np.ndarray) -> int:
    """Display ``canvas`` and return the pressed key code (-1 when none)."""
    try:
        cv2.imshow(window_name, canvas)
        return int(cv2.waitKey(1)) & 0xFF
    except cv2.error as error:
        raise RuntimeError(
            "OpenCV has no display support here (headless build or no display); "
            "install opencv-python or run with show=False"
        ) from error


def _close_windows() -> None:
    """Destroy OpenCV windows; a missing GUI backend is not an error during cleanup."""
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass
