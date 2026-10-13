"""Capture loop contracts; decode/capture may use OpenCV but not flow APIs.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from nta_zerodce.models.nta_zerodce import NTAZeroDCE
from nta_zerodce.training.config import Config


def run_webcam(model: NTAZeroDCE, config: Config, camera_id: int = 0) -> None:
    """Run a real-time webcam demonstration.

    Args:
        model: Loaded eval-mode model on config device.
        config: Validated inference/flow settings.
        camera_id: Nonnegative OpenCV capture index.

    Returns:
        None; release capture/window resources on exit or interruption.

    Method:
        REAL_TIME budget/profiling; sequential VideoEnhancer state and RGB/BGR boundary.

    Raises:
        NotImplementedError: Scaffold only; implement this contract before use.
    """
    raise NotImplementedError("TODO: Run a real-time webcam demonstration")
