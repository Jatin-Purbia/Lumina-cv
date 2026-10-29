"""Capture loop contracts; decode/capture may use OpenCV but not flow APIs.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

import cv2
import torch

from nta_zerodce.inference.video_enhancer import VideoEnhancer
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
        ValueError: Invalid inputs or incompatible configuration.
    """
    capture = cv2.VideoCapture(camera_id)
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Cannot open camera {camera_id}")
    enhancer = VideoEnhancer(model, config)
    device = next(model.parameters()).device
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            image = (
                torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                .permute(2, 0, 1)
                .float()
                .div(255)
                .unsqueeze(0)
                .to(device)
            )
            output = enhancer.enhance_frame(image)["enhanced"][0].cpu().permute(1, 2, 0).numpy()
            cv2.imshow("Lumina CV - Esc to exit", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
