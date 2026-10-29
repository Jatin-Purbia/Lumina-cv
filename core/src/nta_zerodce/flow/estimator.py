"""Only owner of OpenCV optical flow; CPU boundary and displacement rescaling.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from nta_zerodce.training.config import FlowConfig


def estimate_flow(first: Tensor, second: Tensor, config: FlowConfig) -> Tensor:
    """Estimate source-to-target pixel displacement.

    Args:
        first/second: CPU RGB float32 (B,3,H,W) in [0,1], matching size.
        config: DIS method/preset, flow scale in (0,1].

    Returns:
        CPU float32 flow (B,2,H,W), dx,dy in original pixels; detached.

    Method:
        INTERFACES: first(p) ~ second(p+flow(p)); DIS [R6]. Resize vectors by full/small dimensions.

    Raises:
        ValueError: Invalid inputs or incompatible configuration.
    """
    if (
        first.shape != second.shape
        or first.ndim != 4
        or first.shape[1] != 3
        or min(first.shape[-2:]) < 16
    ):
        raise ValueError("DIS expects matching RGB frames with H,W >=16")
    if config.method != "dis" or config.preset != "medium" or not 0 < config.scale <= 1:
        raise ValueError("Unsupported DIS settings")
    h, w = first.shape[-2:]
    sh, sw = max(16, round(h * config.scale)), max(16, round(w * config.scale))
    results = []
    for a, b in zip(first.detach().cpu(), second.detach().cpu(), strict=True):
        gray_a = (a.mean(0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
        gray_b = (b.mean(0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
        adapter = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        initial = np.zeros((sh, sw, 2), dtype=np.float32)
        estimated = adapter.calc(
            cv2.resize(gray_a, (sw, sh)), cv2.resize(gray_b, (sw, sh)), initial
        )
        value: NDArray[np.float32] = np.asarray(cv2.resize(estimated, (w, h)), dtype=np.float32)
        value[..., 0] *= w / sw
        value[..., 1] *= h / sh
        results.append(torch.from_numpy(value).permute(2, 0, 1))
    return torch.stack(results).float()


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
        ValueError: Invalid inputs or incompatible configuration.
    """
    return estimate_flow(first, second, config), estimate_flow(second, first, config)
