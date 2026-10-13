"""Stateful video inference; flow warp of curves, masks and scene-cut resets.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from nta_zerodce.models.nta_zerodce import ModelOutput, NTAZeroDCE
from nta_zerodce.training.config import Config


@dataclass
class VideoState:
    """One stream's detached state; METHOD temporal inference.

    previous_image/output: Optional RGB (1,3,H,W) in [0,1].
    previous_A: Optional full-resolution curve (1,24,H,W) in [-1,1].
    All share device and dimensions; None denotes new stream or cut/reset.
    Returns a state record, not an executed enhancer. No cross-stream sharing.
    """

    previous_image: Tensor | None = None
    previous_A: Tensor | None = None
    previous_output: Tensor | None = None


class VideoEnhancer:
    """Sequential one-stream enhancement; caller resets between videos."""

    def __init__(self, model: NTAZeroDCE, config: Config) -> None:
        """Construct a video enhancer with empty state.

        Args:
            model: Loaded eval-mode model.
            config: Validated temporal, curve-scale and flow settings.

        Returns:
            No value; future implementation owns VideoState.

        Method:
            ARCHITECTURE inference graph; temporal flag false is frame-independent.

        Raises:
            NotImplementedError: Scaffold only; implement this contract before use.
        """
        raise NotImplementedError("TODO: Construct a video enhancer with empty state")

    def reset(self) -> None:
        """Clear history on stream changes or detected cuts.

        Args:
            self: Enhancer associated with one stream.

        Returns:
            None; clear all three prior tensors.

        Method:
            METHOD scene cut handling; resolution changes also force reset.

        Raises:
            NotImplementedError: Scaffold only; implement this contract before use.
        """
        raise NotImplementedError("TODO: Clear history on stream changes or detected cuts")

    def enhance_frame(self, image: Tensor) -> ModelOutput:
        """Enhance one frame with optional reliable curve blending.

        Args:
            image: RGB (1,3,H,W) in [0,1] on model device.

        Returns:
            Full-resolution ModelOutput; update detached previous state.

        Method:
            METHOD backward current->previous flow, mask-aware convex A blend and optional output blend.

        Raises:
            NotImplementedError: Scaffold only; implement this contract before use.
        """
        raise NotImplementedError("TODO: Enhance one frame with optional reliable curve blending")
