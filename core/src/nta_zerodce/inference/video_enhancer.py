"""Stateful video inference; flow warp of curves, masks and scene-cut resets.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from nta_zerodce.flow.estimator import estimate_bidirectional
from nta_zerodce.flow.masks import is_scene_cut, reliability_mask
from nta_zerodce.flow.warp import warp
from nta_zerodce.inference.enhancer import enhance_image
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
            No value; owns VideoState.

        Method:
            ARCHITECTURE inference graph; temporal flag false is frame-independent.

        Raises:
            ValueError: Invalid inputs or incompatible configuration.
        """
        config.validate()
        self.model = model
        self.config = config
        self.state = VideoState()

    def reset(self) -> None:
        """Clear history on stream changes or detected cuts.

        Args:
            self: Enhancer associated with one stream.

        Returns:
            None; clear all three prior tensors.

        Method:
            METHOD scene cut handling; resolution changes also force reset.

        Raises:
            ValueError: Invalid inputs or incompatible configuration.
        """
        self.state = VideoState()

    def enhance_frame(self, image: Tensor) -> ModelOutput:
        """Enhance one frame with optional reliable curve blending.

        Args:
            image: RGB (1,3,H,W) in [0,1] on model device.

        Returns:
            Full-resolution ModelOutput; update detached previous state.

        Method:
            METHOD backward current->previous flow, mask-aware convex A blend and optional output blend.

        Raises:
            ValueError: Invalid inputs or incompatible configuration.
        """
        if image.shape[0] != 1:
            raise ValueError("Video enhancer supports one stream (B=1)")
        cfg = self.config
        self.model.eval()
        with torch.no_grad():
            result = enhance_image(self.model, image, cfg.model.curve_scale)
            previous = self.state.previous_image
            if not cfg.flags.temporal:
                self.reset()
                return result
            if previous is not None and (
                previous.shape != image.shape
                or is_scene_cut(
                    previous, image, cfg.temporal.scene_cut_threshold, cfg.temporal.histogram_bins
                )
            ):
                self.reset()
                previous = None
            if previous is not None:
                forward, backward = estimate_bidirectional(previous, image, cfg.flow)
                forward, backward = forward.to(image.device), backward.to(image.device)
                mask = reliability_mask(
                    forward,
                    backward,
                    previous,
                    image,
                    cfg.flow.fb_alpha,
                    cfg.flow.fb_beta,
                    cfg.flow.photometric_threshold,
                )
                assert self.state.previous_A is not None and self.state.previous_output is not None
                prior_maps, _ = warp(self.state.previous_A, backward)
                blend = cfg.temporal.curve_blend * mask
                maps = (1 - blend) * result["A"] + blend * prior_maps
                result = self.model(image, A_override=maps)
                if cfg.temporal.output_blend > 0:
                    prior_output, _ = warp(self.state.previous_output, backward)
                    beta = cfg.temporal.output_blend * mask
                    result["enhanced"] = (1 - beta) * result["enhanced"] + beta * prior_output
            self.state = VideoState(
                image.detach(), result["A"].detach(), result["enhanced"].detach()
            )
            return result
