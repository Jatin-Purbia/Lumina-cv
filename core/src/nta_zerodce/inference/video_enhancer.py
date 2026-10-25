"""Stateful video inference; flow warp of curves, masks and scene-cut resets.

See docs/INTERFACES.md and docs/METHOD.md (extension 5) for the normative contract.

Per frame the enhancer estimates full-resolution maps ``(A_t, sigma_t)``, then, unless history
is unusable, blends the previous *smoothed* curve map warped by the backward (current to
previous) DIS flow, restricted to the reliable pixels ``m`` (current coordinates)::

    A_bar = (1 - w) * A_t + w * warp(A_bar_prev, b),   w = curve_blend * m * in_bounds(b)

The blend is convex, so ``A_bar`` stays in [-1,1]. The curves are applied to the current full
resolution RGB only; pixels are never taken from the previous frame unless the optional output
blend (``temporal.output_blend`` > 0, off by default) is enabled.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor

from nta_zerodce.flow.estimator import MIN_FLOW_SIDE, estimate_bidirectional
from nta_zerodce.flow.masks import is_scene_cut, reliability_mask
from nta_zerodce.flow.warp import warp
from nta_zerodce.inference.enhancer import estimate_full_res_maps
from nta_zerodce.models.nta_zerodce import ModelOutput, NTAZeroDCE
from nta_zerodce.training.config import Config, FlowConfig


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


def _effective_flow_config(flow: FlowConfig, height: int, width: int) -> FlowConfig:
    """Raise ``flow.scale`` just enough that DIS sees at least MIN_FLOW_SIDE pixels per side.

    Only very small frames are affected (for example 24x24 at the default scale 0.5); the
    configured scale is used whenever it already satisfies the estimator's minimum.
    """
    smallest = min(height, width)
    if round(smallest * flow.scale) >= MIN_FLOW_SIDE:
        return flow
    return replace(flow, scale=min(1.0, (MIN_FLOW_SIDE + 0.5) / smallest))


class VideoEnhancer:
    """Sequential one-stream enhancement; caller resets between videos.

    One instance owns the history of exactly one stream. Create a new instance (or call
    ``reset``) for every video or camera session; never feed two streams through one object.

    Read-only diagnostics, updated by every ``enhance_frame`` call:
        frames_processed: Frames enhanced since construction.
        last_was_reset: True when the last frame used no history (first frame, after
            ``reset()``, scene cut, size change, or the temporal flag is off).
        last_scene_cut: True when the last frame was recognised as a scene cut.
        last_mask_coverage: Mean reliability mask value in [0,1] for the last frame, or None
            when no mask was computed (``last_was_reset``).

    Tensors returned in the ModelOutput (notably ``A``) may alias the stored state: treat them
    as read-only, or clone them before modifying.
    """

    def __init__(self, model: NTAZeroDCE, config: Config) -> None:
        """Construct a video enhancer with empty state.

        Args:
            model: Loaded eval-mode model.
            config: Validated temporal, curve-scale and flow settings.

        Returns:
            No value; the enhancer owns an empty VideoState.

        Method:
            ARCHITECTURE inference graph; temporal flag false is frame-independent.

        Raises:
            ValueError: The model has no parameters (device cannot be determined).
        """
        reference = next(model.parameters(), None)
        if reference is None:
            raise ValueError("model has no parameters; cannot determine its device")
        self.model = model
        self.config = config
        self.device: torch.device = reference.device
        self.state = VideoState()
        self.frames_processed = 0
        self.last_was_reset = True
        self.last_scene_cut = False
        self.last_mask_coverage: float | None = None

    def reset(self) -> None:
        """Clear history on stream changes or detected cuts.

        Args:
            self: Enhancer associated with one stream.

        Returns:
            None; clears previous image, smoothed curve map and previous output. The next
            frame is processed as the first of a new stream.

        Method:
            METHOD scene cut handling; resolution changes also force reset.
        """
        self.state = VideoState()

    def _history_usable(self, image: Tensor) -> bool:
        """True when stored history matches the frame's size, device and dtype."""
        state = self.state
        if state.previous_image is None or state.previous_A is None:
            return False
        if self.config.temporal.output_blend > 0.0 and state.previous_output is None:
            return False
        return bool(
            state.previous_image.shape == image.shape
            and state.previous_A.shape[2:] == image.shape[2:]
            and state.previous_image.device == image.device
            and state.previous_image.dtype == image.dtype
        )

    def enhance_frame(self, image: Tensor) -> ModelOutput:
        """Enhance one frame with optional reliable curve blending.

        Args:
            image: RGB (1,3,H,W) in [0,1] on model device, H,W >= 16.

        Returns:
            Full-resolution ModelOutput whose ``A`` is the temporally smoothed map and whose
            ``enhanced`` is the final output; the detached current frame, smoothed map and
            final output become the new state. With ``flags.temporal`` False the result equals
            ``enhance_image`` and no state is kept.

        Method:
            METHOD backward current->previous flow, mask-aware convex A blend and optional
            output blend. Flow is estimated on the unenhanced RGB frames (CPU DIS, both
            directions) and moved to the model device explicitly.

        Raises:
            ValueError: Batch size other than one, wrong shape/dtype/device or tiny image.
        """
        if image.ndim == 4 and image.shape[0] != 1:
            raise ValueError(
                f"VideoEnhancer processes one frame at a time; got batch {image.shape[0]}"
            )
        config = self.config
        with torch.no_grad():
            image = image.detach()
            a_cur, sigma_cur = estimate_full_res_maps(self.model, image, config.model.curve_scale)
            self.frames_processed += 1
            if not config.flags.temporal:
                self.last_was_reset, self.last_scene_cut, self.last_mask_coverage = (
                    True,
                    False,
                    None,
                )
                return self.model.apply_maps(image, a_cur, sigma_cur)

            usable = self._history_usable(image)
            cut = False
            if usable:
                previous = self.state.previous_image
                assert previous is not None  # narrowed by _history_usable
                cut = is_scene_cut(
                    previous,
                    image,
                    config.temporal.scene_cut_threshold,
                    config.temporal.histogram_bins,
                )
            self.last_scene_cut = cut
            if not usable or cut:
                return self._start_stream(image, a_cur, sigma_cur)
            return self._blend_with_history(image, a_cur, sigma_cur)

    def _store(self, image: Tensor, a_bar: Tensor, output: Tensor) -> None:
        """Replace the state with detached copies of the current frame and output."""
        self.state = VideoState(
            previous_image=image.detach().clone(),
            previous_A=a_bar.detach(),
            previous_output=output.detach().clone(),
        )

    def _start_stream(self, image: Tensor, a_cur: Tensor, sigma_cur: Tensor) -> ModelOutput:
        """Drop all history and render with the current maps (A_bar = A_t)."""
        self.reset()
        out = self.model.apply_maps(image, a_cur, sigma_cur)
        self._store(image, a_cur, out["enhanced"])
        self.last_was_reset = True
        self.last_mask_coverage = None
        return out

    def _blend_with_history(self, image: Tensor, a_cur: Tensor, sigma_cur: Tensor) -> ModelOutput:
        """Warp history with the backward flow and blend it where the mask is reliable."""
        config = self.config
        state = self.state
        previous, previous_a = state.previous_image, state.previous_A
        assert previous is not None and previous_a is not None
        flow_config = _effective_flow_config(config.flow, int(image.shape[2]), int(image.shape[3]))
        # DIS runs on the CPU (OpenCV); frames are clamped for it only, the model sees `image`.
        previous_cpu = previous.to("cpu", torch.float32).clamp(0.0, 1.0).contiguous()
        current_cpu = image.to("cpu", torch.float32).clamp(0.0, 1.0).contiguous()
        forward_flow, backward_flow = estimate_bidirectional(previous_cpu, current_cpu, flow_config)
        forward_flow = forward_flow.to(device=image.device, dtype=image.dtype)
        backward_flow = backward_flow.to(device=image.device, dtype=image.dtype)
        mask = reliability_mask(
            forward_flow,
            backward_flow,
            previous,
            image,
            alpha=config.flow.fb_alpha,
            beta=config.flow.fb_beta,
            photometric_threshold=config.flow.photometric_threshold,
        )
        warped_a, a_in_bounds = warp(previous_a, backward_flow)
        weight_a = config.temporal.curve_blend * mask * a_in_bounds
        a_bar = (1.0 - weight_a) * a_cur + weight_a * warped_a
        out = self.model.apply_maps(image, a_bar, sigma_cur)
        beta = config.temporal.output_blend
        if beta > 0.0:
            previous_output = state.previous_output
            assert previous_output is not None
            warped_output, o_in_bounds = warp(previous_output, backward_flow)
            weight_o = beta * mask * o_in_bounds
            out["enhanced"] = (1.0 - weight_o) * out["enhanced"] + weight_o * warped_output
        self._store(image, a_bar, out["enhanced"])
        self.last_was_reset = False
        self.last_mask_coverage = float(mask.mean())
        return out
