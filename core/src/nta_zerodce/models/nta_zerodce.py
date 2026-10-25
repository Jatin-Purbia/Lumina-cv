"""Compose noise estimation, curve estimation and optional residual denoising.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nta_zerodce.models.dce_net import DCENet, LightDCENet, apply_curves
from nta_zerodce.models.denoiser import ResidualDenoiser
from nta_zerodce.models.noise_branch import NoiseBranch

if TYPE_CHECKING:
    from nta_zerodce.training.config import Config


class ModelOutput(TypedDict):
    """All outputs are full-resolution float32, on the input device.

    enhanced/enhanced_raw: RGB (B,3,H,W) in [0,1]. A: (B,24,H,W) in [-1,1].
    sigma: (B,1,H,W) in [0,0.2]; exact zeros when noise_aware is disabled.
    METHOD composition; keys are always present, including in baseline mode.
    """

    enhanced: Tensor
    enhanced_raw: Tensor
    A: Tensor
    sigma: Tensor


def compute_gain(
    image: Tensor,
    enhanced_raw: Tensor,
    window: int = 5,
    gain_min: float = 1.0,
    gain_max: float = 8.0,
    eps: float = 1e-6,
) -> Tensor:
    """Detached local brightness gain used by the denoiser and the edge loss.

    Args:
        image: RGB (B,3,H,W) in [0,1]; the (possibly noisy) model input x.
        enhanced_raw: RGB (B,3,H,W) in [0,1]; curve output z of the same image.
        window: Odd averaging width (model.gain_window), default 5; needs H,W > window//2.
        gain_min: Lower clamp, >= 1.
        gain_max: Upper clamp, >= gain_min.
        eps: Positive division guard.

    Returns:
        G (B,1,H,W) in [gain_min, gain_max], computed without autograd (a constant).

    Method:
        METHOD extension 1: G = stopgrad(clamp(avgpool_w(Y(z)) / (avgpool_w(Y(x)) + eps),
        gain_min, gain_max)) with stride-1 average pooling and reflection padding. A bounded
        heuristic, not the derivative of the curve.

    Raises:
        ValueError: If window is not odd and positive or the shapes disagree.
    """
    if window < 1 or window % 2 == 0:
        raise ValueError(f"window must be odd and positive, got {window}")
    if image.shape != enhanced_raw.shape or image.dim() != 4 or image.shape[1] != 3:
        raise ValueError(
            f"image {tuple(image.shape)} and enhanced_raw {tuple(enhanced_raw.shape)} must be "
            "equal (B,3,H,W) shapes"
        )
    pad = window // 2
    with torch.no_grad():
        y_in = image.detach().mean(dim=1, keepdim=True)
        y_out = enhanced_raw.detach().mean(dim=1, keepdim=True)
        if pad > 0:
            y_in = F.pad(y_in, (pad, pad, pad, pad), mode="reflect")
            y_out = F.pad(y_out, (pad, pad, pad, pad), mode="reflect")
        y_in = F.avg_pool2d(y_in, window, stride=1)
        y_out = F.avg_pool2d(y_out, window, stride=1)
        return (y_out / (y_in + eps)).clamp(gain_min, gain_max)


class NTAZeroDCE(nn.Module):
    """Unified flag-controlled enhancement module; METHOD composition.

    No training, dataset, file IO or optical-flow dependencies are allowed here. The module
    never resizes: ``config.model.curve_scale`` is consumed by the inference layer, which
    estimates maps on a downscaled image (``estimate_maps``), upsamples them and renders at full
    resolution (``apply_maps``).
    """

    def __init__(self, config: Config) -> None:
        """Construct the flag-controlled composition.

        Args:
            config: Validated schema from training/config.py; denoise implies noise_aware.

        Returns:
            No value; registers the curve estimator and, per flag, the noise branch and denoiser.

        Method:
            ARCHITECTURE composition; flags gate construction. noise_aware adds NoiseBranch and
            a fourth curve-estimator input channel; denoise adds ResidualDenoiser; light selects
            LightDCENet.

        Raises:
            ValueError: If denoise is enabled without noise_aware.
        """
        super().__init__()
        flags = config.flags
        model = config.model
        if flags.denoise and not flags.noise_aware:
            raise ValueError("denoise requires noise_aware")
        self.noise_aware = bool(flags.noise_aware)
        self.denoise = bool(flags.denoise)
        self.light = bool(flags.light)
        self.iterations = int(model.iterations)
        self.gain_window = int(model.gain_window)
        self.gain_min = float(model.gain_min)
        self.gain_max = float(model.gain_max)
        self.eps = float(model.eps)
        in_channels = 4 if self.noise_aware else 3
        self.dce: nn.Module
        if self.light:
            self.dce = LightDCENet(in_channels, int(model.channels), self.iterations)
        else:
            self.dce = DCENet(in_channels, int(model.channels), self.iterations)
        self.noise_branch: NoiseBranch | None = None
        self.denoiser: ResidualDenoiser | None = None
        if self.noise_aware:
            self.noise_branch = NoiseBranch(int(model.noise_channels), float(model.sigma_max))
        if self.denoise:
            self.denoiser = ResidualDenoiser(
                int(model.denoise_channels), float(model.residual_scale)
            )

    def _estimate_sigma(self, image: Tensor) -> Tensor:
        """Per-pixel sigma (B,1,H,W); exact zeros (same device/dtype) when noise is disabled."""
        if self.noise_branch is None:
            return torch.zeros_like(image[:, :1])
        return self.noise_branch(image)

    def estimate_maps(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """Estimate curve maps and noise sigma without rendering any image.

        Args:
            image: RGB float32 tensor (B,3,H,W) in [0,1]; any size >= 1 pixel per side. The
                call is device-agnostic and never resizes, so a caller may pass a downscaled
                copy (curve_scale < 1) and upsample the returned maps itself.

        Returns:
            (A, sigma): A (B,24,H,W) in [-1,1] and sigma (B,1,H,W) in [0,sigma_max], both at
            the resolution of ``image``. sigma is exact zeros when noise_aware is disabled.

        Method:
            sigma = NoiseBranch(x); A = DCE(concat(x, sigma)) when noise_aware, else DCE(x).
            Gradients flow from A back into the noise head through the fourth input channel.
        """
        sigma = self._estimate_sigma(image)
        dce_input = torch.cat([image, sigma], dim=1) if self.noise_aware else image
        return self.dce(dce_input), sigma

    def apply_maps(self, image: Tensor, A: Tensor, sigma: Tensor) -> ModelOutput:
        """Render an image from given curve maps and sigma, then optionally denoise.

        Args:
            image: RGB float32 tensor (B,3,H,W) in [0,1] at the output resolution.
            A: (B,24,H,W) curve maps in [-1,1] at the same resolution as ``image`` (for
                example upsampled from a downscaled estimate or temporally smoothed).
            sigma: (B,1,H,W) noise standard deviation at the same resolution; ignored (but
                echoed back) when denoise is disabled.

        Returns:
            ModelOutput: enhanced_raw = apply_curves(image, A); enhanced = ResidualDenoiser(
            enhanced_raw, sigma, compute_gain(image, enhanced_raw)) when denoise is on, else
            enhanced is enhanced_raw. A and sigma are returned unchanged.

        Method:
            METHOD composition; the gain uses the same image the maps were applied to.
        """
        raw = apply_curves(image, A, self.iterations)
        enhanced = raw
        if self.denoiser is not None:
            gain = compute_gain(image, raw, self.gain_window, self.gain_min, self.gain_max, self.eps)
            enhanced = self.denoiser(raw, sigma, gain)
        return {"enhanced": enhanced, "enhanced_raw": raw, "A": A, "sigma": sigma}

    def forward(self, image: Tensor, A_override: Tensor | None = None) -> ModelOutput:
        """Estimate and apply curves, then optionally denoise.

        Args:
            image: RGB float32 tensor (B,3,H,W) in [0,1]; no implicit device transfer.
            A_override: Optional full-resolution (B,24,H,W) in [-1,1] from temporal smoothing.

        Returns:
            ModelOutput with enhanced, enhanced_raw, A, sigma; baseline enhanced equals raw.

        Method:
            METHOD composition and temporal inference; equals
            ``apply_maps(image, *estimate_maps(image))``. The override bypasses curve estimation
            only: sigma is still computed when noise_aware is enabled.

        Raises:
            ValueError: If A_override does not have shape (B,24,H,W).
        """
        if A_override is None:
            A, sigma = self.estimate_maps(image)
        else:
            expected = (image.shape[0], 3 * self.iterations, image.shape[2], image.shape[3])
            if tuple(A_override.shape) != expected:
                raise ValueError(
                    f"A_override must have shape {expected}, got {tuple(A_override.shape)}"
                )
            A = A_override
            sigma = self._estimate_sigma(image)
        return self.apply_maps(image, A, sigma)
