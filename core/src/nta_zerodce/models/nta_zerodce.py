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
from nta_zerodce.tensor_ops import check_image, local_gain

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


class NTAZeroDCE(nn.Module):
    """Unified flag-controlled enhancement module; METHOD composition.

    No training, dataset, file IO or optical-flow dependencies are allowed here.
    """

    def __init__(self, config: Config) -> None:
        """Construct the flag-controlled composition.

        Args:
            config: Validated schema from training/config.py; denoise implies noise_aware.

        Returns:
            No value; owns curve/noise/denoise submodules.

        Method:
            ARCHITECTURE composition; flags gate construction.

        Raises:
            ValueError: Invalid inputs or incompatible configuration.
        """
        super().__init__()
        config.validate()
        self.config = config
        self.estimator: DCENet | LightDCENet
        if config.flags.light:
            self.estimator = LightDCENet(
                4 if config.flags.noise_aware else 3, config.model.channels
            )
        else:
            self.estimator = DCENet(4 if config.flags.noise_aware else 3, config.model.channels)
        self.noise = (
            NoiseBranch(config.model.noise_channels, config.model.sigma_max)
            if config.flags.noise_aware
            else None
        )
        self.denoiser = (
            ResidualDenoiser(config.model.denoise_channels, config.model.residual_scale)
            if config.flags.denoise
            else None
        )

    def forward(self, image: Tensor, A_override: Tensor | None = None) -> ModelOutput:
        """Estimate and apply curves, then optionally denoise.

        Args:
            image: RGB float32 tensor (B,3,H,W) in [0,1]; no implicit device transfer.
            A_override: Optional full-resolution (B,24,H,W) in [-1,1] from temporal smoothing.

        Returns:
            ModelOutput with enhanced, enhanced_raw, A, sigma; baseline enhanced equals raw.

        Method:
            METHOD composition and temporal inference; override bypasses curve estimation only.

        Raises:
            ValueError: Invalid inputs or incompatible configuration.
        """
        check_image(image)
        estimation_image = image
        scale = self.config.model.curve_scale
        if scale != 1:
            size = tuple(max(16, round(d * scale)) for d in image.shape[-2:])
            estimation_image = F.interpolate(image, size=size, mode="bilinear", align_corners=False)
        small_sigma = (
            self.noise(estimation_image)
            if self.noise is not None
            else torch.zeros_like(estimation_image[:, :1])
        )
        sigma = small_sigma
        if sigma.shape[-2:] != image.shape[-2:]:
            sigma = F.interpolate(
                sigma, size=image.shape[-2:], mode="bilinear", align_corners=False
            )
        if A_override is None:
            inputs = (
                torch.cat((estimation_image, small_sigma), 1)
                if self.noise is not None
                else estimation_image
            )
            A = self.estimator(inputs)
            if A.shape[-2:] != image.shape[-2:]:
                A = F.interpolate(A, size=image.shape[-2:], mode="bilinear", align_corners=False)
        else:
            A = A_override
        raw = apply_curves(image, A)
        gain = local_gain(image, raw, self.config.model)
        output = self.denoiser(raw, sigma, gain) if self.denoiser is not None else raw
        return {"enhanced": output, "enhanced_raw": raw, "A": A, "sigma": sigma}
