"""Compose noise estimation, curve estimation and optional residual denoising.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from torch import Tensor, nn

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
            No value; future implementation owns curve/noise/denoise submodules.

        Method:
            ARCHITECTURE composition; flags gate construction.

        Raises:
            NotImplementedError: Scaffold only; implement this contract before use.
        """
        raise NotImplementedError("TODO: Construct the flag-controlled composition")

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
            NotImplementedError: Scaffold only; implement this contract before use.
        """
        raise NotImplementedError("TODO: Estimate and apply curves, then optionally denoise")
