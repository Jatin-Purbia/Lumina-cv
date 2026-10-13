"""Still-image enhancement with optional downscaled curve estimation.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from torch import Tensor

from nta_zerodce.models.nta_zerodce import ModelOutput, NTAZeroDCE


def enhance_image(model: NTAZeroDCE, image: Tensor, curve_scale: float = 1.0) -> ModelOutput:
    """Enhance RGB using inference-only model execution.

    Args:
        image: RGB float32 tensor (B,3,H,W) in [0,1]; no implicit device transfer.
        model: Eval-mode module on the image device.
        curve_scale: In (0,1]; estimate sigma/A at reduced size and resize maps.

    Returns:
        Full-resolution ModelOutput; no gradients, RGB in [0,1].

    Method:
        REAL_TIME downsample estimator then apply eight curves to original RGB; no resized final RGB.

    Raises:
        NotImplementedError: Scaffold only; implement this contract before use.
    """
    raise NotImplementedError("TODO: Enhance RGB using inference-only model execution")
