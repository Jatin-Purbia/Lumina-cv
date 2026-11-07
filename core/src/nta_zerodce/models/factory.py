"""Construct baseline or light models without data or training side effects.

See docs/ARCHITECTURE.md for the normative contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nta_zerodce.models.nta_zerodce import NTAZeroDCE

if TYPE_CHECKING:
    from nta_zerodce.training.config import Config


def build_model(config: Config) -> NTAZeroDCE:
    """Build a model from validated feature flags.

    Args:
        config: Validated Config; light changes convolution type, noise changes input channels.

    Returns:
        Uninitialized-for-training NTAZeroDCE (default PyTorch initialization, float32, CPU);
        the caller chooses the device.

    Method:
        ARCHITECTURE flag matrix and METHOD architecture. Only the invariants the model itself
        needs are checked (denoise requires noise_aware); schema validation is the caller's job.

    Raises:
        ValueError: If denoise is enabled without noise_aware.
    """
    return NTAZeroDCE(config)
