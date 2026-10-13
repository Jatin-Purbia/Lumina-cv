"""Construct baseline or light models without data or training side effects.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
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
        Uninitialized-for-training NTAZeroDCE; caller chooses device.

    Method:
        ARCHITECTURE flag matrix and METHOD architecture.

    Raises:
        NotImplementedError: Scaffold only; implement this contract before use.
    """
    raise NotImplementedError("TODO: Build a model from validated feature flags")
