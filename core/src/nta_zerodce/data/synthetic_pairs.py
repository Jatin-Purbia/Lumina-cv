"""Affine still-image pairs with analytic flow and independent corruption.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from torch import Generator, Tensor

from nta_zerodce.training.config import Config


def make_pair(
    image: Tensor, config: Config, generator: Generator | None = None
) -> dict[str, Tensor]:
    """Create synthetic temporal supervision.

    Args:
        image: RGB float32 tensor (B,3,H,W) in [0,1]; no implicit device transfer.
        config: Validated synthetic/noise/flow constants.
        generator: Device-matched RNG; independent draws for each frame.

    Returns:
        image1/image2 RGB (B,3,H,W); forward/backward_flow (B,2,H,W); mask (B,1,H,W).

    Method:
        METHOD extension 5: affine inverse coordinate maps, exposure flicker and independent noise; DATA generation.

    Raises:
        NotImplementedError: Scaffold only; implement this contract before use.
    """
    raise NotImplementedError("TODO: Create synthetic temporal supervision")
