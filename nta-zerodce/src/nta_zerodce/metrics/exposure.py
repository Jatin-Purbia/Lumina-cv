"""Luminance clipping fraction; EXPERIMENTS highlight cautions.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from torch import Tensor


def clipping_fraction(output: Tensor, threshold: float = 0.99) -> Tensor:
    """Measure near-saturated luminance fraction.

    Args:
        output: RGB float32 tensor (B,3,H,W) in [0,1], on the input device.
        threshold: In [0,1], default .99.

    Returns:
        Per-image (B,) fraction [0,1], lower conditionally better.

    Method:
        EXPERIMENTS: mean(Y(output)>=threshold); report input clipping alongside output.

    Raises:
        NotImplementedError: Scaffold only; implement this contract before use.
    """
    raise NotImplementedError("TODO: Measure near-saturated luminance fraction")
