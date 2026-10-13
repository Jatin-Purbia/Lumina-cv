"""Pure tensor losses/targets; no IO, OpenCV, optimizer or registry side effects.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from torch import Tensor


def highlight_loss(image: Tensor, output: Tensor, threshold: float = 0.8) -> Tensor:
    """Protect already bright input pixels.

    Args:
        image: RGB float32 tensor (B,3,H,W) in [0,1]; no implicit device transfer.
        output: RGB float32 tensor (B,3,H,W) in [0,1], on the input device.
        threshold: Input luminance threshold in [0,1], default 0.8.

    Returns:
        Scalar float32 nonnegative differentiable Tensor on the input device.

    Method:
        Extension 4; masked positive luminance increase squared; empty mask returns graph-connected zero. See docs/METHOD.md; baseline attribution [R1,R2], extensions are project proposals.

    Raises:
        NotImplementedError: Scaffold only; implement this contract before use.
    """
    raise NotImplementedError("TODO: Protect already bright input pixels")
