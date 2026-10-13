"""Predict one per-pixel noise standard deviation shared across RGB.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from torch import Tensor, nn


class NoiseBranch(nn.Module):
    """Predict one per-pixel noise standard deviation shared across RGB. See METHOD extension 1."""

    def __init__(self, channels: int = 16, sigma_max: float = 0.2) -> None:
        """Construct the auxiliary head.

        Args:
            channels: Positive hidden width.
            sigma_max: Positive bound in normalized RGB units.

        Returns:
            No value; future implementation registers convolutional layers.

        Method:
            METHOD extension 1: bounded sigmoid sigma; synthetic target sqrt(a*meanRGB(x)+b^2).

        Raises:
            NotImplementedError: Scaffold only; implement this contract before use.
        """
        raise NotImplementedError("TODO: Construct the auxiliary head")

    def forward(self, image: Tensor) -> Tensor:
        """Predict one per-pixel noise standard deviation shared across RGB.

        Args:
            image: RGB float32 tensor (B,3,H,W) in [0,1]; no implicit device transfer.

        Returns:
            sigma (B,1,H,W) in [0,sigma_max], same device.

        Method:
            METHOD extension 1: bounded sigmoid sigma; synthetic target sqrt(a*meanRGB(x)+b^2).

        Raises:
            NotImplementedError: Scaffold only; implement this contract before use.
        """
        raise NotImplementedError(
            "TODO: Predict one per-pixel noise standard deviation shared across RGB"
        )
