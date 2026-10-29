"""Predict one per-pixel noise standard deviation shared across RGB.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from typing import cast

from torch import Tensor, nn


class NoiseBranch(nn.Module):
    """Predict one per-pixel noise standard deviation shared across RGB. See METHOD extension 1."""

    def __init__(self, channels: int = 16, sigma_max: float = 0.2) -> None:
        """Construct the auxiliary head.

        Args:
            channels: Positive hidden width.
            sigma_max: Positive bound in normalized RGB units.

        Returns:
            No value; registers convolutional layers.

        Method:
            METHOD extension 1: bounded sigmoid sigma; synthetic target sqrt(a*meanRGB(x)+b^2).

        Raises:
            ValueError: Invalid inputs or incompatible configuration.
        """
        super().__init__()
        if channels < 1 or not 0 < sigma_max <= 1:
            raise ValueError("Invalid noise head dimensions or bound")
        self.sigma_max = sigma_max
        self.net = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, 1, 3, padding=1),
        )

    def forward(self, image: Tensor) -> Tensor:
        """Predict one per-pixel noise standard deviation shared across RGB.

        Args:
            image: RGB float32 tensor (B,3,H,W) in [0,1]; no implicit device transfer.

        Returns:
            sigma (B,1,H,W) in [0,sigma_max], same device.

        Method:
            METHOD extension 1: bounded sigmoid sigma; synthetic target sqrt(a*meanRGB(x)+b^2).

        Raises:
            ValueError: Invalid inputs or incompatible configuration.
        """
        return cast(Tensor, self.net(image)).sigmoid() * self.sigma_max
