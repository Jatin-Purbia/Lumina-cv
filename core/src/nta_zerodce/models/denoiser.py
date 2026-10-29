"""Remove a bounded residual conditioned on estimated amplified noise.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn


class ResidualDenoiser(nn.Module):
    """Remove a bounded residual conditioned on estimated amplified noise. See METHOD extension 1."""

    def __init__(self, channels: int = 16, residual_scale: float = 0.1) -> None:
        """Construct the auxiliary head.

        Args:
            channels: Positive hidden width.
            residual_scale: Positive bound in normalized RGB units.

        Returns:
            No value; registers convolutional layers.

        Method:
            METHOD extension 1: clamp(z - residual_scale*tanh(head([z,sigma*gain])),0,1).

        Raises:
            ValueError: Invalid inputs or incompatible configuration.
        """
        super().__init__()
        if channels < 1 or not 0 < residual_scale <= 1:
            raise ValueError("Invalid denoiser width or residual bound")
        self.residual_scale = residual_scale
        self.net = nn.Sequential(
            nn.Conv2d(4, 4, 3, padding=1, groups=4),
            nn.Conv2d(4, channels, 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(),
            nn.Conv2d(channels, 3, 3, padding=1),
        )

    def forward(self, enhanced_raw: Tensor, sigma: Tensor, gain: Tensor) -> Tensor:
        """Remove a bounded residual conditioned on estimated amplified noise.

        Args:
            enhanced_raw: RGB (B,3,H,W) in [0,1].
            sigma: Nonnegative float32 tensor (B,1,H,W), standard deviation in RGB units.
            gain: Detached (B,1,H,W) in [1,8].

        Returns:
            enhanced (B,3,H,W) in [0,1].

        Method:
            METHOD extension 1: clamp(z - residual_scale*tanh(head([z,sigma*gain])),0,1).

        Raises:
            ValueError: Invalid inputs or incompatible configuration.
        """
        residual = cast(
            Tensor, self.net(torch.cat((enhanced_raw, sigma * gain.detach()), dim=1))
        ).tanh()
        return (enhanced_raw - self.residual_scale * residual).clamp(0, 1)
