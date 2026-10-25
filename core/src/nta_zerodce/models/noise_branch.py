"""Auxiliary per-pixel noise-standard-deviation head.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class NoiseBranch(nn.Module):
    """Predict one per-pixel noise standard deviation shared across RGB. See METHOD extension 1.

    RGB -> 16 -> 16 -> 1 with 3x3 convolutions, hidden ReLU and a bounded sigmoid output
    scaled by sigma_max.
    """

    def __init__(self, channels: int = 16, sigma_max: float = 0.2) -> None:
        """Construct the auxiliary head.

        Args:
            channels: Positive hidden width.
            sigma_max: Positive bound in normalized RGB units.

        Returns:
            No value; registers three convolutional layers.

        Method:
            METHOD extension 1: bounded sigmoid sigma; synthetic target sqrt(a*meanRGB(x)+b^2).

        Raises:
            ValueError: If channels or sigma_max is not positive.
        """
        super().__init__()
        if channels < 1 or not sigma_max > 0:
            raise ValueError(f"need positive channels and sigma_max, got {channels}, {sigma_max}")
        self.sigma_max = float(sigma_max)
        self.conv1 = nn.Conv2d(3, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels, 1, 3, padding=1)

    def forward(self, image: Tensor) -> Tensor:
        """Predict one per-pixel noise standard deviation shared across RGB.

        Args:
            image: RGB float32 tensor (B,3,H,W) in [0,1]; no implicit device transfer.

        Returns:
            sigma (B,1,H,W) in [0,sigma_max], same device.

        Method:
            METHOD extension 1: sigma = sigma_max * sigmoid(head(x)).
        """
        h = torch.relu(self.conv1(image))
        h = torch.relu(self.conv2(h))
        return self.sigma_max * torch.sigmoid(self.conv3(h))
