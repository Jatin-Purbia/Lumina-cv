"""Optional residual denoiser conditioned on estimated amplified noise.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from nta_zerodce.models.dce_net import SeparableConv


class ResidualDenoiser(nn.Module):
    """Remove a bounded residual conditioned on estimated amplified noise. See METHOD extension 1.

    Input is concat[z, sigma*gain] (four channels); two depthwise/pointwise blocks with ReLU
    and a regular 3x3 RGB residual head.
    """

    def __init__(self, channels: int = 16, residual_scale: float = 0.1) -> None:
        """Construct the auxiliary head.

        Args:
            channels: Positive hidden width.
            residual_scale: Positive bound in normalized RGB units.

        Returns:
            No value; registers two separable blocks and a regular residual head.

        Method:
            METHOD extension 1: clamp(z - residual_scale*tanh(head([z,sigma*gain])),0,1).

        Raises:
            ValueError: If channels or residual_scale is not positive.
        """
        super().__init__()
        if channels < 1 or not residual_scale > 0:
            raise ValueError(
                f"need positive channels and residual_scale, got {channels}, {residual_scale}"
            )
        self.residual_scale = float(residual_scale)
        self.block1 = SeparableConv(4, channels)
        self.block2 = SeparableConv(channels, channels)
        self.head = nn.Conv2d(channels, 3, 3, padding=1)

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
            The gain is detached here as a safeguard; sigma keeps its graph.
        """
        conditioned = torch.cat([enhanced_raw, sigma * gain.detach()], dim=1)
        h = torch.relu(self.block1(conditioned))
        h = torch.relu(self.block2(h))
        residual = self.residual_scale * torch.tanh(self.head(h))
        return (enhanced_raw - residual).clamp(0.0, 1.0)
