"""Remove a bounded residual conditioned on estimated amplified noise.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from torch import Tensor, nn


class ResidualDenoiser(nn.Module):
    """Remove a bounded residual conditioned on estimated amplified noise. See METHOD extension 1."""

    def __init__(self, channels: int = 16, residual_scale: float = 0.1) -> None:
        """Construct the auxiliary head.

        Args:
            channels: Positive hidden width.
            residual_scale: Positive bound in normalized RGB units.

        Returns:
            No value; future implementation registers convolutional layers.

        Method:
            METHOD extension 1: clamp(z - residual_scale*tanh(head([z,sigma*gain])),0,1).

        Raises:
            NotImplementedError: Scaffold only; implement this contract before use.
        """
        raise NotImplementedError("TODO: Construct the auxiliary head")

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
            NotImplementedError: Scaffold only; implement this contract before use.
        """
        raise NotImplementedError(
            "TODO: Remove a bounded residual conditioned on estimated amplified noise"
        )
