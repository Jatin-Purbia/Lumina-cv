"""DCE-Net architectures and iterative curves; ARCHITECTURE baseline path.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from torch import Tensor, nn


class DCENet(nn.Module):
    """Estimate eight RGB curve maps using seven symmetric skip convolutions.

    Constructor and forward contracts are below; all methods are stubs.
    """

    def __init__(self, in_channels: int = 3, channels: int = 32, iterations: int = 8) -> None:
        """Construct the curve estimator.

        Args:
            in_channels: 3 for baseline or 4 for noise-aware RGB+sigma.
            channels: Positive hidden width, default 32.
            iterations: Number of updates; fixed to 8 in this project.

        Returns:
            No value; future implementation registers layers.

        Method:
            METHOD baseline architecture; 3x3 stride 1, symmetric concatenations, tanh head.

        Raises:
            NotImplementedError: Scaffold only; implement this contract before use.
        """
        raise NotImplementedError("TODO: Construct the curve estimator")

    def forward(self, image: Tensor) -> Tensor:
        """Estimate eight RGB curve maps using seven symmetric skip convolutions.

        Args:
            image: (B,in_channels,H,W) float32; RGB in [0,1], optional sigma >=0.

        Returns:
            A: (B,24,H,W) float32 in [-1,1], iteration-major RGB ordering.

        Method:
            METHOD baseline network, with identical skips for the light variant.

        Raises:
            NotImplementedError: Scaffold only; implement this contract before use.
        """
        raise NotImplementedError(
            "TODO: Estimate eight RGB curve maps using seven symmetric skip convolutions"
        )


class LightDCENet(nn.Module):
    """Estimate curves with a project depthwise-separable variant; not a Zero-DCE++ reproduction.

    Constructor and forward contracts are below; all methods are stubs.
    """

    def __init__(self, in_channels: int = 3, channels: int = 32, iterations: int = 8) -> None:
        """Construct the curve estimator.

        Args:
            in_channels: 3 for baseline or 4 for noise-aware RGB+sigma.
            channels: Positive hidden width, default 32.
            iterations: Number of updates; fixed to 8 in this project.

        Returns:
            No value; future implementation registers layers.

        Method:
            METHOD baseline architecture; 3x3 stride 1, symmetric concatenations, tanh head.

        Raises:
            NotImplementedError: Scaffold only; implement this contract before use.
        """
        raise NotImplementedError("TODO: Construct the curve estimator")

    def forward(self, image: Tensor) -> Tensor:
        """Estimate curves with a project depthwise-separable variant; not a Zero-DCE++ reproduction.

        Args:
            image: (B,in_channels,H,W) float32; RGB in [0,1], optional sigma >=0.

        Returns:
            A: (B,24,H,W) float32 in [-1,1], iteration-major RGB ordering.

        Method:
            METHOD baseline network, with identical skips for the light variant.

        Raises:
            NotImplementedError: Scaffold only; implement this contract before use.
        """
        raise NotImplementedError(
            "TODO: Estimate curves with a project depthwise-separable variant; not a Zero-DCE++ reproduction"
        )


def apply_curves(image: Tensor, A: Tensor, iterations: int = 8) -> Tensor:
    """Apply iterative light-enhancement curves.

    Args:
        image: RGB float32 tensor (B,3,H,W) in [0,1]; no implicit device transfer.
        A: (B,3*iterations,H,W) in [-1,1], iteration-major RGB.
        iterations: Positive count, default 8.

    Returns:
        Enhanced RGB (B,3,H,W) in [0,1] on the same device.

    Method:
        Zero-DCE [R1], METHOD baseline: x_next = x + A*x*(1-x).

    Raises:
        NotImplementedError: Scaffold only; implement this contract before use.
    """
    raise NotImplementedError("TODO: Apply iterative light-enhancement curves")
