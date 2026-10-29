"""DCE-Net architectures and iterative curves; ARCHITECTURE baseline path.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn


class DCENet(nn.Module):
    """Estimate eight RGB curve maps using seven symmetric skip convolutions.

    Constructor and forward contracts are below; methods implement the documented tensor contract.
    """

    def __init__(self, in_channels: int = 3, channels: int = 32, iterations: int = 8) -> None:
        """Construct the curve estimator.

        Args:
            in_channels: 3 for baseline or 4 for noise-aware RGB+sigma.
            channels: Positive hidden width, default 32.
            iterations: Number of updates; fixed to 8 in this project.

        Returns:
            No value; registers layers.

        Method:
            METHOD baseline architecture; 3x3 stride 1, symmetric concatenations, tanh head.

        Raises:
            ValueError: Invalid inputs or incompatible configuration.
        """
        super().__init__()
        if in_channels not in (3, 4) or channels <= 0 or iterations != 8:
            raise ValueError("Expected 3/4 input channels, positive width and eight iterations")
        widths = [
            in_channels,
            channels,
            channels,
            channels,
            channels * 2,
            channels * 2,
            channels * 2,
        ]
        self.layers = nn.ModuleList(
            nn.Conv2d(w, channels if i < 6 else 24, 3, padding=1) for i, w in enumerate(widths)
        )

    def forward(self, image: Tensor) -> Tensor:
        """Estimate eight RGB curve maps using seven symmetric skip convolutions.

        Args:
            image: (B,in_channels,H,W) float32; RGB in [0,1], optional sigma >=0.

        Returns:
            A: (B,24,H,W) float32 in [-1,1], iteration-major RGB ordering.

        Method:
            METHOD baseline network, with identical skips for the light variant.

        Raises:
            ValueError: Invalid inputs or incompatible configuration.
        """
        h1 = self.layers[0](image).relu()
        h2 = self.layers[1](h1).relu()
        h3 = self.layers[2](h2).relu()
        h4 = self.layers[3](h3).relu()
        h5 = self.layers[4](torch.cat((h3, h4), 1)).relu()
        h6 = self.layers[5](torch.cat((h2, h5), 1)).relu()
        return cast(Tensor, self.layers[6](torch.cat((h1, h6), 1))).tanh()


class LightDCENet(DCENet):
    """Estimate curves with a project depthwise-separable variant; not a Zero-DCE++ reproduction.

    Constructor and forward contracts are below; methods implement the documented tensor contract.
    """

    def __init__(self, in_channels: int = 3, channels: int = 32, iterations: int = 8) -> None:
        """Construct the curve estimator.

        Args:
            in_channels: 3 for baseline or 4 for noise-aware RGB+sigma.
            channels: Positive hidden width, default 32.
            iterations: Number of updates; fixed to 8 in this project.

        Returns:
            No value; registers layers.

        Method:
            METHOD baseline architecture; 3x3 stride 1, symmetric concatenations, tanh head.

        Raises:
            ValueError: Invalid inputs or incompatible configuration.
        """
        nn.Module.__init__(self)
        if in_channels not in (3, 4) or channels <= 0 or iterations != 8:
            raise ValueError("Invalid estimator dimensions")
        widths = [in_channels, channels, channels, channels, channels * 2, channels * 2]
        self.layers = nn.ModuleList(
            nn.Sequential(nn.Conv2d(w, w, 3, padding=1, groups=w), nn.Conv2d(w, channels, 1))
            for w in widths
        )
        self.layers.append(nn.Conv2d(channels * 2, 24, 3, padding=1))

    def forward(self, image: Tensor) -> Tensor:
        """Estimate curves with a project depthwise-separable variant; not a Zero-DCE++ reproduction.

        Args:
            image: (B,in_channels,H,W) float32; RGB in [0,1], optional sigma >=0.

        Returns:
            A: (B,24,H,W) float32 in [-1,1], iteration-major RGB ordering.

        Method:
            METHOD baseline network, with identical skips for the light variant.

        Raises:
            ValueError: Invalid inputs or incompatible configuration.
        """
        return super().forward(image)


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
        ValueError: Invalid inputs or incompatible configuration.
    """
    if (
        iterations < 1
        or image.ndim != 4
        or image.shape[1] != 3
        or A.shape != (image.shape[0], 3 * iterations, *image.shape[2:])
    ):
        raise ValueError("Expected RGB BCHW and iteration-major matching curve maps")
    if image.device != A.device or image.dtype != A.dtype:
        raise ValueError("Curve and image device/dtype must match")
    result = image
    for index in range(iterations):
        curve = A[:, 3 * index : 3 * (index + 1)]
        result = result + curve * result * (1 - result)
    return result
