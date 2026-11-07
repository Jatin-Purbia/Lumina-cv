"""DCE-Net architectures and iterative curves; ARCHITECTURE baseline path.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SeparableConv(nn.Module):
    """Depthwise 3x3 convolution followed by a pointwise 1x1 convolution (project helper).

    Same-size zero padding, stride one. Shared by the light curve estimator and the auxiliary
    heads; not a Zero-DCE++ reproduction.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Construct the depthwise/pointwise pair.

        Args:
            in_channels: Positive input width (depthwise groups).
            out_channels: Positive output width.
        """
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Apply depthwise then pointwise convolution; (B,in,H,W) -> (B,out,H,W)."""
        return self.pointwise(self.depthwise(x))


def _check_widths(in_channels: int, channels: int, iterations: int) -> None:
    if in_channels < 1 or channels < 1 or iterations < 1:
        raise ValueError(
            f"in_channels, channels and iterations must be positive, got "
            f"{in_channels}, {channels}, {iterations}"
        )


class DCENet(nn.Module):
    """Estimate eight RGB curve maps using seven symmetric skip convolutions.

    Seven 3x3 stride-one convolutions with same-size zero padding, ReLU hidden activations and
    a tanh head; no normalization or downsampling. Skips follow METHOD exactly:
    h5 sees concat(h3,h4), h6 sees concat(h2,h5) and the head sees concat(h1,h6).
    """

    def __init__(self, in_channels: int = 3, channels: int = 32, iterations: int = 8) -> None:
        """Construct the curve estimator.

        Args:
            in_channels: 3 for baseline or 4 for noise-aware RGB+sigma.
            channels: Positive hidden width, default 32.
            iterations: Number of updates; fixed to 8 in this project.

        Returns:
            No value; registers seven convolutions.

        Method:
            METHOD baseline architecture; 3x3 stride 1, symmetric concatenations, tanh head.

        Raises:
            ValueError: If a width or the iteration count is not positive.
        """
        super().__init__()
        _check_widths(in_channels, channels, iterations)
        self.in_channels = in_channels
        self.channels = channels
        self.iterations = iterations
        self.conv1 = nn.Conv2d(in_channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv4 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv5 = nn.Conv2d(2 * channels, channels, 3, padding=1)
        self.conv6 = nn.Conv2d(2 * channels, channels, 3, padding=1)
        self.head = nn.Conv2d(2 * channels, 3 * iterations, 3, padding=1)

    def forward(self, image: Tensor) -> Tensor:
        """Estimate eight RGB curve maps using seven symmetric skip convolutions.

        Args:
            image: (B,in_channels,H,W) float32; RGB in [0,1], optional sigma >=0.

        Returns:
            A: (B,24,H,W) float32 in [-1,1], iteration-major RGB ordering.

        Method:
            METHOD baseline network, with identical skips for the light variant.
        """
        h1 = torch.relu(self.conv1(image))
        h2 = torch.relu(self.conv2(h1))
        h3 = torch.relu(self.conv3(h2))
        h4 = torch.relu(self.conv4(h3))
        h5 = torch.relu(self.conv5(torch.cat([h3, h4], dim=1)))
        h6 = torch.relu(self.conv6(torch.cat([h2, h5], dim=1)))
        return torch.tanh(self.head(torch.cat([h1, h6], dim=1)))


class LightDCENet(nn.Module):
    """Estimate curves with a project depthwise-separable variant; not a Zero-DCE++ reproduction.

    Each hidden 3x3 convolution of DCENet becomes depthwise 3x3 then pointwise 1x1; widths and
    skips are identical and the final 3x3 tanh head stays a regular convolution.
    """

    def __init__(self, in_channels: int = 3, channels: int = 32, iterations: int = 8) -> None:
        """Construct the curve estimator.

        Args:
            in_channels: 3 for baseline or 4 for noise-aware RGB+sigma.
            channels: Positive hidden width, default 32.
            iterations: Number of updates; fixed to 8 in this project.

        Returns:
            No value; registers six separable convolutions and one regular head.

        Method:
            METHOD baseline architecture; 3x3 stride 1, symmetric concatenations, tanh head.

        Raises:
            ValueError: If a width or the iteration count is not positive.
        """
        super().__init__()
        _check_widths(in_channels, channels, iterations)
        self.in_channels = in_channels
        self.channels = channels
        self.iterations = iterations
        self.conv1 = SeparableConv(in_channels, channels)
        self.conv2 = SeparableConv(channels, channels)
        self.conv3 = SeparableConv(channels, channels)
        self.conv4 = SeparableConv(channels, channels)
        self.conv5 = SeparableConv(2 * channels, channels)
        self.conv6 = SeparableConv(2 * channels, channels)
        self.head = nn.Conv2d(2 * channels, 3 * iterations, 3, padding=1)

    def forward(self, image: Tensor) -> Tensor:
        """Estimate curves with a project depthwise-separable variant.

        Args:
            image: (B,in_channels,H,W) float32; RGB in [0,1], optional sigma >=0.

        Returns:
            A: (B,24,H,W) float32 in [-1,1], iteration-major RGB ordering.

        Method:
            METHOD baseline network, with identical skips for the light variant.
        """
        h1 = torch.relu(self.conv1(image))
        h2 = torch.relu(self.conv2(h1))
        h3 = torch.relu(self.conv3(h2))
        h4 = torch.relu(self.conv4(h3))
        h5 = torch.relu(self.conv5(torch.cat([h3, h4], dim=1)))
        h6 = torch.relu(self.conv6(torch.cat([h2, h5], dim=1)))
        return torch.tanh(self.head(torch.cat([h1, h6], dim=1)))


def apply_curves(image: Tensor, A: Tensor, iterations: int = 8) -> Tensor:
    """Apply iterative light-enhancement curves.

    Args:
        image: RGB float32 tensor (B,3,H,W) in [0,1]; no implicit device transfer.
        A: (B,3*iterations,H,W) in [-1,1], iteration-major RGB.
        iterations: Positive count, default 8.

    Returns:
        Enhanced RGB (B,3,H,W) in [0,1] on the same device.

    Method:
        Zero-DCE [R1], METHOD baseline: x_next = x + A*x*(1-x) with the plus-sign convention;
        iteration i uses channels 3i..3i+2 (R,G,B). The result is clamped to [0,1] only to
        absorb float rounding, since the map is mathematically bounded for |A|<=1.

    Raises:
        ValueError: If iterations is not positive or the shapes are inconsistent.
    """
    if iterations < 1:
        raise ValueError(f"iterations must be positive, got {iterations}")
    if image.dim() != 4 or image.shape[1] != 3:
        raise ValueError(f"image must be (B,3,H,W), got {tuple(image.shape)}")
    expected = (image.shape[0], 3 * iterations, image.shape[2], image.shape[3])
    if tuple(A.shape) != expected:
        raise ValueError(f"A must have shape {expected}, got {tuple(A.shape)}")
    x = image
    for i in range(iterations):
        x = x + A[:, 3 * i : 3 * i + 3] * x * (1.0 - x)
    return x.clamp(0.0, 1.0)
