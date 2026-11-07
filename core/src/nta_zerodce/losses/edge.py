"""Pure tensor losses/targets; no IO, OpenCV, optimizer or registry side effects.

See docs/ARCHITECTURE.md for the normative contract.
"""

from __future__ import annotations

from torch import Tensor

from nta_zerodce.losses._ops import gaussian_blur_reflect, luminance, masked_mean, sobel_magnitude


def edge_loss(
    image: Tensor,
    output: Tensor,
    sigma: Tensor,
    gain: Tensor,
    smooth_sigma: float = 0.8,
    kernel_size: int = 5,
    noise_multiplier: float = 2.0,
    eps: float = 1e-06,
) -> Tensor:
    """Match noise-reliable gain-scaled input edges.

    Args:
        image: RGB float32 tensor (B,3,H,W) in [0,1]; no implicit device transfer.
        output: RGB float32 tensor (B,3,H,W) in [0,1], on the input device.
        sigma: Nonnegative float32 tensor (B,1,H,W), standard deviation in RGB units.
        gain: Detached (B,1,H,W) in [1,8].
        smooth_sigma: Positive Gaussian standard deviation in pixels.
        kernel_size: Odd Gaussian width, default 5.
        noise_multiplier: Positive edge reliability threshold, default 2.
        eps: Positive gradient/division guard.

    Returns:
        Scalar float32 nonnegative differentiable Tensor on the input device.

    Method:
        Extension 2. s=GaussianBlur(Y(x)) (reflect padding); D(v)=sqrt(Sx^2+Sy^2+eps^2)-eps with
        Sobel kernels divided by 8 and reflect padding. d_in=stopgrad(D(s)), d_out=D(Y(o)),
        w=stopgrad(clamp((d_in-noise_multiplier*sigma)/(d_in+eps),0,1)) and
        L=masked_mean_1(|d_out-stopgrad(gain*d_in)|, w). Sigma, gain, input and weights carry no
        gradient. See docs/ARCHITECTURE.md; extensions are project proposals.

    Raises:
        ValueError: If image and output shapes differ or sigma/gain are not (B,1,H,W).
    """
    if image.shape != output.shape:
        raise ValueError(f"image {tuple(image.shape)} and output {tuple(output.shape)} differ")
    expected = (image.shape[0], 1, image.shape[2], image.shape[3])
    if tuple(sigma.shape) != expected or tuple(gain.shape) != expected:
        raise ValueError(f"sigma and gain must have shape {expected}")
    smoothed = gaussian_blur_reflect(luminance(image.detach()), smooth_sigma, kernel_size)
    d_in = sobel_magnitude(smoothed, eps).detach()
    d_out = sobel_magnitude(luminance(output), eps)
    weights = ((d_in - noise_multiplier * sigma.detach()) / (d_in + eps)).clamp(0.0, 1.0).detach()
    reference = (gain.detach() * d_in).detach()
    return masked_mean((d_out - reference).abs(), weights, channels=1)
