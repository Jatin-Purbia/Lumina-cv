"""Pure tensor losses/targets; no IO, OpenCV, optimizer or registry side effects.

See docs/ARCHITECTURE.md for the normative contract.
"""

from __future__ import annotations

from torch import Tensor

from nta_zerodce.losses._ops import avg_pool_reflect, luminance, masked_mean, smoothstep


def chroma_loss(
    image: Tensor,
    output: Tensor,
    sigma: Tensor,
    window: int = 5,
    eps: float = 1e-06,
    reliability_low: float = 0.05,
    reliability_high: float = 0.2,
    reliability_noise: float = 2.0,
) -> Tensor:
    """Preserve reliable local chromaticity.

    Args:
        image: RGB float32 tensor (B,3,H,W) in [0,1]; no implicit device transfer.
        output: RGB float32 tensor (B,3,H,W) in [0,1], on the input device.
        sigma: Nonnegative float32 tensor (B,1,H,W), standard deviation in RGB units.
        window: Odd local averaging width, default 5.
        eps: Positive division guard.
        reliability_low/high: Luminance ramp endpoints, low<high.
        reliability_noise: Sigma multiplier subtracted from input luminance.

    Returns:
        Scalar float32 nonnegative differentiable Tensor on the input device.

    Method:
        Extension 3; c(v)=v/(sum_RGB(v)+eps). The reference chromaticity is the stride-1,
        reflection-padded window average of c(x), taken after normalization and detached; the
        weights are smoothstep(low,high,Y(x)-noise*sigma), also detached. Loss is
        masked_mean_3(|c(o)-c_ref|, w) and is exactly zero (graph-connected) with no reliable
        pixels. See docs/ARCHITECTURE.md; extensions are project proposals.

    Raises:
        ValueError: If image and output shapes differ or sigma does not match their spatial size.
    """
    if image.shape != output.shape:
        raise ValueError(f"image {tuple(image.shape)} and output {tuple(output.shape)} differ")
    if sigma.shape[0] != image.shape[0] or sigma.shape[-2:] != image.shape[-2:]:
        raise ValueError(f"sigma {tuple(sigma.shape)} does not match image {tuple(image.shape)}")
    x = image.detach()
    c_in = x / (x.sum(dim=1, keepdim=True) + eps)
    c_ref = avg_pool_reflect(c_in, window).detach()
    weights = smoothstep(
        reliability_low,
        reliability_high,
        luminance(x) - reliability_noise * sigma.detach(),
    ).detach()
    c_out = output / (output.sum(dim=1, keepdim=True) + eps)
    return masked_mean((c_out - c_ref).abs(), weights, channels=3)
