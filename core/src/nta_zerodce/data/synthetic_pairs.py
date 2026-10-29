"""Affine still-image pairs with analytic flow and independent corruption.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

import math

import torch
from torch import Generator, Tensor

from nta_zerodce.data.noise_synth import sample_noise
from nta_zerodce.flow.warp import warp
from nta_zerodce.training.config import Config


def make_pair(
    image: Tensor, config: Config, generator: Generator | None = None
) -> dict[str, Tensor]:
    """Create synthetic temporal supervision.

    Args:
        image: RGB float32 tensor (B,3,H,W) in [0,1]; no implicit device transfer.
        config: Validated synthetic/noise/flow constants.
        generator: Device-matched RNG; independent draws for each frame.

    Returns:
        image1/image2 RGB (B,3,H,W); forward/backward_flow (B,2,H,W); mask (B,1,H,W).

    Method:
        METHOD extension 5: affine inverse coordinate maps, exposure flicker and independent noise; DATA generation.

    Raises:
        ValueError: Invalid inputs or incompatible configuration.
    """
    config.validate()
    batch, _, h, w = image.shape
    if min(h, w) < 2:
        raise ValueError("Synthetic pairs require H,W >1")
    s = config.synthetic

    def uniform(low: float, high: float) -> Tensor:
        return (
            torch.rand(batch, device=image.device, dtype=image.dtype, generator=generator)
            * (high - low)
            + low
        )

    angle = uniform(-s.rotation_degrees, s.rotation_degrees) * math.pi / 180
    scale = uniform(s.scale_min, s.scale_max)
    matrix = torch.zeros(batch, 3, 3, device=image.device, dtype=image.dtype)
    matrix[:, 0, 0] = matrix[:, 1, 1] = scale * angle.cos()
    matrix[:, 0, 1] = -scale * angle.sin()
    matrix[:, 1, 0] = scale * angle.sin()
    matrix[:, 2, 2] = 1
    center = image.new_tensor([(w - 1) / 2, (h - 1) / 2])
    translations = torch.stack(
        (
            uniform(-s.translation_pixels, s.translation_pixels),
            uniform(-s.translation_pixels, s.translation_pixels),
        ),
        1,
    )
    matrix[:, :2, 2] = center - torch.einsum("bij,j->bi", matrix[:, :2, :2], center) + translations
    yy, xx = torch.meshgrid(
        torch.arange(h, device=image.device, dtype=image.dtype),
        torch.arange(w, device=image.device, dtype=image.dtype),
        indexing="ij",
    )
    points = torch.stack((xx, yy, torch.ones_like(xx))).reshape(1, 3, -1).expand(batch, -1, -1)
    forward = (matrix.bmm(points)[:, :2] - points[:, :2]).reshape(batch, 2, h, w)
    backward = (torch.linalg.inv(matrix).bmm(points)[:, :2] - points[:, :2]).reshape(batch, 2, h, w)
    second, bounds = warp(image, backward)
    warped_f, _ = warp(forward, backward)
    consistent = (backward + warped_f).square().sum(1, keepdim=True) <= config.flow.fb_alpha * (
        backward.square().sum(1, keepdim=True) + warped_f.square().sum(1, keepdim=True)
    ) + config.flow.fb_beta
    flicker = uniform(s.flicker_min, s.flicker_max)
    second = (second * flicker[:, None, None, None]).clamp(0, 1)
    a = uniform(config.noise.a_min, config.noise.a_max)
    b = uniform(config.noise.b_min, config.noise.b_max)
    active = uniform(0, 1) < config.noise.probability
    second, _ = sample_noise(second, a * active, b * active, generator)
    first_a = uniform(config.noise.a_min, config.noise.a_max)
    first_b = uniform(config.noise.b_min, config.noise.b_max)
    first_active = uniform(0, 1) < config.noise.probability
    first, _ = sample_noise(image, first_a * first_active, first_b * first_active, generator)
    return {
        "image1": first,
        "image2": second,
        "forward_flow": forward.detach(),
        "backward_flow": backward.detach(),
        "mask": (bounds * consistent).detach(),
        "affine": matrix.detach(),
        "flicker": flicker.detach(),
        "noise_a": (a * active).detach(),
        "noise_b": (b * active).detach(),
        "first_noise_a": (first_a * first_active).detach(),
        "first_noise_b": (first_b * first_active).detach(),
    }
