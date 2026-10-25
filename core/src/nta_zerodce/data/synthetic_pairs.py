"""Affine still-image pairs with analytic flow and independent corruption.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from torch import Generator, Tensor

from nta_zerodce.data.noise_synth import check_generator, sample_noise
from nta_zerodce.training.config import Config


def _uniform(
    low: float, high: float, count: int, generator: Generator | None, device: torch.device
) -> Tensor:
    """Draw ``count`` float64 values uniformly in [low, high] on the host-side compute dtype."""
    draw = torch.rand(count, generator=generator, device=device, dtype=torch.float64)
    return low + (high - low) * draw


def _noisy(
    clean: Tensor, config: Config, generator: Generator | None
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Add per-image heteroscedastic noise to one frame.

    Coefficients a~U(a_min,a_max), b~U(b_min,b_max) are drawn per image; each image receives
    noise with probability ``config.noise.probability`` (otherwise a=b=0 is recorded and the
    frame passes through unchanged).

    Returns:
        (noisy image, sigma_target (B,1,H,W), a (B,), b (B,)).
    """
    batch = clean.shape[0]
    noise = config.noise
    device = clean.device
    a = _uniform(noise.a_min, noise.a_max, batch, generator, device).to(torch.float32)
    b = _uniform(noise.b_min, noise.b_max, batch, generator, device).to(torch.float32)
    applied = torch.rand(batch, generator=generator, device=device) < noise.probability
    a = torch.where(applied, a, torch.zeros_like(a))
    b = torch.where(applied, b, torch.zeros_like(b))
    noisy, sigma = sample_noise(clean, a.view(batch, 1, 1, 1), b.view(batch, 1, 1, 1), generator)
    return noisy, sigma, a, b


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
        Additive extras (documented, not part of the minimal contract):
        ``clean1`` / ``clean2`` (frames before noise; ``clean2`` includes flicker and is
        clamped to [0,1] before noise), ``sigma_target1`` / ``sigma_target2`` (B,1,H,W,
        pre-clipping noise targets), ``affine`` (B,2,3) forward map T and ``affine_inverse``
        (B,2,3) inverse map T^-1 in pixel coordinates (rows give x', y' from [x, y, 1]),
        ``flicker`` (B,), ``noise_a`` / ``noise_b`` (B,2) with column 0 the first frame and
        column 1 the second frame (zero where noise was not applied).

    Method:
        METHOD extension 5. Pixel coordinates p=(x,y), x in [0,W-1], y in [0,H-1], with
        centre c=((W-1)/2,(H-1)/2). T(p)=c+t+s*R(theta)(p-c), theta~U(+/-rotation_degrees),
        t~U(+/-translation_pixels) per axis, s~U(scale_min,scale_max). The second frame uses
        inverse mapping, image2(q)=image1(T^-1(q)), with bilinear sampling, zero padding and
        align_corners=True (grid_x=2*x/(W-1)-1). Forward flow f(p)=T(p)-p lives in image1
        coordinates; backward flow b(q)=T^-1(q)-q lives in image2 (current) coordinates; both
        are computed analytically and independently. The mask is 1 where T^-1(q) lies inside
        [0,W-1]x[0,H-1] (geometric validity in current coordinates); the real-pair photometric
        gate is deliberately not applied. Then image2<-clamp(flicker*image2) with flicker ~
        U(flicker_min,flicker_max), and independent noise is added to both frames through
        ``sample_noise``. Note the flicker product is clamped before noise (the noise std is
        then relative to the clamped frame); this differs from clamping once only where
        flicker*image2 exceeds 1.

    Raises:
        TypeError: If ``image`` or ``config`` has the wrong type.
        ValueError: On wrong shape/dtype/range, H or W below 2, an inverted config range, or a
            generator on a different device type than ``image``.
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"image must be a torch.Tensor, got {type(image).__name__}")
    if not isinstance(config, Config):
        raise TypeError(f"config must be a Config, got {type(config).__name__}")
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"image must have shape (B,3,H,W), got {tuple(image.shape)}")
    if image.dtype != torch.float32:
        raise ValueError(f"image must be float32, got {image.dtype}")
    batch, _, height, width = (int(v) for v in image.shape)
    if batch < 1 or height < 2 or width < 2:
        raise ValueError(f"need B>=1 and H,W>=2, got shape {tuple(image.shape)}")
    if not torch.isfinite(image).all() or bool((image < 0).any()) or bool((image > 1).any()):
        raise ValueError("image values must be finite and within [0,1]")
    syn = config.synthetic
    for name, low, high in (
        ("scale", syn.scale_min, syn.scale_max),
        ("flicker", syn.flicker_min, syn.flicker_max),
        ("noise a", config.noise.a_min, config.noise.a_max),
        ("noise b", config.noise.b_min, config.noise.b_max),
    ):
        if not low <= high:
            raise ValueError(f"{name} range is inverted: [{low}, {high}]")
    if syn.scale_min <= 0:
        raise ValueError("scale_min must be positive")
    if syn.rotation_degrees < 0 or syn.translation_pixels < 0:
        raise ValueError("rotation_degrees and translation_pixels must be nonnegative")
    device = image.device
    check_generator(generator, device)

    # Transform parameters (float64 for coordinate math), drawn per sample.
    theta = _uniform(-syn.rotation_degrees, syn.rotation_degrees, batch, generator, device)
    theta = theta * (math.pi / 180.0)
    shift_x = _uniform(-syn.translation_pixels, syn.translation_pixels, batch, generator, device)
    shift_y = _uniform(-syn.translation_pixels, syn.translation_pixels, batch, generator, device)
    scale = _uniform(syn.scale_min, syn.scale_max, batch, generator, device)
    flicker = _uniform(syn.flicker_min, syn.flicker_max, batch, generator, device)

    cos, sin = torch.cos(theta), torch.sin(theta)
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    # Forward map T(p) = c + t + s R (p - c) as a 2x3 matrix [sR | c + t - sR c].
    m00, m01 = scale * cos, -scale * sin
    m10, m11 = scale * sin, scale * cos
    fwd = torch.stack(
        [
            torch.stack([m00, m01, cx + shift_x - (m00 * cx + m01 * cy)], dim=1),
            torch.stack([m10, m11, cy + shift_y - (m10 * cx + m11 * cy)], dim=1),
        ],
        dim=1,
    )
    # Inverse map T^-1(q) = c + R(-theta)(q - c - t) / s.
    i00, i01 = cos / scale, sin / scale
    i10, i11 = -sin / scale, cos / scale
    tx_c, ty_c = cx + shift_x, cy + shift_y
    inv = torch.stack(
        [
            torch.stack([i00, i01, cx - (i00 * tx_c + i01 * ty_c)], dim=1),
            torch.stack([i10, i11, cy - (i10 * tx_c + i11 * ty_c)], dim=1),
        ],
        dim=1,
    )

    ys = torch.arange(height, device=device, dtype=torch.float64)
    xs = torch.arange(width, device=device, dtype=torch.float64)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=0).view(1, 3, -1)

    def apply(matrix: Tensor) -> Tensor:
        """Map the pixel grid through a (B,2,3) affine matrix; returns (B,2,H,W)."""
        return torch.matmul(matrix, coords.expand(batch, -1, -1)).view(batch, 2, height, width)

    base = coords[:, :2].view(1, 2, height, width)
    forward_target = apply(fwd)  # T(p) for p on the image1 grid
    source = apply(inv)  # T^-1(q) for q on the image2 grid
    forward_flow = forward_target - base
    backward_flow = source - base

    inside = (
        (source[:, 0] >= 0)
        & (source[:, 0] <= width - 1)
        & (source[:, 1] >= 0)
        & (source[:, 1] <= height - 1)
    )
    mask = inside.unsqueeze(1).to(torch.float32)

    # Inverse mapping: image2(q) = image1(T^-1(q)); bilinear, zero padding, align_corners=True.
    sample_grid = torch.stack(
        [2.0 * source[:, 0] / (width - 1) - 1.0, 2.0 * source[:, 1] / (height - 1) - 1.0], dim=-1
    ).to(torch.float32)
    warped = functional.grid_sample(
        image, sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    flicker32 = flicker.to(torch.float32)
    clean2 = torch.clamp(flicker32.view(batch, 1, 1, 1) * warped, 0.0, 1.0)

    noisy1, sigma1, a1, b1 = _noisy(image, config, generator)
    noisy2, sigma2, a2, b2 = _noisy(clean2, config, generator)

    return {
        "image1": noisy1,
        "image2": noisy2,
        "forward_flow": forward_flow.to(torch.float32),
        "backward_flow": backward_flow.to(torch.float32),
        "mask": mask,
        "clean1": image.clone(),
        "clean2": clean2,
        "sigma_target1": sigma1,
        "sigma_target2": sigma2,
        "affine": fwd.to(torch.float32),
        "affine_inverse": inv.to(torch.float32),
        "flicker": flicker32,
        "noise_a": torch.stack([a1, a2], dim=1),
        "noise_b": torch.stack([b1, b2], dim=1),
    }
