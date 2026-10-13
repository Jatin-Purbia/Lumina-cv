"""Heteroscedastic synthetic corruption; DATA generation protocol.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from torch import Generator, Tensor


def sample_noise(
    image: Tensor, a: Tensor, b: Tensor, generator: Generator | None = None
) -> tuple[Tensor, Tensor]:
    """Synthesize Gaussian-approximated Poisson-Gaussian noise.

    Args:
        image: RGB float32 tensor (B,3,H,W) in [0,1]; no implicit device transfer.
        a/b: Nonnegative (B,1,1,1); b is read-noise std, not variance.
        generator: Device-matched RNG or None; do not reset global seed.

    Returns:
        (clipped noisy RGB, sigma_target (B,1,H,W)); targets are pre-clipping.

    Method:
        METHOD extension 1: n=sqrt(a*x+b^2)*epsilon; sigma_target=sqrt(a*meanRGB(x)+b^2).

    Raises:
        NotImplementedError: Scaffold only; implement this contract before use.
    """
    raise NotImplementedError("TODO: Synthesize Gaussian-approximated Poisson-Gaussian noise")
