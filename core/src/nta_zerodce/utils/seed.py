"""Reproducible RNG setup; EXPERIMENTS seed protocol.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch random sources.

    Args:
        seed: Nonnegative integer.
        deterministic: Request deterministic algorithms with explicit unsupported-op failures.

    Returns:
        None; future implementation seeds CPU/CUDA and documents worker seeding.

    Method:
        EXPERIMENTS [17,29,43]; determinism does not guarantee identical cross-device results.

    Raises:
        NotImplementedError: Scaffold only; implement this contract before use.
    """
    raise NotImplementedError("TODO: Seed Python, NumPy and PyTorch random sources")
