"""Reproducible RNG setup; EXPERIMENTS seed protocol.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch random sources.

    Args:
        seed: Nonnegative integer.
        deterministic: Request deterministic algorithms with explicit unsupported-op failures.

    Returns:
        None; seeds Python, NumPy, CPU and available CUDA streams.

    Method:
        EXPERIMENTS [17,29,43]; determinism does not guarantee identical cross-device results.

    Raises:
        ValueError: Invalid inputs or incompatible configuration.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    torch.backends.cudnn.benchmark = not deterministic
