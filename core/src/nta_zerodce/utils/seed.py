"""Reproducible RNG setup; EXPERIMENTS seed protocol.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

_MAX_SEED = 2**32 - 1  # NumPy's legacy seeding accepts 32-bit unsigned seeds only.


def _check_seed(seed: int) -> None:
    """Reject non-integers, bools, negative and over-wide seeds."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an integer; got {seed!r}")
    if not 0 <= seed <= _MAX_SEED:
        raise ValueError(f"seed must be in [0, {_MAX_SEED}]; got {seed!r}")


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch random sources.

    Args:
        seed: Nonnegative integer (at most 2**32 - 1, the NumPy limit).
        deterministic: Request deterministic algorithms with explicit unsupported-op failures.

    Returns:
        None. Seeds ``random``, ``numpy`` and ``torch`` (CPU, and every CUDA device when CUDA is
        available). Sets cudnn benchmark off. When ``deterministic`` it also enables
        ``torch.use_deterministic_algorithms(True)`` (operations without a deterministic
        implementation raise instead of silently varying), turns on cudnn deterministic mode and,
        if CUDA is present, sets ``CUBLAS_WORKSPACE_CONFIG`` (kept if already set). When not
        deterministic those switches are explicitly turned off. DataLoader workers are seeded by
        passing ``worker_init_fn`` and a generator from ``make_generator``.

    Method:
        EXPERIMENTS [17,29,43]; determinism does not guarantee identical cross-device results.
        PYTHONHASHSEED is exported for child processes only: hash randomization of the running
        interpreter is fixed at startup and cannot be changed here.

    Raises:
        ValueError: The seed is not a nonnegative integer within range.
    """
    _check_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def worker_init_fn(worker_id: int) -> None:
    """Seed ``random`` and NumPy inside a DataLoader worker.

    Args:
        worker_id: Worker index supplied by ``torch.utils.data.DataLoader``.

    Returns:
        None. PyTorch already gives each worker the distinct seed ``base_seed + worker_id``
        (``base_seed`` is drawn from the loader's generator); this mirrors it into ``random`` and
        NumPy, so a seeded generator reproduces augmentation across runs.

    Method:
        Derive from ``torch.initial_seed()`` modulo 2**32 for NumPy compatibility.

    Raises:
        None.
    """
    seed = torch.initial_seed() % (_MAX_SEED + 1)
    random.seed(seed)
    np.random.seed(seed)


def make_generator(seed: int, device: torch.device | str = "cpu") -> torch.Generator:
    """Create a seeded ``torch.Generator`` (for DataLoader shuffling and sampling).

    Args:
        seed: Nonnegative integer.
        device: Device the generator lives on; default CPU.

    Returns:
        A new generator seeded with ``seed``; independent of the global RNG.

    Method:
        ``torch.Generator(device).manual_seed(seed)``.

    Raises:
        ValueError: The seed is invalid.
    """
    _check_seed(seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator
