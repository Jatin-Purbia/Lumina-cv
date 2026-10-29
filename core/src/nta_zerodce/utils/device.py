"""Explicit device selection without hidden tensor transfers.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

import torch
from torch import device


def resolve_device(requested: str = "auto") -> device:
    """Resolve a requested compute device.

    Args:
        requested: auto, cpu or cuda[:index]; explicit unavailable CUDA must fail.

    Returns:
        torch.device; auto chooses available CUDA else CPU.

    Method:
        INTERFACES device rules; OpenCV flow remains CPU then caller transfers.

    Raises:
        ValueError: Invalid inputs or incompatible configuration.
    """
    if requested == "auto":
        requested = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    result = device(requested)
    if result.type == "cuda":
        if not torch.cuda.is_available() or (
            result.index is not None and result.index >= torch.cuda.device_count()
        ):
            raise ValueError(f"CUDA device unavailable: {requested}")
    elif result.type == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS unavailable")
    elif result.type != "cpu":
        raise ValueError(f"Unsupported device: {requested}")
    return result
