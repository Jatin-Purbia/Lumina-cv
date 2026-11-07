"""Explicit device selection without hidden tensor transfers.

See docs/ARCHITECTURE.md for the normative contract.
"""

from __future__ import annotations

import re

import torch
from torch import device

_CUDA = re.compile(r"^cuda(?::([0-9]+))?$")


def resolve_device(requested: str = "auto") -> device:
    """Resolve a requested compute device.

    Args:
        requested: auto, cpu or cuda[:index]; explicit unavailable CUDA must fail.

    Returns:
        torch.device; auto chooses available CUDA else CPU.

    Method:
        INTERFACES device rules; OpenCV flow remains CPU then caller transfers. Strings are
        matched exactly (lowercase). No tensor is moved and no CUDA context is created here.

    Raises:
        ValueError: The string is not auto, cpu, cuda or cuda:N.
        RuntimeError: CUDA was requested explicitly but is unavailable, or the index is out of range.
    """
    if not isinstance(requested, str):
        raise ValueError(f"device must be a string; got {requested!r}")
    if requested == "auto":
        return device("cuda") if torch.cuda.is_available() else device("cpu")
    if requested == "cpu":
        return device("cpu")
    match = _CUDA.match(requested)
    if match is None:
        raise ValueError(f"invalid device {requested!r}; use 'auto', 'cpu', 'cuda' or 'cuda:N'")
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"device {requested!r} was requested but CUDA is not available; use 'cpu' or 'auto'"
        )
    if match.group(1) is not None:
        index = int(match.group(1))
        count = torch.cuda.device_count()
        if index >= count:
            raise RuntimeError(
                f"device {requested!r} is out of range; this machine has {count} CUDA device(s)"
            )
    return device(requested)
