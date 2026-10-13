"""Explicit device selection without hidden tensor transfers.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

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
        NotImplementedError: Scaffold only; implement this contract before use.
    """
    raise NotImplementedError("TODO: Resolve a requested compute device")
