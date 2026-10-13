"""Structured CSV experiment records; EXPERIMENTS table fields.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from pathlib import Path


def log_metrics(path: Path, step: int, metrics: dict[str, float]) -> None:
    """Append one stable-schema CSV log record.

    Args:
        path: Run-local CSV destination.
        step: Nonnegative global step.
        metrics: Stable named finite scalar values; missing evaluations use NaN explicitly.

    Returns:
        None; include run/seed through run directory and resolved config.

    Method:
        EXPERIMENTS reporting; do not silently mix train/eval columns.

    Raises:
        NotImplementedError: Scaffold only; implement this contract before use.
    """
    raise NotImplementedError("TODO: Append one stable-schema CSV log record")
