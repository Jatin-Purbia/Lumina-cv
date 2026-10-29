"""Structured CSV experiment records; EXPERIMENTS table fields.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

import csv
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
        ValueError: Invalid inputs or incompatible configuration.
    """
    names = ["step", *sorted(metrics)]
    if "step" in metrics:
        raise ValueError("step is reserved")
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(newline="", encoding="utf-8") as handle:
            if next(csv.reader(handle)) != names:
                raise ValueError("CSV metric schema changed")
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        if not exists:
            writer.writeheader()
        writer.writerow({"step": step, **metrics})
