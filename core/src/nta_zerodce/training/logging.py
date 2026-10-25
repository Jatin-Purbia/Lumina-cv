"""Structured CSV experiment logs with a stable schema.

See docs/EXPERIMENTS.md reporting rules: train and eval columns are never silently mixed.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path


def log_metrics(path: Path, step: int, metrics: dict[str, float]) -> None:
    """Append one stable-schema CSV log record.

    The first call writes the header ``step`` plus the metric names in the given order.
    Later calls must pass exactly the same metric names; a missing evaluation must be passed
    explicitly as ``float('nan')`` rather than omitted.

    Args:
        path: Run-local CSV destination; the parent directory must exist.
        step: Nonnegative global step.
        metrics: Named scalar values; NaN allowed, infinities rejected.

    Returns:
        None.

    Raises:
        ValueError: Negative step, empty metrics, infinite values, or a column-set change.
        FileNotFoundError: The parent directory does not exist.
    """
    path = Path(path)
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"step must be a nonnegative int; got {step!r}")
    if not metrics:
        raise ValueError("metrics must not be empty")
    for name, value in metrics.items():
        if not name or "," in name or "\n" in name:
            raise ValueError(f"invalid metric name {name!r}")
        if math.isinf(float(value)):
            raise ValueError(f"metric {name!r} is infinite")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"log directory does not exist: {path.parent}")
    columns = ["step", *metrics]
    exists = path.is_file() and path.stat().st_size > 0
    if exists:
        with path.open(encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle), [])
        if header != columns:
            raise ValueError(
                f"metric columns changed for {path}: file has {header}, call passes {columns}"
            )
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(columns)
        writer.writerow([step, *(repr(float(v)) for v in metrics.values())])
