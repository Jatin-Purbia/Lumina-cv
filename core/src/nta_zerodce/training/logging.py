"""Structured CSV experiment logs with a stable schema.

See docs/ARCHITECTURE.md reporting rules: train and eval columns are never silently mixed.
Each CSV file has one fixed column set (``step`` plus the metric names of its first record);
use separate files for training and validation records.
"""

from __future__ import annotations

import csv
import math
import operator
from pathlib import Path


def _check_step(step: int) -> int:
    """Return ``step`` as a plain int, rejecting bools and negative values."""
    if isinstance(step, bool):
        raise ValueError(f"step must be a nonnegative int; got {step!r}")
    try:
        value = operator.index(step)
    except TypeError:
        raise ValueError(f"step must be a nonnegative int; got {step!r}") from None
    if value < 0:
        raise ValueError(f"step must be a nonnegative int; got {step!r}")
    return value


def _header(path: Path) -> list[str]:
    """Header row of an existing non-empty CSV (empty list when the file has no content)."""
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return next(csv.reader(handle), [])


def log_metrics(path: Path, step: int, metrics: dict[str, float]) -> None:
    """Append one stable-schema CSV log record.

    The first call writes the header ``step`` plus the metric names in the given order and so
    fixes the columns of that file. Later calls may pass the metrics in any order; a column that
    is not passed is written as NaN (a missing evaluation), but a name that is not one of the
    file's columns is rejected so train and evaluation columns are never mixed silently.

    Args:
        path: Run-local CSV destination; the parent directory must exist.
        step: Nonnegative global step.
        metrics: Named scalar values; NaN allowed, infinities rejected.

    Returns:
        None.

    Raises:
        ValueError: Negative step, empty metrics, non-numeric or infinite values, an invalid
            column name, or a name that is not a column of an existing file.
        FileNotFoundError: The parent directory does not exist.
    """
    path = Path(path)
    step = _check_step(step)
    if not metrics:
        raise ValueError("metrics must not be empty")
    values: dict[str, float] = {}
    for name, value in metrics.items():
        if not name or name == "step" or "," in name or "\n" in name or '"' in name:
            raise ValueError(f"invalid metric name {name!r}")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"metric {name!r} is not a number: {value!r}") from None
        if math.isinf(number):
            raise ValueError(f"metric {name!r} is infinite")
        values[name] = number
    if not path.parent.is_dir():
        raise FileNotFoundError(f"log directory does not exist: {path.parent}")
    header = _header(path)
    if header:
        if header[0] != "step":
            raise ValueError(f"{path} is not a metrics log (first column {header[0]!r})")
        extra = [name for name in values if name not in header[1:]]
        if extra:
            raise ValueError(
                f"metric columns changed for {path}: {extra} are not among the file's "
                f"columns {header[1:]}; use a separate log file for a different column set"
            )
        columns = header[1:]
    else:
        columns = list(values)
    row = [repr(values.get(name, math.nan)) for name in columns]
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if not header:
            writer.writerow(["step", *columns])
        writer.writerow([step, *row])


def read_log(path: Path) -> list[dict[str, float]]:
    """Read a metrics CSV back as one ``{column: float}`` dict per row (``step`` included).

    Args:
        path: CSV written by ``log_metrics``.

    Returns:
        Rows in file order; ``step`` is returned as a float like every other column.

    Raises:
        FileNotFoundError: The file does not exist.
        ValueError: A row cannot be parsed.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"log not found: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def truncate_log(path: Path, max_step: int) -> int:
    """Drop every row whose step exceeds ``max_step`` (used when resuming from a checkpoint).

    Rows after the checkpoint belong to an interrupted epoch and would otherwise appear twice.
    A trailing row cut off mid-write is dropped as well. A missing file is left alone.

    Args:
        path: CSV written by ``log_metrics``.
        max_step: Last step to keep.

    Returns:
        Number of rows removed.
    """
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return 0
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    header, body = rows[0], rows[1:]
    kept = []
    for row in body:
        try:
            keep = len(row) == len(header) and int(row[0]) <= max_step
        except ValueError:
            keep = False
        if keep:
            kept.append(row)
    removed = len(body) - len(kept)
    if removed:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(kept)
    return removed
