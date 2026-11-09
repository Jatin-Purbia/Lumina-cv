"""Per-photo regression gates for checkpoint review, not perceptual quality scores."""

from __future__ import annotations

import math
from dataclasses import dataclass

from nta_zerodce.training.evaluation import Row

_OUTPUT_METRICS = ("mean_brightness", "contrast_std", "clipping_fraction")
_INPUT_METRICS = ("mean_brightness_input", "contrast_std_input")
_REQUIRED_METRICS = frozenset((*_OUTPUT_METRICS, *_INPUT_METRICS))


@dataclass(frozen=True)
class PhotoRegression:
    """One candidate sample exceeds a user-chosen change limit relative to the reference."""

    sample_id: str
    metric: str
    reference: float
    candidate: float
    change: float
    limit: float


def _samples(rows: list[Row], dataset: str) -> dict[str, dict[str, float]]:
    """Index a single evaluated dataset, rejecting missing, duplicate or invalid measurements."""
    samples: dict[str, dict[str, float]] = {}
    for row in rows:
        if row["dataset"] != dataset or row["metric"] not in _REQUIRED_METRICS:
            continue
        name, metric = str(row["sample_id"]), str(row["metric"])
        value = row["value"]
        if not isinstance(value, float | int) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{dataset}/{name}: {metric} must be finite and in [0,1]")
        sample = samples.setdefault(name, {})
        if metric in sample:
            raise ValueError(f"{dataset}/{name}: duplicate {metric}; use one checkpoint per CSV")
        sample[metric] = float(value)
    if not samples:
        raise ValueError(f"no photo measurements for dataset {dataset!r}; re-run evaluation")
    for name, sample in samples.items():
        missing = _REQUIRED_METRICS - sample.keys()
        if missing:
            raise ValueError(f"{dataset}/{name}: missing {sorted(missing)}; re-run evaluation")
    return samples


def photo_regressions(
    reference_rows: list[Row],
    candidate_rows: list[Row],
    *,
    dataset: str,
    max_brightness_increase: float,
    max_contrast_drop: float,
    max_clipping_increase: float,
) -> list[PhotoRegression]:
    """Flag per-image drift relative to a reference checkpoint on the same held-out photos.

    Limits are explicit absolute intensity/fraction changes in [0,1], chosen before viewing
    candidate results. They are operational guardrails, not universal quality thresholds.
    Passing means only that these three regressions were not found; it does not rank models.
    Use the same image manifest for both evaluations. Input statistics must also match, as a
    useful mismatch check (equal statistics do not prove identical image content).
    """
    limits = (max_brightness_increase, max_contrast_drop, max_clipping_increase)
    if not dataset.strip():
        raise ValueError("dataset must not be empty")
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in limits):
        raise ValueError("regression limits must be finite absolute changes in [0,1]")
    reference, candidate = _samples(reference_rows, dataset), _samples(candidate_rows, dataset)
    if reference.keys() != candidate.keys():
        raise ValueError("reference and candidate must contain the same held-out sample IDs")
    failures: list[PhotoRegression] = []
    for name in sorted(reference):
        before, after = reference[name], candidate[name]
        for metric in _INPUT_METRICS:
            if not math.isclose(before[metric], after[metric], rel_tol=0, abs_tol=1e-6):
                raise ValueError(f"{dataset}/{name}: {metric} differs; evaluate the same inputs")
        for metric, limit, direction in zip(_OUTPUT_METRICS, limits, (1, -1, 1), strict=True):
            change = direction * (after[metric] - before[metric])
            if change > limit + 1e-7:
                failures.append(
                    PhotoRegression(name, metric, before[metric], after[metric], change, limit)
                )
    return failures
