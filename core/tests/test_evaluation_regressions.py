"""Expose brightening regressions and keep summary denominators truthful."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest
import torch

from nta_zerodce.training.config import load_config
from nta_zerodce.training.evaluation import _image_rows, aggregate, format_aggregate
from nta_zerodce.training.selection import photo_regressions


def test_aggregate_reports_only_the_samples_used_in_the_mean() -> None:
    rows = [
        {"dataset": "unpaired", "metric": "noise_estimate", "value": value}
        for value in (0.01, math.nan, 0.03, math.inf, -math.inf)
    ]
    result = aggregate(rows)
    assert len(result) == 1
    assert result[0].mean == pytest.approx(0.02)
    assert result[0].count == 2
    assert result[0].excluded_count == 3
    assert "excluded" in format_aggregate(result)


def test_aggregate_without_finite_samples_has_zero_count() -> None:
    result = aggregate([{"dataset": "paired", "metric": "psnr", "value": math.inf}])[0]
    assert math.isnan(result.mean)
    assert result.count == 0
    assert result.excluded_count == 1


def test_evaluation_records_brightness_and_contrast_loss_on_ordinary_photos() -> None:
    original = torch.zeros(1, 3, 16, 16)
    original[:, :, :, 8:] = 0.2
    washed_out = torch.full_like(original, 0.6)
    rows = _image_rows("regression", 17, "unpaired/ordinary", "scene", original, washed_out, None)
    metrics = {str(row["metric"]): row["value"] for row in rows}
    assert metrics["mean_brightness_input"] == pytest.approx(0.1)
    assert metrics["mean_brightness"] == pytest.approx(0.6)
    assert metrics["contrast_std_input"] == pytest.approx(0.1)
    assert metrics["contrast_std"] == pytest.approx(0.0, abs=1e-6)
    # This regression is invisible to clipping alone: no output pixel is near white.
    assert metrics["clipping_fraction"] == 0.0


def test_photo_preset_keeps_architecture_and_uses_a_short_inspectable_schedule() -> None:
    configs = Path(__file__).resolve().parents[1] / "configs"
    photo, full = load_config(configs / "photo.py"), load_config(configs / "full.py")
    assert photo.model == full.model
    assert photo.loss == dataclasses.replace(full.loss, exposure_target=0.6)
    assert photo.weights == full.weights
    assert photo.flags == dataclasses.replace(full.flags, temporal=False)
    assert photo.training.epochs == 10
    assert photo.training.checkpoint_every == 1
    assert photo.training == dataclasses.replace(full.training, epochs=10, checkpoint_every=1)
    target05 = load_config(configs / "photo_target05.py")
    assert target05 == dataclasses.replace(
        photo, loss=dataclasses.replace(photo.loss, exposure_target=0.5)
    )


def _photo_rows(brightness: float, contrast: float, clipping: float) -> list[dict[str, object]]:
    return [
        {"dataset": "unpaired/ordinary", "sample_id": "scene", "metric": metric, "value": value}
        for metric, value in (
            ("mean_brightness", brightness),
            ("contrast_std", contrast),
            ("clipping_fraction", clipping),
            ("mean_brightness_input", 0.3),
            ("contrast_std_input", 0.2),
        )
    ]


def test_photo_gate_rejects_washing_out_even_without_clipped_pixels() -> None:
    reference = _photo_rows(0.4, 0.2, 0.0)
    candidate = _photo_rows(0.65, 0.1, 0.0)
    failures = photo_regressions(
        reference,
        candidate,
        dataset="unpaired/ordinary",
        max_brightness_increase=0.05,
        max_contrast_drop=0.02,
        max_clipping_increase=0.01,
    )
    assert {f.metric for f in failures} == {"mean_brightness", "contrast_std"}
    assert failures[0].change == pytest.approx(0.25)


def test_photo_gate_accepts_boundary_changes_and_unchanged_reference() -> None:
    limits = {
        "max_brightness_increase": 0.05,
        "max_contrast_drop": 0.02,
        "max_clipping_increase": 0.01,
    }
    reference = _photo_rows(0.4, 0.2, 0.0)
    assert not photo_regressions(reference, reference, dataset="unpaired/ordinary", **limits)
    assert not photo_regressions(
        reference, _photo_rows(0.45, 0.18, 0.01), dataset="unpaired/ordinary", **limits
    )


@pytest.mark.parametrize("problem", ["missing", "duplicate", "nonfinite", "different_input"])
def test_photo_gate_rejects_incomplete_or_incomparable_measurements(problem: str) -> None:
    reference, candidate = _photo_rows(0.4, 0.2, 0.0), _photo_rows(0.4, 0.2, 0.0)
    if problem == "missing":
        candidate.pop()
    elif problem == "duplicate":
        candidate.append(candidate[0].copy())
    elif problem == "nonfinite":
        candidate[0]["value"] = math.nan
    else:
        candidate[-1]["value"] = 0.25
    with pytest.raises(ValueError):
        photo_regressions(
            reference,
            candidate,
            dataset="unpaired/ordinary",
            max_brightness_increase=0.05,
            max_contrast_drop=0.02,
            max_clipping_increase=0.01,
        )
