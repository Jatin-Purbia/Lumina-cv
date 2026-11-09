"""Inference benchmarks must preserve checkpoints and report drift from the same model."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from nta_zerodce.data.image_dataset import make_synthetic_images
from nta_zerodce.models.factory import build_model
from nta_zerodce.training.checkpoint import save_checkpoint
from nta_zerodce.training.config import Config, override_config

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_inference.py"
spec = importlib.util.spec_from_file_location("benchmark_inference", SCRIPT)
assert spec is not None and spec.loader is not None
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_benchmark_records_reference_drift_and_preserves_weights(tmp_path: Path) -> None:
    config = override_config(Config(), {"model.channels": 4})
    model = build_model(config)
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, model, torch.optim.Adam(model.parameters()), config, 0)
    original = checkpoint.read_bytes()
    images = tmp_path / "images"
    make_synthetic_images(images, 2, (32, 48), 17)
    destination = tmp_path / "result.json"
    assert (
        benchmark.main(
            [
                "--checkpoint",
                str(checkpoint),
                "--input",
                str(images),
                "--output",
                str(destination),
                "--scales",
                "0.5",
                "0.5",
                "--repeats",
                "1",
                "--warmup",
                "0",
            ]
        )
        == 0
    )
    report = json.loads(destination.read_text())
    assert checkpoint.read_bytes() == original
    assert report["checkpoint_sha256"] == hashlib.sha256(original).hexdigest()
    assert report["threads"] == 1
    assert len(report["images"]) == 4  # Two images, duplicate scales removed.
    reference = report["summary"][0]
    assert reference["curve_scale"] == reference["speedup"] == 1
    assert reference["mean_image_abs_error_255"] == 0
    assert reference["worst_pixel_abs_error_255"] == 0
    assert all(row["median_ms"] > 0 for row in report["images"])
    assert all(len(row["samples_ms"]) == 1 for row in report["images"])


@pytest.mark.parametrize(
    "extra",
    [["--repeats", "0"], ["--warmup", "-1"], ["--threads", "0"], ["--scales", "nan"]],
)
def test_invalid_benchmark_options_fail_before_loading_files(
    extra: list[str], tmp_path: Path
) -> None:
    assert (
        benchmark.main(
            [
                "--checkpoint",
                "missing.pt",
                "--input",
                "missing.png",
                "--output",
                str(tmp_path / "result.json"),
                *extra,
            ]
        )
        == 2
    )
    assert not (tmp_path / "result.json").exists()


def test_existing_benchmark_result_is_not_overwritten(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    image.touch()
    destination = tmp_path / "result.json"
    destination.write_text("previous measurement")
    assert (
        benchmark.main(
            ["--checkpoint", "missing.pt", "--input", str(image), "--output", str(destination)]
        )
        == 2
    )
    assert destination.read_text() == "previous measurement"
