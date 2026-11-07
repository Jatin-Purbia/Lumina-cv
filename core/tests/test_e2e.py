"""End to end: train (CLI) -> checkpoint -> enhance (CLI) -> HTTP API, with every extension on.

The pieces are covered in depth by their own test modules; this one proves they fit together:
the checkpoint a CLI training run writes loads in both consumers, logs stay finite, and the CLI and the HTTP API produce the same pixels for the same upload.
"""

from __future__ import annotations

import csv
import importlib.util
import io
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from _training_helpers import tiny_config
from fastapi.testclient import TestClient
from PIL import Image

from nta_zerodce.api import Settings, create_app
from nta_zerodce.data.image_dataset import make_synthetic_images
from nta_zerodce.training.checkpoint import read_checkpoint
from nta_zerodce.training.config import save_config

ROOT = Path(__file__).resolve().parents[1]
FULL_FLAGS = {
    "noise_aware": True,
    "denoise": True,
    "edge": True,
    "chroma": True,
    "adaptive_exp": True,
    "temporal": True,
}


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"e2e_script_{name}", ROOT / "scripts" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Run the real training CLI for three epochs; return (checkpoint, run directory)."""
    root = tmp_path_factory.mktemp("e2e")
    config = tiny_config(root, FULL_FLAGS, epochs=3, images=8, size=64)
    config_path = root / "cfg.yaml"
    save_config(config, config_path)
    code = load_script("train").main(["--config", str(config_path), "--run-name", "e2e"])
    assert code == 0
    checkpoint = Path(config.paths.checkpoint_dir) / "e2e" / "final.pt"
    assert checkpoint.is_file()
    return checkpoint, Path(config.paths.run_dir) / "e2e"


def test_training_writes_finite_logs_and_a_loadable_checkpoint(
    trained: tuple[Path, Path],
) -> None:
    checkpoint, run_dir = trained
    payload = read_checkpoint(checkpoint)
    assert payload["format_version"] == 1
    assert payload["epoch"] == 2  # epochs 0..2
    assert all(payload["flags"][name] for name in FULL_FLAGS)
    logs = sorted(run_dir.glob("*.csv"))
    assert logs, "training should write CSV logs"
    for log in logs:
        with log.open(newline="") as handle:
            for row in csv.DictReader(handle):
                for value in row.values():
                    try:
                        assert np.isfinite(float(value))
                    except ValueError:
                        pass  # non-numeric columns (names, flags)


def test_cli_and_api_enhance_identically(trained: tuple[Path, Path], tmp_path: Path) -> None:
    checkpoint, _ = trained
    source = make_synthetic_images(tmp_path / "in", 1, 48, seed=3)[0]
    output = tmp_path / "out.png"
    code = load_script("enhance").main(
        [
            "--checkpoint",
            str(checkpoint),
            "--input",
            str(source),
            "--output",
            str(output),
            "--device",
            "cpu",
        ]
    )
    assert code == 0
    expected = np.asarray(Image.open(output).convert("RGB"))
    assert expected.shape == (48, 48, 3)

    settings = Settings(checkpoint=checkpoint, device="cpu")
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/v1/health").json()["model_loaded"] is True
        response = client.post(
            "/api/v1/enhance", files={"file": ("x.png", source.read_bytes(), "image/png")}
        )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    served = np.asarray(Image.open(io.BytesIO(response.content)).convert("RGB"))
    np.testing.assert_array_equal(served, expected)
    # Enhancement must change the (dark) input and stay a valid image.
    original = np.asarray(Image.open(source).convert("RGB")).astype(np.float32)
    assert served.astype(np.float32).mean() > original.mean()


def test_enhancement_is_deterministic_across_runs(
    trained: tuple[Path, Path], tmp_path: Path
) -> None:
    checkpoint, _ = trained
    source = make_synthetic_images(tmp_path / "in", 1, 40, seed=4)[0]
    results = []
    for index in range(2):
        out = tmp_path / f"o{index}.png"
        assert (
            load_script("enhance").main(
                [
                    "--checkpoint",
                    str(checkpoint),
                    "--input",
                    str(source),
                    "--output",
                    str(out),
                    "--device",
                    "cpu",
                ]
            )
            == 0
        )
        results.append(out.read_bytes())
    assert results[0] == results[1]
