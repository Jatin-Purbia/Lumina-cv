"""Executable documented contracts."""

from __future__ import annotations

import argparse
import re
from dataclasses import asdict
from pathlib import Path
from urllib.parse import unquote

import pytest
import yaml

from nta_zerodce.cli import dispatch
from nta_zerodce.losses.registry import LOSS_CONTRACTS, LOSS_REGISTRY
from nta_zerodce.training.config import Config
from nta_zerodce.training.validation import config_from_dict


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("weights", "edge", -1),
        ("weights", "edge", float("nan")),
        ("weights", "edge", float("inf")),
        ("model", "channels", True),
        ("model", "iterations", 7),
        ("model", "curve_scale", 0),
        ("model", "gain_window", 4),
        ("training", "crop_size", 17),
        ("training", "workers", -1),
        ("training", "lr", "0.1"),
        ("flow", "align_corners", False),
        ("flags", "edge", 1),
        ("flags", "denoise", True),
        ("paths", "run_dir", ""),
        ("temporal", "pair_source", "other"),
        ("noise", "a_max", 0.5),
        ("loss", "unknown", 0.1),
    ],
)
def test_invalid_config(section: str, key: str, value: object) -> None:
    """Reject malformed scalar types, nonfinite values and cross-field violations."""
    data = asdict(Config())
    data[section][key] = value
    with pytest.raises(ValueError):
        config_from_dict(data)


def test_ablation_matrix_preflight(tmp_path: Path) -> None:
    """Malformed matrices fail before scheduling training."""
    matrix = {"base_config": "configs/baseline.yaml", "seeds": [17, 17], "runs": []}
    path = tmp_path / "matrix.yaml"
    path.write_text(yaml.safe_dump(matrix))
    with pytest.raises(ValueError, match="seeds"):
        dispatch("ablate", argparse.Namespace(matrix=str(path), output=str(tmp_path / "ablations")))


# Collection-time documentation and complete-configuration audits.
# make docs-check collects this module; broken links fail during collection.
ROOT = Path(__file__).resolve().parents[1]
for document in [
    ROOT / "README.md",
    ROOT.parent / "README.md",
    ROOT.parent / "web" / "README.md",
    *sorted((ROOT / "docs").glob("*.md")),
]:
    for target in re.findall(r"(?<!!)\[[^\]]*\]\(([^)]+)\)", document.read_text(encoding="utf-8")):
        target = target.split("#", 1)[0]
        if target and not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target):
            assert (document.parent / unquote(target.strip("<>"))).exists(), (document, target)

defaults = asdict(Config())
for config_file in ["baseline.yaml", "full.yaml"]:
    actual = yaml.safe_load((ROOT / "configs" / config_file).read_text(encoding="utf-8"))
    assert actual.keys() == defaults.keys()
    for section, fields in defaults.items():
        if isinstance(fields, dict):
            assert actual[section].keys() == fields.keys(), (config_file, section)
    assert not actual["flags"]["denoise"] or actual["flags"]["noise_aware"]
matrix = yaml.safe_load((ROOT / "configs" / "ablation_matrix.yaml").read_text(encoding="utf-8"))
assert len(matrix["runs"]) == 13
assert matrix["seeds"] == [17, 29, 43]
for row in matrix["runs"]:
    assert row["flags"].keys() == defaults["flags"].keys()
    assert not row["flags"]["denoise"] or row["flags"]["noise_aware"]

assert LOSS_REGISTRY.keys() == LOSS_CONTRACTS.keys()
for loss_name, loss_spec in LOSS_REGISTRY.items():
    assert callable(loss_spec.callable)
    assert (loss_spec.flag, loss_spec.default_weight, loss_spec.required_inputs) == LOSS_CONTRACTS[
        loss_name
    ][1:]
