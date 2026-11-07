"""train / evaluate / ablate command-line scripts and the ablation matrix (CPU, tiny data)."""

from __future__ import annotations

import copy
import csv
import dataclasses
import importlib.util
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import yaml
from _training_helpers import (
    CONFIGS,
    make_model_and_optimizer,
    tiny_config,
    write_test_video,
)
from PIL import Image

from nta_zerodce.data.image_dataset import make_synthetic_images
from nta_zerodce.training.ablation import REQUIRED_ROWS, expand_matrix, parse_matrix
from nta_zerodce.training.checkpoint import read_checkpoint, save_checkpoint
from nta_zerodce.training.config import Config, load_config, save_config
from nta_zerodce.training.evaluation import EVAL_COLUMNS, heldout_video_dir, read_rows

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
MATRIX = CONFIGS / "ablation_matrix.yaml"
Capture = pytest.CaptureFixture[str]


def load_script(name: str) -> ModuleType:
    """Import ``scripts/<name>.py`` as a module (scripts are not a package)."""
    spec = importlib.util.spec_from_file_location(f"script_{name}", SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def train_script() -> ModuleType:
    return load_script("train")


@pytest.fixture(scope="module")
def evaluate_script() -> ModuleType:
    return load_script("evaluate")


@pytest.fixture(scope="module")
def ablate_script() -> ModuleType:
    return load_script("ablate")


def save_yaml_config(config: Config, root: Path) -> Path:
    """Write ``config`` as ``root/cfg.yaml`` and return the path."""
    path = root / "cfg.yaml"
    save_config(config, path)
    return path


def write_pairs(root: Path, count: int = 2) -> None:
    """Write low/high LOL-style pairs plus unpaired images (nested one level)."""
    low = make_synthetic_images(root / "eval_low", count, 48, seed=5)
    (root / "eval_high").mkdir(exist_ok=True)
    for path in low:
        brighter = np.clip(np.asarray(Image.open(path)).astype(np.float32) * 3, 0, 255)
        Image.fromarray(brighter.astype(np.uint8)).save(root / "eval_high" / path.name)
    make_synthetic_images(root / "unpaired" / "SETA", 2, 40, seed=9)
    make_synthetic_images(root / "unpaired", 1, 40, seed=10)


def make_checkpoint(config: Config, path: Path) -> Path:
    """Save an untrained checkpoint for ``config`` (fast; evaluation does not need training)."""
    model, optimizer = make_model_and_optimizer(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(path, model, optimizer, config, 0)
    return path


# --------------------------------------------------------------------------- --help


@pytest.mark.parametrize("name", ["train", "evaluate", "ablate"])
def test_help_needs_no_data(name: str, capsys: Capture) -> None:
    module = load_script(name)
    with pytest.raises(SystemExit) as exit_info:
        module.main(["--help"])
    assert exit_info.value.code == 0
    text = capsys.readouterr().out
    for option in (
        "--config",
        "--noise",
        "--no-noise",
        "--adaptive-exp",
        "--curve-scale",
        "--seed",
    ):
        assert option in text


def test_scripts_run_as_programs() -> None:
    for argv in (["train.py", "--help"], ["ablate.py", "--dry-run", "--only", "baseline"]):
        done = subprocess.run(
            [sys.executable, str(SCRIPTS / argv[0]), *argv[1:]],
            capture_output=True,
            text=True,
            cwd=SCRIPTS.parent,
            check=False,
        )
        assert done.returncode == 0, done.stderr
        assert "usage" in done.stdout.lower() or "3 run(s)" in done.stdout


# --------------------------------------------------------------------------- train


def test_train_main_runs_and_applies_overrides(
    train_script: ModuleType, tmp_path: Path, capsys: Capture
) -> None:
    config = tiny_config(tmp_path, epochs=5)
    cfg = save_yaml_config(config, tmp_path)
    code = train_script.main(
        ["--config", str(cfg), "--epochs", "1", "--noise", "--seed", "29", "--run-name", "cli"]
    )
    assert code == 0
    final = tmp_path / "checkpoints" / "cli" / "final.pt"
    assert str(final) in capsys.readouterr().out
    payload = read_checkpoint(final)
    assert payload["flags"]["noise_aware"] is True and payload["epoch"] == 0
    assert payload["config"]["training"]["seed"] == 29
    assert (tmp_path / "runs" / "cli" / "metrics.csv").is_file()


def test_train_main_resume(train_script: ModuleType, tmp_path: Path) -> None:
    cfg = save_yaml_config(tiny_config(tmp_path, epochs=1), tmp_path)
    assert train_script.main(["--config", str(cfg), "--run-name", "r"]) == 0
    first = tmp_path / "checkpoints" / "r" / "epoch_0001.pt"
    assert (
        train_script.main(
            ["--config", str(cfg), "--run-name", "r", "--epochs", "2", "--resume", str(first)]
        )
        == 0
    )
    assert read_checkpoint(tmp_path / "checkpoints" / "r" / "final.pt")["epoch"] == 1


def test_train_main_error_codes(train_script: ModuleType, tmp_path: Path, capsys: Capture) -> None:
    cfg = save_yaml_config(tiny_config(tmp_path, make_images=False), tmp_path)
    assert train_script.main(["--config", str(cfg)]) == 2  # data missing
    assert "train_images" in capsys.readouterr().err
    assert train_script.main(["--config", str(cfg), "--denoise", "--no-noise"]) == 2
    assert "denoise" in capsys.readouterr().err
    assert train_script.main(["--config", str(tmp_path / "absent.yaml")]) == 2
    assert train_script.main(["--config", str(cfg), "--resume", str(tmp_path / "no.pt")]) == 2


# --------------------------------------------------------------------------- evaluate


def test_evaluate_writes_the_csv_schema_and_prints_means(
    evaluate_script: ModuleType, tmp_path: Path, capsys: Capture
) -> None:
    config = tiny_config(tmp_path, {"noise_aware": True}, make_images=False)
    write_pairs(tmp_path)
    checkpoint = make_checkpoint(config, tmp_path / "checkpoints" / "myrun" / "final.pt")
    output = tmp_path / "out" / "eval.csv"
    assert evaluate_script.main(["--checkpoint", str(checkpoint), "--output", str(output)]) == 0
    with output.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == EVAL_COLUMNS
        rows = list(reader)
    assert {r["run"] for r in rows} == {"myrun"} and {r["seed"] for r in rows} == {"17"}
    datasets = {r["dataset"] for r in rows}
    assert datasets == {"paired", "unpaired", "unpaired/SETA"}
    paired = {r["metric"] for r in rows if r["dataset"] == "paired"}
    assert paired == {
        "psnr",
        "ssim",
        "angular_error",
        "clipping_fraction",
        "noise_estimate",
        "noise_estimate_input",
    }
    unpaired = {r["metric"] for r in rows if r["dataset"].startswith("unpaired")}
    assert unpaired == {"clipping_fraction", "noise_estimate", "noise_estimate_input"}
    clip = [r for r in rows if r["metric"] == "clipping_fraction"]
    assert all(r["input_clipping"] != "" for r in clip)
    assert all(r["mask_coverage"] != "" for r in rows if r["metric"].startswith("noise_estimate"))
    assert all(r["mask_coverage"] == "" for r in rows if r["metric"] == "psnr")
    ids = {r["sample_id"] for r in rows if r["dataset"] == "paired"}
    assert ids == {"synthetic_0000", "synthetic_0001"}
    parsed = read_rows(output)
    assert len(parsed) == len(rows) and all(isinstance(r["value"], float) for r in parsed)
    printed = capsys.readouterr().out
    assert "paired" in printed and "psnr" in printed and "unpaired/SETA" in printed
    assert sum(1 for r in rows if r["dataset"] == "paired" and r["metric"] == "psnr") == 2


def test_evaluate_skips_missing_optional_sets_with_a_notice(
    evaluate_script: ModuleType, tmp_path: Path, capsys: Capture
) -> None:
    config = tiny_config(tmp_path, make_images=False)
    make_synthetic_images(tmp_path / "unpaired", 2, 32, seed=3)
    checkpoint = make_checkpoint(config, tmp_path / "m.pt")
    output = tmp_path / "eval.csv"
    code = evaluate_script.main(
        ["--checkpoint", str(checkpoint), "--output", str(output), "--run-name", "named"]
    )
    assert code == 0
    assert "skipping paired evaluation" in capsys.readouterr().out
    rows = read_rows(output)
    assert {r["dataset"] for r in rows} == {"unpaired"} and {r["run"] for r in rows} == {"named"}


def test_evaluate_without_any_data_exits_2_and_lists_expected_paths(
    evaluate_script: ModuleType, tmp_path: Path, capsys: Capture
) -> None:
    config = tiny_config(tmp_path, {"temporal": True}, make_images=False)
    checkpoint = make_checkpoint(config, tmp_path / "m.pt")
    code = evaluate_script.main(
        ["--checkpoint", str(checkpoint), "--output", str(tmp_path / "e.csv")]
    )
    assert code == 2
    err = capsys.readouterr().err
    for fragment in (
        "no evaluation data found",
        "paths.eval_low",
        "paths.unpaired",
        "held-out videos",
    ):
        assert fragment in err
    assert str(tmp_path / "eval_low") in err and not (tmp_path / "e.csv").exists()


def test_evaluate_argument_errors(
    evaluate_script: ModuleType, tmp_path: Path, capsys: Capture
) -> None:
    assert evaluate_script.main([]) == 2
    assert "--checkpoint is required" in capsys.readouterr().err
    assert evaluate_script.main(["--checkpoint", str(tmp_path / "none.pt")]) == 2
    config = tiny_config(tmp_path, make_images=False)
    checkpoint = make_checkpoint(config, tmp_path / "m.pt")
    assert evaluate_script.main(["--checkpoint", str(checkpoint), "--noise"]) == 2  # arch mismatch
    assert "flags.noise_aware" in capsys.readouterr().err


def test_evaluate_uses_the_given_config_and_curve_scale(
    evaluate_script: ModuleType, tmp_path: Path
) -> None:
    config = tiny_config(tmp_path, make_images=False)
    make_synthetic_images(tmp_path / "unpaired", 1, 48, seed=3)
    checkpoint = make_checkpoint(config, tmp_path / "m.pt")
    moved = tiny_config(tmp_path / "elsewhere", make_images=False)
    make_synthetic_images(tmp_path / "elsewhere" / "unpaired", 3, 32, seed=4)
    cfg = save_yaml_config(moved, tmp_path / "elsewhere")
    output = tmp_path / "scaled.csv"
    args = ["--checkpoint", str(checkpoint), "--config", str(cfg), "--output", str(output)]
    assert evaluate_script.main([*args, "--curve-scale", "0.5"]) == 0
    assert {r["sample_id"] for r in read_rows(output)} == {
        "synthetic_0000",
        "synthetic_0001",
        "synthetic_0002",
    }


def test_evaluate_temporal_reports_flow_aligned_error_on_held_out_clips(
    evaluate_script: ModuleType, tmp_path: Path
) -> None:
    config = tiny_config(tmp_path, {"temporal": True}, make_images=False)
    write_test_video(tmp_path / "videos" / "test" / "s1" / "clip.avi", frames=4, size=48)
    write_test_video(tmp_path / "videos" / "train" / "s2" / "never.avi", frames=4, size=48)
    checkpoint = make_checkpoint(config, tmp_path / "m.pt")
    output = tmp_path / "video.csv"
    assert evaluate_script.main(["--checkpoint", str(checkpoint), "--output", str(output)]) == 0
    rows = read_rows(output)
    assert {r["dataset"] for r in rows} == {"video"} and len(rows) == 3
    assert all(r["metric"] == "temporal_error" and "never" not in str(r["sample_id"]) for r in rows)
    assert all(0.0 <= float(str(r["mask_coverage"])) <= 1.0 for r in rows)
    assert all(r["input_clipping"] is not None for r in rows)
    assert heldout_video_dir(tmp_path / "videos" / "train") == tmp_path / "videos" / "test"
    assert heldout_video_dir(tmp_path / "videos" / "test") == tmp_path / "videos" / "test"
    assert heldout_video_dir(tmp_path / "nowhere" / "train") is None


# --------------------------------------------------------------------------- matrix


def test_real_matrix_expands_to_39_deterministic_runs() -> None:
    specs = expand_matrix(MATRIX)
    assert len(specs) == 39 == 13 * 3
    names = [s.name for s in specs]
    assert names == [name for name in dict.fromkeys(names) for _ in range(3)]
    assert [s.seed for s in specs[:6]] == [17, 29, 43, 17, 29, 43]
    assert names[0] == "baseline" and names[-1] == "loo_temporal"
    assert [(s.name, s.seed) for s in expand_matrix(MATRIX)] == [(s.name, s.seed) for s in specs]
    assert set(REQUIRED_ROWS) <= set(names)
    full = next(s for s in specs if s.name == "full")
    assert all(
        vars(full.flags)[f] for f in ("noise_aware", "denoise", "edge", "chroma", "temporal")
    )
    assert not full.flags.light and full.config.flags == full.flags


def test_runs_inherit_base_hyperparameters_and_override_only_flags_and_seed() -> None:
    base = load_config(CONFIGS / "baseline.yaml")
    for spec in expand_matrix(MATRIX):
        expected = dataclasses.replace(
            base, flags=spec.flags, training=dataclasses.replace(base.training, seed=spec.seed)
        )
        assert spec.config == expected, (spec.name, spec.seed)


def test_expand_matrix_global_overrides_and_forbidden_overrides() -> None:
    specs = expand_matrix(MATRIX, {"training.epochs": 2, "paths.run_dir": None})
    assert {s.config.training.epochs for s in specs} == {2}
    assert specs[0].config.paths.run_dir == load_config(CONFIGS / "baseline.yaml").paths.run_dir
    with pytest.raises(ValueError, match="flags"):
        expand_matrix(MATRIX, {"flags.edge": True})
    with pytest.raises(ValueError, match="seed"):
        expand_matrix(MATRIX, {"training.seed": 3})
    other = expand_matrix(MATRIX, base_config=CONFIGS / "full.yaml")
    assert other[0].flags == expand_matrix(MATRIX)[0].flags  # run flags still win over the base


def _matrix(tmp_path: Path, edit: Callable[[dict[str, Any]], None]) -> Path:
    """Write a modified copy of the real matrix and return its path."""
    data = yaml.safe_load(MATRIX.read_text())
    data["base_config"] = str(CONFIGS / "baseline.yaml")
    edit(data)
    path = tmp_path / "matrix.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def _run(data: dict[str, Any], name: str) -> dict[str, Any]:
    return next(r for r in data["runs"] if r["name"] == name)


INVALID: list[tuple[str, Callable[[dict[str, Any]], None], str]] = [
    (
        "duplicate_name",
        lambda d: d["runs"].append(copy.deepcopy(d["runs"][0])),
        "duplicate run name",
    ),
    ("duplicate_seed", lambda d: d.__setitem__("seeds", [17, 17]), "unique"),
    ("negative_seed", lambda d: d.__setitem__("seeds", [-1, 2]), "nonnegative"),
    ("bool_seed", lambda d: d.__setitem__("seeds", [True, 2]), "nonnegative"),
    ("empty_seeds", lambda d: d.__setitem__("seeds", []), "nonempty"),
    ("missing_flag", lambda d: _run(d, "edge_only")["flags"].pop("light"), "complete mapping"),
    ("unknown_flag", lambda d: _run(d, "edge_only")["flags"].__setitem__("fast", True), "complete"),
    (
        "non_bool_flag",
        lambda d: _run(d, "edge_only")["flags"].__setitem__("edge", 1),
        "true or false",
    ),
    (
        "denoise_without_noise",
        lambda d: _run(d, "edge_only")["flags"].__setitem__("denoise", True),
        "denoise requires noise_aware",
    ),
    ("missing_baseline", lambda d: d["runs"].remove(_run(d, "baseline")), "lacks required rows"),
    ("missing_loo", lambda d: d["runs"].remove(_run(d, "loo_chroma")), "loo_chroma"),
    (
        "wrong_protocol_flags",
        lambda d: _run(d, "chroma_only")["flags"].__setitem__("edge", True),
        "differ from the protocol row",
    ),
    ("unknown_top_key", lambda d: d.__setitem__("extra", 1), "unknown \\['extra'\\]"),
    ("missing_runs", lambda d: d.pop("runs"), "missing \\['runs'\\]"),
    ("run_extra_key", lambda d: d["runs"][0].__setitem__("seed", 1), "exactly the keys"),
    ("bad_name", lambda d: d["runs"][0].__setitem__("name", "a/b"), "name must match"),
    ("empty_base", lambda d: d.__setitem__("base_config", " "), "base_config"),
]


@pytest.mark.parametrize(
    ("edit", "match"), [(e, m) for _, e, m in INVALID], ids=[case[0] for case in INVALID]
)
def test_matrix_validation_errors(
    tmp_path: Path, edit: Callable[[dict[str, Any]], None], match: str
) -> None:
    path = _matrix(tmp_path, edit)
    with pytest.raises(ValueError, match=match):
        parse_matrix(path)
    with pytest.raises(ValueError, match=match):
        expand_matrix(path)


def test_matrix_file_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="matrix"):
        expand_matrix(tmp_path / "none.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a list\n")
    with pytest.raises(ValueError, match="mapping"):
        parse_matrix(bad)
    bad.write_text("a: [unclosed\n")
    with pytest.raises(ValueError, match="cannot parse"):
        parse_matrix(bad)
    missing_base = _matrix(tmp_path, lambda d: d.__setitem__("base_config", "nope/base.yaml"))
    with pytest.raises(FileNotFoundError, match="base_config"):
        expand_matrix(missing_base)


# --------------------------------------------------------------------------- ablate script


def test_ablate_dry_run_prints_the_plan_and_creates_nothing(
    ablate_script: ModuleType, tmp_path: Path, capsys: Capture
) -> None:
    out = tmp_path / "ablations"
    assert ablate_script.main(["--dry-run", "--output", str(out)]) == 0
    text = capsys.readouterr().out
    assert "39 run(s)" in text and text.count("[train]") == 39
    assert "[1/39] baseline seed=17" in text and "[39/39] loo_temporal seed=43" in text
    assert not out.exists()
    assert ablate_script.main(["--dry-run", "--only", "full", "--only", "baseline"]) == 0
    only = capsys.readouterr().out
    assert "6 run(s)" in only and "noise_only" not in only
    assert ablate_script.main(["--dry-run", "--only", "nope"]) == 2
    assert "--only names not in the matrix" in capsys.readouterr().err
    assert ablate_script.main(["--dry-run", "--noise"]) == 2
    assert "--noise" in capsys.readouterr().err
    assert ablate_script.main(["--dry-run", "--seed", "3"]) == 2
    assert ablate_script.main(["--dry-run", "--matrix", str(tmp_path / "none.yaml")]) == 2


def _tiny_matrix(tmp_path: Path, config: Config) -> Path:
    """Matrix over the real rows with one seed and a tiny base config."""
    base = save_yaml_config(config, tmp_path)

    def edit(data: dict[str, Any]) -> None:
        data["base_config"] = str(base)
        data["seeds"] = [3]

    return _matrix(tmp_path, edit)


def test_ablate_trains_evaluates_skips_forces_and_resumes(
    ablate_script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Capture
) -> None:
    config = tiny_config(tmp_path, epochs=2)
    write_pairs(tmp_path, count=1)
    matrix = _tiny_matrix(tmp_path, config)
    out = tmp_path / "ab"
    args = [
        "--matrix",
        str(matrix),
        "--output",
        str(out),
        "--only",
        "baseline",
        "--only",
        "edge_only",
    ]
    assert ablate_script.main(args) == 0
    for run in ("baseline", "edge_only"):
        leaf = out / run / "seed_3"
        assert sorted(p.name for p in leaf.iterdir()) == [
            "config.yaml",
            "epoch_0001.pt",
            "epoch_0002.pt",
            "eval.csv",
            "final.pt",
            "metrics.csv",
            "val_metrics.csv",
        ]
        assert read_checkpoint(leaf / "final.pt")["config"]["training"]["seed"] == 3
        assert load_config(leaf / "config.yaml").flags.edge is (run == "edge_only")
    with (out / "summary.csv").open(newline="") as handle:
        summary = list(csv.DictReader(handle))
    assert {r["run"] for r in summary} == {"baseline", "edge_only"}
    assert {r["status"] for r in summary} == {"trained"}
    assert {"psnr", "temporal_error"} & {r["metric"] for r in summary} == {"psnr"}
    assert [r["seed"] for r in summary if r["metric"] == "psnr"] == ["3", "3"]

    # A finished cell is skipped (train is never called) and its summary is reused.
    def refuse(*_args: object, **_kwargs: object) -> Path:
        raise AssertionError("train must not run for a finished cell")

    original = ablate_script.train
    monkeypatch.setattr(ablate_script, "train", refuse)
    assert ablate_script.main(args) == 0
    with (out / "summary.csv").open(newline="") as handle:
        assert {r["status"] for r in csv.DictReader(handle)} == {"skipped"}
    assert "[done (skip)]" in capsys.readouterr().out

    # --force retrains.
    calls: list[tuple[str, Path | None]] = []

    def spy(config: Config, resume: Path | None = None, **kwargs: Path | str | None) -> Path:
        calls.append((str(kwargs.get("run_name")), resume))
        return Path(original(config, resume, **kwargs))

    monkeypatch.setattr(ablate_script, "train", spy)
    assert ablate_script.main([*args, "--force", "--epochs", "1"]) == 0
    assert calls == [("seed_3", None), ("seed_3", None)]
    assert read_checkpoint(out / "baseline" / "seed_3" / "final.pt")["epoch"] == 0

    # An interrupted cell (no final.pt) resumes from its latest epoch checkpoint.
    leaf = out / "baseline" / "seed_3"
    (leaf / "final.pt").unlink()
    calls.clear()
    assert ablate_script.main([*args, "--only", "baseline"]) == 0
    assert calls == [("seed_3", leaf / "epoch_0001.pt")]
    assert read_checkpoint(leaf / "final.pt")["epoch"] == 1  # resumed at epoch 1 of 2


def test_ablate_without_evaluation_data_still_summarises(
    ablate_script: ModuleType, tmp_path: Path, capsys: Capture
) -> None:
    matrix = _tiny_matrix(tmp_path, tiny_config(tmp_path, epochs=1))
    out = tmp_path / "ab"
    code = ablate_script.main(["--matrix", str(matrix), "--output", str(out), "--only", "baseline"])
    assert code == 0
    assert "no evaluation data found" in capsys.readouterr().out
    assert not (out / "baseline" / "seed_3" / "eval.csv").exists()
    with (out / "summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1 and rows[0]["status"] == "trained" and rows[0]["metric"] == ""


def test_ablate_uses_the_epochs_override_for_every_run(ablate_script: ModuleType) -> None:
    args = ablate_script.build_parser().parse_args(["--epochs", "1", "--dry-run"])
    specs = ablate_script.expand_matrix(MATRIX, ablate_script._overrides(args))
    assert {s.config.training.epochs for s in specs} == {1}


def test_ablate_reexports_the_importable_expansion(ablate_script: ModuleType) -> None:
    assert ablate_script.expand_matrix is expand_matrix
