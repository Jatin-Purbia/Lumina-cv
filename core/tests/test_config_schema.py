"""Config schema behaviour (bounds, types, YAML IO, paths) plus active collection audits.

See docs/ARCHITECTURE.md (Schema bounds and units). The module-level audits at
the bottom run at collection time and must keep passing.
"""

from __future__ import annotations

import ast
import dataclasses
import itertools
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, get_type_hints
from urllib.parse import unquote

import pytest
import yaml

from nta_zerodce.losses.registry import LOSS_CONTRACTS, LOSS_REGISTRY
from nta_zerodce.training.config import (
    Config,
    check_paths,
    load_config,
    load_python_mapping,
    parse_config,
    resolve_path,
    save_config,
    workspace_root,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
BASELINE = CONFIGS / "baseline.py"
FULL = CONFIGS / "full.py"


def _sections() -> dict[str, type[Any]]:
    """Map section name to its dataclass type."""
    hints = get_type_hints(Config)
    return {
        item.name: hints[item.name]
        for item in dataclasses.fields(Config)
        if dataclasses.is_dataclass(hints[item.name])
    }


def _leaf_fields() -> list[tuple[str, str, Any]]:
    """Every (section, field, type) leaf of the schema."""
    out: list[tuple[str, str, Any]] = []
    for section, cls in _sections().items():
        hints = get_type_hints(cls)
        out.extend((section, item.name, hints[item.name]) for item in dataclasses.fields(cls))
    return out


def _with(path: str, value: object, base: Config | None = None) -> Config:
    """Return a copy of base (default Config()) with one dotted leaf replaced, unvalidated."""
    config = Config() if base is None else base
    if "." not in path:
        top: dict[str, Any] = {path: value}
        return dataclasses.replace(config, **top)
    section, name = path.split(".")
    record = dataclasses.replace(getattr(config, section), **{name: value})
    return dataclasses.replace(config, **{section: record})


def _plain(config: Config) -> dict[str, Any]:
    """Config as a nested plain dict (what a complete YAML contains)."""
    return asdict(config)


def _write(tmp_path: Path, data: object, name: str = "c.yaml") -> Path:
    """Dump a python object as YAML into tmp_path."""
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


# (dotted field, valid values, invalid values). Defaults of the other fields stay in force, so a
# few cases exercise cross-field rules (noted inline).
BOUNDS: dict[str, tuple[list[Any], list[Any]]] = {
    "model.channels": ([1, 64], [0, -1]),
    "model.iterations": ([8], [7, 9, 0, -8]),
    "model.noise_channels": ([1, 16], [0, -3]),
    "model.denoise_channels": ([1, 16], [0, -3]),
    # 0.1 < sqrt(a_max + b_max^2) = 0.1020 for the default noise ranges.
    "model.sigma_max": ([0.11, 1.0, 1], [0.0, -0.1, 1.01, 0.1]),
    "model.residual_scale": ([1e-6, 1.0, 1], [0.0, 1.5, -0.1]),
    "model.curve_scale": ([0.25, 1.0, 1, 1e-3], [0.0, 1.0001, -0.5, 2]),
    "model.gain_window": ([1, 3, 5, 9], [0, 2, 4, -3]),
    "model.gain_min": ([1.0, 2.0, 8.0], [0.99, 0.0, 8.5, -1.0]),
    "model.gain_max": ([1.0, 8.0, 100.0], [0.99, 0.5]),
    "model.eps": ([1e-12, 1.0], [0.0, -1e-6]),
    "loss.spatial_patch": ([1, 4, 16, 32], [0, -4, 3, 512]),
    "loss.exposure_patch": ([1, 16, 64], [0, -16, 3, 512]),
    "loss.exposure_target": ([0.0, 0.6, 1.0, 1], [-0.01, 1.01]),
    "loss.adaptive_low": ([0.0, 0.84], [-0.1, 0.85, 0.9]),
    "loss.adaptive_high": ([0.56, 1.0], [0.55, 0.5, 1.2]),
    "loss.adaptive_noise_scale": ([1e-6, 1.0], [0.0, -0.05]),
    "loss.adaptive_noise_penalty": ([0.0, 1.0], [-0.01]),
    "loss.highlight_threshold": ([0.0, 1.0], [-0.1, 1.5]),
    "loss.chroma_window": ([1, 3, 5], [0, 4, -1]),
    "loss.reliability_low": ([0.0, 0.19], [0.2, 0.3, -0.1]),
    "loss.reliability_high": ([0.06, 1.0], [0.05, 1.1]),
    "loss.reliability_noise": ([0.1, 2], [0.0, -2.0]),
    "loss.edge_smooth_sigma": ([0.1, 3.0], [0.0, -0.8]),
    "loss.edge_kernel": ([1, 3, 7], [0, 6, -5]),
    "loss.edge_noise_multiplier": ([0.5, 2], [0.0, -1.0]),
    "loss.eps": ([1e-9, 1.0], [0.0, -1e-6]),
    "noise.a_min": ([0.0, 0.01], [-0.001, 0.02]),
    # 0.04 violates model.sigma_max >= sqrt(a_max + b_max^2) for the default sigma_max 0.2.
    "noise.a_max": ([0.0, 0.039], [-0.01, 0.04]),
    "noise.b_min": ([0.0, 0.02], [-0.1, 0.03]),
    "noise.b_max": ([0.0, 0.1], [-0.01, 0.2]),
    "noise.probability": ([0.0, 0.5, 1.0, 1], [-0.1, 1.1]),
    "synthetic.rotation_degrees": ([0.0, 45.0], [-1.0]),
    "synthetic.translation_pixels": ([0.0, 10], [-0.5]),
    "synthetic.scale_min": ([0.5, 1.02], [0.0, -1.0, 1.03]),
    "synthetic.scale_max": ([0.98, 3.0], [0.0, 0.97]),
    "synthetic.flicker_min": ([0.1, 1.1], [0.0, 1.2, -1.0]),
    "synthetic.flicker_max": ([0.9, 2.0], [0.0, 0.8]),
    "flow.method": (["dis"], ["farneback", "DIS", ""]),
    "flow.preset": (["ultrafast", "fast", "medium"], ["slow", "", "Medium"]),
    "flow.scale": ([0.1, 0.5, 1.0, 1], [0.0, 1.01, -0.5]),
    "flow.fb_alpha": ([0.0, 1.0], [-0.01]),
    "flow.fb_beta": ([0.0, 2], [-0.1]),
    "flow.photometric_threshold": ([0.0, 1.0], [-0.1, 1.1]),
    "flow.align_corners": ([True], [False]),
    "temporal.curve_blend": ([0.0, 1.0], [-0.1, 1.1]),
    "temporal.output_blend": ([0.0, 1.0], [-0.1, 1.1]),
    "temporal.histogram_bins": ([2, 64], [1, 0, -5]),
    "temporal.scene_cut_threshold": ([0.0, 1.0], [-0.1, 1.1]),
    "temporal.pair_source": (["synthetic", "real"], ["video", "", "Real"]),
    "temporal.frame_gap": ([1, 5], [0, -1]),
    "training.lr": ([1e-9, 1.0], [0.0, -1e-4]),
    "training.batch_size": ([1, 64], [0, -1]),
    "training.crop_size": ([16, 32, 48, 256], [0, 8, 15, 17, 20, 24, -16]),
    "training.epochs": ([1, 5], [0, -1]),
    "training.optimizer": (["Adam"], ["adam", "SGD", "AdamW", ""]),
    "training.adam_beta1": ([0.0, 0.99], [1.0, -0.1, 1.5]),
    "training.adam_beta2": ([0.0, 0.9999], [1.0, -0.1, 1.5]),
    "training.adam_eps": ([1e-12, 1.0], [0.0, -1e-8]),
    "training.weight_decay": ([0.0, 0.1], [-1e-4]),
    "training.grad_clip": ([0.01, 10], [0.0, -1.0]),
    "training.seed": ([0, 17, 2**32 - 1], [-1, 2**32]),
    "training.workers": ([0, 4], [-1]),
    "training.device": (
        ["auto", "cpu", "cuda", "cuda:0", "cuda:12"],
        ["gpu", "cuda:", "cuda:-1", "CPU", "cuda:a", "mps", "", " cpu"],
    ),
    "training.checkpoint_every": ([1, 5], [0, -1]),
    "training.log_every": ([1, 10], [0, -1]),
    "schema_version": ([1], [0, 2, -1]),
}
for _name in (item.name for item in dataclasses.fields(_sections()["weights"])):
    BOUNDS[f"weights.{_name}"] = ([0.0, 1.0, 1000, 1e-9], [-1e-9, -1.0])
for _name in (item.name for item in dataclasses.fields(_sections()["paths"])):
    BOUNDS[f"paths.{_name}"] = (["x", "/abs/path", "a b", "data/x/"], ["", "   "])


def _bound_cases(valid: bool) -> list[Any]:
    """Flatten BOUNDS into pytest params for the valid or the invalid side."""
    return [
        pytest.param(path, value, id=f"{path}={value!r}")
        for path, (ok, bad) in BOUNDS.items()
        for value in (ok if valid else bad)
    ]


def test_bounds_table_covers_every_schema_field() -> None:
    """Every leaf of the schema has a bounds row, so no field escapes validation."""
    expected = {"schema_version"} | {f"{s}.{n}" for s, n, _ in _leaf_fields() if s != "flags"}
    assert set(BOUNDS) == expected


def test_defaults_are_valid() -> None:
    """Config() passes validate() and baseline/full YAML load to valid configs."""
    Config().validate()
    for path in (BASELINE, FULL):
        load_config(path).validate()


@pytest.mark.parametrize(("path", "value"), _bound_cases(valid=True))
def test_bound_accepts_valid_value(path: str, value: object) -> None:
    """Edge and interior values on the valid side of each bound are accepted."""
    _with(path, value).validate()


@pytest.mark.parametrize(("path", "value"), _bound_cases(valid=False))
def test_bound_rejects_invalid_value(path: str, value: object) -> None:
    """Each violating value raises ValueError whose message names the dotted field."""
    with pytest.raises(ValueError) as info:
        _with(path, value).validate()
    assert path in str(info.value)


# ---------------------------------------------------------------------------
# Scalar types
# ---------------------------------------------------------------------------

FLOAT_FIELDS = [f"{s}.{n}" for s, n, t in _leaf_fields() if t is float]
INT_FIELDS = [f"{s}.{n}" for s, n, t in _leaf_fields() if t is int]
BOOL_FIELDS = [f"{s}.{n}" for s, n, t in _leaf_fields() if t is bool]
STR_FIELDS = [f"{s}.{n}" for s, n, t in _leaf_fields() if t is str]


def test_field_type_groups_are_nonempty() -> None:
    """Sanity check that the type partition used below sees the whole schema."""
    assert len(FLOAT_FIELDS + INT_FIELDS + BOOL_FIELDS + STR_FIELDS) == len(_leaf_fields())
    assert "training.lr" in FLOAT_FIELDS and "model.iterations" in INT_FIELDS
    assert "flags.denoise" in BOOL_FIELDS and "flow.method" in STR_FIELDS


@pytest.mark.parametrize("path", FLOAT_FIELDS)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_float_fields_reject_nonfinite(path: str, value: float) -> None:
    """NaN and infinities are rejected for every float field."""
    with pytest.raises(ValueError, match=re.escape(path)):
        _with(path, value).validate()


@pytest.mark.parametrize("path", FLOAT_FIELDS + INT_FIELDS)
@pytest.mark.parametrize("value", [True, False, "1", None, [1]])
def test_numeric_fields_reject_bool_and_non_numbers(path: str, value: object) -> None:
    """bool is never accepted as int or float, nor are strings, None or lists."""
    with pytest.raises(ValueError, match=re.escape(path)):
        _with(path, value).validate()


@pytest.mark.parametrize("path", INT_FIELDS)
@pytest.mark.parametrize("value", [8.0, 1.5])
def test_int_fields_reject_floats(path: str, value: float) -> None:
    """Float literals (even integral ones) are not accepted for integer fields."""
    with pytest.raises(ValueError, match=re.escape(path)):
        _with(path, value).validate()


@pytest.mark.parametrize("path", BOOL_FIELDS)
@pytest.mark.parametrize("value", [0, 1, "true", "false", None, 1.0])
def test_bool_fields_reject_non_bool(path: str, value: object) -> None:
    """Flags and align_corners must be real booleans, not ints or strings."""
    with pytest.raises(ValueError, match=re.escape(path)):
        _with(path, value).validate()


@pytest.mark.parametrize("path", STR_FIELDS)
@pytest.mark.parametrize("value", [1, 1.5, None, True, ["a"]])
def test_string_fields_reject_non_strings(path: str, value: object) -> None:
    """String fields reject numbers, bools, None and lists."""
    with pytest.raises(ValueError, match=re.escape(path)):
        _with(path, value).validate()


def test_schema_version_must_be_int_one() -> None:
    """schema_version rejects bool, float and string spellings of 1."""
    for bad in (True, 1.0, "1", None):
        with pytest.raises(ValueError, match="schema_version"):
            _with("schema_version", bad).validate()


def test_section_must_be_the_right_record() -> None:
    """A wrong record type in a section slot is rejected with the section name."""
    with pytest.raises(ValueError, match="flags"):
        dataclasses.replace(Config(), flags={"noise_aware": True}).validate()  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="model"):
        dataclasses.replace(Config(), model=Config().flags).validate()  # type: ignore[arg-type]


def test_integers_are_accepted_for_float_fields() -> None:
    """Integer literals are valid for float fields (and load converts them to float)."""
    config = _with("model.curve_scale", 1)
    config.validate()


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------


def test_denoise_implies_noise_aware() -> None:
    """denoise without noise_aware is rejected with an actionable message."""
    bad = _with("flags.denoise", True)
    with pytest.raises(ValueError, match=r"flags\.denoise.*flags\.noise_aware"):
        bad.validate()


@pytest.mark.parametrize("bits", list(itertools.product([False, True], repeat=7)))
def test_every_flag_combination_validity(bits: tuple[bool, ...]) -> None:
    """All 128 flag combinations are valid except denoise without noise_aware."""
    names = [item.name for item in dataclasses.fields(Config().flags)]
    flags = dataclasses.replace(Config().flags, **dict(zip(names, bits, strict=True)))
    config = dataclasses.replace(Config(), flags=flags)
    if flags.denoise and not flags.noise_aware:
        with pytest.raises(ValueError):
            config.validate()
    else:
        config.validate()


# ---------------------------------------------------------------------------
# Cross-field rules
# ---------------------------------------------------------------------------


def test_loss_patches_must_divide_crop() -> None:
    """Both loss patch sizes must divide training.crop_size."""
    config = _with("training.crop_size", 48)
    config = _with("loss.exposure_patch", 32, config)
    with pytest.raises(ValueError, match=r"loss\.exposure_patch.*training\.crop_size"):
        config.validate()
    _with("loss.exposure_patch", 16, config).validate()


def test_sigma_max_must_cover_noise_range() -> None:
    """model.sigma_max >= sqrt(noise.a_max + noise.b_max**2), with equality allowed."""
    needed = math.sqrt(0.01 + 0.02**2)
    _with("model.sigma_max", needed + 1e-9).validate()
    with pytest.raises(ValueError, match=r"model\.sigma_max"):
        _with("model.sigma_max", needed - 1e-6).validate()


def test_validate_does_not_touch_the_filesystem(tmp_path: Path) -> None:
    """Nonexistent data paths do not fail validation (checked at execution time)."""
    config = _with("paths.train_images", str(tmp_path / "missing"))
    config = _with("temporal.pair_source", "real", config)
    config.validate()


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


def test_load_baseline_and_full_match_flags() -> None:
    """The complete YAML files load; full enables every extension except light."""
    base = load_config(BASELINE)
    full = load_config(FULL)
    assert not any(asdict(base.flags).values())
    assert {k for k, v in asdict(full.flags).items() if v} == {
        "noise_aware",
        "denoise",
        "edge",
        "chroma",
        "adaptive_exp",
        "temporal",
    }
    assert base.training.lr == 1e-4 and isinstance(base.training.lr, float)
    assert base.model.eps == 1e-6 and base.training.adam_eps == 1e-8
    assert base.paths.train_images == "data/LOL/train_low"  # paths stay as written


def test_baseline_equals_schema_defaults() -> None:
    """baseline.py carries the schema defaults exactly (every field is explicit)."""
    assert load_config(BASELINE) == Config()


def test_full_yaml_differs_from_baseline_only_in_flags() -> None:
    """full.py changes only extension flags relative to baseline.py."""
    base, full = _plain(load_config(BASELINE)), _plain(load_config(FULL))
    assert {k: v for k, v in base.items() if k != "flags"} == {
        k: v for k, v in full.items() if k != "flags"
    }
    assert base["flags"] != full["flags"]


@pytest.mark.parametrize("literal", ["1e-4", "1E-4", "+1e-4", "1e-04", "0.0001", "1.0e-4"])
def test_exponent_literals_load_as_floats(tmp_path: Path, literal: str) -> None:
    """PyYAML reads 1e-4 as a string; the loader must treat all spellings as the float."""
    path = tmp_path / "p.yaml"
    path.write_text(f"training:\n  lr: {literal}\n", encoding="utf-8")
    config = load_config(path, partial=True)
    assert config.training.lr == 1e-4 and isinstance(config.training.lr, float)


def test_quoted_number_is_wrong_type(tmp_path: Path) -> None:
    """A quoted number is a string and is rejected for float fields."""
    path = tmp_path / "p.yaml"
    path.write_text('training:\n  lr: "1e-4"\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"training\.lr"):
        load_config(path, partial=True)


@pytest.mark.parametrize(
    ("text", "field"),
    [
        ("model:\n  curve_scale: true\n", "model.curve_scale"),
        ("model:\n  channels: true\n", "model.channels"),
        ("model:\n  channels: 32.0\n", "model.channels"),
        ("model:\n  channels: '32'\n", "model.channels"),
        ("model:\n  sigma_max: .nan\n", "model.sigma_max"),
        ("model:\n  sigma_max: .inf\n", "model.sigma_max"),
        ("weights:\n  exp: -.inf\n", "weights.exp"),
        ("weights:\n  exp: -1\n", "weights.exp"),
        ("flags:\n  edge: 1\n", "flags.edge"),
        ("flags:\n  edge: 'yes'\n", "flags.edge"),
        ("flags:\n  edge: ~\n", "flags.edge"),
        ("flow:\n  scale: 1.5\n", "flow.scale"),
        ("flow:\n  scale: 0\n", "flow.scale"),
        ("loss:\n  exposure_patch: 7\n", "loss.exposure_patch"),
        ("training:\n  crop_size: 250\n", "training.crop_size"),
        ("paths:\n  run_dir: ''\n", "paths.run_dir"),
        ("schema_version: 2\n", "schema_version"),
    ],
)
def test_load_rejects_bad_values(tmp_path: Path, text: str, field: str) -> None:
    """Wrong scalar types and out-of-range values from YAML raise ValueError naming the field."""
    path = tmp_path / "bad.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(field)):
        load_config(path, partial=True)


def test_int_literals_become_floats_on_load(tmp_path: Path) -> None:
    """curve_scale: 1 is accepted and stored as the float 1.0."""
    path = tmp_path / "p.yaml"
    path.write_text("model:\n  curve_scale: 1\n", encoding="utf-8")
    config = load_config(path, partial=True)
    assert config.model.curve_scale == 1.0 and type(config.model.curve_scale) is float


@pytest.mark.parametrize("section", ["root", *_sections()])
def test_unknown_field_rejected_at_every_level(tmp_path: Path, section: str) -> None:
    """Unknown keys at the root or in any section raise with the dotted path."""
    data = _plain(Config())
    if section == "root":
        data["bogus"] = 1
        dotted = "bogus"
    else:
        data[section]["bogus"] = 1
        dotted = f"{section}.bogus"
    with pytest.raises(ValueError, match="unknown field") as info:
        load_config(_write(tmp_path, data))
    assert dotted in str(info.value)


def test_unknown_field_suggests_close_match(tmp_path: Path) -> None:
    """A typo gets a did-you-mean suggestion."""
    data = _plain(Config())
    data["flags"]["noise_awar"] = data["flags"].pop("noise_aware")
    with pytest.raises(ValueError, match="did you mean 'flags.noise_aware'"):
        load_config(_write(tmp_path, data))


def test_unknown_field_is_rejected_even_when_partial(tmp_path: Path) -> None:
    """partial=True fills missing keys but never tolerates unknown ones."""
    path = tmp_path / "p.yaml"
    path.write_text("flags:\n  nope: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"flags\.nope"):
        load_config(path, partial=True)


@pytest.mark.parametrize("section", ["root", *_sections()])
def test_missing_keys_rejected_in_strict_mode(tmp_path: Path, section: str) -> None:
    """The default strict mode requires every key at every level."""
    data = _plain(Config())
    if section == "root":
        del data["flags"]
        dotted = "flags"
    else:
        name = next(iter(data[section]))
        del data[section][name]
        dotted = f"{section}.{name}"
    with pytest.raises(ValueError, match="missing field") as info:
        load_config(_write(tmp_path, data))
    assert dotted in str(info.value)


def test_partial_fills_defaults_and_yaml_beats_defaults(tmp_path: Path) -> None:
    """partial=True: absent keys take schema defaults, present keys win over them."""
    path = tmp_path / "p.yaml"
    path.write_text("flags:\n  edge: true\ntraining:\n  seed: 5\n", encoding="utf-8")
    config = load_config(path, partial=True)
    assert config.flags.edge is True and config.training.seed == 5
    assert config.model == Config().model and config.training.lr == Config().training.lr
    with pytest.raises(ValueError, match="missing field"):
        load_config(path)


def test_empty_partial_yaml_is_the_default_config(tmp_path: Path) -> None:
    """An empty mapping with partial=True equals Config()."""
    path = tmp_path / "p.yaml"
    path.write_text("{}\n", encoding="utf-8")
    assert load_config(path, partial=True) == Config()


@pytest.mark.parametrize("text", ["", "- a\n- b\n", "42\n", "just text\n", "~\n"])
def test_root_must_be_a_mapping(tmp_path: Path, text: str) -> None:
    """Non-mapping documents are rejected."""
    path = tmp_path / "p.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)


def test_section_must_be_a_mapping(tmp_path: Path) -> None:
    """A scalar where a section is expected is rejected naming the section."""
    data = _plain(Config())
    data["flags"] = True
    with pytest.raises(ValueError, match="flags"):
        load_config(_write(tmp_path, data))


def test_invalid_yaml_syntax_and_duplicate_keys(tmp_path: Path) -> None:
    """Syntax errors and duplicate keys are ValueErrors, not silent last-wins."""
    path = tmp_path / "p.yaml"
    path.write_text("flags: [unclosed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot parse"):
        load_config(path)
    path.write_text("training:\n  seed: 1\n  seed: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_config(path, partial=True)


def test_non_utf8_file_is_rejected(tmp_path: Path) -> None:
    """Invalid UTF-8 bytes raise ValueError."""
    path = tmp_path / "p.yaml"
    path.write_bytes(b"flags: \xff\xfe\n")
    with pytest.raises(ValueError, match="UTF-8"):
        load_config(path, partial=True)


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    """A missing YAML file is a FileNotFoundError naming the resolved path."""
    with pytest.raises(FileNotFoundError, match="nope.yaml"):
        load_config(tmp_path / "nope.yaml")


def test_load_does_not_use_cwd_for_relative_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Relative YAML paths resolve against the workspace root, not the current directory."""
    monkeypatch.chdir(tmp_path)
    assert load_config("configs/baseline.py") == load_config(BASELINE)
    (tmp_path / "local.yaml").write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_config("local.yaml", partial=True)
    assert load_config(tmp_path / "local.yaml", partial=True) == Config()


def test_loaded_config_is_frozen() -> None:
    """The returned records are immutable."""
    config = load_config(BASELINE)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.flags.edge = True  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.training = Config().training  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------


def test_override_precedence_and_none_is_absent() -> None:
    """Explicit override > YAML; None leaves YAML untouched; False is a real value."""
    full = load_config(FULL)
    assert full.flags.edge is True
    same = load_config(FULL, {"flags.edge": None, "training.seed": None})
    assert same == full
    off = load_config(FULL, {"flags.edge": False})
    assert off.flags.edge is False and off.flags.chroma is True
    base = load_config(
        BASELINE, {"model.curve_scale": 0.5, "paths.video_dir": "v", "training.seed": 3}
    )
    assert (base.model.curve_scale, base.paths.video_dir, base.training.seed) == (0.5, "v", 3)


def test_override_int_for_float_field_is_converted() -> None:
    """An int override for a float field becomes a float."""
    config = load_config(BASELINE, {"model.curve_scale": 1})
    assert type(config.model.curve_scale) is float


def test_overrides_are_validated() -> None:
    """Overrides go through the same bounds as YAML values."""
    with pytest.raises(ValueError, match=r"model\.curve_scale"):
        load_config(BASELINE, {"model.curve_scale": 1.5})
    with pytest.raises(ValueError, match=r"training\.seed"):
        load_config(BASELINE, {"training.seed": -1})
    with pytest.raises(ValueError, match=r"training\.device"):
        load_config(BASELINE, {"training.device": "tpu"})
    with pytest.raises(ValueError, match=r"flags\.edge"):
        load_config(BASELINE, {"flags.edge": 1})


def test_override_denoise_without_noise_fails() -> None:
    """Explicit noise=False with YAML denoise=true fails validation after resolution."""
    with pytest.raises(ValueError, match=r"flags\.denoise"):
        load_config(FULL, {"flags.noise_aware": False})
    load_config(FULL, {"flags.noise_aware": False, "flags.denoise": False})
    with pytest.raises(ValueError, match=r"flags\.denoise"):
        load_config(BASELINE, {"flags.denoise": True})


@pytest.mark.parametrize(
    "name", ["flags.bogus", "bogus", "model.", "flags", "model", "flags.edge.x", "a.b.c"]
)
def test_bad_override_names_are_rejected(name: str) -> None:
    """Unknown names, section names and over-deep names raise ValueError."""
    with pytest.raises(ValueError, match="override"):
        load_config(BASELINE, {name: 1})


def test_override_none_with_unknown_name_is_still_ignored() -> None:
    """None means absent, so it never triggers name checks (CLI passes absent flags as None)."""
    assert load_config(BASELINE, {"flags.bogus": None}) == load_config(BASELINE)


def test_overrides_do_not_mutate_inputs() -> None:
    """Loading with overrides twice gives independent equal results."""
    overrides: dict[str, object] = {"flags.edge": True}
    first = load_config(BASELINE, overrides)
    assert overrides == {"flags.edge": True}
    assert first == load_config(BASELINE, overrides)


def test_parse_config_defers_bounds_but_checks_structure(tmp_path: Path) -> None:
    """parse_config returns an unvalidated Config (used by the CLI) but still rejects bad keys."""
    parsed = parse_config(BASELINE, {"flags.denoise": True})
    assert parsed.flags.denoise is True
    with pytest.raises(ValueError):
        parsed.validate()
    path = _write(tmp_path, {**_plain(Config()), "extra": 1})
    with pytest.raises(ValueError, match="extra"):
        parse_config(path)


# ---------------------------------------------------------------------------
# Save and round trip
# ---------------------------------------------------------------------------


def _round_trip(config: Config, tmp_path: Path) -> Config:
    """save_config then load_config in the strict mode."""
    path = tmp_path / "rt.yaml"
    save_config(config, path)
    return load_config(path)


@pytest.mark.parametrize("name", ["baseline.py", "full.py"])
def test_round_trip_of_shipped_configs(tmp_path: Path, name: str) -> None:
    """load(save(c)) == c exactly for the shipped complete YAML files."""
    config = load_config(CONFIGS / name)
    assert _round_trip(config, tmp_path) == config


def test_saved_file_contains_every_field_in_order(tmp_path: Path) -> None:
    """The file lists schema_version first and every record field in declaration order."""
    path = tmp_path / "s.yaml"
    save_config(Config(), path)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("schema_version: 1\nflags:\n")
    data = yaml.safe_load(text)
    assert list(data) == list(_plain(Config()))
    for section, fields in _plain(Config()).items():
        if isinstance(fields, dict):
            assert list(data[section]) == list(fields)


@pytest.mark.parametrize(
    "value",
    [1e-4, 1e-6, 1e-8, 0.1 + 0.2, 1 / 3, 2.5e-12, 123456.789, 1e22, 5e-324, 1.7976931348623157e308],
)
def test_floats_round_trip_bit_exactly(tmp_path: Path, value: float) -> None:
    """Dumped floats parse back to the identical double (repr is shortest round-trip)."""
    config = _with("weights.tv", value)
    config = _with("training.lr", min(value, 1.0) or 1e-4, config)
    restored = _round_trip(config, tmp_path)
    assert restored == config
    assert restored.weights.tv.hex() == value.hex()


def test_int_in_float_field_round_trips_equal(tmp_path: Path) -> None:
    """A float field holding an int is written as a float and compares equal after loading."""
    config = _with("model.curve_scale", 1)
    restored = _round_trip(config, tmp_path)
    assert restored == config and type(restored.model.curve_scale) is float


def test_save_validates_before_writing(tmp_path: Path) -> None:
    """An invalid config raises and leaves no file."""
    path = tmp_path / "bad.yaml"
    with pytest.raises(ValueError, match=r"flags\.denoise"):
        save_config(_with("flags.denoise", True), path)
    assert not path.exists()


def test_save_does_not_create_parents(tmp_path: Path) -> None:
    """Parent creation is the caller's job."""
    with pytest.raises(FileNotFoundError):
        save_config(Config(), tmp_path / "missing_dir" / "c.yaml")
    assert not (tmp_path / "missing_dir").exists()


def test_save_keeps_path_strings_verbatim(tmp_path: Path) -> None:
    """Configured paths are not rewritten on save."""
    config = _with("paths.video_dir", "relative/./clips")
    assert _round_trip(config, tmp_path).paths.video_dir == "relative/./clips"


# ---------------------------------------------------------------------------
# Ablation matrix
# ---------------------------------------------------------------------------


def _matrix() -> dict[str, Any]:
    """Parsed ablation matrix."""
    return dict(load_python_mapping(CONFIGS / "ablation_matrix.py", "MATRIX"))


def _rows() -> list[Any]:
    """One param per (run, seed) pair."""
    matrix = _matrix()
    return [
        pytest.param(row, seed, id=f"{row['name']}-{seed}")
        for row in matrix["runs"]
        for seed in matrix["seeds"]
    ]


def test_matrix_structure_and_row_names() -> None:
    """13 unique rows including baseline, full, single-flag and leave-one-out rows."""
    matrix = _matrix()
    names = [row["name"] for row in matrix["runs"]]
    assert len(names) == len(set(names)) == 13
    assert {"baseline", "full", "noise_only", "loo_noise", "loo_temporal"} <= set(names)
    assert matrix["base_config"] == "configs/baseline.py"
    assert matrix["seeds"] == [17, 29, 43] and all(isinstance(s, int) for s in matrix["seeds"])


def test_leave_one_out_rows_remove_exactly_one_extension() -> None:
    """Each loo row equals full minus one extension (noise also removes its dependent denoise)."""
    rows = {row["name"]: row["flags"] for row in _matrix()["runs"]}
    removed = {
        "loo_noise": {"noise_aware", "denoise"},
        "loo_edge": {"edge"},
        "loo_chroma": {"chroma"},
        "loo_adaptive": {"adaptive_exp"},
        "loo_temporal": {"temporal"},
    }
    for name, drop in removed.items():
        expected = {k: (v and k not in drop) for k, v in rows["full"].items()}
        assert rows[name] == expected, name


@pytest.mark.parametrize(("row", "seed"), _rows())
def test_ablation_row_changes_only_flags_and_seed(row: dict[str, Any], seed: int) -> None:
    """Applying a row to the base config validates and differs only in flags and training.seed."""
    base = load_config(CONFIGS / _matrix()["base_config"].split("/")[-1])
    overrides: dict[str, object] = {f"flags.{k}": v for k, v in row["flags"].items()}
    overrides["training.seed"] = seed
    config = load_config(BASELINE, overrides)
    assert config.flags == dataclasses.replace(base.flags, **row["flags"])
    before, after = _plain(base), _plain(config)
    for data in (before, after):
        data.pop("flags")
        data["training"].pop("seed")
    assert before == after
    assert config.training.seed == seed


@pytest.mark.parametrize("row", [pytest.param(r, id=r["name"]) for r in _matrix()["runs"]])
def test_ablation_rows_round_trip(row: dict[str, Any], tmp_path: Path) -> None:
    """Every ablation row survives save/load with identical flags and weights."""
    config = load_config(BASELINE, {f"flags.{k}": v for k, v in row["flags"].items()})
    restored = _round_trip(config, tmp_path)
    assert restored == config and restored.flags == config.flags
    assert restored.weights == config.weights


def test_defaults_and_full_yaml_carry_every_field_and_exposure_gate() -> None:
    """Defaults and shipped YAML carry every dataclass field; adaptive replaces baseline exp."""
    defaults = _plain(Config())
    for name in ("baseline.py", "full.py"):
        raw = load_python_mapping(CONFIGS / name, "CONFIG")
        assert raw.keys() == defaults.keys()
        for section, fields in defaults.items():
            if isinstance(fields, dict):
                assert raw[section].keys() == fields.keys()
    full = load_config(FULL)
    assert full.flags.denoise and full.flags.noise_aware
    assert LOSS_REGISTRY["exp"].flag == "not adaptive_exp"
    assert LOSS_REGISTRY["highlight"].flag == "adaptive_exp"
    # Adaptive exposure reuses weights.exp, so enabling it changes no weight.
    adaptive = load_config(BASELINE, {"flags.adaptive_exp": True})
    assert adaptive.weights == load_config(BASELINE).weights


# ---------------------------------------------------------------------------
# Workspace root and path checks
# ---------------------------------------------------------------------------


def test_workspace_root_contains_pyproject() -> None:
    """workspace_root is the directory with pyproject.toml, independent of the cwd."""
    root = workspace_root()
    assert (root / "pyproject.toml").is_file() and root.is_absolute()
    assert root == Path(__file__).resolve().parents[1]


def test_workspace_root_ignores_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Changing directory does not change the root."""
    expected = workspace_root()
    monkeypatch.chdir(tmp_path)
    assert workspace_root() == expected


def test_resolve_path() -> None:
    """Absolute paths are kept; relative paths join the workspace root; no existence check."""
    absolute = Path(Path.cwd().anchor) / "some" / "abs" / "dir"  # absolute on POSIX and Windows
    assert resolve_path(absolute) == absolute
    assert resolve_path(str(absolute)) == absolute
    assert resolve_path("data/x") == workspace_root() / "data" / "x"
    assert resolve_path(Path("runs")) == workspace_root() / "runs"


def _config_with_roots(tmp_path: Path, **overrides: str) -> Config:
    """Config whose path roots point inside tmp_path (missing unless created by the test)."""
    config = Config()
    for name in (item.name for item in dataclasses.fields(config.paths)):
        config = _with(f"paths.{name}", overrides.get(name, str(tmp_path / name)), config)
    return config


def test_check_paths_passes_for_existing_roots(tmp_path: Path) -> None:
    """Existing roots pass; absolute strings are used as given."""
    (tmp_path / "train_images").mkdir()
    (tmp_path / "unpaired").mkdir()
    config = _config_with_roots(tmp_path)
    check_paths(config, ["train_images", "unpaired"])
    check_paths(config, [])


def test_check_paths_reports_all_missing_roots(tmp_path: Path) -> None:
    """Missing roots raise FileNotFoundError naming each field and the resolved path."""
    (tmp_path / "train_images").mkdir()
    config = _config_with_roots(tmp_path)
    with pytest.raises(FileNotFoundError) as info:
        check_paths(config, ["train_images", "eval_low", "eval_high"])
    message = str(info.value)
    assert "paths.eval_low" in message and "paths.eval_high" in message
    assert "paths.train_images" not in message


def test_check_paths_resolves_relative_against_workspace_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Relative roots are looked up under the workspace root, not the cwd."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()  # a decoy in the cwd must not count
    config = _with("paths.run_dir", "configs")
    check_paths(config, ["run_dir"])  # exists under the workspace root
    config = _with("paths.run_dir", "definitely_not_here_dir")
    with pytest.raises(FileNotFoundError, match="definitely_not_here_dir"):
        check_paths(config, ["run_dir"])


def test_check_paths_video_dir_only_for_real_pairs(tmp_path: Path) -> None:
    """video_dir is required only when temporal.pair_source is 'real'."""
    config = _config_with_roots(tmp_path)
    check_paths(config, ["video_dir"])  # synthetic: skipped
    real = _with("temporal.pair_source", "real", config)
    with pytest.raises(FileNotFoundError, match=r"paths\.video_dir"):
        check_paths(real, ["video_dir"])
    (tmp_path / "video_dir").mkdir()
    check_paths(real, ["video_dir"])


def test_check_paths_rejects_unknown_root_and_empty_value(tmp_path: Path) -> None:
    """Unknown names and empty path strings are ValueErrors, not missing files."""
    config = _config_with_roots(tmp_path)
    with pytest.raises(ValueError, match="did you mean 'train_images'"):
        check_paths(config, ["train_image"])
    with pytest.raises(ValueError, match="unknown path root"):
        check_paths(config, ["bogus"])
    with pytest.raises(ValueError, match=r"paths\.run_dir"):
        check_paths(_with("paths.run_dir", ""), ["run_dir"])


# Active collection-time checks avoid implementing a function in this stub-only scaffold.
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
for config_file in ["baseline.py", "full.py"]:
    actual = load_python_mapping(ROOT / "configs" / config_file, "CONFIG")
    assert actual.keys() == defaults.keys()
    for section, fields in defaults.items():
        if isinstance(fields, dict):
            assert actual[section].keys() == fields.keys(), (config_file, section)
    assert not actual["flags"]["denoise"] or actual["flags"]["noise_aware"]
matrix = load_python_mapping(ROOT / "configs" / "ablation_matrix.py", "MATRIX")
assert len(matrix["runs"]) == 13
assert matrix["seeds"] == [17, 29, 43]
for row in matrix["runs"]:
    assert row["flags"].keys() == defaults["flags"].keys()
    assert not row["flags"]["denoise"] or row["flags"]["noise_aware"]

# M0 stub-only AST audit retired: implementation has begun (docs/ARCHITECTURE.md).
# Every public function must still have a docstring and full annotations.
for source in [*(ROOT / "src").rglob("*.py"), *(ROOT / "scripts").glob("*.py")]:
    for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            assert node.returns is not None, (source, node.name)
            assert all(a.annotation is not None for a in node.args.args if a.arg != "self")

assert LOSS_REGISTRY.keys() == LOSS_CONTRACTS.keys()
for loss_name, loss_spec in LOSS_REGISTRY.items():
    assert callable(loss_spec.callable)
    assert (loss_spec.flag, loss_spec.default_weight, loss_spec.required_inputs) == LOSS_CONTRACTS[
        loss_name
    ][1:]
