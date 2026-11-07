"""Flag combinations, YAML round trips and CLI precedence; TESTING strategy.

See docs/INTERFACES.md (CLI and checkpoint contracts) and docs/TESTING.md. Checkpoint-level flag
rejection (noise/light/iteration mismatch) lives with the checkpoint implementation; here the
config-side halves are covered: schema_version gating and exact flag/weight restoration.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
from pathlib import Path

import pytest
import yaml

from nta_zerodce.cli import add_common_arguments, resolve_config
from nta_zerodce.training.config import Config, FeatureFlags, load_config, save_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
FLAG_NAMES = [item.name for item in dataclasses.fields(FeatureFlags)]
COMBINATIONS = [
    dict(zip(FLAG_NAMES, bits, strict=True))
    for bits in itertools.product([False, True], repeat=len(FLAG_NAMES))
]
VALID = [c for c in COMBINATIONS if c["noise_aware"] or not c["denoise"]]
INVALID = [c for c in COMBINATIONS if c["denoise"] and not c["noise_aware"]]


def _id(combo: dict[str, bool]) -> str:
    """Compact readable id such as 'noise+edge' or 'none'."""
    return "+".join(name for name, on in combo.items() if on) or "none"


def _parse(*argv: str) -> argparse.Namespace:
    """Parse a command line with only the shared flags."""
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    return parser.parse_args(list(argv))


def test_combination_counts() -> None:
    """7 flags give 128 combinations; 32 of them enable denoise without noise."""
    assert len(COMBINATIONS) == 128 and len(INVALID) == 32 and len(VALID) == 96


@pytest.mark.parametrize("combo", VALID, ids=_id)
def test_valid_combination_round_trips_flags_and_weights(
    combo: dict[str, bool], tmp_path: Path
) -> None:
    """Every valid extension combination saves to YAML and restores exact flags and weights."""
    config = dataclasses.replace(Config(), flags=FeatureFlags(**combo))
    path = tmp_path / "c.yaml"
    save_config(config, path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["flags"] == combo and raw["schema_version"] == 1
    restored = load_config(path)
    assert restored == config
    assert dataclasses.asdict(restored.flags) == combo
    assert restored.weights == config.weights


@pytest.mark.parametrize("combo", INVALID, ids=_id)
def test_denoise_without_noise_is_rejected(combo: dict[str, bool], tmp_path: Path) -> None:
    """denoise without noise_aware raises ValueError at validate, save and load."""
    config = dataclasses.replace(Config(), flags=FeatureFlags(**combo))
    with pytest.raises(ValueError, match="denoise"):
        config.validate()
    with pytest.raises(ValueError, match="denoise"):
        save_config(config, tmp_path / "c.yaml")
    data = dataclasses.asdict(Config())
    data["flags"] = combo
    path = tmp_path / "raw.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="denoise"):
        load_config(path)


def test_changed_weights_round_trip(tmp_path: Path) -> None:
    """Non-default weights and flags survive a round trip together."""
    flags = FeatureFlags(noise_aware=True, denoise=True, edge=True)
    weights = dataclasses.replace(Config().weights, exp=7.25, tv=0.1 + 0.2, temporal=0)
    config = dataclasses.replace(Config(), flags=flags, weights=weights)
    path = tmp_path / "c.yaml"
    save_config(config, path)
    restored = load_config(path)
    assert restored.flags == flags and restored.weights == weights


def test_unknown_schema_version_is_rejected(tmp_path: Path) -> None:
    """Only schema_version 1 loads; the saved file always records 1."""
    path = tmp_path / "c.yaml"
    save_config(Config(), path)
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["schema_version"] == 1
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["schema_version"] = 2
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_config(path)


# ---------------------------------------------------------------------------
# CLI precedence: schema defaults < YAML < explicit CLI; None is absent, not false.
# ---------------------------------------------------------------------------


def test_absent_flag_preserves_yaml_true_and_false() -> None:
    """With no flags given, the YAML values are untouched (including true flags)."""
    full_config = resolve_config(_parse("--config", str(CONFIGS / "full.py")))
    assert full_config == load_config(CONFIGS / "full.py")
    assert full_config.flags.noise_aware and full_config.flags.denoise
    base_config = resolve_config(_parse("--config", str(CONFIGS / "baseline.py")))
    assert not any(dataclasses.asdict(base_config.flags).values())


def test_default_config_argument_is_baseline() -> None:
    """The default --config is configs/baseline.py resolved against the workspace root."""
    assert resolve_config(_parse()) == load_config(CONFIGS / "baseline.py")


@pytest.mark.parametrize(
    ("option", "field"),
    [
        ("noise", "noise_aware"),
        ("edge", "edge"),
        ("chroma", "chroma"),
        ("adaptive-exp", "adaptive_exp"),
        ("temporal", "temporal"),
        ("light", "light"),
    ],
)
def test_explicit_false_overrides_yaml_true(option: str, field: str) -> None:
    """--no-<flag> beats YAML true, and --<flag> beats YAML false."""
    full = str(CONFIGS / "full.py")
    base = str(CONFIGS / "baseline.py")
    off = resolve_config(_parse("--config", full, "--no-denoise", f"--no-{option}"))
    assert getattr(off.flags, field) is False
    on = resolve_config(_parse("--config", base, f"--{option}"))
    assert getattr(on.flags, field) is True
    assert not on.flags.denoise


def test_explicit_no_noise_with_yaml_denoise_fails_validation() -> None:
    """--no-noise while YAML denoise stays on is rejected with a message naming the flag."""
    with pytest.raises(ValueError, match=r"--no-noise") as info:
        resolve_config(_parse("--config", str(CONFIGS / "full.py"), "--no-noise"))
    assert "denoise" in str(info.value)


def test_no_noise_together_with_no_denoise_is_valid() -> None:
    """Turning both off resolves cleanly and leaves the other YAML flags alone."""
    config = resolve_config(
        _parse("--config", str(CONFIGS / "full.py"), "--no-noise", "--no-denoise")
    )
    assert not config.flags.noise_aware and not config.flags.denoise
    assert config.flags.edge and config.flags.chroma and config.flags.temporal


def test_denoise_alone_on_baseline_fails_and_with_noise_passes() -> None:
    """--denoise needs --noise when YAML has noise disabled."""
    base = str(CONFIGS / "baseline.py")
    with pytest.raises(ValueError, match="denoise"):
        resolve_config(_parse("--config", base, "--denoise"))
    config = resolve_config(_parse("--config", base, "--noise", "--denoise"))
    assert config.flags.noise_aware and config.flags.denoise


def test_value_flags_override_yaml() -> None:
    """--curve-scale, --video-dir, --seed and --device override their YAML fields."""
    config = resolve_config(
        _parse(
            "--config",
            str(CONFIGS / "full.py"),
            "--curve-scale",
            "0.5",
            "--video-dir",
            "elsewhere/videos",
            "--seed",
            "29",
            "--device",
            "cpu",
        )
    )
    assert config.model.curve_scale == 0.5
    assert config.paths.video_dir == "elsewhere/videos"
    assert config.training.seed == 29 and config.training.device == "cpu"
    assert config.training.lr == load_config(CONFIGS / "full.py").training.lr


@pytest.mark.parametrize(
    "argv",
    [
        ("--curve-scale", "1.5"),
        ("--curve-scale", "0"),
        ("--curve-scale", "nan"),
        ("--seed", "-1"),
        ("--device", "tpu"),
    ],
)
def test_invalid_cli_values_fail_validation(argv: tuple[str, ...]) -> None:
    """Out-of-range CLI values are rejected by the same validation as YAML values."""
    with pytest.raises(ValueError):
        resolve_config(_parse("--config", str(CONFIGS / "baseline.py"), *argv))


def test_every_flag_set_at_once_matches_manual_replace() -> None:
    """The full-flags command line equals editing the baseline flags directly."""
    argv = ["--config", str(CONFIGS / "baseline.py")]
    argv += ["--noise", "--denoise", "--edge", "--chroma", "--adaptive-exp", "--temporal"]
    config = resolve_config(_parse(*argv))
    expected = dataclasses.replace(
        Config(),
        flags=FeatureFlags(
            noise_aware=True, denoise=True, edge=True, chroma=True, adaptive_exp=True, temporal=True
        ),
    )
    assert config == expected == load_config(CONFIGS / "full.py")
