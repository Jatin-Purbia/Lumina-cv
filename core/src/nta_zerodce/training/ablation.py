"""Ablation matrix parsing, validation and expansion (docs/EXPERIMENTS.md, INTERFACES).

``configs/ablation_matrix.py`` (``MATRIX``; a YAML file is also accepted) has exactly three keys::

    base_config: configs/baseline.py       # complete experiment config, workspace-relative
    seeds: [17, 29, 43]                    # unique nonnegative integers
    runs:                                  # unique names, complete FeatureFlags mappings
      - name: baseline
        flags: {noise_aware: false, denoise: false, edge: false, ...}

Every run inherits all base hyperparameters and overrides only its flags and the seed. The
matrix must contain the rows ``baseline``, ``full``, each single extension (``noise_only``,
``edge_only``, ``chroma_only``, ``adaptive_only``, ``temporal_only``) and each leave-one-out row
(``loo_noise``, ``loo_edge``, ``loo_chroma``, ``loo_adaptive``, ``loo_temporal``) with exactly
the flags EXPERIMENTS.md assigns them; ``noise_denoise`` is validated when present.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from nta_zerodce.training.config import (
    Config,
    FeatureFlags,
    load_python_mapping,
    override_config,
    parse_config,
    resolve_path,
)

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_FLAG_NAMES = tuple(item.name for item in dataclasses.fields(FeatureFlags))
_EXTENSIONS: dict[str, tuple[str, ...]] = {
    "noise": ("noise_aware", "denoise"),
    "edge": ("edge",),
    "chroma": ("chroma",),
    "adaptive": ("adaptive_exp",),
    "temporal": ("temporal",),
}
_SINGLE_FLAGS: dict[str, tuple[str, ...]] = {
    "noise_only": ("noise_aware",),
    "edge_only": ("edge",),
    "chroma_only": ("chroma",),
    "adaptive_only": ("adaptive_exp",),
    "temporal_only": ("temporal",),
    "noise_denoise": ("noise_aware", "denoise"),
}


def _on(*names: str) -> dict[str, bool]:
    """Flag mapping with exactly ``names`` enabled."""
    return {flag: flag in names for flag in _FLAG_NAMES}


def required_flags() -> dict[str, dict[str, bool]]:
    """Flags every required row must carry (plus the optional ``noise_denoise`` row).

    Returns:
        Row name to complete flag mapping: baseline, full, the five singles, ``noise_denoise``
        and the five leave-one-out rows (``loo_<extension>`` removes all flags of that
        extension, so ``loo_noise`` also drops denoise).
    """
    full = {flag for flags in _EXTENSIONS.values() for flag in flags}
    rows: dict[str, dict[str, bool]] = {"baseline": _on(), "full": _on(*full)}
    for name, flags in _SINGLE_FLAGS.items():
        rows[name] = _on(*flags)
    for extension, flags in _EXTENSIONS.items():
        rows[f"loo_{extension}"] = _on(*(full - set(flags)))
    return rows


REQUIRED_ROWS = tuple(name for name in required_flags() if name != "noise_denoise")


@dataclass(frozen=True)
class RunSpec:
    """One (run, seed) cell of the matrix.

    Attributes:
        name: Run name from the matrix (``baseline``, ``loo_edge``, ...).
        seed: Training seed.
        flags: The run's feature flags.
        config: Fully resolved, validated Config: base hyperparameters with the run's flags and
            seed (and any global overrides given to ``expand_matrix``).
    """

    name: str
    seed: int
    flags: FeatureFlags
    config: Config

    def leaf(self, output: Path) -> Path:
        """Directory ``<output>/<run>/seed_<seed>`` holding this cell's config, logs, checkpoints."""
        return Path(output) / self.name / f"seed_{self.seed}"


def _is_int(value: object) -> bool:
    """True for real integers (``bool`` is not accepted)."""
    return isinstance(value, int) and not isinstance(value, bool)


def parse_matrix(
    path: str | Path,
) -> tuple[str, tuple[int, ...], tuple[tuple[str, FeatureFlags], ...]]:
    """Parse and validate the matrix file.

    Args:
        path: Matrix YAML (workspace-relative or absolute).

    Returns:
        ``(base_config, seeds, runs)`` with ``runs`` as ``(name, FeatureFlags)`` pairs in file
        order.

    Raises:
        FileNotFoundError: The matrix file does not exist.
        ValueError: Any structural, duplicate, dependency or missing-row problem; the message
            names the offending entry.
    """
    resolved = resolve_path(path)
    if resolved.suffix == ".py":
        raw = load_python_mapping(resolved, "MATRIX")
    else:
        try:
            text = resolved.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise FileNotFoundError(f"ablation matrix not found: {resolved}") from None
        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as err:
            raise ValueError(f"cannot parse ablation matrix {resolved}: {err}") from None
    if not isinstance(raw, dict):
        raise ValueError(f"ablation matrix {resolved} must be a YAML mapping")
    expected = {"base_config", "seeds", "runs"}
    if set(raw) != expected:
        raise ValueError(
            f"ablation matrix keys must be exactly {sorted(expected)}; missing "
            f"{sorted(expected - set(raw))}, unknown {sorted(set(raw) - expected)}"
        )
    base = raw["base_config"]
    if not isinstance(base, str) or not base.strip():
        raise ValueError("base_config must be a nonempty path string")
    seeds = raw["seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seeds must be a nonempty list of integers")
    for seed in seeds:
        if not _is_int(seed) or seed < 0 or seed > 2**32 - 1:
            raise ValueError(f"seeds must be nonnegative integers (< 2**32); got {seed!r}")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds must be unique; got {seeds}")
    entries = raw["runs"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("runs must be a nonempty list")
    runs: list[tuple[str, FeatureFlags]] = []
    seen: set[str] = set()
    for position, entry in enumerate(entries):
        where = f"runs[{position}]"
        if not isinstance(entry, dict) or set(entry) != {"name", "flags"}:
            raise ValueError(f"{where} must be a mapping with exactly the keys name and flags")
        name = entry["name"]
        if not isinstance(name, str) or not _NAME.match(name):
            raise ValueError(f"{where}: name must match {_NAME.pattern}; got {name!r}")
        if name in seen:
            raise ValueError(f"duplicate run name {name!r}")
        seen.add(name)
        flags = entry["flags"]
        if not isinstance(flags, dict) or set(flags) != set(_FLAG_NAMES):
            have = sorted(flags) if isinstance(flags, dict) else flags
            raise ValueError(
                f"run {name!r}: flags must be a complete mapping of {list(_FLAG_NAMES)}; got {have}"
            )
        for flag, value in flags.items():
            if not isinstance(value, bool):
                raise ValueError(f"run {name!r}: flags.{flag} must be true or false; got {value!r}")
        if flags["denoise"] and not flags["noise_aware"]:
            raise ValueError(f"run {name!r}: denoise requires noise_aware")
        runs.append((name, FeatureFlags(**flags)))
    table = required_flags()
    by_name = dict(runs)
    missing = [name for name in REQUIRED_ROWS if name not in by_name]
    if missing:
        raise ValueError(f"ablation matrix lacks required rows: {missing}")
    for name, flags in runs:
        if name in table and dataclasses.asdict(flags) != table[name]:
            raise ValueError(
                f"run {name!r} flags {dataclasses.asdict(flags)} differ from the protocol row "
                f"{table[name]} (docs/EXPERIMENTS.md)"
            )
    return base, tuple(int(s) for s in seeds), tuple(runs)


def _resolve_base(base: str, matrix: Path) -> Path:
    """Resolve ``base_config``: workspace-relative first, then next to the matrix file."""
    primary = resolve_path(base)
    if primary.is_file():
        return primary
    sibling = resolve_path(matrix).parent / base
    if sibling.is_file():
        return sibling
    raise FileNotFoundError(f"base_config not found: {primary}")


def expand_matrix(
    path: str | Path,
    overrides: Mapping[str, object] | None = None,
    *,
    base_config: str | Path | None = None,
) -> list[RunSpec]:
    """Expand the matrix into runs x seeds in a deterministic order.

    Args:
        path: Matrix YAML (see the module docstring).
        overrides: Dotted overrides applied to the base config for EVERY run before the run's
            flags and seed (for example ``{"training.epochs": 1}`` for smoke runs, or paths).
            Values of None are ignored. ``flags.*`` and ``training.seed`` are rejected because
            the matrix owns them.
        base_config: Optional replacement for the matrix's ``base_config``.

    Returns:
        One RunSpec per (run, seed): runs in file order, seeds in file order within each run.

    Raises:
        FileNotFoundError: Matrix or base config missing.
        ValueError: Invalid matrix or config, or an override of flags or seed.
    """
    base, seeds, runs = parse_matrix(path)
    forbidden = [
        key
        for key, value in (overrides or {}).items()
        if value is not None and (key.startswith("flags.") or key == "training.seed")
    ]
    if forbidden:
        raise ValueError(
            f"{forbidden} cannot be overridden for an ablation: the matrix sets flags and seeds"
        )
    base_path = (
        resolve_path(base_config) if base_config is not None else _resolve_base(base, Path(path))
    )
    base_cfg = override_config(parse_config(base_path), overrides)
    specs: list[RunSpec] = []
    for name, flags in runs:
        for seed in seeds:
            cfg = dataclasses.replace(
                base_cfg,
                flags=flags,
                training=dataclasses.replace(base_cfg.training, seed=seed),
            )
            cfg.validate()
            specs.append(RunSpec(name, seed, flags, cfg))
    return specs
