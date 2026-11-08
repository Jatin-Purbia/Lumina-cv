"""Typed YAML schema, validation and YAML IO; INTERFACES config schema and METHOD constants.

Public helpers: ``Config.validate``, ``load_config`` (strict, complete YAML), ``parse_config``
(structure only), ``save_config``, ``check_paths``, ``workspace_root`` and ``resolve_path``.
See docs/ARCHITECTURE.md for the normative contract.
"""

from __future__ import annotations

import copy
import dataclasses
import difflib
import importlib.util
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn, cast, get_type_hints

import yaml


@dataclass(frozen=True)
class FeatureFlags:
    """FeatureFlags fields and defaults; see INTERFACES config table.

    Args:
        noise_aware: bool; default False. Units and bounds in INTERFACES.
        denoise: bool; default False. Units and bounds in INTERFACES.
        edge: bool; default False. Units and bounds in INTERFACES.
        chroma: bool; default False. Units and bounds in INTERFACES.
        adaptive_exp: bool; default False. Units and bounds in INTERFACES.
        temporal: bool; default False. Units and bounds in INTERFACES.
        light: bool; default False. Units and bounds in INTERFACES.

    Returns:
        Immutable schema record; no tensor computation.

    Method:
        Project configuration contract, not a paper algorithm.
    """

    noise_aware: bool = False
    denoise: bool = False
    edge: bool = False
    chroma: bool = False
    adaptive_exp: bool = False
    temporal: bool = False
    light: bool = False


@dataclass(frozen=True)
class ModelConfig:
    """ModelConfig fields and defaults; see INTERFACES config table.

    Args:
        channels: int; default 32. Units and bounds in INTERFACES.
        iterations: int; default 8. Units and bounds in INTERFACES.
        noise_channels: int; default 16. Units and bounds in INTERFACES.
        denoise_channels: int; default 16. Units and bounds in INTERFACES.
        sigma_max: float; default 0.2. Units and bounds in INTERFACES.
        residual_scale: float; default 0.1. Units and bounds in INTERFACES.
        curve_scale: float; default 1.0. Units and bounds in INTERFACES.
        gain_window: int; default 5, odd stride-one local pooling width.
        gain_min: float; default 1.0. Units and bounds in INTERFACES.
        gain_max: float; default 8.0. Units and bounds in INTERFACES.
        eps: float; default 1e-6. Units and bounds in INTERFACES.

    Returns:
        Immutable schema record; no tensor computation.

    Method:
        Project configuration contract, not a paper algorithm.
    """

    channels: int = 32
    iterations: int = 8
    noise_channels: int = 16
    denoise_channels: int = 16
    sigma_max: float = 0.2
    residual_scale: float = 0.1
    curve_scale: float = 1.0
    gain_window: int = 5
    gain_min: float = 1.0
    gain_max: float = 8.0
    eps: float = 1e-6


@dataclass(frozen=True)
class LossWeights:
    """LossWeights fields and defaults; see INTERFACES config table.

    Args:
        spa: float; default 1.0. Units and bounds in INTERFACES.
        exp: float; default 10.0. Units and bounds in INTERFACES.
        col: float; default 5.0. Units and bounds in INTERFACES.
        tv: float; default 200.0. Units and bounds in INTERFACES.
        edge: float; default 20.0. Units and bounds in INTERFACES.
        chroma: float; default 20.0. Units and bounds in INTERFACES.
        highlight: float; default 20.0. Units and bounds in INTERFACES.
        noise: float; default 20.0. Units and bounds in INTERFACES.
        invariance: float; default 10.0. Units and bounds in INTERFACES.
        fidelity: float; default 1.0. Units and bounds in INTERFACES.
        temporal: float; default 10.0. Units and bounds in INTERFACES.

    Returns:
        Immutable schema record; no tensor computation.

    Method:
        Project configuration contract, not a paper algorithm.
    """

    spa: float = 1.0
    exp: float = 10.0
    col: float = 5.0
    tv: float = 200.0
    edge: float = 20.0
    chroma: float = 20.0
    highlight: float = 20.0
    noise: float = 20.0
    invariance: float = 10.0
    fidelity: float = 1.0
    temporal: float = 10.0


@dataclass(frozen=True)
class LossConfig:
    """LossConfig fields and defaults; see INTERFACES config table.

    Args:
        spatial_patch: int; default 4. Units and bounds in INTERFACES.
        exposure_patch: int; default 16. Units and bounds in INTERFACES.
        exposure_target: float; default 0.5 (lowered from the original 0.6 as the first
            over-brightening mitigation named in docs/IMPLEMENTATION.md Limits; unverified by
            retraining). Units and bounds in INTERFACES.
        adaptive_low: float; default 0.55. Units and bounds in INTERFACES.
        adaptive_high: float; default 0.85. Units and bounds in INTERFACES.
        adaptive_noise_scale: float; default 0.05. Units and bounds in INTERFACES.
        adaptive_noise_penalty: float; default 0.15. Units and bounds in INTERFACES.
        highlight_threshold: float; default 0.8. Units and bounds in INTERFACES.
        chroma_window: int; default 5. Units and bounds in INTERFACES.
        reliability_low: float; default 0.05. Units and bounds in INTERFACES.
        reliability_high: float; default 0.2. Units and bounds in INTERFACES.
        reliability_noise: float; default 2.0. Units and bounds in INTERFACES.
        edge_smooth_sigma: float; default 0.8. Units and bounds in INTERFACES.
        edge_kernel: int; default 5. Units and bounds in INTERFACES.
        edge_noise_multiplier: float; default 2.0. Units and bounds in INTERFACES.
        eps: float; default 1e-6. Units and bounds in INTERFACES.

    Returns:
        Immutable schema record; no tensor computation.

    Method:
        Project configuration contract, not a paper algorithm.
    """

    spatial_patch: int = 4
    exposure_patch: int = 16
    exposure_target: float = 0.5
    adaptive_low: float = 0.55
    adaptive_high: float = 0.85
    adaptive_noise_scale: float = 0.05
    adaptive_noise_penalty: float = 0.15
    highlight_threshold: float = 0.8
    chroma_window: int = 5
    reliability_low: float = 0.05
    reliability_high: float = 0.2
    reliability_noise: float = 2.0
    edge_smooth_sigma: float = 0.8
    edge_kernel: int = 5
    edge_noise_multiplier: float = 2.0
    eps: float = 1e-6


@dataclass(frozen=True)
class NoiseConfig:
    """NoiseConfig fields and defaults; see INTERFACES config table.

    Args:
        a_min: float; default 0.0. Units and bounds in INTERFACES.
        a_max: float; default 0.01. Units and bounds in INTERFACES.
        b_min: float; default 0.0. Units and bounds in INTERFACES.
        b_max: float; default 0.02. Units and bounds in INTERFACES.
        probability: float; default 1.0. Units and bounds in INTERFACES.

    Returns:
        Immutable schema record; no tensor computation.

    Method:
        Project configuration contract, not a paper algorithm.
    """

    a_min: float = 0.0
    a_max: float = 0.01
    b_min: float = 0.0
    b_max: float = 0.02
    probability: float = 1.0


@dataclass(frozen=True)
class SyntheticConfig:
    """SyntheticConfig fields and defaults; see INTERFACES config table.

    Args:
        rotation_degrees: float; default 3.0. Units and bounds in INTERFACES.
        translation_pixels: float; default 8.0. Units and bounds in INTERFACES.
        scale_min: float; default 0.98. Units and bounds in INTERFACES.
        scale_max: float; default 1.02. Units and bounds in INTERFACES.
        flicker_min: float; default 0.9. Units and bounds in INTERFACES.
        flicker_max: float; default 1.1. Units and bounds in INTERFACES.

    Returns:
        Immutable schema record; no tensor computation.

    Method:
        Project configuration contract, not a paper algorithm.
    """

    rotation_degrees: float = 3.0
    translation_pixels: float = 8.0
    scale_min: float = 0.98
    scale_max: float = 1.02
    flicker_min: float = 0.9
    flicker_max: float = 1.1


@dataclass(frozen=True)
class FlowConfig:
    """FlowConfig fields and defaults; see INTERFACES config table.

    Args:
        method: str; default "dis". Units and bounds in INTERFACES.
        preset: str; default "medium". Units and bounds in INTERFACES.
        scale: float; default 0.5. Units and bounds in INTERFACES.
        fb_alpha: float; default 0.01. Units and bounds in INTERFACES.
        fb_beta: float; default 0.5. Units and bounds in INTERFACES.
        photometric_threshold: float; default 0.1. Units and bounds in INTERFACES.
        align_corners: bool; default True. Units and bounds in INTERFACES.

    Returns:
        Immutable schema record; no tensor computation.

    Method:
        Project configuration contract, not a paper algorithm.
    """

    method: str = "dis"
    preset: str = "medium"
    scale: float = 0.5
    fb_alpha: float = 0.01
    fb_beta: float = 0.5
    photometric_threshold: float = 0.1
    align_corners: bool = True


@dataclass(frozen=True)
class TemporalConfig:
    """TemporalConfig fields and defaults; see INTERFACES config table.

    Args:
        curve_blend: float; default 0.7. Units and bounds in INTERFACES.
        output_blend: float; default 0.0. Units and bounds in INTERFACES.
        histogram_bins: int; default 32, normalized scene-cut histogram resolution.
        scene_cut_threshold: float; default 0.3. Units and bounds in INTERFACES.
        pair_source: str; default "synthetic". Units and bounds in INTERFACES.
        frame_gap: int; default 1. Units and bounds in INTERFACES.

    Returns:
        Immutable schema record; no tensor computation.

    Method:
        Project configuration contract, not a paper algorithm.
    """

    curve_blend: float = 0.7
    output_blend: float = 0.0
    histogram_bins: int = 32
    scene_cut_threshold: float = 0.3
    pair_source: str = "synthetic"
    frame_gap: int = 1


@dataclass(frozen=True)
class TrainingConfig:
    """TrainingConfig fields and defaults; see INTERFACES config table.

    Args:
        lr: float; default 1e-4. Units and bounds in INTERFACES.
        batch_size: int; default 8. Units and bounds in INTERFACES.
        crop_size: int; default 128. Units and bounds in INTERFACES.
        epochs: int; default 100. Units and bounds in INTERFACES.
        optimizer: str; default "Adam". Units and bounds in INTERFACES.
        adam_beta1: float; default .9, first-moment decay in [0,1).
        adam_beta2: float; default .999, second-moment decay in [0,1).
        adam_eps: float; default 1e-8, positive denominator stability.
        weight_decay: float; default 1e-4. Units and bounds in INTERFACES.
        grad_clip: float; default 0.1. Units and bounds in INTERFACES.
        seed: int; default 17. Units and bounds in INTERFACES.
        workers: int; default 2. Units and bounds in INTERFACES.
        device: str; default "auto". Units and bounds in INTERFACES.
        checkpoint_every: int; default 5. Units and bounds in INTERFACES.
        log_every: int; default 10. Units and bounds in INTERFACES.

    Returns:
        Immutable schema record; no tensor computation.

    Method:
        Project configuration contract, not a paper algorithm.
    """

    lr: float = 1e-4
    batch_size: int = 8
    crop_size: int = 128
    epochs: int = 100
    optimizer: str = "Adam"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    weight_decay: float = 1e-4
    grad_clip: float = 0.1
    seed: int = 17
    workers: int = 2
    device: str = "auto"
    checkpoint_every: int = 5
    log_every: int = 10


@dataclass(frozen=True)
class DataPaths:
    """DataPaths fields and defaults; see INTERFACES config table.

    Args:
        train_images: str; default "data/LOL/train_low". Units and bounds in INTERFACES.
        eval_low: str; default "data/LOL/eval15/low". Units and bounds in INTERFACES.
        eval_high: str; default "data/LOL/eval15/high". Units and bounds in INTERFACES.
        unpaired: str; default "data/unpaired". Units and bounds in INTERFACES.
        video_dir: str; default "data/videos/train". Units and bounds in INTERFACES.
        run_dir: str; default "runs". Units and bounds in INTERFACES.
        checkpoint_dir: str; default "checkpoints". Units and bounds in INTERFACES.

    Returns:
        Immutable schema record; no tensor computation.

    Method:
        Project configuration contract, not a paper algorithm.
    """

    train_images: str = "data/LOL/train_low"
    eval_low: str = "data/LOL/eval15/low"
    eval_high: str = "data/LOL/eval15/high"
    unpaired: str = "data/unpaired"
    video_dir: str = "data/videos/train"
    run_dir: str = "runs"
    checkpoint_dir: str = "checkpoints"


@dataclass(frozen=True)
class Config:
    """Root schema; nested sections below, schema_version=1.

    Args: Nested immutable records with no tensor payloads.
    Returns: Serializable experiment specification; call validate before execution.
    Method: INTERFACES config schema and EXPERIMENTS fixed protocol.
    """

    schema_version: int = 1
    flags: FeatureFlags = field(default_factory=FeatureFlags)
    model: ModelConfig = field(default_factory=ModelConfig)
    weights: LossWeights = field(default_factory=LossWeights)
    loss: LossConfig = field(default_factory=LossConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: DataPaths = field(default_factory=DataPaths)

    def validate(self) -> None:
        """Validate schema bounds and cross-field constraints.

        Args:
            self: Config with every YAML field resolved; reject unknown fields and nonfinite values.

        Returns:
            None on success; ValueError for invalid flags, bounds, paths or schema version.

        Method:
            INTERFACES validation checklist; denoise implies noise_aware; temporal pair source must be valid.
            Scalar types are checked first (bool is never accepted as int or float, ints are
            accepted for float fields, NaN and infinity are rejected), then every per-section bound
            and cross-section constraint (loss patches divide training.crop_size, model.sigma_max
            covers sqrt(noise.a_max + noise.b_max**2)). Path existence is NOT checked here; see
            check_paths.

        Raises:
            ValueError: Message names the offending dotted field, the requirement and the value.
        """
        _check_types(self, "")
        if self.schema_version != SCHEMA_VERSION:
            _fail("schema_version", f"exactly {SCHEMA_VERSION}", self.schema_version)
        _validate_flags(self.flags)
        _validate_model(self.model)
        _validate_weights(self.weights)
        _validate_loss(self.loss)
        _validate_noise(self.noise)
        _validate_synthetic(self.synthetic)
        _validate_flow(self.flow)
        _validate_temporal(self.temporal)
        _validate_training(self.training)
        _validate_paths(self.paths)
        _validate_cross(self)


SCHEMA_VERSION = 1
"""The only supported ``schema_version``."""

_DEVICE_PATTERN = re.compile(r"^(auto|cpu|cuda|cuda:[0-9]+)$")
_FLOAT_NO_DOT = re.compile(r"^[-+]?[0-9][0-9_]*[eE][-+]?[0-9]+$")
_FLOW_PRESETS = ("ultrafast", "fast", "medium")
_PAIR_SOURCES = ("synthetic", "real")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _hints(cls: type[Any]) -> dict[str, Any]:
    """Return resolved field annotations of a dataclass ."""
    return get_type_hints(cls)


def _is_section(hint: object) -> bool:
    """True when an annotation (or a type) is a nested config dataclass."""
    return isinstance(hint, type) and dataclasses.is_dataclass(hint)


def _fail(path: str, requirement: str, value: object) -> NoReturn:
    """Raise the standard actionable ValueError naming the dotted field."""
    raise ValueError(f"{path} must be {requirement}; got {value!r}")


def _check_types(obj: object, prefix: str) -> None:
    """Check every field of a dataclass tree against its annotation (no bounds)."""
    hints = _hints(type(obj))
    for item in dataclasses.fields(cast(Any, obj)):
        path = prefix + item.name
        value = getattr(obj, item.name)
        hint = hints[item.name]
        if _is_section(hint):
            if not isinstance(value, hint):
                _fail(path, f"a {hint.__name__} record", value)
            _check_types(value, path + ".")
        elif hint is bool:
            if not isinstance(value, bool):
                _fail(path, "a boolean (true or false, not an int or string)", value)
        elif hint is int:
            if isinstance(value, bool) or not isinstance(value, int):
                _fail(path, "an integer (bool is not accepted)", value)
        elif hint is float:
            if isinstance(value, bool) or not isinstance(value, int | float):
                _fail(path, "a finite number (bool is not accepted)", value)
            if not math.isfinite(value):
                _fail(path, "finite (NaN and infinity are rejected)", value)
        elif hint is str:
            if not isinstance(value, str):
                _fail(path, "a string", value)


def _gt(path: str, value: float, low: float) -> None:
    """Require value > low."""
    if not value > low:
        _fail(path, f"> {low:g}", value)


def _ge(path: str, value: float, low: float) -> None:
    """Require value >= low."""
    if not value >= low:
        _fail(path, f">= {low:g}", value)


def _closed01(path: str, value: float) -> None:
    """Require value in [0, 1]."""
    if not 0.0 <= value <= 1.0:
        _fail(path, "in [0, 1]", value)


def _unit_half_open(path: str, value: float) -> None:
    """Require value in (0, 1]."""
    if not 0.0 < value <= 1.0:
        _fail(path, "in (0, 1]", value)


def _odd(path: str, value: int) -> None:
    """Require a positive odd integer."""
    if value < 1 or value % 2 == 0:
        _fail(path, "a positive odd integer", value)


def _ordered(low_path: str, low: float, high_path: str, high: float, strict: bool) -> None:
    """Require low <= high (or low < high when strict)."""
    if (low >= high) if strict else (low > high):
        op = "<" if strict else "<="
        raise ValueError(f"{low_path} must be {op} {high_path}; got {low!r} and {high!r}")


def _validate_flags(flags: FeatureFlags) -> None:
    """Flags are booleans (checked earlier); denoise implies noise_aware."""
    if flags.denoise and not flags.noise_aware:
        raise ValueError(
            "flags.denoise=true requires flags.noise_aware=true "
            "(the denoiser consumes the noise estimate); enable noise_aware or disable denoise"
        )


def _validate_model(model: ModelConfig) -> None:
    """INTERFACES model row."""
    for name in ("channels", "noise_channels", "denoise_channels"):
        _ge(f"model.{name}", getattr(model, name), 1)
    if model.iterations != 8:
        _fail(
            "model.iterations", "exactly 8 (the curve contract has 24 channels)", model.iterations
        )
    _unit_half_open("model.sigma_max", model.sigma_max)
    _unit_half_open("model.residual_scale", model.residual_scale)
    _unit_half_open("model.curve_scale", model.curve_scale)
    _odd("model.gain_window", model.gain_window)
    _ge("model.gain_min", model.gain_min, 1.0)
    _ordered("model.gain_min", model.gain_min, "model.gain_max", model.gain_max, strict=False)
    _gt("model.eps", model.eps, 0.0)


def _validate_weights(weights: LossWeights) -> None:
    """All loss weights are finite and nonnegative."""
    for item in dataclasses.fields(weights):
        _ge(f"weights.{item.name}", getattr(weights, item.name), 0.0)


def _validate_loss(loss: LossConfig) -> None:
    """INTERFACES loss row (the crop divisibility is a cross-section check)."""
    _ge("loss.spatial_patch", loss.spatial_patch, 1)
    _ge("loss.exposure_patch", loss.exposure_patch, 1)
    for name in (
        "exposure_target",
        "adaptive_low",
        "adaptive_high",
        "highlight_threshold",
        "reliability_low",
        "reliability_high",
    ):
        _closed01(f"loss.{name}", getattr(loss, name))
    _ordered("loss.adaptive_low", loss.adaptive_low, "loss.adaptive_high", loss.adaptive_high, True)
    _ordered(
        "loss.reliability_low",
        loss.reliability_low,
        "loss.reliability_high",
        loss.reliability_high,
        True,
    )
    for name in ("adaptive_noise_scale", "reliability_noise", "edge_noise_multiplier", "eps"):
        _gt(f"loss.{name}", getattr(loss, name), 0.0)
    _ge("loss.adaptive_noise_penalty", loss.adaptive_noise_penalty, 0.0)
    _gt("loss.edge_smooth_sigma", loss.edge_smooth_sigma, 0.0)
    _odd("loss.edge_kernel", loss.edge_kernel)
    _odd("loss.chroma_window", loss.chroma_window)


def _validate_noise(noise: NoiseConfig) -> None:
    """INTERFACES noise row (the sigma_max coupling is a cross-section check)."""
    for name in ("a_min", "a_max", "b_min", "b_max"):
        _ge(f"noise.{name}", getattr(noise, name), 0.0)
    _ordered("noise.a_min", noise.a_min, "noise.a_max", noise.a_max, strict=False)
    _ordered("noise.b_min", noise.b_min, "noise.b_max", noise.b_max, strict=False)
    _closed01("noise.probability", noise.probability)


def _validate_synthetic(synthetic: SyntheticConfig) -> None:
    """INTERFACES synthetic row."""
    _ge("synthetic.rotation_degrees", synthetic.rotation_degrees, 0.0)
    _ge("synthetic.translation_pixels", synthetic.translation_pixels, 0.0)
    for low, high in (("scale_min", "scale_max"), ("flicker_min", "flicker_max")):
        _gt(f"synthetic.{low}", getattr(synthetic, low), 0.0)
        _gt(f"synthetic.{high}", getattr(synthetic, high), 0.0)
        _ordered(
            f"synthetic.{low}",
            getattr(synthetic, low),
            f"synthetic.{high}",
            getattr(synthetic, high),
            strict=False,
        )


def _validate_flow(flow: FlowConfig) -> None:
    """INTERFACES flow row."""
    if flow.method != "dis":
        _fail("flow.method", "exactly 'dis'", flow.method)
    if flow.preset not in _FLOW_PRESETS:
        _fail("flow.preset", f"one of {', '.join(_FLOW_PRESETS)}", flow.preset)
    _unit_half_open("flow.scale", flow.scale)
    _ge("flow.fb_alpha", flow.fb_alpha, 0.0)
    _ge("flow.fb_beta", flow.fb_beta, 0.0)
    _closed01("flow.photometric_threshold", flow.photometric_threshold)
    if flow.align_corners is not True:
        _fail("flow.align_corners", "true (the flow convention is fixed)", flow.align_corners)


def _validate_temporal(temporal: TemporalConfig) -> None:
    """INTERFACES temporal row."""
    _closed01("temporal.curve_blend", temporal.curve_blend)
    _closed01("temporal.output_blend", temporal.output_blend)
    _closed01("temporal.scene_cut_threshold", temporal.scene_cut_threshold)
    _ge("temporal.histogram_bins", temporal.histogram_bins, 2)
    if temporal.pair_source not in _PAIR_SOURCES:
        _fail("temporal.pair_source", f"one of {', '.join(_PAIR_SOURCES)}", temporal.pair_source)
    _ge("temporal.frame_gap", temporal.frame_gap, 1)


def _validate_training(training: TrainingConfig) -> None:
    """INTERFACES training row (crop divisibility by loss patches is cross-section)."""
    _gt("training.lr", training.lr, 0.0)
    for name in ("batch_size", "epochs", "checkpoint_every", "log_every"):
        _ge(f"training.{name}", getattr(training, name), 1)
    _ge("training.workers", training.workers, 0)
    _ge("training.crop_size", training.crop_size, 16)
    if training.crop_size % 16 != 0:
        _fail("training.crop_size", "divisible by 16", training.crop_size)
    if training.optimizer != "Adam":
        _fail("training.optimizer", "exactly 'Adam'", training.optimizer)
    for name in ("adam_beta1", "adam_beta2"):
        value = getattr(training, name)
        if not 0.0 <= value < 1.0:
            _fail(f"training.{name}", "in [0, 1)", value)
    _gt("training.adam_eps", training.adam_eps, 0.0)
    _ge("training.weight_decay", training.weight_decay, 0.0)
    _gt("training.grad_clip", training.grad_clip, 0.0)
    if not 0 <= training.seed <= 2**32 - 1:
        _fail("training.seed", "in [0, 2**32 - 1] (the NumPy seed range)", training.seed)
    if not _DEVICE_PATTERN.match(training.device):
        _fail("training.device", "'auto', 'cpu', 'cuda' or 'cuda:N'", training.device)


def _validate_paths(paths: DataPaths) -> None:
    """Paths are nonempty strings; existence is checked by check_paths."""
    for item in dataclasses.fields(paths):
        value = getattr(paths, item.name)
        if not value.strip():
            _fail(f"paths.{item.name}", "a nonempty path string", value)


def _validate_cross(config: Config) -> None:
    """Constraints that span sections."""
    crop = config.training.crop_size
    for name in ("spatial_patch", "exposure_patch"):
        patch = getattr(config.loss, name)
        if crop % patch != 0:
            raise ValueError(
                f"loss.{name} must divide training.crop_size; got {patch!r} and {crop!r}"
            )
    needed = math.sqrt(config.noise.a_max + config.noise.b_max**2)
    if config.model.sigma_max < needed:
        raise ValueError(
            "model.sigma_max must be >= sqrt(noise.a_max + noise.b_max**2) "
            f"= {needed:.6g}; got {config.model.sigma_max!r}"
        )


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def workspace_root() -> Path:
    """Return the core workspace root, the directory containing ``pyproject.toml``.

    Args:
        None.

    Returns:
        Absolute path found by walking up from this source file; relative config and data paths
        are resolved against it, never against the current working directory.

    Method:
        Parent walk from ``__file__``; the first directory with a ``pyproject.toml`` wins.

    Raises:
        FileNotFoundError: No ``pyproject.toml`` above this file (for example a broken install).
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError(
        "cannot locate the core workspace root (no pyproject.toml above config.py)"
    )


def resolve_path(path: str | Path) -> Path:
    """Resolve a configured path string against the workspace root.

    Args:
        path: Path as written in a config (``paths.*``) or on a command line.

    Returns:
        The path itself when absolute, else ``workspace_root() / path``. No existence check and
        no symlink resolution; the config keeps the original string.

    Method:
        INTERFACES path rule: relative paths resolve against the core workspace root.

    Raises:
        None.
    """
    candidate = Path(path)
    return candidate if candidate.is_absolute() else workspace_root() / candidate


def check_paths(config: Config, require: Iterable[str]) -> None:
    """Check that required data roots exist; called at execution time, never by validate.

    Args:
        config: Validated Config.
        require: Names of ``DataPaths`` fields that must exist, for example ``["train_images"]``.
            ``video_dir`` is only enforced when ``temporal.pair_source == "real"``; synthetic
            pairs need only training images, so it is skipped otherwise.

    Returns:
        None when every required root exists.

    Method:
        Each value is resolved with resolve_path; all missing roots are reported together.

    Raises:
        ValueError: A requested name is not a DataPaths field, or its value is empty.
        FileNotFoundError: One or more required roots do not exist.
    """
    valid = [item.name for item in dataclasses.fields(DataPaths)]
    missing: list[str] = []
    for name in require:
        if name not in valid:
            hint = difflib.get_close_matches(str(name), valid, n=1)
            suffix = f"; did you mean {hint[0]!r}?" if hint else ""
            raise ValueError(f"unknown path root {name!r}; valid roots: {', '.join(valid)}{suffix}")
        if name == "video_dir" and config.temporal.pair_source != "real":
            continue
        value = getattr(config.paths, name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"paths.{name} must be a nonempty path string; got {value!r}")
        resolved = resolve_path(value)
        if not resolved.exists():
            missing.append(f"paths.{name} = {value!r} (resolved to {resolved})")
    if missing:
        raise FileNotFoundError(
            "required data path(s) not found: "
            + "; ".join(missing)
            + ". Create them, edit the YAML paths section or pass the matching CLI option."
        )


# ---------------------------------------------------------------------------
# YAML IO
# ---------------------------------------------------------------------------


class _ConfigLoader(yaml.SafeLoader):  # type: ignore[misc]
    """SafeLoader with YAML 1.2 style floats and duplicate-key rejection.

    PyYAML implements YAML 1.1, whose float rule needs a dot, so a literal such as ``1e-4`` would
    load as a string. This loader also resolves dotless exponent literals to floats.
    """

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        """Build a mapping and reject duplicate keys instead of keeping the last one."""
        seen: set[object] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=True)
            if key in seen:
                raise ValueError(
                    f"duplicate YAML key {key!r} (line {key_node.start_mark.line + 1})"
                )
            seen.add(key)
        mapping: dict[Any, Any] = super().construct_mapping(node, deep)
        return mapping


_ConfigLoader.add_implicit_resolver("tag:yaml.org,2002:float", _FLOAT_NO_DOT, list("-+0123456789"))


def _section_fields(cls: type[Any]) -> dict[str, Any]:
    """Return {field name: resolved type} for a dataclass."""
    hints = _hints(cls)
    return {item.name: hints[item.name] for item in dataclasses.fields(cls)}


def _build(cls: type[Any], data: object, prefix: str, partial: bool) -> object:
    """Structurally build a dataclass from a mapping: key sets, nesting and int-to-float only."""
    where = prefix.rstrip(".") or "<root>"
    if not isinstance(data, Mapping):
        raise ValueError(f"{where} must be a mapping; got {type(data).__name__}")
    fields = _section_fields(cls)
    for key in data:
        if not isinstance(key, str):
            raise ValueError(f"{where} has a non-string key {key!r}")
    unknown = [key for key in data if key not in fields]
    if unknown:
        key = unknown[0]
        close = difflib.get_close_matches(key, list(fields), n=1)
        suffix = f"; did you mean {prefix + close[0]!r}?" if close else ""
        raise ValueError(f"unknown field {prefix + key!r}{suffix}")
    missing = [prefix + name for name in fields if name not in data]
    if missing and not partial:
        raise ValueError(
            "missing field(s) "
            + ", ".join(repr(name) for name in missing)
            + "; complete YAML is required"
        )
    kwargs: dict[str, Any] = {}
    for name, hint in fields.items():
        if name not in data:
            continue
        value = data[name]
        if _is_section(hint):
            kwargs[name] = _build(hint, value, prefix + name + ".", partial)
        elif hint is float and isinstance(value, int) and not isinstance(value, bool):
            try:
                kwargs[name] = float(value)
            except OverflowError:
                raise ValueError(
                    f"{prefix + name} is too large for a float; got {value!r}"
                ) from None
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _apply_overrides(
    data: dict[str, Any], overrides: Mapping[str, object] | None, partial: bool
) -> dict[str, Any]:
    """Return a copy of the raw mapping with dotted overrides applied; None means absent."""
    merged = copy.deepcopy(data)
    for dotted, value in (overrides or {}).items():
        if value is None:
            continue
        parts = dotted.split(".")
        cls: type[Any] = Config
        node = merged
        for depth, part in enumerate(parts):
            fields = _section_fields(cls)
            if part not in fields:
                close = difflib.get_close_matches(dotted, _leaf_names(), n=1)
                suffix = f"; did you mean {close[0]!r}?" if close else ""
                raise ValueError(f"unknown override {dotted!r}{suffix}")
            hint = fields[part]
            last = depth == len(parts) - 1
            if _is_section(hint):
                if last:
                    raise ValueError(f"override {dotted!r} names a section; name a field inside it")
                cls = hint
                child = node.get(part)
                if child is None and partial:
                    child = {}
                    node[part] = child
                if not isinstance(child, dict):
                    raise ValueError(
                        f"cannot override {dotted!r}: {'.'.join(parts[: depth + 1])} is not a mapping"
                    )
                node = child
            else:
                if not last:
                    raise ValueError(f"unknown override {dotted!r}: {part!r} is a scalar field")
                node[part] = value
    return merged


def _leaf_names() -> list[str]:
    """All valid dotted leaf names, for error suggestions."""
    names = ["schema_version"]
    for section, hint in _section_fields(Config).items():
        if _is_section(hint):
            names.extend(f"{section}.{leaf}" for leaf in _section_fields(hint))
    return names


def load_python_mapping(path: Path, variable: str) -> Any:  # noqa: ANN401
    """Import a Python config module and return its module-level ``variable``.

    Args:
        path: Existing ``.py`` file (trusted project code, imported like any module).
        variable: Name of the mapping the module must define, e.g. ``CONFIG``.

    Raises:
        FileNotFoundError: The file does not exist.
        ValueError: The module cannot be imported or does not define ``variable``.
    """
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    spec = importlib.util.spec_from_file_location(f"_lumina_config_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load config module {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as err:  # noqa: BLE001 - surface any import failure as a config error
        raise ValueError(f"cannot load config module {path}: {err}") from None
    if not hasattr(module, variable):
        raise ValueError(f"config module {path} must define {variable}")
    return copy.deepcopy(getattr(module, variable))


def parse_config(
    path: str | Path,
    overrides: Mapping[str, object] | None = None,
    *,
    partial: bool = False,
) -> Config:
    """Load YAML and overrides into a Config WITHOUT bound validation.

    Args:
        path: UTF-8 YAML path, resolved with resolve_path.
        overrides: Dotted field names to explicit values; None leaves the YAML untouched.
        partial: When True missing keys are filled from schema defaults; default False.

    Returns:
        Structurally correct but unvalidated Config; callers must call ``validate()``. This split
        lets the CLI explain flag conflicts before the generic validation message.

    Method:
        safe_load -> strict key-set check at every level -> overrides -> dataclass construction.

    Raises:
        FileNotFoundError: The YAML file does not exist.
        ValueError: Invalid YAML, non-mapping root, unknown or missing keys, bad override names.
    """
    resolved = resolve_path(path)
    if resolved.suffix == ".py":
        raw = load_python_mapping(resolved, "CONFIG")
    else:
        try:
            text = resolved.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise FileNotFoundError(f"config file not found: {resolved}") from None
        except UnicodeDecodeError as err:
            raise ValueError(f"config file {resolved} is not valid UTF-8: {err}") from None
        try:
            raw = yaml.load(text, Loader=_ConfigLoader)  # noqa: S506 - SafeLoader subclass
        except (yaml.YAMLError, ValueError) as err:
            raise ValueError(f"cannot parse config {resolved}: {err}") from None
    if not isinstance(raw, dict):
        raise ValueError(f"config {resolved} must be a YAML mapping; got {type(raw).__name__}")
    try:
        return cast(Config, _build(Config, _apply_overrides(raw, overrides, partial), "", partial))
    except ValueError as err:
        raise ValueError(f"{err} (config: {resolved})") from None


def load_config(
    path: str | Path,
    overrides: Mapping[str, object] | None = None,
    *,
    partial: bool = False,
) -> Config:
    """Load, override and validate a YAML experiment.

    Args:
        path: UTF-8 YAML path, resolved relative to the core workspace root unless absolute.
        overrides: Dotted field names to explicit CLI values; None leaves YAML unchanged.
        partial: Strict by default (every key at every level must be present, as in the complete
            baseline and full YAML). True fills missing keys from schema defaults.

    Returns:
        Validated immutable Config; no model or dataset construction. Paths inside stay as the
        strings written in the YAML; use resolve_path to turn them into filesystem paths.

    Method:
        INTERFACES precedence defaults < YAML < explicit CLI; reject unknown keys at any level.
        Dotless exponent literals such as ``1e-4`` are read as floats; integers are accepted
        for float fields and converted to float.

    Raises:
        FileNotFoundError: The YAML file does not exist.
        ValueError: Parse, key-set, override or validation failure naming the dotted field.
    """
    config = parse_config(path, overrides, partial=partial)
    config.validate()
    return config


def save_config(config: Config, path: str | Path) -> None:
    """Serialize the resolved experiment schema.

    Args:
        config: Validated Config.
        path: UTF-8 YAML destination (relative paths resolve like load_config); parent
            creation is caller responsibility.

    Returns:
        None; writes all fields, flags and schema_version in declaration order.

    Method:
        INTERFACES round-trip/checkpoint contract; no implicit path rewriting. Floats are dumped
        with ``repr`` (shortest string that round-trips a double, for example ``1.0e-06``), which
        the loader parses back to the identical value, so ``load(save(c)) == c`` exactly.
        Float fields holding ints are written as floats.

    Raises:
        ValueError: The config fails validation.
        FileNotFoundError: The destination directory does not exist.
    """
    config.validate()
    data = _to_plain(config)
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=False)
    resolve_path(path).write_text(text, encoding="utf-8")


def _to_plain(obj: object) -> dict[str, Any]:
    """Convert a dataclass tree to plain dict/scalars, normalizing float fields to float."""
    hints = _hints(type(obj))
    out: dict[str, Any] = {}
    for item in dataclasses.fields(cast(Any, obj)):
        value = getattr(obj, item.name)
        if _is_section(type(value)):
            out[item.name] = _to_plain(value)
        elif hints[item.name] is float:
            out[item.name] = float(value)
        else:
            out[item.name] = value
    return out


def config_to_dict(config: Config) -> dict[str, Any]:
    """Return the validated config as plain nested dicts and scalars (checkpoint payloads).

    Args:
        config: Config to serialize; validated first.

    Returns:
        Plain ``dict`` tree with float fields normalized to float; inverse of
        ``config_from_dict``.

    Raises:
        ValueError: The config fails validation.
    """
    config.validate()
    return _to_plain(config)


def config_from_dict(data: Mapping[str, object]) -> Config:
    """Build and validate a Config from the plain dict produced by ``config_to_dict``.

    Args:
        data: Complete nested mapping (every key present, no unknown keys).

    Returns:
        Validated frozen Config.

    Raises:
        ValueError: Missing/unknown keys, wrong types, or bound violations.
    """
    config = cast(Config, _build(Config, dict(data), "", False))
    config.validate()
    return config


def override_config(config: Config, overrides: Mapping[str, object] | None) -> Config:
    """Apply dotted CLI-style overrides to an existing Config; ``None`` values mean absent.

    Args:
        config: Base config (for example one restored from a checkpoint).
        overrides: Dotted names such as ``flags.light`` or ``model.curve_scale``.

    Returns:
        New validated Config; the input is unchanged.

    Raises:
        ValueError: Unknown override names or a result that fails validation.
    """
    merged = _apply_overrides(_to_plain(config), overrides, False)
    return config_from_dict(merged)
