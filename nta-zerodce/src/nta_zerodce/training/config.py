"""Typed YAML schema; INTERFACES config schema and METHOD constants.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


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
        exposure_target: float; default 0.6. Units and bounds in INTERFACES.
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
    exposure_target: float = 0.6
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
        crop_size: int; default 256. Units and bounds in INTERFACES.
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
    crop_size: int = 256
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
        train_images: str; default "data/SICE/part1". Units and bounds in INTERFACES.
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

    train_images: str = "data/SICE/part1"
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

        Raises:
            NotImplementedError: Scaffold only; implement this contract before use.
        """
        raise NotImplementedError("TODO: Validate schema bounds and cross-field constraints")


def load_config(path: Path, overrides: dict[str, object] | None = None) -> Config:
    """Load, override and validate a YAML experiment.

    Args:
        path: UTF-8 YAML path, resolved relative to repository root.
        overrides: Dotted field names to explicit CLI values; None leaves YAML unchanged.

    Returns:
        Validated immutable Config; no model or dataset construction.

    Method:
        INTERFACES precedence defaults < YAML < explicit CLI; reject unknown keys.

    Raises:
        NotImplementedError: Scaffold only; implement this contract before use.
    """
    raise NotImplementedError("TODO: Load, override and validate a YAML experiment")


def save_config(config: Config, path: Path) -> None:
    """Serialize the resolved experiment schema.

    Args:
        config: Validated Config.
        path: UTF-8 YAML destination; parent creation is caller responsibility.

    Returns:
        None; writes all fields, flags and schema_version.

    Method:
        INTERFACES round-trip/checkpoint contract; no implicit path rewriting.

    Raises:
        NotImplementedError: Scaffold only; implement this contract before use.
    """
    raise NotImplementedError("TODO: Serialize the resolved experiment schema")
