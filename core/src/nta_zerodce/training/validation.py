"""Strict schema parsing and bounds, independent of datasets and model creation."""

from __future__ import annotations

import math
from dataclasses import asdict, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nta_zerodce.training.config import Config

CORE_ROOT = Path(__file__).resolve().parents[3]


def core_path(path: Path | str) -> Path:
    """Resolve local runtime paths against the independent core workspace."""
    path = Path(path)
    return path if path.is_absolute() else CORE_ROOT / path


def config_from_dict(data: dict[str, Any]) -> Config:
    """Reject unknown keys and wrong scalar types before creating records."""
    from nta_zerodce.training.config import Config

    defaults = Config()
    if set(data) - set(asdict(defaults)):
        raise ValueError(f"Unknown config fields: {set(data) - set(asdict(defaults))}")
    kwargs: dict[str, Any] = {}
    for section in fields(defaults):
        default = getattr(defaults, section.name)
        value = data.get(section.name, default)
        if section.name == "schema_version":
            if type(value) is not int:
                raise ValueError("schema_version must be an integer")
            kwargs[section.name] = value
            continue
        if section.name not in data:
            kwargs[section.name] = default
            continue
        if not isinstance(value, dict) or set(value) - set(asdict(default)):
            raise ValueError(f"Unknown fields or non-mapping section: {section.name}")
        section_kwargs = {}
        for field in fields(default):
            expected = getattr(default, field.name)
            actual = value.get(field.name, expected)
            if type(expected) is float and type(actual) in (float, int):
                actual = float(actual)
            elif type(actual) is not type(expected):
                raise ValueError(f"{section.name}.{field.name}: expected {type(expected).__name__}")
            section_kwargs[field.name] = actual
        kwargs[section.name] = type(default)(**section_kwargs)
    config = Config(**kwargs)
    config.validate()
    return config


def validate_config(config: Config) -> None:
    """Validate types, finite bounds and cross-field constraints without requiring data."""
    from nta_zerodce.training.config import Config

    defaults = asdict(Config())
    values = asdict(config)
    for section, mapping in values.items():
        if section == "schema_version":
            continue
        for key, value in mapping.items():
            expected = defaults[section][key]
            if type(value) is not type(expected) and not (
                type(expected) is float and type(value) is int
            ):
                raise ValueError(f"{section}.{key}: invalid type")
            if type(value) in (float, int) and not math.isfinite(value):
                raise ValueError(f"{section}.{key}: must be finite")

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    require(
        type(config.schema_version) is int and config.schema_version == 1,
        "Unsupported schema_version",
    )
    require(not config.flags.denoise or config.flags.noise_aware, "denoise requires noise_aware")
    m, loss_config, n, s, f, t, tr = (
        config.model,
        config.loss,
        config.noise,
        config.synthetic,
        config.flow,
        config.temporal,
        config.training,
    )
    require(
        m.iterations == 8 and min(m.channels, m.noise_channels, m.denoise_channels) > 0,
        "Invalid model widths/iterations",
    )
    require(
        all(0 < v <= 1 for v in (m.sigma_max, m.residual_scale, m.curve_scale, f.scale)),
        "Model/flow scales must be in (0,1]",
    )
    require(1 <= m.gain_min <= m.gain_max and m.eps > 0, "Invalid gain bounds or epsilon")
    require(all(v >= 0 for v in asdict(config.weights).values()), "Weights must be nonnegative")
    require(tr.crop_size >= 16 and tr.crop_size % 16 == 0, "crop_size must be a multiple of 16")
    require(
        loss_config.spatial_patch > 0
        and loss_config.exposure_patch > 0
        and tr.crop_size % loss_config.spatial_patch == 0
        and tr.crop_size % loss_config.exposure_patch == 0,
        "Patches must divide crop_size",
    )
    require(
        all(
            v > 0 and v % 2 == 1 and v // 2 < tr.crop_size
            for v in (m.gain_window, loss_config.chroma_window, loss_config.edge_kernel)
        ),
        "Local kernels must be odd, positive and fit crop",
    )
    require(
        all(
            0 <= v <= 1
            for v in (
                loss_config.exposure_target,
                loss_config.adaptive_low,
                loss_config.adaptive_high,
                loss_config.highlight_threshold,
                loss_config.reliability_low,
                loss_config.reliability_high,
                f.photometric_threshold,
            )
        ),
        "Loss thresholds must be in [0,1]",
    )
    require(
        loss_config.adaptive_low < loss_config.adaptive_high
        and loss_config.reliability_low < loss_config.reliability_high,
        "Thresholds must be ordered",
    )
    require(
        min(loss_config.eps, loss_config.adaptive_noise_scale, loss_config.edge_smooth_sigma) > 0
        and min(
            loss_config.adaptive_noise_penalty,
            loss_config.reliability_noise,
            loss_config.edge_noise_multiplier,
        )
        >= 0,
        "Invalid loss constants",
    )
    require(
        0 <= n.a_min <= n.a_max and 0 <= n.b_min <= n.b_max and 0 <= n.probability <= 1,
        "Invalid noise distribution",
    )
    require(
        m.sigma_max >= math.sqrt(n.a_max + n.b_max**2), "sigma_max cannot represent synthetic noise"
    )
    require(
        min(s.rotation_degrees, s.translation_pixels) >= 0
        and 0 < s.scale_min <= s.scale_max
        and 0 < s.flicker_min <= s.flicker_max,
        "Invalid synthetic transform range",
    )
    require(
        f.method == "dis"
        and f.preset == "medium"
        and f.align_corners
        and min(f.fb_alpha, f.fb_beta) >= 0,
        "Invalid DIS/warp configuration",
    )
    require(
        all(0 <= v <= 1 for v in (t.curve_blend, t.output_blend, t.scene_cut_threshold))
        and t.histogram_bins >= 2
        and t.frame_gap >= 1
        and t.pair_source in ("real", "synthetic"),
        "Invalid temporal settings",
    )
    require(
        tr.optimizer == "Adam"
        and min(tr.lr, tr.adam_eps, tr.grad_clip) > 0
        and tr.weight_decay >= 0
        and 0 <= tr.adam_beta1 < 1
        and 0 <= tr.adam_beta2 < 1,
        "Invalid Adam settings",
    )
    require(
        min(tr.batch_size, tr.epochs, tr.checkpoint_every, tr.log_every) >= 1
        and min(tr.seed, tr.workers) >= 0,
        "Invalid training counts",
    )
    require(
        tr.device in ("auto", "cpu", "cuda", "mps")
        or (tr.device.startswith("cuda:") and tr.device[5:].isdigit()),
        "Unsupported device",
    )
    require(
        all(
            isinstance(v, str) and bool(v.strip()) and "://" not in v
            for v in asdict(config.paths).values()
        ),
        "Expected nonempty local paths",
    )
