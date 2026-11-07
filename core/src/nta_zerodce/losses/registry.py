"""Loss metadata and single-decorator registration; ARCHITECTURE loss graph.

See docs/ARCHITECTURE.md for the normative contract.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

from torch import Tensor

from nta_zerodce.losses.adaptive_exposure import adaptive_exposure_loss
from nta_zerodce.losses.chroma import chroma_loss
from nta_zerodce.losses.color import color_loss
from nta_zerodce.losses.edge import edge_loss
from nta_zerodce.losses.exposure import exposure_loss
from nta_zerodce.losses.highlight import highlight_loss
from nta_zerodce.losses.noise import fidelity_loss, invariance_loss, noise_loss
from nta_zerodce.losses.smoothness import smoothness_loss
from nta_zerodce.losses.spatial import spatial_loss
from nta_zerodce.losses.temporal import temporal_loss

LossCallable = Callable[..., Tensor]


@dataclass(frozen=True)
class LossSpec:
    """Registration contract; INTERFACES loss graph and METHOD weight table.

    callable: Tensor-only differentiable loss; inputs specified by required_inputs.
    flag: Config activation expression (flag or not flag), or None for unconditional baseline terms.
    default_weight: Nonnegative scalar starting weight, not a tuned result.
    required_inputs: Ordered names from the loop-step context, including scalar options.
    Returns an immutable metadata record; no loss execution at construction.
    """

    callable: LossCallable
    flag: str | None
    default_weight: float
    required_inputs: tuple[str, ...]


# Built-in metadata only; no loss or decorator is executed at import.
LOSS_REGISTRY: dict[str, LossSpec] = {
    "spa": LossSpec(spatial_loss, None, 1.0, ("image", "enhanced")),
    "exp": LossSpec(exposure_loss, "not adaptive_exp", 10.0, ("enhanced",)),
    "col": LossSpec(color_loss, None, 5.0, ("enhanced",)),
    "tv": LossSpec(smoothness_loss, None, 200.0, ("A",)),
    "edge": LossSpec(edge_loss, "edge", 20.0, ("image", "enhanced", "sigma", "gain")),
    "chroma": LossSpec(chroma_loss, "chroma", 20.0, ("image", "enhanced", "sigma")),
    "highlight": LossSpec(highlight_loss, "adaptive_exp", 20.0, ("image", "enhanced")),
    "noise": LossSpec(noise_loss, "noise_aware", 20.0, ("sigma", "target_sigma")),
    "invariance": LossSpec(
        invariance_loss, "noise_aware", 10.0, ("noisy_enhanced", "teacher_enhanced")
    ),
    "fidelity": LossSpec(fidelity_loss, "denoise", 1.0, ("noisy_enhanced", "teacher_raw")),
    "temporal": LossSpec(
        temporal_loss,
        "temporal",
        10.0,
        ("previous_output", "current_output", "backward_flow", "mask"),
    ),
    "adaptive_exp": LossSpec(adaptive_exposure_loss, "adaptive_exp", 10.0, ("enhanced", "targets")),
}
# name -> (qualified callable, gate expression, default weight, tensor inputs)
LOSS_CONTRACTS: dict[str, tuple[str, str | None, float, tuple[str, ...]]] = {
    "spa": ("spatial.spatial_loss", None, 1.0, ("image", "enhanced")),
    "exp": ("exposure.exposure_loss", "not adaptive_exp", 10.0, ("enhanced",)),
    "col": ("color.color_loss", None, 5.0, ("enhanced",)),
    "tv": ("smoothness.smoothness_loss", None, 200.0, ("A",)),
    "edge": ("edge.edge_loss", "edge", 20.0, ("image", "enhanced", "sigma", "gain")),
    "chroma": ("chroma.chroma_loss", "chroma", 20.0, ("image", "enhanced", "sigma")),
    "highlight": ("highlight.highlight_loss", "adaptive_exp", 20.0, ("image", "enhanced")),
    "noise": ("noise.noise_loss", "noise_aware", 20.0, ("sigma", "target_sigma")),
    "invariance": (
        "noise.invariance_loss",
        "noise_aware",
        10.0,
        ("noisy_enhanced", "teacher_enhanced"),
    ),
    "fidelity": ("noise.fidelity_loss", "denoise", 1.0, ("noisy_enhanced", "teacher_raw")),
    "temporal": (
        "temporal.temporal_loss",
        "temporal",
        10.0,
        ("previous_output", "current_output", "backward_flow", "mask"),
    ),
    "adaptive_exp": (
        "adaptive_exposure.adaptive_exposure_loss",
        "adaptive_exp",
        10.0,
        ("enhanced", "targets"),
    ),
}


# Flag names a loss gate may reference (the FeatureFlags fields). Kept here as plain strings so
# the losses package never imports the training schema.
KNOWN_FLAGS: frozenset[str] = frozenset(
    {"noise_aware", "denoise", "edge", "chroma", "adaptive_exp", "temporal", "light"}
)

# register_loss writes here, never into LOSS_REGISTRY. LOSS_REGISTRY stays a table of the twelve
# built-in specs (a collection-time audit compares it with LOSS_CONTRACTS), so importing this
# module and registering extra losses never mutate the built-in metadata. build_losses
# consults both tables.
_DYNAMIC_REGISTRY: dict[str, LossSpec] = {}


def _check_gate(flag: str | None) -> None:
    """Validate a gate expression: None, a known flag name, or "not <known flag>"."""
    if flag is None:
        return
    name = flag[4:] if flag.startswith("not ") else flag
    if name not in KNOWN_FLAGS:
        raise ValueError(f"unknown flag expression {flag!r}; known flags: {sorted(KNOWN_FLAGS)}")


def _gate_active(flag: str | None, flags: dict[str, bool]) -> bool:
    """Evaluate a validated gate expression against a flag map (missing flags are False)."""
    if flag is None:
        return True
    if flag.startswith("not "):
        return not flags.get(flag[4:], False)
    return flags.get(flag, False)


def register_loss(
    name: str, flag: str | None, default_weight: float, required_inputs: tuple[str, ...]
) -> Callable[[LossCallable], LossCallable]:
    """Register one loss through a decorator.

    Args:
        name: Unique registry key.
        flag: Schema activation flag, ``"not <flag>"``, or None for an unconditional term.
        default_weight: Nonnegative finite scalar.
        required_inputs: Ordered tensor/context argument names.

    Returns:
        Decorator preserving the callable and installing one LossSpec in the dynamic registry
        (``_DYNAMIC_REGISTRY``), not in the module-level ``LOSS_REGISTRY`` metadata table.

    Method:
        CONTRIBUTING extension checklist; reject duplicate keys and unknown flags.

    Raises:
        ValueError: When register_loss is called, for an empty or duplicate name (built-in or
            previously registered), an unknown flag expression, or a negative or non-finite
            weight.
    """
    if not name:
        raise ValueError("loss name must be non-empty")
    if name in LOSS_REGISTRY or name in _DYNAMIC_REGISTRY:
        raise ValueError(f"loss {name!r} is already registered")
    _check_gate(flag)
    if not math.isfinite(default_weight) or default_weight < 0:
        raise ValueError(f"default_weight must be finite and nonnegative, got {default_weight}")
    inputs = tuple(required_inputs)

    def decorator(function: LossCallable) -> LossCallable:
        if name in LOSS_REGISTRY or name in _DYNAMIC_REGISTRY:
            raise ValueError(f"loss {name!r} is already registered")
        _DYNAMIC_REGISTRY[name] = LossSpec(function, flag, float(default_weight), inputs)
        return function

    return decorator


def build_losses(flags: dict[str, bool], weights: dict[str, float]) -> dict[str, LossSpec]:
    """Resolve active loss specifications.

    Args:
        flags: Validated feature map, including denoise=>noise_aware. Missing known flags are
            treated as False; unknown keys are rejected.
        weights: Full nonnegative weight map keyed like LossWeights (spa, exp, col, tv, edge,
            chroma, highlight, noise, invariance, fidelity, temporal). ``adaptive_exp`` inherits
            ``weights["exp"]``; ``highlight`` uses its own weight.

    Returns:
        Active names mapped to immutable specs, in registry order (built-ins first, then
        dynamically registered). Each returned spec carries the resolved weight in
        ``default_weight``. ``exp`` and ``adaptive_exp`` are mutually exclusive.

    Method:
        METHOD total objective; the gate of each spec is evaluated against ``flags``.
        Runtime tensor presence is checked by the training step, which owns the context.

    Raises:
        ValueError: Unknown flag key, denoise without noise_aware, or a missing, negative or
            non-finite weight for an active loss.
    """
    unknown = sorted(set(flags) - KNOWN_FLAGS)
    if unknown:
        raise ValueError(f"unknown flags: {unknown}")
    if flags.get("denoise", False) and not flags.get("noise_aware", False):
        raise ValueError("denoise requires noise_aware")
    active: dict[str, LossSpec] = {}
    for name, spec in {**LOSS_REGISTRY, **_DYNAMIC_REGISTRY}.items():
        if not _gate_active(spec.flag, flags):
            continue
        key = "exp" if name == "adaptive_exp" else name
        if key not in weights:
            raise ValueError(f"missing weight {key!r} for active loss {name!r}")
        weight = float(weights[key])
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"weight {key!r} must be finite and nonnegative, got {weight}")
        active[name] = replace(spec, default_weight=weight)
    return active
