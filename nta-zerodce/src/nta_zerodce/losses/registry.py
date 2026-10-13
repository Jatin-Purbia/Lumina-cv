"""Loss metadata and future single-decorator registration; ARCHITECTURE loss graph.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

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


# Default metadata references stubs; no loss or decorator is executed at import.
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


def register_loss(
    name: str, flag: str | None, default_weight: float, required_inputs: tuple[str, ...]
) -> Callable[[LossCallable], LossCallable]:
    """Register one loss through a decorator.

    Args:
        name: Unique registry key.
        flag: Schema activation flag, or None.
        default_weight: Nonnegative scalar.
        required_inputs: Ordered tensor/context argument names.

    Returns:
        Decorator preserving the callable and installing one LossSpec.

    Method:
        CONTRIBUTING extension checklist; reject duplicate keys and unknown flags.

    Raises:
        NotImplementedError: Scaffold only; implement this contract before use.
    """
    raise NotImplementedError("TODO: Register one loss through a decorator")


def build_losses(flags: dict[str, bool], weights: dict[str, float]) -> dict[str, LossSpec]:
    """Resolve active loss specifications.

    Args:
        flags: Validated feature map, including denoise=>noise_aware.
        weights: Full nonnegative weight map; adaptive_exp inherits exp weight.

    Returns:
        Active names and immutable specs; exp and adaptive_exp mutually exclusive.

    Method:
        METHOD total objective; missing runtime tensor inputs must be rejected.

    Raises:
        NotImplementedError: Scaffold only; implement this contract before use.
    """
    raise NotImplementedError("TODO: Resolve active loss specifications")
