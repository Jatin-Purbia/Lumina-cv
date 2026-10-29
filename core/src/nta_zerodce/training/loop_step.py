"""One training update; pseudocode follows.

1. Read unpaired RGB x, shared crop; synthesize independent n when noise-aware.
2. Forward E(x+n), obtain enhanced/raw/A/sigma; form detached gain and masks.
3. Compute spa/exp-or-adaptive/col/tv; gate edge/chroma/highlight by flags.
4. Under no_grad forward E(x): teacher enhanced for invariance, raw for fidelity.
5. Add synthetic sigma supervision, invariance and denoise fidelity where enabled.
6. If temporal, forward pair; obtain detached DIS or analytic bidirectional flow.
7. Build current-coordinate mask; add temporal loss on enhanced pair outputs.
8. Sum weighted scalar terms; zero_grad, backward, clip global norm to .1, step.
9. Return detached scalar logs; log zero-valued inactive terms explicitly.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer

from nta_zerodce.data.noise_synth import sample_noise
from nta_zerodce.data.synthetic_pairs import make_pair
from nta_zerodce.flow.estimator import estimate_bidirectional
from nta_zerodce.flow.masks import reliability_mask
from nta_zerodce.losses.adaptive_exposure import adaptive_targets
from nta_zerodce.losses.registry import LOSS_REGISTRY, build_losses
from nta_zerodce.models.nta_zerodce import NTAZeroDCE
from nta_zerodce.tensor_ops import local_gain
from nta_zerodce.training.config import Config


def train_step(
    model: NTAZeroDCE, batch: dict[str, Tensor], optimizer: Optimizer, config: Config
) -> dict[str, float]:
    """Perform exactly one gated optimization step.

    Args:
        model: Train-mode model on batch device.
        batch: image (B,3,H,W) in [0,1]; temporal fields as INTERFACES specifies.
        optimizer: Adam bound to model parameters.
        config: Validated flags/weights and gradient norm clip.

    Returns:
        Detached Python scalars: total and every loss name, finite for valid batches.

    Method:
        METHOD objective; teacher/masks/gain/targets detached, all active student paths differentiable.

    Raises:
        ValueError: Invalid inputs or incompatible configuration.
    """
    config.validate()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    device = next(model.parameters()).device
    batch = {key: value.to(device) for key, value in batch.items()}
    image = batch["image"] if "image" in batch else batch["image1"]
    student = image
    context: dict[str, Tensor] = {}
    if config.flags.noise_aware:
        shape = (image.shape[0],)
        a = torch.empty(shape, device=device).uniform_(config.noise.a_min, config.noise.a_max)
        b = torch.empty(shape, device=device).uniform_(config.noise.b_min, config.noise.b_max)
        active = torch.rand(shape, device=device) < config.noise.probability
        student, target_sigma = sample_noise(image, a * active, b * active)
        with torch.no_grad():
            teacher = model(image)
        context.update(
            target_sigma=target_sigma,
            teacher_enhanced=teacher["enhanced"],
            teacher_raw=teacher["enhanced_raw"],
        )
    output = model(student)
    context.update(image=student, **output)
    context["gain"] = local_gain(student, output["enhanced_raw"], config.model)
    context["noisy_enhanced"] = output["enhanced"]
    loss_config = config.loss
    if config.flags.adaptive_exp:
        context["targets"] = adaptive_targets(
            student,
            output["sigma"],
            loss_config.exposure_target,
            loss_config.adaptive_low,
            loss_config.adaptive_high,
            loss_config.adaptive_noise_scale,
            loss_config.adaptive_noise_penalty,
            loss_config.exposure_patch,
        )
    if config.flags.temporal:
        pair = batch if "image2" in batch else make_pair(image, config)
        previous = model(pair["image1"])["enhanced"]
        current = model(pair["image2"])["enhanced"]
        if "backward_flow" in pair and "mask" in pair:
            backward, mask = pair["backward_flow"].detach(), pair["mask"].detach()
        else:
            forward, backward = estimate_bidirectional(pair["image1"], pair["image2"], config.flow)
            forward, backward = forward.to(device), backward.to(device)
            f = config.flow
            mask = reliability_mask(
                forward,
                backward,
                pair["image1"],
                pair["image2"],
                f.fb_alpha,
                f.fb_beta,
                f.photometric_threshold,
            )
        context.update(
            previous_output=previous, current_output=current, backward_flow=backward, mask=mask
        )
    options: dict[str, dict[str, Any]] = {
        "spa": {"patch_size": loss_config.spatial_patch},
        "exp": {"target": loss_config.exposure_target, "patch_size": loss_config.exposure_patch},
        "adaptive_exp": {"patch_size": loss_config.exposure_patch},
        "highlight": {"threshold": loss_config.highlight_threshold},
        "chroma": {
            "window": loss_config.chroma_window,
            "eps": loss_config.eps,
            "reliability_low": loss_config.reliability_low,
            "reliability_high": loss_config.reliability_high,
            "reliability_noise": loss_config.reliability_noise,
        },
        "edge": {
            "smooth_sigma": loss_config.edge_smooth_sigma,
            "kernel_size": loss_config.edge_kernel,
            "noise_multiplier": loss_config.edge_noise_multiplier,
            "eps": loss_config.eps,
        },
    }
    logs = {name: 0.0 for name in LOSS_REGISTRY}
    logs.update({f"weighted_{name}": 0.0 for name in LOSS_REGISTRY})
    total = output["enhanced"].sum() * 0
    for name, spec in build_losses(asdict(config.flags), asdict(config.weights)).items():
        missing = set(spec.required_inputs) - context.keys()
        if missing:
            raise ValueError(f"{name}: missing loss inputs {missing}")
        value = spec.callable(
            *(context[key] for key in spec.required_inputs), **options.get(name, {})
        )
        weighted = value * spec.default_weight
        total = total + weighted
        logs[name] = float(value.detach())
        logs[f"weighted_{name}"] = float(weighted.detach())
    if not torch.isfinite(total):
        raise ValueError("Nonfinite training objective")
    total.backward()
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), config.training.grad_clip, error_if_nonfinite=True
    )
    optimizer.step()
    logs.update(
        total=float(total.detach()),
        grad_norm=float(norm),
        mask_coverage=float(context["mask"].mean()) if "mask" in context else 0.0,
    )
    return logs
