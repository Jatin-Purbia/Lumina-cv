"""One gated optimization step and the batch preparation that feeds it.

``prepare_batch`` turns a raw dataset batch into the training context (noisy copy and known
sigma target, synthetic or DIS temporal pair with flow and reliability mask). ``train_step``
then runs the METHOD objective exactly once: student forward on the noisy image, a no-grad
teacher forward of the same model on the clean image, detached sigma/gain/targets/masks, the
flag-gated losses of ``losses.registry.build_losses``, one weighted sum, backward, global
gradient-norm clipping and one Adam step.

Batch contract (all tensors on the model device, float32 RGB in [0,1], BCHW):

* ``image``: clean unpaired crop (the teacher input and, without noise, the student input).
* ``noisy`` and ``target_sigma`` (B,1,H,W): required when ``flags.noise_aware``; the student
  input is ``noisy`` and ``target_sigma`` is the known pre-clipping sigma.
* ``image1``, ``image2`` (B,3,H,W), ``backward_flow`` (B,2,H,W) and ``mask`` (B,1,H,W):
  required when ``flags.temporal``; ``backward_flow`` points from ``image2`` (current) to
  ``image1`` (previous) as in docs/ARCHITECTURE.md.

See docs/ARCHITECTURE.md (objective, gating, detach policy, loss table).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict

import torch
from torch import Generator, Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer

from nta_zerodce.data.noise_synth import sample_noise
from nta_zerodce.data.synthetic_pairs import make_pair
from nta_zerodce.flow.estimator import estimate_bidirectional
from nta_zerodce.flow.masks import reliability_mask
from nta_zerodce.losses.adaptive_exposure import adaptive_targets
from nta_zerodce.losses.registry import LossSpec, build_losses
from nta_zerodce.models.nta_zerodce import NTAZeroDCE, compute_gain
from nta_zerodce.training.config import Config

# Context names that depend on which optional forward passes the active losses need.
_TEACHER_INPUTS = frozenset({"teacher_enhanced", "teacher_raw"})
_TEMPORAL_INPUTS = frozenset({"previous_output", "current_output", "backward_flow", "mask"})
_TEMPORAL_BATCH_KEYS = ("image1", "image2", "backward_flow", "mask")


# ---------------------------------------------------------------------------
# Batch preparation
# ---------------------------------------------------------------------------


def _draw_noise(clean: Tensor, config: Config, generator: Generator) -> tuple[Tensor, Tensor]:
    """Per-image Poisson-Gaussian-style corruption; returns (noisy, sigma_target).

    ``a ~ U(a_min, a_max)`` and ``b ~ U(b_min, b_max)`` are drawn per image; each image is
    corrupted with probability ``noise.probability`` (otherwise a = b = 0, so the image passes
    through unchanged with a zero target). All randomness comes from ``generator``.
    """
    batch = clean.shape[0]
    noise = config.noise
    device = clean.device
    draws = torch.rand(3, batch, generator=generator, device=device, dtype=torch.float32)
    a = noise.a_min + (noise.a_max - noise.a_min) * draws[0]
    b = noise.b_min + (noise.b_max - noise.b_min) * draws[1]
    applied = draws[2] < noise.probability
    a = torch.where(applied, a, torch.zeros_like(a)).view(batch, 1, 1, 1)
    b = torch.where(applied, b, torch.zeros_like(b)).view(batch, 1, 1, 1)
    return sample_noise(clean, a, b, generator)


def _real_pair(batch: dict[str, Tensor], config: Config, device: torch.device) -> dict[str, Tensor]:
    """Real adjacent frames: CPU DIS flow in both directions, reliability mask, then device."""
    first = batch["image1"].detach().cpu().contiguous()
    second = batch["image2"].detach().cpu().contiguous()
    forward, backward = estimate_bidirectional(first, second, config.flow)
    mask = reliability_mask(
        forward,
        backward,
        first,
        second,
        alpha=config.flow.fb_alpha,
        beta=config.flow.fb_beta,
        photometric_threshold=config.flow.photometric_threshold,
    )
    return {
        "image1": first.to(device),
        "image2": second.to(device),
        "backward_flow": backward.to(device),
        "mask": mask.to(device),
    }


def prepare_batch(
    batch: dict[str, Tensor], config: Config, generator: Generator, device: torch.device
) -> dict[str, Tensor]:
    """Build the full ``train_step`` batch from a raw dataset batch.

    Args:
        batch: Raw collated dataset batch: ``image`` (B,3,H,W); with a real temporal source
            additionally ``image1``/``image2`` (a pure temporal batch without ``image`` is also
            accepted and uses ``image1`` as the still-image input). Extra keys are ignored.
        config: Validated Config; ``flags.noise_aware`` adds the noisy copy and sigma target,
            ``flags.temporal`` with ``temporal.pair_source`` ``synthetic`` builds an affine
            pair (analytic flow, geometric mask) from ``image``, ``real`` computes DIS flow on
            the CPU and the reliability mask from the two frames.
        generator: Torch generator on ``device``; the only source of randomness (noise
            coefficients, noise samples, synthetic affine and flicker).
        device: Target device. Tensors are moved here once, explicitly.

    Returns:
        Context for ``train_step``: ``image`` (clean), plus ``noisy`` and ``target_sigma`` when
        noise-aware, plus ``image1``, ``image2``, ``backward_flow``, ``mask`` when temporal.

    Raises:
        ValueError: A required raw key is missing, or the generator lives on another device.
    """
    if generator.device.type != device.type:
        raise ValueError(
            f"generator is on {generator.device.type} but the training device is {device.type}"
        )
    flags = config.flags
    real_pair = flags.temporal and config.temporal.pair_source == "real"
    if "image" in batch:
        clean = batch["image"].to(device)
    elif real_pair and "image1" in batch:
        clean = batch["image1"].to(device)
    else:
        raise ValueError(f"raw batch needs an 'image' tensor; got keys {sorted(batch)}")
    out: dict[str, Tensor] = {"image": clean}
    if flags.noise_aware:
        noisy, target_sigma = _draw_noise(clean, config, generator)
        out["noisy"] = noisy
        out["target_sigma"] = target_sigma
    if flags.temporal:
        if real_pair:
            missing = [key for key in ("image1", "image2") if key not in batch]
            if missing:
                raise ValueError(
                    f"temporal.pair_source='real' needs {missing} in the raw batch "
                    f"(VideoPairDataset); got keys {sorted(batch)}"
                )
            out.update(_real_pair(batch, config, device))
        else:
            pair = make_pair(clean, config, generator)
            for key in _TEMPORAL_BATCH_KEYS:
                out[key] = pair[key]
    return out


# ---------------------------------------------------------------------------
# Loss graph
# ---------------------------------------------------------------------------


def _device_of(model: NTAZeroDCE) -> torch.device:
    """Device of the model parameters (models in this project are never split)."""
    return next(model.parameters()).device


def _require(batch: dict[str, Tensor], keys: Iterable[str], why: str) -> None:
    """Raise a clear ValueError naming missing batch keys and the flag that needs them."""
    missing = [key for key in keys if key not in batch]
    if missing:
        raise ValueError(
            f"train batch is missing {missing} required by {why}; "
            "build the batch with training.loop_step.prepare_batch"
        )


def _check_devices(batch: dict[str, Tensor], device: torch.device) -> None:
    """Training never moves tensors implicitly: every batch tensor must be on the model device."""
    for key, value in batch.items():
        if value.device != device:
            raise ValueError(
                f"batch['{key}'] is on {value.device} but the model is on {device}; "
                "move the batch explicitly (prepare_batch does it)"
            )


def _context(
    needed: frozenset[str],
    model: NTAZeroDCE,
    batch: dict[str, Tensor],
    config: Config,
    student_input: Tensor,
    student: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Compute exactly the context tensors named in ``needed`` (METHOD detach policy).

    ``sigma`` is the live noise-head output (the noise loss needs its gradient); the adapters
    that must not push gradient into it (edge, chroma) detach it themselves. ``gain`` and
    ``targets`` are constants by construction.
    """
    flags = config.flags
    ctx: dict[str, Tensor] = {
        "image": student_input,
        "enhanced": student["enhanced"],
        "A": student["A"],
        "sigma": student["sigma"],
    }
    if "gain" in needed:
        ctx["gain"] = compute_gain(
            student_input,
            student["enhanced_raw"],
            config.model.gain_window,
            config.model.gain_min,
            config.model.gain_max,
            config.model.eps,
        )
    if "targets" in needed:
        loss = config.loss
        ctx["targets"] = adaptive_targets(
            student_input,
            student["sigma"].detach(),
            target=loss.exposure_target,
            low=loss.adaptive_low,
            high=loss.adaptive_high,
            noise_scale=loss.adaptive_noise_scale,
            noise_penalty=loss.adaptive_noise_penalty,
            patch_size=loss.exposure_patch,
        )
    if "target_sigma" in needed:
        _require(batch, ["target_sigma"], "the noise loss (flags.noise_aware)")
        ctx["target_sigma"] = batch["target_sigma"].detach()
    if "noisy_enhanced" in needed:
        if not flags.noise_aware:
            raise ValueError("noisy_enhanced exists only for noise-aware training")
        ctx["noisy_enhanced"] = student["enhanced"]
    if needed & _TEACHER_INPUTS:
        _require(batch, ["image"], "the clean-input teacher")
        with torch.no_grad():
            teacher = model(batch["image"])
        ctx["teacher_enhanced"] = teacher["enhanced"].detach()
        ctx["teacher_raw"] = teacher["enhanced_raw"].detach()
    if needed & _TEMPORAL_INPUTS:
        _require(batch, _TEMPORAL_BATCH_KEYS, "the temporal loss (flags.temporal)")
        frames = torch.cat([batch["image1"], batch["image2"]], dim=0)
        pair = model(frames)
        previous, current = pair["enhanced"].chunk(2, dim=0)
        ctx["previous_output"] = previous
        ctx["current_output"] = current
        ctx["backward_flow"] = batch["backward_flow"].detach()
        ctx["mask"] = batch["mask"].detach()
    return ctx


def _call_loss(name: str, spec: LossSpec, ctx: dict[str, Tensor], config: Config) -> Tensor:
    """Explicit adapter from a loss name to its call, using ``config.loss`` scalar options."""
    missing = [key for key in spec.required_inputs if key not in ctx]
    if missing:
        raise ValueError(
            f"loss {name!r} requires context inputs {missing} that the train step did not "
            f"provide (available: {sorted(ctx)})"
        )
    loss = config.loss
    fn = spec.callable
    if name == "spa":
        return fn(ctx["image"], ctx["enhanced"], patch_size=loss.spatial_patch)
    if name == "exp":
        return fn(ctx["enhanced"], target=loss.exposure_target, patch_size=loss.exposure_patch)
    if name == "adaptive_exp":
        return fn(ctx["enhanced"], ctx["targets"], patch_size=loss.exposure_patch)
    if name == "col":
        return fn(ctx["enhanced"])
    if name == "tv":
        return fn(ctx["A"])
    if name == "edge":
        return fn(
            ctx["image"],
            ctx["enhanced"],
            ctx["sigma"].detach(),
            ctx["gain"].detach(),
            smooth_sigma=loss.edge_smooth_sigma,
            kernel_size=loss.edge_kernel,
            noise_multiplier=loss.edge_noise_multiplier,
            eps=loss.eps,
        )
    if name == "chroma":
        return fn(
            ctx["image"],
            ctx["enhanced"],
            ctx["sigma"].detach(),
            window=loss.chroma_window,
            eps=loss.eps,
            reliability_low=loss.reliability_low,
            reliability_high=loss.reliability_high,
            reliability_noise=loss.reliability_noise,
        )
    if name == "highlight":
        return fn(ctx["image"], ctx["enhanced"], threshold=loss.highlight_threshold)
    if name == "noise":
        return fn(ctx["sigma"], ctx["target_sigma"])
    if name == "invariance":
        return fn(ctx["noisy_enhanced"], ctx["teacher_enhanced"])
    if name == "fidelity":
        return fn(ctx["noisy_enhanced"], ctx["teacher_raw"])
    if name == "temporal":
        return fn(
            ctx["previous_output"],
            ctx["current_output"],
            ctx["backward_flow"],
            ctx["mask"],
        )
    raise ValueError(f"no train-step adapter for loss {name!r}; add one in loop_step._call_loss")


def active_losses(config: Config) -> dict[str, LossSpec]:
    """Active loss specs for ``config`` with the resolved weights (exp XOR adaptive_exp).

    Args:
        config: Validated Config.

    Returns:
        Name to LossSpec (``default_weight`` holds the resolved weight), in registry order.
    """
    return build_losses(asdict(config.flags), asdict(config.weights))


def compute_losses(
    model: NTAZeroDCE, batch: dict[str, Tensor], config: Config
) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
    """Forward the student (and teacher/pair as needed) and evaluate the gated losses.

    Args:
        model: Model on the batch device. Gradients flow through the student paths only.
        batch: Train batch as described in the module docstring.
        config: Validated Config.

    Returns:
        ``(total, raw, weighted)``: total is the scalar sum of ``weight * loss``; raw and
        weighted map every active loss name to its scalar tensor.

    Raises:
        ValueError: A required batch key is missing, devices disagree or a loss has no input.
    """
    flags = config.flags
    device = _device_of(model)
    if "image" not in batch:
        raise ValueError(f"train batch needs 'image'; got keys {sorted(batch)}")
    _check_devices(batch, device)
    specs = active_losses(config)
    if flags.noise_aware:
        _require(batch, ["noisy", "target_sigma"], "flags.noise_aware")
        student_input = batch["noisy"]
    else:
        student_input = batch["image"]
    needed = frozenset(name for spec in specs.values() for name in spec.required_inputs)
    student = dict(model(student_input))
    ctx = _context(needed, model, batch, config, student_input, student)
    raw: dict[str, Tensor] = {}
    weighted: dict[str, Tensor] = {}
    for name, spec in specs.items():
        value = _call_loss(name, spec, ctx, config)
        raw[name] = value
        weighted[name] = spec.default_weight * value
    total = torch.stack(list(weighted.values())).sum()
    return total, raw, weighted


def _scalars(
    total: Tensor, raw: dict[str, Tensor], weighted: dict[str, Tensor]
) -> dict[str, float]:
    """Detached Python floats: total, raw losses by name and ``weighted/<name>``."""
    out: dict[str, float] = {"total": float(total.detach())}
    for name, value in raw.items():
        out[name] = float(value.detach())
    for name, value in weighted.items():
        out[f"weighted/{name}"] = float(value.detach())
    return out


def _check_finite(total: Tensor, raw: dict[str, Tensor]) -> None:
    """Raise FloatingPointError naming the first non-finite loss (before any step)."""
    for name, value in raw.items():
        if not bool(torch.isfinite(value.detach())):
            raise FloatingPointError(f"non-finite loss {name!r}: {float(value.detach())}")
    if not bool(torch.isfinite(total.detach())):
        raise FloatingPointError(f"non-finite total loss: {float(total.detach())}")


def train_step(
    model: NTAZeroDCE, batch: dict[str, Tensor], optimizer: Optimizer, config: Config
) -> dict[str, float]:
    """Perform exactly one gated optimization step.

    Args:
        model: Model on the batch device (set to train mode here).
        batch: Context from ``prepare_batch``: ``image`` (B,3,H,W) in [0,1]; ``noisy`` and
            ``target_sigma`` when noise-aware; ``image1``, ``image2``, ``backward_flow``,
            ``mask`` when temporal.
        optimizer: Adam bound to the model parameters.
        config: Validated flags/weights and the gradient-norm clip.

    Returns:
        Detached Python floats: ``total``; every active raw loss under its name;
        ``grad_norm`` (global norm before clipping); ``weighted/<name>`` for each weighted term.

    Method:
        METHOD objective. The student sees the noisy image when noise-aware; the teacher is the
        same model under no_grad on the clean image. Sigma used by adaptive targets, edge and
        chroma, the gain, targets, teacher outputs, flow and mask are all detached; the noise
        loss alone sends gradient into the noise head, and the temporal loss sends it to both
        frame forwards. total = sum(weight * loss); zero_grad, backward, clip, step.

    Raises:
        FloatingPointError: A loss (or the gradient norm) is non-finite; the optimizer is not
            stepped and the parameters are unchanged.
        ValueError: Missing batch keys or tensors on a different device than the model.
    """
    model.train()
    total, raw, weighted = compute_losses(model, batch, config)
    _check_finite(total, raw)
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    grad_norm = clip_grad_norm_(model.parameters(), config.training.grad_clip)
    norm = float(grad_norm)
    if not math.isfinite(norm):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError(f"non-finite gradient norm ({norm}); optimizer step skipped")
    optimizer.step()
    out = _scalars(total, raw, weighted)
    out["grad_norm"] = norm
    return out


def validation_step(
    model: NTAZeroDCE, batch: dict[str, Tensor], config: Config
) -> dict[str, float]:
    """Evaluate the objective on a batch without gradients or an optimizer step.

    Args:
        model: Model on the batch device; its training mode is restored afterwards.
        batch: Train-style batch (see the module docstring).
        config: Validated Config.

    Returns:
        ``total``, every active raw loss and ``weighted/<name>`` (no ``grad_norm``). Non-finite
        values are reported as they are, not raised.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            total, raw, weighted = compute_losses(model, batch, config)
    finally:
        model.train(was_training)
    return _scalars(total, raw, weighted)
