"""Gated train step, batch preparation and the trainer (CPU, seeded, tiny)."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest
import torch
from _training_helpers import (
    FULL_FLAGS,
    make_model_and_optimizer,
    states_equal,
    tiny_config,
    write_test_video,
)
from torch import Tensor

from nta_zerodce.training.ablation import parse_matrix
from nta_zerodce.training.checkpoint import read_checkpoint
from nta_zerodce.training.config import Config, load_config
from nta_zerodce.training.logging import read_log
from nta_zerodce.training.loop_step import (
    _call_loss,
    _context,
    active_losses,
    compute_losses,
    prepare_batch,
    train_step,
    validation_step,
)
from nta_zerodce.training.trainer import default_run_name, split_hash, train
from nta_zerodce.utils.seed import make_generator

CPU = torch.device("cpu")
MATRIX = Path(__file__).resolve().parents[1] / "configs" / "ablation_matrix.py"
ROWS = [(name, dict(vars(flags))) for name, flags in parse_matrix(MATRIX)[2]]


def expected_losses(flags: dict[str, bool]) -> set[str]:
    """Loss names the METHOD total objective activates for ``flags``."""
    names = {"spa", "col", "tv"}
    names.add("adaptive_exp" if flags["adaptive_exp"] else "exp")
    if flags["adaptive_exp"]:
        names.add("highlight")
    for flag, name in (("edge", "edge"), ("chroma", "chroma"), ("temporal", "temporal")):
        if flags[flag]:
            names.add(name)
    if flags["noise_aware"]:
        names |= {"noise", "invariance"}
    if flags["denoise"]:
        names.add("fidelity")
    return names


def make_batch(config: Config, seed: int = 0, count: int = 2) -> dict[str, Tensor]:
    """Prepared train batch of dim 32x32 images (low light with a bright patch)."""
    generator = make_generator(seed)
    image = torch.rand(count, 3, 32, 32, generator=generator) * 0.3
    image[:, :, :8, :8] = 0.9
    return prepare_batch({"image": image}, config, generator, CPU)


# --------------------------------------------------------------------------- gating


@pytest.mark.parametrize(("name", "flags"), ROWS, ids=[row[0] for row in ROWS])
def test_gating_matches_the_objective_for_every_ablation_row(
    name: str, flags: dict[str, bool], tmp_path: Path
) -> None:
    config = tiny_config(tmp_path, flags, make_images=False)
    model, optimizer = make_model_and_optimizer(config)
    result = train_step(model, make_batch(config), optimizer, config)
    names = expected_losses(flags)
    assert set(active_losses(config)) == names
    assert not {"exp", "adaptive_exp"} <= set(result)  # mutually exclusive
    assert set(result) == (
        {"total", "grad_norm"} | names | {f"weighted/{loss}" for loss in names}
    ), name
    assert all(math.isfinite(value) for value in result.values())
    weights = dataclasses.asdict(config.weights)
    for loss in names:
        key = "exp" if loss == "adaptive_exp" else loss
        assert result[f"weighted/{loss}"] == pytest.approx(weights[key] * result[loss], rel=1e-5)
    assert result["total"] == pytest.approx(
        sum(result[f"weighted/{loss}"] for loss in names), rel=1e-5
    )


def test_adaptive_replaces_exp_at_the_exp_weight(tmp_path: Path) -> None:
    config = tiny_config(tmp_path, {"adaptive_exp": True}, make_images=False)
    config = dataclasses.replace(
        config, weights=dataclasses.replace(config.weights, exp=3.0, highlight=7.0)
    )
    specs = active_losses(config)
    assert "exp" not in specs and specs["adaptive_exp"].default_weight == 3.0
    assert specs["highlight"].default_weight == 7.0


def test_every_loss_is_computed_with_a_gradient_path(tmp_path: Path) -> None:
    config = tiny_config(tmp_path, FULL_FLAGS, make_images=False)
    model, _ = make_model_and_optimizer(config)
    _, raw, _ = compute_losses(model, make_batch(config), config)
    assert set(raw) == expected_losses(FULL_FLAGS | {"light": False})
    for name, value in raw.items():
        assert value.ndim == 0 and value.requires_grad, name
        assert torch.isfinite(value), name


def test_light_model_trains(tmp_path: Path) -> None:
    config = tiny_config(tmp_path, {"light": True, "noise_aware": True}, make_images=False)
    model, optimizer = make_model_and_optimizer(config)
    result = train_step(model, make_batch(config), optimizer, config)
    assert math.isfinite(result["total"])


# --------------------------------------------------------------------------- optimisation


def test_step_changes_parameters_and_reports_the_pre_clip_norm(tmp_path: Path) -> None:
    config = tiny_config(tmp_path, {"noise_aware": True}, make_images=False)
    model, optimizer = make_model_and_optimizer(config)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    result = train_step(model, make_batch(config), optimizer, config)
    after = model.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before)
    assert result["grad_norm"] > 0 and model.training


def test_gradient_clipping_bound_is_respected(tmp_path: Path) -> None:
    config = tiny_config(tmp_path, {"noise_aware": True}, make_images=False)
    clip = 1e-3
    config = dataclasses.replace(
        config, training=dataclasses.replace(config.training, grad_clip=clip)
    )
    model, optimizer = make_model_and_optimizer(config)
    result = train_step(model, make_batch(config), optimizer, config)
    assert result["grad_norm"] > clip  # clipping was active
    clipped = torch.sqrt(sum((p.grad**2).sum() for p in model.parameters() if p.grad is not None))
    assert float(clipped) <= clip * (1 + 1e-4)


def test_loss_decreases_over_repeated_steps_on_one_batch(tmp_path: Path) -> None:
    config = tiny_config(tmp_path, make_images=False)
    config = dataclasses.replace(
        config, training=dataclasses.replace(config.training, lr=1e-2, grad_clip=10.0)
    )
    model, optimizer = make_model_and_optimizer(config)
    batch = make_batch(config)
    first = train_step(model, batch, optimizer, config)["exp"]
    for _ in range(14):
        last = train_step(model, batch, optimizer, config)["exp"]
    assert last < first


def test_non_finite_loss_raises_and_does_not_step(tmp_path: Path) -> None:
    config = tiny_config(tmp_path, make_images=False)
    model, optimizer = make_model_and_optimizer(config)
    batch = make_batch(config)
    batch["image"] = batch["image"].clone()
    batch["image"][0, 0, 0, 0] = float("nan")
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with pytest.raises(FloatingPointError, match=r"non-finite loss '(spa|exp|col|tv)'"):
        train_step(model, batch, optimizer, config)
    assert states_equal(before, model.state_dict())


def test_validation_step_has_no_side_effects(tmp_path: Path) -> None:
    config = tiny_config(tmp_path, {"noise_aware": True, "temporal": True}, make_images=False)
    model, optimizer = make_model_and_optimizer(config)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    result = validation_step(model, make_batch(config), config)
    assert "grad_norm" not in result and "temporal" in result
    assert states_equal(before, model.state_dict()) and model.training
    assert all(p.grad is None for p in model.parameters())
    assert optimizer.state_dict()["state"] == {}


# --------------------------------------------------------------------------- batch contract


def test_missing_batch_keys_raise_a_clear_error(tmp_path: Path) -> None:
    config = tiny_config(tmp_path, {"noise_aware": True, "temporal": True}, make_images=False)
    model, optimizer = make_model_and_optimizer(config)
    with pytest.raises(ValueError, match="image"):
        train_step(model, {}, optimizer, config)
    with pytest.raises(ValueError, match=r"\['noisy', 'target_sigma'\].*prepare_batch"):
        train_step(model, {"image": torch.rand(1, 3, 32, 32)}, optimizer, config)
    batch = make_batch(config)
    del batch["mask"]
    with pytest.raises(ValueError, match="'mask'"):
        train_step(model, batch, optimizer, config)


def test_unknown_loss_and_missing_context_inputs_are_rejected(tmp_path: Path) -> None:
    config = tiny_config(tmp_path, {"edge": True}, make_images=False)
    spec = active_losses(config)["edge"]
    with pytest.raises(ValueError, match=r"requires context inputs \['gain'\]"):
        _call_loss("edge", spec, {"image": 0, "enhanced": 0, "sigma": 0}, config)
    with pytest.raises(ValueError, match="no train-step adapter"):
        _call_loss("mystery", spec.__class__(spec.callable, None, 1.0, ()), {}, config)


def test_prepare_batch_noise_is_deterministic_and_matches_the_target(tmp_path: Path) -> None:
    config = tiny_config(tmp_path, {"noise_aware": True}, make_images=False)
    image = torch.rand(3, 3, 32, 32, generator=make_generator(1)) * 0.5
    one = prepare_batch({"image": image}, config, make_generator(5), CPU)
    two = prepare_batch({"image": image}, config, make_generator(5), CPU)
    other = prepare_batch({"image": image}, config, make_generator(6), CPU)
    assert set(one) == {"image", "noisy", "target_sigma"}
    assert torch.equal(one["image"], image)
    assert torch.equal(one["noisy"], two["noisy"]) and not torch.equal(one["noisy"], other["noisy"])
    assert one["noisy"].min() >= 0 and one["noisy"].max() <= 1
    sigma = one["target_sigma"]
    assert sigma.shape == (3, 1, 32, 32)
    assert sigma.max() <= math.sqrt(config.noise.a_max + config.noise.b_max**2) + 1e-6
    # per-image coefficients: the target is constant per image up to the luminance term
    assert sigma.amin() >= config.noise.b_min - 1e-6


def test_noise_probability_zero_leaves_images_clean(tmp_path: Path) -> None:
    config = tiny_config(tmp_path, {"noise_aware": True}, make_images=False)
    config = dataclasses.replace(config, noise=dataclasses.replace(config.noise, probability=0.0))
    image = torch.rand(2, 3, 32, 32, generator=make_generator(1))
    batch = prepare_batch({"image": image}, config, make_generator(0), CPU)
    assert torch.equal(batch["noisy"], image) and float(batch["target_sigma"].abs().max()) == 0.0


def test_prepare_batch_without_flags_adds_nothing(tmp_path: Path) -> None:
    config = tiny_config(tmp_path, make_images=False)
    batch = prepare_batch({"image": torch.rand(1, 3, 32, 32)}, config, make_generator(0), CPU)
    assert set(batch) == {"image"}
    with pytest.raises(ValueError, match="image"):
        prepare_batch({}, config, make_generator(0), CPU)


def test_synthetic_temporal_batch_geometry(tmp_path: Path) -> None:
    config = tiny_config(tmp_path, {"temporal": True}, make_images=False)
    image = torch.rand(2, 3, 32, 32, generator=make_generator(2))
    batch = prepare_batch({"image": image}, config, make_generator(0), CPU)
    assert batch["image1"].shape == batch["image2"].shape == (2, 3, 32, 32)
    assert batch["backward_flow"].shape == (2, 2, 32, 32)
    assert set(batch["mask"].unique().tolist()) <= {0.0, 1.0} and batch["mask"].mean() > 0.5


def test_real_temporal_batch_uses_dis_flow(tmp_path: Path) -> None:
    config = tiny_config(tmp_path, {"temporal": True}, make_images=False)
    config = dataclasses.replace(
        config, temporal=dataclasses.replace(config.temporal, pair_source="real")
    )
    texture = torch.rand(1, 3, 64, 64, generator=make_generator(3)).repeat(2, 1, 1, 1)
    first = texture[:, :, :32, :32].contiguous()
    second = texture[:, :, :32, 1:33].contiguous()  # content shifted left by one pixel
    batch = prepare_batch(
        {"image": first, "image1": first, "image2": second}, config, make_generator(0), CPU
    )
    flow, mask = batch["backward_flow"], batch["mask"]
    assert flow.shape == (2, 2, 32, 32) and mask.shape == (2, 1, 32, 32)
    assert torch.isfinite(flow).all() and set(mask.unique().tolist()) <= {0.0, 1.0}
    with pytest.raises(ValueError, match="image2"):
        prepare_batch({"image": first, "image1": first}, config, make_generator(0), CPU)
    only_pair = prepare_batch({"image1": first, "image2": second}, config, make_generator(0), CPU)
    assert torch.equal(only_pair["image"], first)
    model, optimizer = make_model_and_optimizer(config)
    assert math.isfinite(train_step(model, batch, optimizer, config)["temporal"])


# --------------------------------------------------------------------------- gradient reach


def test_noise_loss_reaches_only_the_noise_head(tmp_path: Path) -> None:
    config = tiny_config(tmp_path, {"noise_aware": True}, make_images=False)
    model, _ = make_model_and_optimizer(config)
    _, raw, _ = compute_losses(model, make_batch(config), config)
    raw["noise"].backward()
    assert model.noise_branch is not None
    assert all(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.noise_branch.parameters()
    )
    assert all(p.grad is None for p in model.dce.parameters())


def test_context_tensors_that_must_be_constants_carry_no_graph(tmp_path: Path) -> None:
    config = tiny_config(tmp_path, FULL_FLAGS, make_images=False)
    model, _ = make_model_and_optimizer(config)
    batch = make_batch(config)
    student_input = batch["noisy"]
    student = dict(model(student_input))
    needed = frozenset(
        name for spec in active_losses(config).values() for name in spec.required_inputs
    )
    ctx = _context(needed, model, batch, config, student_input, student)
    for name in ("gain", "targets", "teacher_enhanced", "teacher_raw", "backward_flow", "mask"):
        assert name in ctx and not ctx[name].requires_grad, name
    for name in ("enhanced", "A", "sigma", "noisy_enhanced", "previous_output", "current_output"):
        assert ctx[name].requires_grad, name


@pytest.mark.parametrize("name", ["edge", "chroma"])
def test_edge_and_chroma_detach_sigma(name: str, tmp_path: Path) -> None:
    config = tiny_config(tmp_path, {name: True}, make_images=False)
    spec = active_losses(config)[name]
    generator = make_generator(4)
    image = torch.rand(1, 3, 32, 32, generator=generator)
    enhanced = (image * 1.5).clamp(0, 1).requires_grad_(True)
    sigma = (torch.rand(1, 1, 32, 32, generator=generator) * 0.05).requires_grad_(True)
    gain = torch.full((1, 1, 32, 32), 2.0, requires_grad=True)
    ctx = {"image": image, "enhanced": enhanced, "sigma": sigma, "gain": gain}
    value = _call_loss(name, spec, ctx, config)
    value.backward()
    assert enhanced.grad is not None and sigma.grad is None and gain.grad is None


def test_teacher_receives_no_gradient(tmp_path: Path) -> None:
    config = tiny_config(tmp_path, {"noise_aware": True, "denoise": True}, make_images=False)
    model, _ = make_model_and_optimizer(config)
    batch = make_batch(config)
    batch["image"].requires_grad_(True)
    _, raw, _ = compute_losses(model, batch, config)
    for name in ("invariance", "fidelity"):
        (grad,) = torch.autograd.grad(raw[name], batch["image"], allow_unused=True)
        assert grad is None, name  # the clean-input teacher is under no_grad
        assert raw[name].requires_grad  # while the noisy student keeps its graph


def test_temporal_loss_sends_gradient_to_both_frames(tmp_path: Path) -> None:
    config = tiny_config(tmp_path, {"temporal": True}, make_images=False)
    model, _ = make_model_and_optimizer(config)
    batch = make_batch(config)
    batch["image1"].requires_grad_(True)
    batch["image2"].requires_grad_(True)
    _, raw, _ = compute_losses(model, batch, config)
    first, second = torch.autograd.grad(raw["temporal"], [batch["image1"], batch["image2"]])
    assert first.abs().sum() > 0 and second.abs().sum() > 0
    assert batch["backward_flow"].grad is None and batch["mask"].grad is None


def test_adaptive_targets_cannot_be_lowered_through_sigma(tmp_path: Path) -> None:
    config = tiny_config(tmp_path, {"adaptive_exp": True, "noise_aware": True}, make_images=False)
    model, _ = make_model_and_optimizer(config)
    batch = make_batch(config)
    student = dict(model(batch["noisy"]))
    ctx = _context(frozenset({"targets"}), model, batch, config, batch["noisy"], student)
    assert ctx["targets"].requires_grad is False and student["sigma"].requires_grad


# --------------------------------------------------------------------------- trainer


def test_default_run_name_is_deterministic(tmp_path: Path) -> None:
    def name(flags: dict[str, bool], seed: int = 17) -> str:
        return default_run_name(tiny_config(tmp_path, flags, seed=seed, make_images=False))

    assert name({}) == "baseline_s17"
    assert name({"noise_aware": True, "denoise": True}, 29) == "noise-denoise_s29"
    assert name({"edge": True, "temporal": True}) == "edge-temporal_s17"
    assert name(FULL_FLAGS) == "full_s17"
    assert name(FULL_FLAGS | {"light": True}, 43) == "full-light_s43"
    assert name({"light": True}) == "baseline-light_s17"


def test_split_hash_ignores_order_and_root(tmp_path: Path) -> None:
    a, b = tmp_path / "x" / "1.png", tmp_path / "x" / "2.png"
    assert split_hash([a, b], tmp_path / "x") == split_hash([b, a], tmp_path / "x")
    assert split_hash([a], tmp_path / "x") != split_hash([b], tmp_path / "x")
    moved = tmp_path / "y"
    assert split_hash([a, b], tmp_path / "x") == split_hash(
        [moved / "1.png", moved / "2.png"], moved
    )
    assert len(split_hash([], tmp_path)) == 64


def test_baseline_run_writes_checkpoints_logs_and_config(tmp_path: Path) -> None:
    config = tiny_config(tmp_path, epochs=2)
    final = train(config)
    run_dir, ckpt_dir = (
        tmp_path / "runs" / "baseline_s17",
        tmp_path / "checkpoints" / "baseline_s17",
    )
    assert final == ckpt_dir / "final.pt"
    assert sorted(p.name for p in ckpt_dir.iterdir()) == [
        "epoch_0001.pt",
        "epoch_0002.pt",
        "final.pt",
    ]
    assert load_config(run_dir / "config.yaml") == config
    train_rows, val_rows = read_log(run_dir / "metrics.csv"), read_log(run_dir / "val_metrics.csv")
    steps = 3  # 8 images, one held-out scene: 7 train images, batch 2, drop_last
    assert [row["step"] for row in train_rows] == [float(i) for i in range(1, 2 * steps + 1)]
    assert [row["epoch"] for row in train_rows] == [0.0] * steps + [1.0] * steps
    assert [row["step"] for row in val_rows] == [float(steps), float(2 * steps)]
    assert "grad_norm" in train_rows[0] and "grad_norm" not in val_rows[0]  # no column mixing
    assert set(train_rows[0]) >= {"step", "epoch", "total", "spa", "exp", "col", "tv"}
    payload = read_checkpoint(final)
    assert payload["epoch"] == 1
    provenance = payload["provenance"]
    assert len(provenance["train_split_sha256"]) == len(provenance["val_split_sha256"]) == 64
    assert provenance["train_split_sha256"] != provenance["val_split_sha256"]
    assert provenance["val_images"] == "1" and provenance["global_step"] == str(2 * steps)


def test_full_run_with_synthetic_temporal_pairs(tmp_path: Path) -> None:
    config = tiny_config(tmp_path, FULL_FLAGS, epochs=2, images=6)
    final = train(config, run_name="full_smoke")
    run_dir = tmp_path / "runs" / "full_smoke"
    rows = read_log(run_dir / "metrics.csv")
    columns = set(rows[0])
    assert {"noise", "invariance", "fidelity", "edge", "chroma", "highlight", "temporal"} <= columns
    assert "exp" not in columns and "adaptive_exp" in columns
    assert all(set(row) == columns for row in rows)
    assert all(math.isfinite(value) for row in rows for value in row.values())
    assert set(read_log(run_dir / "val_metrics.csv")[0]) == columns - {"grad_norm"}
    assert final.is_file() and read_checkpoint(final)["flags"]["temporal"] is True


def test_log_every_averages_the_window(tmp_path: Path) -> None:
    config = tiny_config(tmp_path, epochs=1, images=10, log_every=2)  # 9 train imgs -> 4 steps
    train(config)
    rows = read_log(tmp_path / "runs" / "baseline_s17" / "metrics.csv")
    assert [row["step"] for row in rows] == [2.0, 4.0]


def test_resume_reproduces_an_uninterrupted_run_exactly(tmp_path: Path) -> None:
    flags = {"noise_aware": True, "denoise": True, "temporal": True, "edge": True}
    straight_root, split_root = tmp_path / "a", tmp_path / "b"
    straight = tiny_config(straight_root, flags, epochs=3, images=6)
    final_a = train(straight)
    config_b = tiny_config(split_root, flags, epochs=2, images=6)
    train(config_b, run_name="b")
    interrupted = tmp_path / "b" / "checkpoints" / "b" / "epoch_0002.pt"
    resumed = dataclasses.replace(
        config_b, training=dataclasses.replace(config_b.training, epochs=3)
    )
    final_b = train(resumed, interrupted, run_name="b")
    state_a = read_checkpoint(final_a)
    state_b = read_checkpoint(final_b)
    assert states_equal(state_a["model"], state_b["model"])
    for key, value in state_a["optimizer"]["state"].items():
        for name, tensor in value.items():
            assert torch.equal(
                torch.as_tensor(tensor), torch.as_tensor(state_b["optimizer"]["state"][key][name])
            )
    rows_a = read_log(straight_root / "runs" / default_run_name(straight) / "metrics.csv")
    rows_b = read_log(split_root / "runs" / "b" / "metrics.csv")
    assert rows_a == rows_b  # log continues without duplicates or gaps
    val_a = read_log(straight_root / "runs" / default_run_name(straight) / "val_metrics.csv")
    assert val_a == read_log(split_root / "runs" / "b" / "val_metrics.csv")


def test_resume_discards_log_rows_after_the_checkpoint(tmp_path: Path) -> None:
    config = tiny_config(tmp_path, epochs=2)
    train(config)
    run_dir = tmp_path / "runs" / "baseline_s17"
    clean = read_log(run_dir / "metrics.csv")
    # Pretend the process died during epoch 1 after the epoch-0 checkpoint: keep its stale rows.
    resumed = train(config, tmp_path / "checkpoints" / "baseline_s17" / "epoch_0001.pt")
    assert read_log(run_dir / "metrics.csv") == clean
    assert resumed.is_file()


def test_resume_rejects_changed_flags_split_and_finished_runs(tmp_path: Path) -> None:
    config = tiny_config(tmp_path, epochs=1, images=8)
    final = train(config)
    edge = tiny_config(tmp_path, {"edge": True}, epochs=2, make_images=False)
    with pytest.raises(ValueError, match="exact resume requires identical flags"):
        train(edge, final)
    again = train(config, final)  # already complete: only final.pt is rewritten
    assert states_equal(read_checkpoint(again)["model"], read_checkpoint(final)["model"])
    shorter = dataclasses.replace(config, training=dataclasses.replace(config.training, epochs=1))
    longer = train(shorter, final, run_name="longer")  # fresh name, same weights
    assert longer.is_file()
    two = train(
        dataclasses.replace(config, training=dataclasses.replace(config.training, epochs=2))
    )
    with pytest.raises(ValueError, match="completed 2 epoch"):
        train(config, two)
    other = tiny_config(tmp_path / "other", epochs=2, images=9)
    with pytest.raises(ValueError, match="split_sha256 changed"):
        train(other, final, run_name="other")
    with pytest.raises(FileNotFoundError):
        train(config, tmp_path / "missing.pt")


def test_fresh_run_replaces_state_of_the_same_run_name(tmp_path: Path) -> None:
    train(tiny_config(tmp_path, epochs=2))
    ckpt_dir = tmp_path / "checkpoints" / "baseline_s17"
    (ckpt_dir / "stray.tmp").write_bytes(b"x")
    train(tiny_config(tmp_path, epochs=1, make_images=False))
    assert sorted(p.name for p in ckpt_dir.iterdir()) == ["epoch_0001.pt", "final.pt"]
    rows = read_log(tmp_path / "runs" / "baseline_s17" / "metrics.csv")
    assert len(rows) == 3 and rows[0]["step"] == 1.0


def test_output_root_overrides_and_same_seed_is_deterministic(tmp_path: Path) -> None:
    config = tiny_config(tmp_path, {"noise_aware": True})
    one = train(config, run_name="one", run_dir=tmp_path / "r1", checkpoint_dir=tmp_path / "c1")
    two = train(config, run_name="two", run_dir=tmp_path / "r2", checkpoint_dir=tmp_path / "c2")
    assert one == tmp_path / "c1" / "one" / "final.pt"
    assert (tmp_path / "r1" / "one" / "config.yaml").is_file()
    assert states_equal(read_checkpoint(one)["model"], read_checkpoint(two)["model"])
    other = train(
        tiny_config(tmp_path, {"noise_aware": True}, seed=29, make_images=False),
        run_dir=tmp_path / "r3",
        checkpoint_dir=tmp_path / "c3",
    )
    assert not states_equal(read_checkpoint(one)["model"], read_checkpoint(other)["model"])


def test_missing_data_fails_fast_with_the_path(tmp_path: Path) -> None:
    config = tiny_config(tmp_path, make_images=False)
    with pytest.raises(FileNotFoundError, match="train_images"):
        train(config)
    assert not (tmp_path / "runs").exists()
    real = tiny_config(tmp_path, {"temporal": True}, images=4)
    real = dataclasses.replace(
        real, temporal=dataclasses.replace(real.temporal, pair_source="real")
    )
    with pytest.raises(FileNotFoundError, match="video_dir"):
        train(real)


def test_training_with_real_video_pairs(tmp_path: Path) -> None:
    config = tiny_config(tmp_path, {"temporal": True, "noise_aware": True}, epochs=1, images=6)
    config = dataclasses.replace(
        config, temporal=dataclasses.replace(config.temporal, pair_source="real")
    )
    write_test_video(tmp_path / "videos" / "train" / "s1" / "clip.avi", frames=6, size=64)
    final = train(config, run_name="real")
    rows = read_log(tmp_path / "runs" / "real" / "metrics.csv")
    assert "temporal" in rows[0] and all(math.isfinite(row["total"]) for row in rows)
    payload = read_checkpoint(final)
    assert payload["provenance"]["video_pairs"] == "5"
    val = read_log(tmp_path / "runs" / "real" / "val_metrics.csv")
    assert "temporal" not in val[0]  # validation covers the still-image objective only


def test_trainer_requires_a_valid_config(tmp_path: Path) -> None:
    config = tiny_config(tmp_path, make_images=False)
    broken = dataclasses.replace(config, training=dataclasses.replace(config.training, lr=-1.0))
    with pytest.raises(ValueError, match="training.lr"):
        train(broken)
