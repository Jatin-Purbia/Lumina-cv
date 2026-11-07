"""Checkpoint format v1 and CSV logging: atomicity, rejection paths, RNG and state round trips."""

from __future__ import annotations

import csv
import dataclasses
import math
import random
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

import numpy as np
import pytest
import torch
from _training_helpers import make_model_and_optimizer, states_equal, tiny_config

from nta_zerodce.models.factory import build_model
from nta_zerodce.training.checkpoint import (
    FORMAT_VERSION,
    checkpoint_config,
    load_checkpoint,
    load_checkpoint_full,
    read_checkpoint,
    save_checkpoint,
)
from nta_zerodce.training.config import Config
from nta_zerodce.training.logging import log_metrics, read_log, truncate_log


@pytest.fixture
def config(tmp_path: Path) -> Config:
    """Baseline-flag tiny config (no dataset needed for checkpoint tests)."""
    return tiny_config(tmp_path, make_images=False)


def _trained_pair(config: Config) -> tuple[Any, Any]:
    """Model and optimizer after a couple of Adam steps so the optimizer state is non-empty."""
    model, optimizer = make_model_and_optimizer(config)
    for _ in range(2):
        optimizer.zero_grad()
        model(torch.rand(1, 3, 16, 16))["enhanced"].mean().backward()
        optimizer.step()
    return model, optimizer


def _rewrite(path: Path, edit: Callable[[dict[str, Any]], None]) -> None:
    """Load a checkpoint payload, apply ``edit`` and save it back (a tamper helper)."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    edit(payload)
    torch.save(payload, path)


# --------------------------------------------------------------------------- round trip


def test_round_trip_restores_weights_optimizer_config_and_provenance(
    config: Config, tmp_path: Path
) -> None:
    model, optimizer = _trained_pair(config)
    path = tmp_path / "a.pt"
    save_checkpoint(path, model, optimizer, config, 3, provenance={"train_split_sha256": "abc"})
    fresh, fresh_opt = make_model_and_optimizer(config, seed=99)
    loaded = load_checkpoint_full(path, fresh, fresh_opt, expected_config=config)
    assert states_equal(model.state_dict(), fresh.state_dict())
    assert loaded.config == config and loaded.epoch == 3
    assert loaded.provenance["train_split_sha256"] == "abc"
    assert {"git_commit", "python", "torch", "numpy"} <= set(loaded.provenance)
    assert checkpoint_config(path) == config
    assert load_checkpoint(path, make_model_and_optimizer(config)[0]) == config
    saved_state = optimizer.state_dict()["state"]
    for index, entry in saved_state.items():
        for key, value in entry.items():
            assert torch.equal(
                torch.as_tensor(value), torch.as_tensor(fresh_opt.state_dict()["state"][index][key])
            )
    payload = read_checkpoint(path)
    assert payload["format_version"] == FORMAT_VERSION == 1
    assert payload["flags"] == dataclasses.asdict(config.flags)


def test_rng_states_round_trip(config: Config, tmp_path: Path) -> None:
    model, optimizer = _trained_pair(config)
    path = tmp_path / "rng.pt"
    random.seed(5)
    np.random.seed(6)
    torch.manual_seed(7)
    save_checkpoint(path, model, optimizer, config, 0)
    expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    load_checkpoint_full(path, model, optimizer, expected_config=config)
    assert (random.random(), float(np.random.rand()), float(torch.rand(1))) == expected


def test_load_without_optimizer_leaves_rng_alone(config: Config, tmp_path: Path) -> None:
    model, optimizer = _trained_pair(config)
    path = tmp_path / "plain.pt"
    save_checkpoint(path, model, optimizer, config, 0)
    torch.manual_seed(11)
    expected = float(torch.rand(1))
    torch.manual_seed(11)
    load_checkpoint(path, model)
    assert float(torch.rand(1)) == expected


# --------------------------------------------------------------------------- atomic write


def test_failed_write_leaves_no_partial_file(
    config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, optimizer = _trained_pair(config)
    path = tmp_path / "atomic.pt"

    def boom(obj: object, handle: IO[bytes]) -> None:
        handle.write(b"partial bytes")
        raise RuntimeError("disk full")

    monkeypatch.setattr(torch, "save", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        save_checkpoint(path, model, optimizer, config, 0)
    assert list(tmp_path.iterdir()) == []


def test_failed_overwrite_keeps_the_previous_checkpoint(
    config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, optimizer = _trained_pair(config)
    path = tmp_path / "keep.pt"
    save_checkpoint(path, model, optimizer, config, 1)
    before = path.read_bytes()

    def boom(obj: object, handle: IO[bytes]) -> None:
        raise RuntimeError("interrupted")

    monkeypatch.setattr(torch, "save", boom)
    with pytest.raises(RuntimeError):
        save_checkpoint(path, model, optimizer, config, 2)
    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["keep.pt"]


def test_save_argument_errors(config: Config, tmp_path: Path) -> None:
    model, optimizer = make_model_and_optimizer(config)
    with pytest.raises(ValueError, match="epoch"):
        save_checkpoint(tmp_path / "x.pt", model, optimizer, config, -1)
    with pytest.raises(ValueError, match="epoch"):
        save_checkpoint(tmp_path / "x.pt", model, optimizer, config, True)
    with pytest.raises(FileNotFoundError, match="directory"):
        save_checkpoint(tmp_path / "missing" / "x.pt", model, optimizer, config, 0)
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- rejection paths


def test_missing_and_unreadable_files(config: Config, tmp_path: Path) -> None:
    model, _ = make_model_and_optimizer(config)
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nope.pt", model)
    garbage = tmp_path / "garbage.pt"
    garbage.write_bytes(b"this is not a checkpoint")
    with pytest.raises(ValueError, match="cannot read"):
        load_checkpoint(garbage, model)
    good = tmp_path / "good.pt"
    save_checkpoint(good, model, make_model_and_optimizer(config)[1], config, 0)
    truncated = tmp_path / "truncated.pt"
    truncated.write_bytes(good.read_bytes()[:200])
    with pytest.raises(ValueError, match="cannot read"):
        load_checkpoint(truncated, model)


def test_unsafe_pickle_is_rejected(config: Config, tmp_path: Path) -> None:
    class Payload:
        def __reduce__(self) -> tuple[Any, ...]:
            return (print, ("executed",))

    path = tmp_path / "evil.pt"
    torch.save({"x": Payload()}, path)
    with pytest.raises(ValueError, match="cannot read"):
        read_checkpoint(path)


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("format_version", 2, "format_version"),
        ("format_version", "1", "format_version"),
        ("schema_version", 2, "schema_version"),
    ],
)
def test_unknown_versions_are_rejected(
    config: Config, tmp_path: Path, key: str, value: object, match: str
) -> None:
    model, optimizer = make_model_and_optimizer(config)
    path = tmp_path / "v.pt"
    save_checkpoint(path, model, optimizer, config, 0)
    _rewrite(path, lambda payload: payload.__setitem__(key, value))
    with pytest.raises(ValueError, match=match):
        load_checkpoint(path, model)


def test_unexpected_and_missing_keys_are_rejected(config: Config, tmp_path: Path) -> None:
    model, optimizer = make_model_and_optimizer(config)
    path = tmp_path / "keys.pt"
    save_checkpoint(path, model, optimizer, config, 0)
    _rewrite(path, lambda payload: payload.__setitem__("extra", 1))
    with pytest.raises(ValueError, match=r"unexpected \['extra'\]"):
        read_checkpoint(path)
    save_checkpoint(path, model, optimizer, config, 0)
    _rewrite(path, lambda payload: payload.pop("rng"))
    with pytest.raises(ValueError, match=r"missing \['rng'\]"):
        read_checkpoint(path)
    path.write_bytes(b"")
    torch.save([1, 2, 3], path)
    with pytest.raises(ValueError, match="mapping"):
        read_checkpoint(path)


@pytest.mark.parametrize(
    "edit",
    [
        lambda p: p.__setitem__("epoch", -1),
        lambda p: p.__setitem__("epoch", True),
        lambda p: p.__setitem__("epoch", 1.5),
        lambda p: p.__setitem__("config", "text"),
        lambda p: p.__setitem__("provenance", [1]),
        lambda p: p.__setitem__("model", {"w": 1}),
        lambda p: p["rng"].pop("torch"),
    ],
)
def test_malformed_fields_are_rejected(
    config: Config, tmp_path: Path, edit: Callable[[dict[str, Any]], None]
) -> None:
    model, optimizer = make_model_and_optimizer(config)
    path = tmp_path / "bad.pt"
    save_checkpoint(path, model, optimizer, config, 0)
    _rewrite(path, edit)
    with pytest.raises(ValueError):
        load_checkpoint(path, model)


def test_tampered_config_is_rejected(config: Config, tmp_path: Path) -> None:
    model, optimizer = make_model_and_optimizer(config)
    path = tmp_path / "cfg.pt"
    save_checkpoint(path, model, optimizer, config, 0)
    _rewrite(path, lambda p: p["config"]["training"].__setitem__("lr", -1.0))
    with pytest.raises(ValueError, match="invalid config"):
        load_checkpoint(path, model)
    save_checkpoint(path, model, optimizer, config, 0)
    _rewrite(path, lambda p: p["config"].pop("flags"))
    with pytest.raises(ValueError, match="config"):
        checkpoint_config(path)


def test_flags_that_disagree_with_the_stored_config_are_rejected(
    config: Config, tmp_path: Path
) -> None:
    model, optimizer = make_model_and_optimizer(config)
    path = tmp_path / "flags.pt"
    save_checkpoint(path, model, optimizer, config, 0)
    _rewrite(path, lambda p: p["flags"].__setitem__("edge", True))
    with pytest.raises(ValueError, match="disagree"):
        load_checkpoint(path, model)


@pytest.mark.parametrize(
    ("flags", "match"),
    [
        ({"noise_aware": True}, "flags.noise_aware"),
        ({"noise_aware": True, "denoise": True}, "flags.noise_aware"),
        ({"light": True}, "flags.light"),
    ],
)
def test_architecture_flag_mismatch_is_rejected(
    config: Config, tmp_path: Path, flags: dict[str, bool], match: str
) -> None:
    model, optimizer = make_model_and_optimizer(config)
    path = tmp_path / "arch.pt"
    save_checkpoint(path, model, optimizer, config, 0)
    other = tiny_config(tmp_path, flags, make_images=False)
    with pytest.raises(ValueError, match=match):
        load_checkpoint(path, build_model(other), expected_config=other)


def test_model_size_mismatch_is_rejected(config: Config, tmp_path: Path) -> None:
    model, optimizer = make_model_and_optimizer(config)
    path = tmp_path / "size.pt"
    save_checkpoint(path, model, optimizer, config, 0)
    wider = dataclasses.replace(config, model=dataclasses.replace(config.model, channels=16))
    with pytest.raises(ValueError, match="model.channels"):
        load_checkpoint(path, build_model(wider), expected_config=wider)
    with pytest.raises(ValueError, match="size mismatch"):
        load_checkpoint(path, build_model(wider))  # shapes differ, no expected config given


def test_loss_only_flag_change_allowed_for_weights_but_not_for_exact_resume(
    config: Config, tmp_path: Path
) -> None:
    model, optimizer = _trained_pair(config)
    path = tmp_path / "loss_flag.pt"
    save_checkpoint(path, model, optimizer, config, 0)
    edge_config = tiny_config(tmp_path, {"edge": True}, make_images=False)
    target = build_model(edge_config)
    load_checkpoint(path, target, expected_config=edge_config)  # same architecture: fine
    assert states_equal(model.state_dict(), target.state_dict())
    with pytest.raises(ValueError, match="exact resume"):
        load_checkpoint(
            path,
            build_model(edge_config),
            Adam_for(build_model(edge_config)),
            expected_config=edge_config,
        )


def Adam_for(model: torch.nn.Module) -> torch.optim.Adam:  # noqa: N802 - test helper name
    """Adam over ``model`` with the default test learning rate."""
    return torch.optim.Adam(model.parameters(), lr=1e-4)


def test_rejected_load_does_not_modify_model_optimizer_or_rng(
    config: Config, tmp_path: Path
) -> None:
    model, optimizer = _trained_pair(config)
    path = tmp_path / "bad_opt.pt"
    save_checkpoint(path, model, optimizer, config, 0)
    # Valid model state, broken optimizer state: nothing may be half loaded.
    _rewrite(path, lambda p: p["optimizer"]["param_groups"].clear())
    target, target_opt = make_model_and_optimizer(config, seed=5)
    weights_before = {k: v.clone() for k, v in target.state_dict().items()}
    opt_before = target_opt.state_dict()
    torch.manual_seed(21)
    rng_before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="cannot be loaded"):
        load_checkpoint_full(path, target, target_opt, expected_config=config)
    assert states_equal(weights_before, target.state_dict())
    assert target_opt.state_dict()["state"] == opt_before["state"]
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_wrong_tensor_shape_is_rejected_without_changing_the_model(
    config: Config, tmp_path: Path
) -> None:
    model, optimizer = make_model_and_optimizer(config)
    path = tmp_path / "shape.pt"
    save_checkpoint(path, model, optimizer, config, 0)

    def edit(payload: dict[str, Any]) -> None:
        name = next(iter(payload["model"]))
        payload["model"][name] = torch.zeros(1)

    _rewrite(path, edit)
    target, _ = make_model_and_optimizer(config, seed=2)
    before = {k: v.clone() for k, v in target.state_dict().items()}
    with pytest.raises(ValueError, match="cannot be loaded"):
        load_checkpoint(path, target)
    assert states_equal(before, target.state_dict())


# --------------------------------------------------------------------------- logging


def test_log_columns_are_fixed_by_the_first_call(tmp_path: Path) -> None:
    path = tmp_path / "m.csv"
    log_metrics(path, 1, {"a": 1.0, "b": 2.0})
    log_metrics(path, 2, {"b": 4.0, "a": 3.0})  # order does not matter
    log_metrics(path, 3, {"a": 5.0})  # missing column becomes NaN
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["step", "a", "b"]
    assert rows[1][1:] == ["1.0", "2.0"] and rows[2][1:] == ["3.0", "4.0"]
    assert rows[3][1] == "5.0" and math.isnan(float(rows[3][2]))
    with pytest.raises(ValueError, match="columns changed"):
        log_metrics(path, 4, {"a": 1.0, "val_loss": 2.0})
    assert len(read_log(path)) == 3  # the rejected call wrote nothing


def test_log_validation(tmp_path: Path) -> None:
    path = tmp_path / "m.csv"
    for bad_step in (-1, True, 1.5):
        with pytest.raises(ValueError, match="step"):
            log_metrics(path, bad_step, {"a": 1.0})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="empty"):
        log_metrics(path, 0, {})
    with pytest.raises(ValueError, match="infinite"):
        log_metrics(path, 0, {"a": math.inf})
    with pytest.raises(ValueError, match="invalid metric name"):
        log_metrics(path, 0, {"a,b": 1.0})
    with pytest.raises(ValueError, match="invalid metric name"):
        log_metrics(path, 0, {"step": 1.0})
    with pytest.raises(ValueError, match="not a number"):
        log_metrics(path, 0, {"a": "x"})  # type: ignore[dict-item]
    with pytest.raises(FileNotFoundError):
        log_metrics(tmp_path / "missing" / "m.csv", 0, {"a": 1.0})
    assert not path.exists()
    log_metrics(path, 0, {"a": math.nan})
    assert math.isnan(read_log(path)[0]["a"])


def test_truncate_log_drops_rows_after_the_checkpoint_step(tmp_path: Path) -> None:
    path = tmp_path / "m.csv"
    for step in (2, 4, 6, 8):
        log_metrics(path, step, {"a": float(step)})
    with path.open("a") as handle:
        handle.write("9,")  # a row cut off mid-write
    assert truncate_log(path, 4) == 3
    assert [row["step"] for row in read_log(path)] == [2.0, 4.0]
    assert truncate_log(path, 4) == 0
    assert truncate_log(tmp_path / "absent.csv", 4) == 0
    log_metrics(path, 6, {"a": 6.0})  # appending after truncation keeps the header
    assert [row["step"] for row in read_log(path)] == [2.0, 4.0, 6.0]
