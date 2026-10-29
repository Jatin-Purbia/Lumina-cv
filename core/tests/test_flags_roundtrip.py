"""Executable documented contracts."""

from __future__ import annotations

import argparse
import random
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from nta_zerodce.cli import cli_config
from nta_zerodce.losses.registry import build_losses
from nta_zerodce.models.factory import build_model
from nta_zerodce.training.checkpoint import load_checkpoint, save_checkpoint
from nta_zerodce.training.config import load_config, save_config
from nta_zerodce.training.loop_step import train_step
from nta_zerodce.training.state import read_checkpoint
from nta_zerodce.training.validation import config_from_dict
from tests.support import small_config


def test_config_flags_roundtrip(tmp_path: Path) -> None:
    """YAML preserves all fields and explicit CLI false overrides YAML true."""
    config = small_config(tmp_path, True)
    path = tmp_path / "config.yaml"
    save_config(config, path)
    assert load_config(path) == config
    args = argparse.Namespace(config=str(path), edge=False)
    result = cli_config(args)
    assert not result.flags.edge and result.flags.noise_aware
    active = build_losses(asdict(result.flags), asdict(result.weights))
    assert "exp" not in active and active["adaptive_exp"].default_weight == config.weights.exp
    assert "edge" not in active
    with pytest.raises(ValueError):
        load_config(path, {"model.typo": 1})
    with pytest.raises(ValueError):
        config_from_dict({"typo": {}})


def test_checkpoint_rng_and_rejections(tmp_path: Path) -> None:
    """Strict state/flag roundtrip, optimizer and three CPU random streams."""
    config = small_config(tmp_path)
    model = build_model(config)
    optimizer = torch.optim.Adam(model.parameters())
    image = torch.rand(2, 3, 32, 32) * 0.2
    train_step(model, {"image": image}, optimizer, config)
    path = tmp_path / "state.pt"
    save_checkpoint(path, model, optimizer, config, 0)
    expected = model(image)["enhanced"].detach()
    rng = (random.random(), np.random.rand(), torch.rand(2))
    restored = build_model(config)
    restored_optimizer = torch.optim.Adam(restored.parameters())
    assert load_checkpoint(path, restored, restored_optimizer, config) == config
    assert random.random() == rng[0] and np.random.rand() == rng[1]
    assert torch.equal(torch.rand(2), rng[2])
    assert torch.equal(restored(image)["enhanced"], expected)
    changed = replace(config, flags=replace(config.flags, edge=True))
    with pytest.raises(ValueError, match="flags"):
        load_checkpoint(path, build_model(changed), expected_config=changed)
    payload = read_checkpoint(path)
    payload["unexpected"] = 1
    torch.save(payload, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="format"):
        load_checkpoint(tmp_path / "bad.pt", restored)
