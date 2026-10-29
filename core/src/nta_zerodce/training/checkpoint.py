"""Checkpoint schema and feature-flag round trips; no silent architecture mismatch.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch.optim import Optimizer

from nta_zerodce.models.nta_zerodce import NTAZeroDCE
from nta_zerodce.training.config import Config
from nta_zerodce.training.state import capture_rng, provenance, read_checkpoint, restore_rng
from nta_zerodce.training.validation import config_from_dict


def save_checkpoint(
    path: Path, model: NTAZeroDCE, optimizer: Optimizer, config: Config, epoch: int
) -> None:
    """Save reproducible training state.

    Args:
        path: Checkpoint destination.
        model/optimizer: Current state owners.
        config: Fully resolved validated Config.
        epoch: Completed zero-based epoch index >=0.

    Returns:
        None; file contains model/optimizer states, config, flags, schema_version, epoch and RNG states.

    Method:
        TESTING flag/config round-trip; atomic write and explicit checkpoint format version 1.

    Raises:
        ValueError: Invalid inputs or incompatible configuration.
    """
    config.validate()
    if type(epoch) is not int or epoch < 0:
        raise ValueError("Expected completed zero-based epoch")
    if config.flags != model.config.flags or config.model != model.config.model:
        raise ValueError("Checkpoint config must describe the model")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "schema_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "config": asdict(config),
        "flags": asdict(config.flags),
        "rng": capture_rng(),
        "provenance": provenance(config),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: Path,
    model: NTAZeroDCE,
    optimizer: Optimizer | None = None,
    expected_config: Config | None = None,
) -> Config:
    """Load compatible model and optional optimizer state.

    Args:
        path: Trusted local checkpoint; no network loading.
        model: Destination architecture on explicit caller device.
        optimizer: Optional resume optimizer.
        expected_config: If set, reject flag/shape/schema mismatches.

    Returns:
        Restored resolved Config; require exact checkpoint architecture flags.

    Method:
        INTERFACES checkpoints; portable CPU map_location then explicit caller transfer.

    Raises:
        ValueError: Invalid inputs or incompatible configuration.
    """
    payload = read_checkpoint(path)
    config = config_from_dict(payload["config"])
    expected = expected_config or model.config
    if config.flags != expected.flags or config.flags != model.config.flags:
        raise ValueError("Checkpoint feature flags do not match")
    # curve_scale is an inference setting; parameter shape and behavior constants must match.
    a, b = asdict(config.model), asdict(expected.model)
    a.pop("curve_scale")
    b.pop("curve_scale")
    if a != b:
        raise ValueError("Checkpoint model configuration mismatch")
    if optimizer is not None:
        if config != expected:
            raise ValueError("Exact resume requires identical resolved configuration")
        if payload["optimizer"] is None:
            raise ValueError("Checkpoint lacks optimizer state")
        if payload["provenance"]["split_hashes"] != provenance(expected)["split_hashes"]:
            raise ValueError("Dataset content changed since checkpoint")
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
        restore_rng(payload["rng"])
    return config
