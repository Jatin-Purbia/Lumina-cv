"""Checkpoint schema and feature-flag round trips; no silent architecture mismatch.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from pathlib import Path

from torch.optim import Optimizer

from nta_zerodce.models.nta_zerodce import NTAZeroDCE
from nta_zerodce.training.config import Config


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
        None; future file contains model/optimizer states, config, flags, schema_version, epoch and RNG states.

    Method:
        TESTING flag/config round-trip; atomic write and explicit checkpoint format version 1.

    Raises:
        NotImplementedError: Scaffold only; implement this contract before use.
    """
    raise NotImplementedError("TODO: Save reproducible training state")


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
        NotImplementedError: Scaffold only; implement this contract before use.
    """
    raise NotImplementedError("TODO: Load compatible model and optional optimizer state")
