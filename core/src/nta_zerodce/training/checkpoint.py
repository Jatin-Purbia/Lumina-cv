"""Checkpoint format v1: model, optimizer, resolved config, RNG and provenance.

See docs/INTERFACES.md ("CLI and checkpoint contracts"). Checkpoints are trusted local files
loaded on CPU with ``weights_only=True`` and then moved by the caller; no network loading.
Unknown format or schema versions, unexpected keys, and architecture mismatches are rejected
instead of being loaded silently.
"""

from __future__ import annotations

import os
import platform
import random
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import Optimizer

from nta_zerodce.models.nta_zerodce import NTAZeroDCE
from nta_zerodce.training.config import (
    Config,
    config_from_dict,
    config_to_dict,
    workspace_root,
)

FORMAT_VERSION = 1
SCHEMA_VERSION = 1
_REQUIRED_KEYS = frozenset(
    {
        "format_version",
        "schema_version",
        "epoch",
        "config",
        "flags",
        "model",
        "optimizer",
        "rng",
        "provenance",
    }
)
_ARCHITECTURE_FLAGS = ("noise_aware", "denoise", "light")
_ARCHITECTURE_MODEL_FIELDS = ("channels", "iterations", "noise_channels", "denoise_channels")


@dataclass(frozen=True)
class LoadedCheckpoint:
    """Result of a checkpoint load: restored config, completed epoch and provenance."""

    config: Config
    epoch: int
    provenance: dict[str, str]


def _git_commit() -> str:
    """Best-effort current git commit hash for provenance; ``unknown`` when unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace_root(),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "unknown"


def _capture_rng() -> dict[str, Any]:
    """Capture Python, NumPy and torch RNG states in ``weights_only``-loadable types."""
    np_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": (
            str(np_state[0]),
            [int(v) for v in np_state[1]],
            int(np_state[2]),
            int(np_state[3]),
            float(np_state[4]),
        ),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    return state


def _restore_rng(state: Mapping[str, Any]) -> None:
    """Restore RNG states captured by ``_capture_rng``."""
    py = state["python"]
    random.setstate((int(py[0]), tuple(int(v) for v in py[1]), py[2]))
    name, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.array(keys, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state["torch"].cpu())
    cuda_states = state["cuda"]
    if cuda_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in cuda_states])


def save_checkpoint(
    path: Path,
    model: NTAZeroDCE,
    optimizer: Optimizer,
    config: Config,
    epoch: int,
    *,
    provenance: Mapping[str, str] | None = None,
) -> None:
    """Save reproducible training state atomically.

    Args:
        path: Checkpoint destination; the parent directory must exist.
        model: Model whose weights are stored on CPU.
        optimizer: Optimizer whose state enables an exact resume.
        config: Fully resolved validated Config.
        epoch: Completed zero-based epoch index, at least 0.
        provenance: Extra string records (for example split hashes) merged into provenance.

    Returns:
        None. The file holds format_version=1, schema_version, epoch, resolved config and
        flags, model and optimizer state, RNG states and provenance.

    Raises:
        ValueError: Negative epoch or an invalid config.
        FileNotFoundError: The destination directory does not exist.
    """
    if epoch < 0:
        raise ValueError(f"epoch must be >= 0; got {epoch}")
    path = Path(path)
    if not path.parent.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {path.parent}")
    record: dict[str, str] = {
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
    }
    record.update({str(k): str(v) for k, v in (provenance or {}).items()})
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "schema_version": config.schema_version,
        "epoch": int(epoch),
        "config": config_to_dict(config),
        "flags": asdict(config.flags),
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "rng": _capture_rng(),
        "provenance": record,
    }
    handle, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    os.close(handle)
    try:
        torch.save(payload, tmp_name)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def read_checkpoint(path: Path) -> dict[str, Any]:
    """Read and structurally validate a checkpoint payload (CPU, ``weights_only``).

    Args:
        path: Trusted local checkpoint file.

    Returns:
        The payload dict.

    Raises:
        FileNotFoundError: Missing file.
        ValueError: Unreadable file, unknown format/schema version, missing or unexpected keys.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as err:  # corrupted pickle, wrong file type, unsafe globals
        raise ValueError(f"cannot read checkpoint {path}: {err}") from None
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint {path} does not contain a mapping")
    keys = set(payload)
    if keys != _REQUIRED_KEYS:
        missing = sorted(_REQUIRED_KEYS - keys)
        unexpected = sorted(keys - _REQUIRED_KEYS)
        raise ValueError(
            f"checkpoint {path} has an unsupported layout (missing {missing}, "
            f"unexpected {unexpected})"
        )
    if payload["format_version"] != FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format_version {payload['format_version']!r}; "
            f"this build reads version {FORMAT_VERSION}"
        )
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported checkpoint schema_version {payload['schema_version']!r}; "
            f"this build reads version {SCHEMA_VERSION}"
        )
    return dict(payload)


def checkpoint_config(path: Path) -> Config:
    """Return the resolved Config stored in a checkpoint without touching any model."""
    return config_from_dict(read_checkpoint(path)["config"])


def _check_compatible(saved: Config, expected: Config, exact: bool) -> None:
    """Reject an expected config that cannot run the saved weights (or resume exactly)."""
    for name in _ARCHITECTURE_FLAGS:
        if getattr(saved.flags, name) != getattr(expected.flags, name):
            raise ValueError(
                f"checkpoint was trained with flags.{name}={getattr(saved.flags, name)} "
                f"but the requested config has {getattr(expected.flags, name)}"
            )
    for name in _ARCHITECTURE_MODEL_FIELDS:
        if getattr(saved.model, name) != getattr(expected.model, name):
            raise ValueError(
                f"checkpoint was trained with model.{name}={getattr(saved.model, name)} "
                f"but the requested config has {getattr(expected.model, name)}"
            )
    if exact:
        if saved.flags != expected.flags:
            raise ValueError(
                "exact resume requires identical flags; checkpoint "
                f"{asdict(saved.flags)} vs requested {asdict(expected.flags)}"
            )
        if saved.model != expected.model:
            raise ValueError("exact resume requires an identical model section")


def load_checkpoint_full(
    path: Path,
    model: NTAZeroDCE,
    optimizer: Optimizer | None = None,
    expected_config: Config | None = None,
) -> LoadedCheckpoint:
    """Load a checkpoint into ``model`` (and ``optimizer``) and return its metadata.

    Args:
        path: Trusted local checkpoint.
        model: Destination architecture on the caller's device; weights are copied in.
        optimizer: If given, optimizer state and RNG states are restored for an exact resume.
        expected_config: If given, architecture flags and shapes must match; with an optimizer
            all flags and the model section must match exactly.

    Returns:
        LoadedCheckpoint with the restored Config, completed epoch and provenance.

    Raises:
        ValueError: Format/schema/layout problems, flag or shape mismatches, bad state dicts.
        FileNotFoundError: Missing file.
    """
    payload = read_checkpoint(path)
    saved = config_from_dict(payload["config"])
    if asdict(saved.flags) != payload["flags"]:
        raise ValueError("checkpoint flags disagree with its stored config")
    if expected_config is not None:
        _check_compatible(saved, expected_config, exact=optimizer is not None)
    try:
        model.load_state_dict(payload["model"], strict=True)
    except RuntimeError as err:
        raise ValueError(f"checkpoint weights do not match the model architecture: {err}") from None
    if optimizer is not None:
        try:
            optimizer.load_state_dict(payload["optimizer"])
        except (ValueError, KeyError) as err:
            raise ValueError(f"checkpoint optimizer state is incompatible: {err}") from None
        _restore_rng(payload["rng"])
    return LoadedCheckpoint(
        config=saved,
        epoch=int(payload["epoch"]),
        provenance={str(k): str(v) for k, v in payload["provenance"].items()},
    )


def load_checkpoint(
    path: Path,
    model: NTAZeroDCE,
    optimizer: Optimizer | None = None,
    expected_config: Config | None = None,
) -> Config:
    """Load compatible model and optional optimizer state; return the restored Config.

    See ``load_checkpoint_full`` for the arguments and failure modes; this wrapper returns
    only the restored Config as in the INTERFACES contract.
    """
    return load_checkpoint_full(path, model, optimizer, expected_config).config
