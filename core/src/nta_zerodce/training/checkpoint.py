"""Checkpoint format v1: model, optimizer, resolved config, RNG and provenance.

See docs/INTERFACES.md ("CLI and checkpoint contracts"). Checkpoints are trusted local files
loaded on CPU with ``weights_only=True`` and then moved by the caller; no network loading.
Unknown format or schema versions, unexpected keys, and architecture mismatches are rejected
instead of being loaded silently.
"""

from __future__ import annotations

import copy
import functools
import os
import platform
import random
import subprocess
import uuid
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


@functools.lru_cache(maxsize=1)
def _git_commit() -> str:
    """Best-effort current git commit hash for provenance; ``unknown`` when unavailable.

    Cached per process: every checkpoint of a run records the commit the run started from.
    """
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
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError(f"epoch must be an int >= 0; got {epoch!r}")
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
    # Write next to the destination under a unique name (default file permissions, unlike
    # mkstemp's 0600), flush to disk, then rename: readers see the old file or the complete new
    # one, never a partial file, and a failure removes the temporary file.
    tmp_path = path.parent / f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        with tmp_path.open("xb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
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
    epoch = payload["epoch"]
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError(f"checkpoint {path} has an invalid epoch {epoch!r}")
    for name in ("config", "flags", "model", "optimizer", "rng", "provenance"):
        if not isinstance(payload[name], dict):
            raise ValueError(
                f"checkpoint {path} field {name!r} must be a mapping; "
                f"got {type(payload[name]).__name__}"
            )
    if not all(isinstance(value, torch.Tensor) for value in payload["model"].values()):
        raise ValueError(f"checkpoint {path} model state holds non-tensor entries")
    for name in ("python", "numpy", "torch", "cuda"):
        if name not in payload["rng"]:
            raise ValueError(f"checkpoint {path} rng state lacks {name!r}")
    return dict(payload)


def _stored_config(payload: Mapping[str, Any], path: Path) -> Config:
    """Rebuild the stored Config, turning any failure into a ValueError naming the file."""
    try:
        return config_from_dict(payload["config"])
    except ValueError as err:
        raise ValueError(f"checkpoint {path} holds an invalid config: {err}") from None
    except (TypeError, KeyError, AttributeError) as err:
        raise ValueError(f"checkpoint {path} holds a malformed config: {err!r}") from None


def checkpoint_config(path: Path) -> Config:
    """Return the resolved Config stored in a checkpoint without touching any model."""
    return _stored_config(read_checkpoint(path), Path(path))


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


def _apply_state(
    payload: Mapping[str, Any], model: NTAZeroDCE, optimizer: Optimizer | None, path: Path
) -> None:
    """Copy model (and optimizer + RNG) state in; on any failure restore everything.

    A rejected checkpoint must not leave a half-loaded model, so the previous model weights,
    optimizer state and RNG states are put back before the ValueError propagates.
    """
    model_backup = {key: value.detach().clone() for key, value in model.state_dict().items()}
    optimizer_backup = copy.deepcopy(optimizer.state_dict()) if optimizer is not None else None
    rng_backup = _capture_rng() if optimizer is not None else None
    try:
        model.load_state_dict(payload["model"], strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(payload["optimizer"])
            _restore_rng(payload["rng"])
    except Exception as err:
        model.load_state_dict(model_backup, strict=True)
        if optimizer is not None and optimizer_backup is not None and rng_backup is not None:
            optimizer.load_state_dict(optimizer_backup)
            _restore_rng(rng_backup)
        raise ValueError(f"checkpoint {path} cannot be loaded here: {err}") from None


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
    saved = _stored_config(payload, Path(path))
    if asdict(saved.flags) != payload["flags"]:
        raise ValueError("checkpoint flags disagree with its stored config")
    if expected_config is not None:
        _check_compatible(saved, expected_config, exact=optimizer is not None)
    _apply_state(payload, model, optimizer, Path(path))
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
