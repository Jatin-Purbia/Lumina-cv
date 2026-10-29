"""Trusted checkpoint IO, RNG restoration and dataset provenance."""

from __future__ import annotations

import hashlib
import platform
import random
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nta_zerodce.data.files import image_files, video_files
from nta_zerodce.training.config import Config
from nta_zerodce.training.validation import CORE_ROOT, core_path


def capture_rng() -> dict[str, Any]:
    """Capture CPU and all available CUDA random streams."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict[str, Any]) -> None:
    """Restore streams for exact training continuation."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def file_hash(path: Path) -> str:
    """Hash content in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance(config: Config) -> dict[str, Any]:
    """Record environment, commit and content-aware manifest hashes."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=CORE_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    hashes: dict[str, str | None] = {}
    for name, value in asdict(config.paths).items():
        if name in ("run_dir", "checkpoint_dir"):
            continue
        root = core_path(value)
        if not root.exists():
            hashes[name] = None
            continue
        files = video_files(root) if name == "video_dir" else image_files(root)
        manifest = "\n".join(
            f"{p.relative_to(root) if root.is_dir() else p.name}:{file_hash(p)}" for p in files
        )
        hashes[name] = hashlib.sha256(manifest.encode()).hexdigest()
    return {
        "git_commit": commit,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "split_hashes": hashes,
    }


def read_checkpoint(path: Path) -> dict[str, Any]:
    """Read only trusted local files; reject unknown format and payload keys."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "format_version",
        "schema_version",
        "model",
        "optimizer",
        "epoch",
        "config",
        "flags",
        "rng",
        "provenance",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != expected
        or payload["format_version"] != 1
        or payload["schema_version"] != 1
    ):
        raise ValueError("Unsupported checkpoint format/schema/keys")
    if (
        type(payload["epoch"]) is not int
        or payload["epoch"] < 0
        or payload["flags"] != payload["config"].get("flags")
    ):
        raise ValueError("Invalid checkpoint epoch/flags")
    return payload


def validate_disjoint(training: Path, evaluation: list[Path], videos: bool = False) -> None:
    """Reject path and content overlap with any existing held-out image roots."""
    train_files = video_files(training) if videos else image_files(training)
    hashes = {file_hash(p) for p in train_files}
    for root in evaluation:
        if root.exists():
            for path in image_files(root):
                if file_hash(path) in hashes:
                    raise ValueError(f"Train/evaluation data overlap: {path}")
