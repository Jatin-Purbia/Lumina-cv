"""Shared builders for the training, checkpoint and script tests (tiny CPU configs and data)."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path

import torch
from torch.optim import Adam

from nta_zerodce.data.image_dataset import make_synthetic_images
from nta_zerodce.inference.io import write_video
from nta_zerodce.models.factory import build_model
from nta_zerodce.models.nta_zerodce import NTAZeroDCE
from nta_zerodce.training.config import Config, load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
FULL_FLAGS: dict[str, bool] = {
    "noise_aware": True,
    "denoise": True,
    "edge": True,
    "chroma": True,
    "adaptive_exp": True,
    "temporal": True,
}


def tiny_config(
    root: Path,
    flags: Mapping[str, bool] | None = None,
    *,
    epochs: int = 1,
    images: int = 8,
    size: int = 64,
    seed: int = 17,
    batch_size: int = 2,
    make_images: bool = True,
    **training: object,
) -> Config:
    """Baseline config shrunk for CPU tests: 32 px crops, batch 2, workers 0, local paths.

    Images are written to ``root/images`` (when ``make_images``); evaluation directories point at
    ``root/eval_low`` etc., which do not exist until a test creates them.
    """
    base = load_config(CONFIGS / "baseline.py")
    settings: dict[str, object] = {
        "crop_size": 32,
        "batch_size": batch_size,
        "epochs": epochs,
        "workers": 0,
        "seed": seed,
        "device": "cpu",
        "checkpoint_every": 1,
        "log_every": 1,
    }
    settings.update(training)
    if make_images:
        make_synthetic_images(root / "images", images, size)
    return dataclasses.replace(
        base,
        flags=dataclasses.replace(base.flags, **dict(flags or {})),
        training=dataclasses.replace(base.training, **settings),  # type: ignore[arg-type]
        paths=dataclasses.replace(
            base.paths,
            train_images=str(root / "images"),
            eval_low=str(root / "eval_low"),
            eval_high=str(root / "eval_high"),
            unpaired=str(root / "unpaired"),
            video_dir=str(root / "videos" / "train"),
            run_dir=str(root / "runs"),
            checkpoint_dir=str(root / "checkpoints"),
        ),
    )


def make_model_and_optimizer(config: Config, seed: int = 0) -> tuple[NTAZeroDCE, Adam]:
    """Freshly initialised model and Adam for ``config`` (seeded initialisation)."""
    torch.manual_seed(seed)
    model = build_model(config)
    return model, Adam(model.parameters(), lr=config.training.lr)


def write_test_video(path: Path, frames: int = 6, size: int = 64, seed: int = 0) -> None:
    """Write a tiny textured clip whose content shifts one pixel per frame."""
    generator = torch.Generator().manual_seed(seed)
    texture = torch.rand(1, 3, size, size + frames, generator=generator) * 0.5 + 0.2
    path.parent.mkdir(parents=True, exist_ok=True)
    write_video((texture[..., i : i + size] for i in range(frames)), path, fps=10.0)


def states_equal(first: Mapping[str, torch.Tensor], second: Mapping[str, torch.Tensor]) -> bool:
    """True when two state dicts have the same keys and bit-identical tensors."""
    return first.keys() == second.keys() and all(
        torch.equal(first[key], second[key]) for key in first
    )
