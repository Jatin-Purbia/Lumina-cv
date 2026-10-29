"""Training orchestration and reproducible runs, not model computation.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from nta_zerodce.data.files import video_files
from nta_zerodce.data.image_dataset import ImageDataset
from nta_zerodce.data.video_pairs import VideoPairDataset
from nta_zerodce.models.factory import build_model
from nta_zerodce.training.checkpoint import load_checkpoint, save_checkpoint
from nta_zerodce.training.config import Config, save_config
from nta_zerodce.training.logging import log_metrics
from nta_zerodce.training.loop_step import train_step
from nta_zerodce.training.state import provenance, read_checkpoint, validate_disjoint
from nta_zerodce.training.validation import core_path
from nta_zerodce.utils.device import resolve_device
from nta_zerodce.utils.seed import seed_everything


def train(config: Config, resume: Path | None = None) -> Path:
    """Run the configured training schedule.

    Args:
        config: Validated Config with dataset paths and gated objectives.
        resume: Optional checkpoint; exact flag/schema compatibility required.

    Returns:
        Final checkpoint Path; persist resolved config, seed and metrics CSV.

    Method:
        EXPERIMENTS 100 epochs, Adam, fixed seeds; loop_step owns updates.

    Raises:
        ValueError: Invalid inputs or incompatible configuration.
    """
    config.validate()
    seed_everything(config.training.seed)
    device = resolve_device(config.training.device)
    root = core_path(config.paths.train_images)
    real = config.flags.temporal and config.temporal.pair_source == "real"
    split: dict[str, Any]
    if real:
        dataset: Dataset[dict[str, Tensor]] = VideoPairDataset(
            core_path(config.paths.video_dir),
            config.training.crop_size,
            frame_gap=config.temporal.frame_gap,
        )
        split = {
            "training_videos": [str(p) for p in video_files(core_path(config.paths.video_dir))],
            "policy": "Manually session-separated video directory",
        }
    else:
        images = ImageDataset(root, config.training.crop_size)
        # Scene split is fixed independently of the run seed. Flat curated roots use file IDs.
        scenes = sorted(
            {
                p.relative_to(root).parts[0] if len(p.relative_to(root).parts) > 1 else p.name
                for p in images.files
            }
        )
        shuffled = scenes.copy()
        random.Random(17).shuffle(shuffled)
        nval = max(1, round(len(scenes) * 0.1)) if len(scenes) > 1 else 0
        validation = set(shuffled[:nval])
        images.files = [
            p
            for p in images.files
            if (p.relative_to(root).parts[0] if len(p.relative_to(root).parts) > 1 else p.name)
            not in validation
        ]
        dataset = images
        split = {
            "split_seed": 17,
            "train_scenes": sorted(set(scenes) - validation),
            "validation_scenes": sorted(validation),
            "training_images": [str(p.relative_to(root)) for p in images.files],
        }
    # Evaluation images must never be training inputs, including copies under another path.
    validate_disjoint(
        root if not real else core_path(config.paths.video_dir),
        [
            core_path(config.paths.eval_low),
            core_path(config.paths.eval_high),
            core_path(config.paths.unpaired),
        ],
        videos=real,
    )
    run = core_path(config.paths.run_dir)
    checkpoints = core_path(config.paths.checkpoint_dir)
    run.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    log = run / "metrics.csv"
    if resume is None and log.exists():
        raise ValueError(
            f"Run already contains metrics; use --resume or a new paths.run_dir: {run}"
        )
    model = build_model(config).to(device)
    t = config.training
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=t.lr,
        betas=(t.adam_beta1, t.adam_beta2),
        eps=t.adam_eps,
        weight_decay=t.weight_decay,
    )
    start = 0
    if resume is not None:
        load_checkpoint(core_path(resume), model, optimizer, config)
        start = read_checkpoint(core_path(resume))["epoch"] + 1
    save_config(config, run / "config.yaml")
    (run / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    (run / "provenance.json").write_text(json.dumps(provenance(config), indent=2), encoding="utf-8")
    final = checkpoints / "last.pt"
    started = time.perf_counter()
    for epoch in range(start, t.epochs):
        # Epoch-local order/worker seeds make checkpoint continuation independent of loader history.
        generator = torch.Generator().manual_seed(t.seed + epoch)
        loader = DataLoader(
            dataset,
            batch_size=t.batch_size,
            shuffle=True,
            num_workers=t.workers,
            generator=generator,
        )
        for index, batch in enumerate(loader):
            metrics = train_step(model, batch, optimizer, config)
            step = epoch * len(loader) + index
            if step % t.log_every == 0:
                log_metrics(
                    log,
                    step,
                    {
                        **metrics,
                        "epoch": float(epoch),
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                )
        save_checkpoint(final, model, optimizer, config, epoch)
        if (epoch + 1) % t.checkpoint_every == 0:
            save_checkpoint(
                checkpoints / f"epoch_{epoch + 1:04d}.pt", model, optimizer, config, epoch
            )
    if start >= t.epochs:
        return core_path(resume) if resume is not None else final
    return final
