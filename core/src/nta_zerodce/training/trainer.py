"""Training orchestration and reproducible runs, not model computation.

``train`` validates the config, seeds everything, builds model, optimizer, data loaders and the
run directory, runs the schedule through ``loop_step`` and writes checkpoints plus CSV logs.

Run layout (``<run_dir>`` and ``<checkpoint_dir>`` come from ``config.paths`` unless the keyword
overrides are given; relative paths resolve against the core workspace root)::

    <run_dir>/<run_name>/config.yaml        resolved config
    <run_dir>/<run_name>/metrics.csv        train losses (step, epoch, total, <losses>, ...)
    <run_dir>/<run_name>/val_metrics.csv    validation losses, one row per epoch
    <checkpoint_dir>/<run_name>/epoch_XXXX.pt   after XXXX completed epochs (checkpoint_every)
    <checkpoint_dir>/<run_name>/final.pt        after the last epoch

The default run name is derived deterministically from the flags and the seed, see
``default_run_name``. A fresh run owns its directories: logs and ``epoch_*.pt``/``final.pt``
from an earlier run of the same name are removed. A resumed run appends to the logs after
dropping rows past the checkpoint.

Reproducibility: the data order, crops, noise and synthetic pairs of epoch ``e`` depend only on
``(training.seed, e)``, so resuming from the checkpoint of epoch ``e - 1`` continues exactly
(with ``workers=0`` and on the same device and software versions). The train/validation scene
split always uses ``val_fraction=0.1`` and split seed 17, independent of the training seed.
"""

from __future__ import annotations

import hashlib
import itertools
import logging
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader

from nta_zerodce.data.image_dataset import ImageDataset, scene_split
from nta_zerodce.data.video_pairs import VideoPairDataset
from nta_zerodce.models.factory import build_model
from nta_zerodce.models.nta_zerodce import NTAZeroDCE
from nta_zerodce.training.checkpoint import load_checkpoint_full, save_checkpoint
from nta_zerodce.training.config import (
    Config,
    check_paths,
    resolve_path,
    save_config,
)
from nta_zerodce.training.logging import log_metrics, truncate_log
from nta_zerodce.training.loop_step import prepare_batch, train_step, validation_step
from nta_zerodce.utils.device import resolve_device
from nta_zerodce.utils.seed import make_generator, seed_everything, worker_init_fn

logger = logging.getLogger(__name__)

SPLIT_VAL_FRACTION = 0.1
SPLIT_SEED = 17
"""Scene split constants fixed across training seeds (docs/DATA.md, docs/EXPERIMENTS.md)."""

_STREAM_SHUFFLE = 0
_STREAM_BATCH = 1
_STREAM_VIDEO_BASE = 100
_STREAM_VAL = 7919


def default_run_name(config: Config) -> str:
    """Deterministic run name from the flags and the seed.

    Args:
        config: Validated Config.

    Returns:
        ``<flags>_s<seed>`` where ``<flags>`` is ``baseline`` when no flag is set, ``full`` when
        noise, denoise, edge, chroma, adaptive and temporal are all set, otherwise the enabled
        flags joined by ``-`` in the order noise, denoise, edge, chroma, adaptive, temporal;
        ``-light`` is appended when ``flags.light`` is set. Examples: ``baseline_s17``,
        ``noise-denoise_s29``, ``full-light_s43``.
    """
    flags = config.flags
    order = [
        ("noise", flags.noise_aware),
        ("denoise", flags.denoise),
        ("edge", flags.edge),
        ("chroma", flags.chroma),
        ("adaptive", flags.adaptive_exp),
        ("temporal", flags.temporal),
    ]
    enabled = [name for name, on in order if on]
    if len(enabled) == len(order):
        base = "full"
    elif enabled:
        base = "-".join(enabled)
    else:
        base = "baseline"
    if flags.light:
        base = f"{base}-light"
    return f"{base}_s{config.training.seed}"


def split_hash(paths: Sequence[Path], root: Path) -> str:
    """SHA-256 of the sorted POSIX paths relative to ``root`` (portable split fingerprint).

    Args:
        paths: Image paths of one split.
        root: Dataset root they live under; paths outside it are hashed as given.

    Returns:
        Hex digest.
    """
    names = []
    for path in paths:
        try:
            names.append(Path(path).relative_to(root).as_posix())
        except ValueError:
            names.append(Path(path).as_posix())
    digest = hashlib.sha256()
    digest.update("\n".join(sorted(names)).encode("utf-8"))
    return digest.hexdigest()


def _derive_seed(seed: int, epoch: int, stream: int) -> int:
    """Mix ``(seed, epoch, stream)`` into a 32-bit seed independent of neighbouring tuples."""
    state = np.random.SeedSequence([int(seed), int(epoch), int(stream)]).generate_state(1)
    return int(state[0])


@dataclass
class _Data:
    """Datasets and loaders of one run."""

    train_dataset: ImageDataset
    train_loader: DataLoader[dict[str, Tensor]]
    val_loader: DataLoader[dict[str, Tensor]] | None
    video_dataset: VideoPairDataset | None
    video_loader: DataLoader[dict[str, Tensor]] | None
    provenance: dict[str, str]


def _build_data(config: Config, device: torch.device) -> _Data:
    """Index the datasets (scene split) and build the loaders; fails fast on missing data."""
    flags = config.flags
    training = config.training
    required = ["train_images"]
    if flags.temporal:
        required.append("video_dir")  # enforced only when temporal.pair_source == "real"
    check_paths(config, required)
    root = resolve_path(config.paths.train_images)
    train_paths, val_paths = scene_split(root, SPLIT_VAL_FRACTION, SPLIT_SEED)
    train_dataset = ImageDataset(root, training.crop_size, paths=train_paths, seed=training.seed)
    pin = device.type == "cuda"
    train_loader: DataLoader[dict[str, Tensor]] = DataLoader(
        train_dataset,
        batch_size=training.batch_size,
        shuffle=True,
        drop_last=len(train_dataset) >= training.batch_size,
        num_workers=training.workers,
        worker_init_fn=worker_init_fn,
        generator=make_generator(training.seed),
        pin_memory=pin,
    )
    val_loader: DataLoader[dict[str, Tensor]] | None = None
    if val_paths:
        try:
            val_dataset = ImageDataset(
                root, training.crop_size, paths=val_paths, seed=training.seed
            )
        except ValueError as err:
            logger.warning("validation disabled: %s", err)
        else:
            val_loader = DataLoader(
                val_dataset, batch_size=training.batch_size, shuffle=False, num_workers=0
            )
    provenance = {
        "train_split_sha256": split_hash(train_dataset.paths, root),
        "val_split_sha256": split_hash(val_paths, root),
        "train_images": str(len(train_dataset)),
        "val_images": str(len(val_paths)),
        "split_seed": str(SPLIT_SEED),
        "split_val_fraction": str(SPLIT_VAL_FRACTION),
    }
    video_dataset: VideoPairDataset | None = None
    video_loader: DataLoader[dict[str, Tensor]] | None = None
    if flags.temporal and config.temporal.pair_source == "real":
        video_dataset = VideoPairDataset(
            resolve_path(config.paths.video_dir),
            training.crop_size,
            frame_gap=config.temporal.frame_gap,
            seed=training.seed,
        )
        video_loader = DataLoader(
            video_dataset,
            batch_size=training.batch_size,
            shuffle=True,
            drop_last=len(video_dataset) >= training.batch_size,
            num_workers=training.workers,
            worker_init_fn=worker_init_fn,
            generator=make_generator(training.seed),
            pin_memory=pin,
        )
        provenance["video_pairs"] = str(len(video_dataset))
    return _Data(train_dataset, train_loader, val_loader, video_dataset, video_loader, provenance)


def _video_stream(
    loader: DataLoader[dict[str, Tensor]], dataset: VideoPairDataset, seed: int, epoch: int
) -> Iterator[dict[str, Tensor]]:
    """Endless, restartable stream of real frame pairs for one epoch (never caches batches)."""
    for restart in itertools.count():
        dataset.set_epoch(epoch * 1024 + restart)
        generator = loader.generator
        assert generator is not None
        generator.manual_seed(_derive_seed(seed, epoch, _STREAM_VIDEO_BASE + restart))
        yield from loader


def _prepare_run_dirs(
    run_path: Path, ckpt_path: Path, fresh: bool, completed_steps: int
) -> tuple[Path, Path]:
    """Create run/checkpoint directories; reset logs on a fresh run, truncate them on resume."""
    run_path.mkdir(parents=True, exist_ok=True)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    train_log = run_path / "metrics.csv"
    val_log = run_path / "val_metrics.csv"
    if fresh:
        for stale in (train_log, val_log, ckpt_path / "final.pt"):
            stale.unlink(missing_ok=True)
        for pattern in ("epoch_*.pt", "*.tmp"):
            for stale in ckpt_path.glob(pattern):
                stale.unlink(missing_ok=True)
    else:
        truncate_log(train_log, completed_steps)
        truncate_log(val_log, completed_steps)
    return train_log, val_log


def _mean_of(sums: dict[str, float], count: float) -> dict[str, float]:
    """Divide accumulated sums by ``count``."""
    return {name: value / count for name, value in sums.items()}


def _validate(
    model: NTAZeroDCE,
    loader: DataLoader[dict[str, Tensor]],
    config: Config,
    device: torch.device,
) -> dict[str, float]:
    """Mean validation losses over the held-out scenes with fixed crops and noise."""
    val_config = config
    if config.flags.temporal and config.temporal.pair_source == "real":
        # Real pairs come from the train videos only; validate the still-image objective.
        val_config = replace(config, flags=replace(config.flags, temporal=False))
    generator = make_generator(_derive_seed(config.training.seed, 0, _STREAM_VAL), device)
    sums: dict[str, float] = {}
    count = 0
    for raw in loader:
        batch = prepare_batch(raw, val_config, generator, device)
        size = int(batch["image"].shape[0])
        result = validation_step(model, batch, val_config)
        for name, value in result.items():
            sums[name] = sums.get(name, 0.0) + value * size
        count += size
    return _mean_of(sums, count)


def train(
    config: Config,
    resume: Path | None = None,
    *,
    run_name: str | None = None,
    run_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
) -> Path:
    """Run the configured training schedule.

    Args:
        config: Validated Config with dataset paths and gated objectives.
        resume: Optional checkpoint; exact flag/model compatibility is required. Training
            continues at the epoch after the checkpoint's, with optimizer and RNG state
            restored; the epoch count comes from ``config.training.epochs``.
        run_name: Run directory name; default ``default_run_name(config)``.
        run_dir: Overrides ``config.paths.run_dir`` (the run directory is ``run_dir/run_name``).
        checkpoint_dir: Overrides ``config.paths.checkpoint_dir``.

    Returns:
        Path of ``final.pt``. The resolved config, ``metrics.csv`` and ``val_metrics.csv`` are
        written to the run directory.

    Method:
        EXPERIMENTS fixed schedule (Adam, grad clip, no scheduler); ``loop_step`` owns updates.
        Train metrics are the mean over the steps since the previous log row, logged every
        ``log_every`` steps and at the end of each epoch.

    Raises:
        ValueError: Invalid config, incompatible or unreadable checkpoint, or a changed
            train/validation split on resume.
        FileNotFoundError: Missing data roots (listed together) or checkpoint.
        FloatingPointError: A non-finite loss; nothing is stepped for that batch.
    """
    config.validate()
    training = config.training
    device = resolve_device(training.device)
    seed_everything(training.seed, deterministic=device.type == "cpu")
    name = run_name if run_name is not None else default_run_name(config)
    run_path = resolve_path(run_dir if run_dir is not None else config.paths.run_dir) / name
    ckpt_path = (
        resolve_path(checkpoint_dir if checkpoint_dir is not None else config.paths.checkpoint_dir)
        / name
    )

    data = _build_data(config, device)
    steps_per_epoch = len(data.train_loader)
    model = build_model(config).to(device)
    optimizer = Adam(
        model.parameters(),
        lr=training.lr,
        betas=(training.adam_beta1, training.adam_beta2),
        eps=training.adam_eps,
        weight_decay=training.weight_decay,
    )

    start_epoch = 0
    elapsed_before = 0.0
    if resume is not None:
        loaded = load_checkpoint_full(Path(resume), model, optimizer, expected_config=config)
        for key in ("train_split_sha256", "val_split_sha256"):
            before = loaded.provenance.get(key)
            if before is not None and before != data.provenance[key]:
                raise ValueError(
                    f"cannot resume: {key} changed (checkpoint {before[:12]}, now "
                    f"{data.provenance[key][:12]}); the training images differ"
                )
        start_epoch = loaded.epoch + 1
        try:
            elapsed_before = float(loaded.provenance.get("elapsed_seconds", "0"))
        except ValueError:
            elapsed_before = 0.0
    train_log, val_log = _prepare_run_dirs(
        run_path, ckpt_path, fresh=resume is None, completed_steps=start_epoch * steps_per_epoch
    )
    save_config(config, run_path / "config.yaml")
    logger.info(
        "run %s: %d train images, %d steps/epoch, epochs %d..%d on %s",
        name,
        len(data.train_dataset),
        steps_per_epoch,
        start_epoch,
        training.epochs - 1,
        device,
    )

    started = time.monotonic()
    global_step = start_epoch * steps_per_epoch
    final_path = ckpt_path / "final.pt"

    def provenance(step: int) -> dict[str, str]:
        record = dict(data.provenance)
        record.update(
            {
                "run_name": name,
                "device": str(device),
                "global_step": str(step),
                "steps_per_epoch": str(steps_per_epoch),
                "elapsed_seconds": f"{elapsed_before + time.monotonic() - started:.3f}",
            }
        )
        return record

    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, training.epochs):
        model.train()
        data.train_dataset.set_epoch(epoch)
        shuffle_generator = data.train_loader.generator
        assert shuffle_generator is not None
        shuffle_generator.manual_seed(_derive_seed(training.seed, epoch, _STREAM_SHUFFLE))
        batch_generator = make_generator(_derive_seed(training.seed, epoch, _STREAM_BATCH), device)
        video = (
            _video_stream(data.video_loader, data.video_dataset, training.seed, epoch)
            if data.video_loader is not None and data.video_dataset is not None
            else None
        )
        window: dict[str, float] = {}
        window_count = 0
        for index, raw in enumerate(data.train_loader):
            if video is not None:
                raw = {**raw, **{k: v for k, v in next(video).items() if k != "index"}}
            batch = prepare_batch(raw, config, batch_generator, device)
            result = train_step(model, batch, optimizer, config)
            global_step += 1
            for key, value in result.items():
                window[key] = window.get(key, 0.0) + value
            window_count += 1
            if global_step % training.log_every == 0 or index == steps_per_epoch - 1:
                row = {"epoch": float(epoch), **_mean_of(window, window_count)}
                log_metrics(train_log, global_step, row)
                window, window_count = {}, 0
        if data.val_loader is not None:
            val = _validate(model, data.val_loader, config, device)
            model.train()
            log_metrics(val_log, global_step, {"epoch": float(epoch), **val})
            logger.info(
                "epoch %d: train total %s val total %.5g", epoch, row["total"], val["total"]
            )
        else:
            logger.info("epoch %d: train total %.5g", epoch, row["total"])
        last_epoch = epoch
        if (epoch + 1) % training.checkpoint_every == 0:
            save_checkpoint(
                ckpt_path / f"epoch_{epoch + 1:04d}.pt",
                model,
                optimizer,
                config,
                epoch,
                provenance=provenance(global_step),
            )

    if last_epoch < 0:
        raise ValueError(
            f"nothing to train: epochs={training.epochs} but the run starts at epoch {start_epoch}"
        )
    save_checkpoint(
        final_path, model, optimizer, config, last_epoch, provenance=provenance(global_step)
    )
    return final_path
