"""Small generated-data configuration shared by tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from nta_zerodce.training.config import Config, load_config


def small_config(tmp_path: Path, full: bool = False) -> Config:
    """Small CPU model retaining the documented math and feature semantics."""
    config = load_config(Path("configs/full.yaml" if full else "configs/baseline.yaml"))
    return replace(
        config,
        model=replace(config.model, channels=4, noise_channels=4, denoise_channels=4),
        training=replace(
            config.training,
            crop_size=32,
            batch_size=2,
            epochs=1,
            workers=0,
            device="cpu",
            log_every=1,
        ),
        paths=replace(
            config.paths,
            train_images=str(tmp_path / "train"),
            eval_low=str(tmp_path / "low"),
            eval_high=str(tmp_path / "high"),
            unpaired=str(tmp_path / "unpaired"),
            video_dir=str(tmp_path / "videos"),
            run_dir=str(tmp_path / "run"),
            checkpoint_dir=str(tmp_path / "checkpoints"),
        ),
    )
