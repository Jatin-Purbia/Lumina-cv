"""Executable documented contracts."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.optim import Optimizer

import nta_zerodce.training.trainer as trainer_module
from nta_zerodce.cli import dispatch
from nta_zerodce.data.image_dataset import ImageDataset
from nta_zerodce.data.video_pairs import VideoPairDataset
from nta_zerodce.inference.io import read_image, read_video, write_image, write_video
from nta_zerodce.metrics.colour import angular_error
from nta_zerodce.metrics.fidelity import psnr, ssim
from nta_zerodce.metrics.noise import noise_estimate
from nta_zerodce.metrics.temporal import warping_error
from nta_zerodce.models.factory import build_model
from nta_zerodce.models.nta_zerodce import NTAZeroDCE
from nta_zerodce.training.checkpoint import load_checkpoint, save_checkpoint
from nta_zerodce.training.config import Config, save_config
from nta_zerodce.training.evaluation import evaluate
from nta_zerodce.training.logging import log_metrics
from nta_zerodce.training.loop_step import train_step
from nta_zerodce.training.trainer import train
from tests.support import small_config


def test_training_overfit_and_full_update(tmp_path: Path) -> None:
    """Synthetic CPU baseline learns brighter output; full objective updates all heads."""
    config = small_config(tmp_path)
    model = build_model(config)
    image = torch.rand(2, 3, 32, 32) * 0.1 + 0.05
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    before = (model(image)["enhanced"].mean() - 0.6).square().item()
    for _ in range(30):
        logs = train_step(model, {"image": image}, optimizer, config)
        assert np.isfinite(logs["total"])
    after = (model(image)["enhanced"].mean() - 0.6).square().item()
    assert after < before * 0.8
    full = small_config(tmp_path, True)
    model = build_model(full)
    logs = train_step(model, {"image": image}, torch.optim.Adam(model.parameters()), full)
    assert all(np.isfinite(value) for value in logs.values())
    assert logs["exp"] == 0 and logs["mask_coverage"] > 0


def test_local_workflow_and_metrics(tmp_path: Path) -> None:
    """Train/checkpoint/CLI-enhance/paired-evaluate work with generated local images."""
    config = small_config(tmp_path)
    for i in range(3):
        write_image(
            torch.rand(1, 3, 40, 40) * 0.2,
            Path(config.paths.train_images) / f"scene{i}" / "low.png",
        )
    image = torch.rand(1, 3, 32, 40) * 0.2
    write_image(image, Path(config.paths.eval_low) / "a.png")
    write_image(image * 2, Path(config.paths.eval_high) / "a.png")
    checkpoint = train(config)
    assert checkpoint.exists()
    model = build_model(config)
    load_checkpoint(checkpoint, model, expected_config=config)
    config_path = tmp_path / "config.yaml"
    save_config(config, config_path)
    dispatch(
        "enhance",
        argparse.Namespace(
            config=str(config_path),
            checkpoint=str(checkpoint),
            input=config.paths.eval_low,
            output=str(tmp_path / "enhanced"),
            webcam=None,
        ),
    )
    assert read_image(tmp_path / "enhanced/a.png").shape == image.shape
    destination = tmp_path / "evaluation.csv"
    evaluate(model, config, destination)
    with destination.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {"psnr", "ssim", "angular_error", "noise_proxy", "clipping"} <= {
        r["metric"] for r in rows
    }
    assert all(r["sample_id"] == "a.png" for r in rows)
    assert torch.isinf(psnr(image, image)).all()
    assert torch.allclose(ssim(image, image), torch.ones(1))
    assert angular_error(image, image).max() < 0.05
    assert torch.isnan(noise_estimate(image, torch.zeros_like(image[:, :1]))).all()
    assert torch.isnan(
        warping_error(image, image, torch.zeros(1, 2, 32, 40), torch.zeros_like(image[:, :1]))
    ).all()
    # Pair validation and stable CSV schemas fail loudly.
    write_image(image, Path(config.paths.eval_high) / "extra.png")
    with pytest.raises(ValueError, match="manifests"):
        ImageDataset(Path(config.paths.eval_low), paired_root=Path(config.paths.eval_high))
    log_metrics(tmp_path / "log.csv", 0, {"loss": 1.0})
    with pytest.raises(ValueError, match="schema"):
        log_metrics(tmp_path / "log.csv", 1, {"other": 1.0})


def test_video_io(tmp_path: Path) -> None:
    """Encoding and decoding preserve frame count, dimensions and RGB ordering."""
    frame = torch.zeros(1, 3, 32, 48)
    frame[:, 0] = 0.8
    path = tmp_path / "clip.mp4"
    write_video(iter([frame, frame, frame]), path, 25.0)
    decoded = list(read_video(path))
    assert len(decoded) == 3 and decoded[0].shape == frame.shape
    assert decoded[0][:, 0].mean() > 0.7 and decoded[0][:, 2].mean() < 0.1
    dataset = VideoPairDataset(tmp_path, crop_size=32, frame_gap=2)
    assert len(dataset) == 1
    assert dataset[0]["image1"].shape == dataset[0]["image2"].shape == (3, 32, 32)


def test_resume_matches_uninterrupted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Epoch-boundary optimizer/RNG continuation produces identical CPU parameters."""
    config = small_config(tmp_path)
    config = replace(config, training=replace(config.training, epochs=2))
    for index in range(3):
        write_image(
            torch.rand(1, 3, 40, 40) * 0.2,
            Path(config.paths.train_images) / f"scene{index}" / "low.png",
        )
    complete = train(config)
    expected = torch.load(complete, weights_only=False, map_location="cpu")["model"]
    resumed = replace(
        config,
        paths=replace(
            config.paths,
            run_dir=str(tmp_path / "interrupted"),
            checkpoint_dir=str(tmp_path / "resumed-checkpoints"),
        ),
    )
    original = save_checkpoint

    def interrupt_after_save(
        path: Path, model: NTAZeroDCE, optimizer: Optimizer, settings: Config, epoch: int
    ) -> None:
        original(path, model, optimizer, settings, epoch)
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(trainer_module, "save_checkpoint", interrupt_after_save)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train(resumed)
    monkeypatch.setattr(trainer_module, "save_checkpoint", original)
    checkpoint = train(resumed, Path(resumed.paths.checkpoint_dir) / "last.pt")
    actual = torch.load(checkpoint, weights_only=False, map_location="cpu")["model"]
    assert expected.keys() == actual.keys()
    assert all(torch.equal(expected[key], actual[key]) for key in expected)
