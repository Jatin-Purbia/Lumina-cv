"""Shared command orchestration for the five documented entry points."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import torch
import yaml

from nta_zerodce.data.files import IMAGE_SUFFIXES, VIDEO_SUFFIXES, image_files
from nta_zerodce.data.synthetic_pairs import make_pair
from nta_zerodce.inference.enhancer import enhance_image
from nta_zerodce.inference.io import read_image, read_video, write_image, write_video
from nta_zerodce.inference.video_enhancer import VideoEnhancer
from nta_zerodce.inference.webcam import run_webcam
from nta_zerodce.models.factory import build_model
from nta_zerodce.models.nta_zerodce import NTAZeroDCE
from nta_zerodce.training.checkpoint import load_checkpoint
from nta_zerodce.training.config import Config, load_config, save_config
from nta_zerodce.training.evaluation import evaluate
from nta_zerodce.training.trainer import train
from nta_zerodce.training.validation import config_from_dict, core_path
from nta_zerodce.utils.device import resolve_device
from nta_zerodce.utils.seed import seed_everything


def cli_config(args: argparse.Namespace) -> Config:
    """Apply only explicitly supplied CLI overrides to the selected YAML."""
    keys = {
        "noise": "flags.noise_aware",
        "adaptive_exp": "flags.adaptive_exp",
        "curve_scale": "model.curve_scale",
        "video_dir": "paths.video_dir",
        "seed": "training.seed",
        "device": "training.device",
    }
    keys.update(
        {name: f"flags.{name}" for name in ("denoise", "edge", "chroma", "temporal", "light")}
    )
    return load_config(
        Path(args.config), {key: getattr(args, name, None) for name, key in keys.items()}
    )


def loaded_model(args: argparse.Namespace, config: Config) -> NTAZeroDCE:
    """Load a compatible checkpoint on CPU, then move to the requested device."""
    if not args.checkpoint:
        raise ValueError("--checkpoint is required")
    model = build_model(config)
    load_checkpoint(core_path(args.checkpoint), model, expected_config=config)
    model.to(resolve_device(config.training.device)).eval()
    return model


def dispatch(command: str, args: argparse.Namespace) -> None:
    """Execute a documented command, with actionable input validation."""
    if command == "ablate":
        run_ablations(args)
        return
    config = cli_config(args)
    seed_everything(config.training.seed)
    if command == "train":
        print(train(config, Path(args.resume) if args.resume else None))
    elif command == "enhance":
        if args.webcam is not None and args.input is not None:
            raise ValueError("--webcam and --input are mutually exclusive")
        if args.webcam is None and (not args.input or not args.output):
            raise ValueError("--input and --output are required unless --webcam is set")
        model = loaded_model(args, config)
        if args.webcam is not None:
            run_webcam(model, config, args.webcam)
        else:
            enhance_path(model, config, core_path(args.input), core_path(args.output))
    elif command == "evaluate":
        if not args.output:
            raise ValueError("--output is required")
        evaluate(
            loaded_model(args, config),
            config,
            core_path(args.output),
            core_path(args.input) if args.input else None,
        )
    elif command == "make_synthetic_data":
        if not args.input or not args.output:
            raise ValueError("--input and --output are required")
        source, output = core_path(args.input), core_path(args.output)
        files = image_files(source)
        if not files:
            raise ValueError(f"No source images: {source}")
        output.mkdir(parents=True, exist_ok=True)
        save_config(config, output / "config.yaml")
        for index, path in enumerate(files):
            sample = output / f"{index:06d}"
            if sample.exists():
                raise ValueError(f"Synthetic sample already exists: {sample}")
            sample.mkdir()
            generator = torch.Generator().manual_seed(config.training.seed + index)
            pair = make_pair(read_image(path), config, generator)
            write_image(pair["image1"], sample / "first.png")
            write_image(pair["image2"], sample / "second.png")
            # Retain unquantized frames, analytic fields and sampled transform metadata.
            torch.save(pair, sample / "pair.pt")
            metadata = {
                "source": str(path),
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "seed": config.training.seed + index,
                "config": asdict(config),
            }
            (sample / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    else:
        raise ValueError(f"Unknown command: {command}")


def enhance_path(model: NTAZeroDCE, config: Config, source: Path, output: Path) -> None:
    """Enhance a still, recursive image directory, or video preserving source FPS."""
    if not source.exists():
        raise ValueError(f"Input does not exist: {source}")
    if source.resolve() == output.resolve() or (
        source.is_dir() and source.resolve() in output.resolve().parents
    ):
        raise ValueError("Output must be separate from the input tree")
    device = next(model.parameters()).device
    if source.is_file() and source.suffix.lower() in VIDEO_SUFFIXES:
        capture = cv2.VideoCapture(str(source))
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
        finally:
            capture.release()
        enhancer = VideoEnhancer(model, config)
        frames = (
            enhancer.enhance_frame(frame.to(device))["enhanced"] for frame in read_video(source)
        )
        write_video(frames, output, fps)
        return
    files = image_files(source)
    if not files:
        raise ValueError(f"No supported input images: {source}")
    for path in files:
        target = (
            output / path.relative_to(source)
            if source.is_dir()
            else output / path.name
            if output.suffix.lower() not in IMAGE_SUFFIXES
            else output
        )
        if target.exists():
            raise ValueError(f"Output already exists: {target}")
        result = enhance_image(model, read_image(path).to(device), config.model.curve_scale)
        write_image(result["enhanced"], target)


def run_ablations(args: argparse.Namespace) -> None:
    """Validate the entire fixed matrix before scheduling independent seed runs."""
    matrix = yaml.safe_load(core_path(args.matrix).read_text(encoding="utf-8"))
    if not isinstance(matrix, dict) or set(matrix) != {"base_config", "seeds", "runs"}:
        raise ValueError("Matrix requires base_config, seeds and runs")
    seeds, rows = matrix["seeds"], matrix["runs"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(s) is not int or s < 0 for s in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Matrix seeds must be unique nonnegative integers")
    canonical = yaml.safe_load(
        core_path("configs/ablation_matrix.yaml").read_text(encoding="utf-8")
    )
    expected = {row["name"]: row["flags"] for row in canonical["runs"]}
    if not isinstance(rows, list) or len(rows) != 13:
        raise ValueError("Expected all 13 documented ablation rows")
    names = set()
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) != {"name", "flags"}
            or row["name"] in names
            or row["name"] not in expected
            or row["flags"] != expected[row["name"]]
        ):
            raise ValueError("Matrix must contain the documented unique flag combinations")
        names.add(row["name"])
    # Feature/seed overrides would silently alter the experimental matrix.
    if any(
        getattr(args, name, None) is not None
        for name in (
            "noise",
            "denoise",
            "edge",
            "chroma",
            "adaptive_exp",
            "temporal",
            "light",
            "seed",
        )
    ):
        raise ValueError("Ablation flags and seeds are controlled by --matrix")
    args.config = matrix["base_config"]
    base = cli_config(args)
    output = core_path(args.output or "runs/ablations")
    configs = []
    for row in rows:
        for seed in seeds:
            folder = output / row["name"] / str(seed)
            data = asdict(base)
            data["flags"] = row["flags"]
            data["training"]["seed"] = seed
            data["paths"]["run_dir"] = str(folder)
            data["paths"]["checkpoint_dir"] = str(folder / "checkpoints")
            configs.append((folder, config_from_dict(data)))
    output.mkdir(parents=True, exist_ok=True)
    with (output / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "seed", "checkpoint", "evaluation"])
        for folder, config in configs:
            checkpoint = train(config)
            model = build_model(config)
            load_checkpoint(checkpoint, model, expected_config=config)
            model.to(resolve_device(config.training.device))
            destination = folder / "evaluation.csv"
            evaluate(model, config, destination)
            writer.writerow(
                [folder.parent.name, config.training.seed, str(checkpoint), str(destination)]
            )
            handle.flush()
