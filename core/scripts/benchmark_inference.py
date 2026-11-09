"""Measure curve-scale latency and output drift against full-resolution estimation.

Example (from core/): python scripts/benchmark_inference.py --checkpoint
checkpoints/lol/final.pt --input ../public --output runs/inference_benchmark.json

Drift is relative to the same checkpoint at scale 1, not a perceptual quality score.
Decoding, file IO and device transfers are excluded from timed model execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from nta_zerodce.cli import main_guard
from nta_zerodce.data.image_dataset import IMAGE_EXTENSIONS, index_files
from nta_zerodce.inference.enhancer import enhance_image
from nta_zerodce.inference.io import read_image
from nta_zerodce.inference.loader import load_inference_model
from nta_zerodce.models.nta_zerodce import NTAZeroDCE


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(
    model: NTAZeroDCE, image: Tensor, scale: float, repeats: int, warmup: int
) -> tuple[Tensor, list[float]]:
    """Return the last output and synchronized per-run milliseconds, excluding warmups."""
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be positive and warmup must be non-negative")
    timings: list[float] = []
    with torch.inference_mode():
        for index in range(warmup + repeats):
            _synchronize(image.device)
            started = time.perf_counter()
            output = enhance_image(model, image, scale)["enhanced"]
            _synchronize(image.device)
            elapsed = (time.perf_counter() - started) * 1000
            if index >= warmup:
                timings.append(elapsed)
    return output, timings


def run(args: argparse.Namespace) -> None:
    """Benchmark every input image, preserving weights and recording reproducible settings."""
    if args.repeats < 1 or args.warmup < 0 or args.threads < 1:
        raise ValueError("repeats/threads must be positive and warmup must be non-negative")
    scales = list(dict.fromkeys([1.0, *args.scales]))
    if any(not math.isfinite(s) or not 0 < s <= 1 for s in scales):
        raise ValueError("all scales must be finite and in (0, 1]")
    source = Path(args.input)
    paths = [source] if source.is_file() else index_files(source, IMAGE_EXTENSIONS)
    if not paths:
        raise ValueError(f"no images found in {source}")
    destination = Path(args.output)
    if destination.exists():
        raise ValueError(f"output already exists: {destination}; choose a new output path")
    torch.set_num_threads(args.threads)
    checkpoint = Path(args.checkpoint)
    model, _, device = load_inference_model(checkpoint, device=args.device)
    rows: list[dict[str, Any]] = []
    for path in paths:
        image = read_image(path).to(device)
        reference, reference_timings = measure(model, image, 1.0, args.repeats, args.warmup)
        for scale in scales:
            if scale == 1.0:
                output, timings = reference, reference_timings
            else:
                output, timings = measure(model, image, scale, args.repeats, args.warmup)
            difference = (output - reference).abs()
            rows.append(
                {
                    "image": str(path),
                    "width": image.shape[3],
                    "height": image.shape[2],
                    "curve_scale": scale,
                    "median_ms": statistics.median(timings),
                    "samples_ms": timings,
                    "mean_abs_error_255": float(difference.mean()) * 255,
                    "max_abs_error_255": float(difference.max()) * 255,
                }
            )
        print(f"measured {path.name}", flush=True)
    summary: list[dict[str, float]] = []
    baseline = statistics.mean(row["median_ms"] for row in rows if row["curve_scale"] == 1)
    for scale in scales:
        selected = [row for row in rows if row["curve_scale"] == scale]
        latency = statistics.mean(row["median_ms"] for row in selected)
        summary.append(
            {
                "curve_scale": scale,
                "mean_image_median_ms": latency,
                "speedup": baseline / latency,
                "mean_image_abs_error_255": statistics.mean(
                    row["mean_abs_error_255"] for row in selected
                ),
                "worst_pixel_abs_error_255": max(row["max_abs_error_255"] for row in selected),
            }
        )
    payload = {
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "torch": str(torch.__version__),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": str(device),
        "threads": torch.get_num_threads(),
        "repeats": args.repeats,
        "warmup": args.warmup,
        "summary": summary,
        "images": rows,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {destination}")


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark; this command never trains or replaces checkpoints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--input", required=True, help="One image or a directory searched recursively"
    )
    parser.add_argument("--output", required=True, help="New JSON result path (never overwritten)")
    parser.add_argument("--scales", type=float, nargs="+", default=[0.5, 0.25])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1, help="CPU intra-op threads")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args(argv)
    return main_guard(lambda: run(args), prog="benchmark_inference")


if __name__ == "__main__":
    raise SystemExit(main())
