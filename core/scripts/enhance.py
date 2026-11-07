"""Enhance images, image folders, videos or a webcam stream with a trained checkpoint.

Examples (see the README):

    python scripts/enhance.py --checkpoint checkpoints/full.pt --input photo.png --output out/
    python scripts/enhance.py --checkpoint checkpoints/full.pt --input data/unpaired --output runs/enhanced
    python scripts/enhance.py --checkpoint checkpoints/full.pt --input clip.mp4 --output clip_enhanced.mp4
    python scripts/enhance.py --checkpoint checkpoints/full.pt --webcam 0 --light --curve-scale 0.5

See docs/ARCHITECTURE.md for the normative contract.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import torch
from torch import Tensor

from nta_zerodce.cli import add_common_arguments, main_guard, overrides_from_args
from nta_zerodce.data.image_dataset import IMAGE_EXTENSIONS, index_files
from nta_zerodce.inference.enhancer import MIN_SIDE, enhance_image
from nta_zerodce.inference.io import read_image, read_video, write_image, write_video
from nta_zerodce.inference.loader import load_inference_model
from nta_zerodce.inference.video_enhancer import VideoEnhancer
from nta_zerodce.inference.webcam import run_webcam
from nta_zerodce.models.nta_zerodce import NTAZeroDCE
from nta_zerodce.training.config import Config

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
"""Input containers accepted for frame-by-frame enhancement."""

VIDEO_OUTPUT_EXTENSIONS = (".mp4", ".avi")
"""Output containers the encoder supports (mp4v and MJPG)."""

DEFAULT_FPS = 30.0
"""Frame rate used when the container reports none."""


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser (shared flags plus enhance-specific options).

    Returns:
        Parser with the common INTERFACES flags, ``--checkpoint``, a mutually exclusive
        ``--input`` / ``--webcam`` group, ``--output`` and ``--overwrite``. ``--config`` defaults
        to None, which means "use the configuration stored in the checkpoint"; flags given on
        the command line are applied on top of whichever configuration is used.
    """
    parser = argparse.ArgumentParser(
        prog="enhance",
        description=(
            "Enhance low-light images, folders of images, videos or a webcam with a trained "
            "checkpoint. Images are written as PNG. A folder is searched recursively and its "
            "relative layout is reproduced under --output. A video (.mp4 .avi .mov .mkv) is "
            "enhanced frame by frame with temporal curve smoothing when flags.temporal is on, "
            "and written at the source frame rate to a .mp4 or .avi file; audio is NOT copied."
        ),
        epilog=(
            "Without --config the configuration stored in the checkpoint is used, so the "
            "architecture always matches the weights. Flags such as --temporal/--no-temporal or "
            "--curve-scale are applied on top. An explicit --config must describe the same "
            "architecture (noise, denoise, light, widths) as the checkpoint. --device defaults "
            "to auto; --seed and --video-dir are accepted for uniformity but unused here."
        ),
    )
    add_common_arguments(parser)
    parser.set_defaults(config=None)
    for action in parser._actions:
        if "--config" in action.option_strings:
            action.help = "Experiment YAML; default: the configuration stored in the checkpoint"
    parser.add_argument("--checkpoint", default=None, help="Trained checkpoint (required to run)")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--input", default=None, help="Image, folder of images (recursive) or video file"
    )
    source.add_argument(
        "--webcam",
        type=int,
        default=None,
        metavar="CAMERA_ID",
        help="Camera index for a live window (q or ESC quits, s toggles split view)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Image input: a .png path or a directory. Folder input: output directory. "
            "Video input: a .mp4 or .avi file. Parent directories are created"
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing output files (default: refuse)"
    )
    return parser


# --------------------------------------------------------------------------- planning


def _is_inside(path: Path, parent: Path) -> bool:
    """Whether ``path`` equals or lies below ``parent`` (after resolving symlinks)."""
    resolved, base = path.resolve(), parent.resolve()
    return resolved == base or base in resolved.parents


def _refuse_existing(targets: list[Path], overwrite: bool) -> None:
    """Fail before any work when outputs exist and ``--overwrite`` was not given."""
    if overwrite:
        return
    existing = [t for t in targets if t.exists()]
    if existing:
        shown = ", ".join(str(t) for t in existing[:3])
        more = f" (and {len(existing) - 3} more)" if len(existing) > 3 else ""
        raise ValueError(f"output exists: {shown}{more}; pass --overwrite to replace")


def _plan_images(source: Path, output: Path) -> list[tuple[Path, Path]]:
    """Map every input image to its PNG output path (single file or folder input)."""
    if source.is_file():
        directory_mode = output.is_dir() or output.suffix == ""
        if not directory_mode and output.suffix.lower() != ".png":
            raise ValueError(
                f"images are written as PNG; use a .png path or a directory for --output "
                f"(got {output})"
            )
        target = output / f"{source.stem}.png" if directory_mode else output
        if target.resolve() == source.resolve():
            raise ValueError("output would overwrite the input file")
        return [(source, target)]
    files = index_files(source, IMAGE_EXTENSIONS)
    if not files:
        raise ValueError(f"no images ({', '.join(IMAGE_EXTENSIONS)}) found under {source}")
    if _is_inside(output, source):
        raise ValueError(f"--output {output} must not be inside --input {source}")
    plan: list[tuple[Path, Path]] = []
    seen: dict[Path, Path] = {}
    for path in files:
        target = (output / path.relative_to(source)).with_suffix(".png")
        if target in seen:
            raise ValueError(
                f"{seen[target]} and {path} would both be written to {target}; "
                "rename one of them or enhance them separately"
            )
        seen[target] = path
        plan.append((path, target))
    return plan


# --------------------------------------------------------------------------- execution


def _enhance_images(
    model: NTAZeroDCE,
    config: Config,
    device: torch.device,
    plan: list[tuple[Path, Path]],
) -> None:
    """Enhance independent images; one bad file is reported and the batch continues."""
    started = time.perf_counter()
    failures = 0
    for source, target in plan:
        file_started = time.perf_counter()
        try:
            image = read_image(source)
            height, width = int(image.shape[2]), int(image.shape[3])
            if min(height, width) < MIN_SIDE:
                raise ValueError(f"image is {width}x{height}; both sides must be >= {MIN_SIDE} px")
            result = enhance_image(model, image.to(device), config.model.curve_scale)
            target.parent.mkdir(parents=True, exist_ok=True)
            write_image(result["enhanced"], target)
        except (ValueError, OSError) as error:
            failures += 1
            print(f"enhance: error: {source}: {error}", file=sys.stderr)
            continue
        millis = 1000.0 * (time.perf_counter() - file_started)
        print(f"{source} -> {target} ({width}x{height}, {millis:.0f} ms)")
    done = len(plan) - failures
    print(f"enhanced {done} of {len(plan)} image(s) in {time.perf_counter() - started:.2f} s")
    if failures:
        raise RuntimeError(f"{failures} of {len(plan)} image(s) failed (see messages above)")


def _check_video_target(source: Path, target: Path, overwrite: bool) -> None:
    """Validate the video output path before any model is loaded."""
    if target.suffix.lower() not in VIDEO_OUTPUT_EXTENSIONS:
        raise ValueError(
            f"video output must end in {' or '.join(VIDEO_OUTPUT_EXTENSIONS)}; got {target}"
        )
    if source.resolve() == target.resolve():
        raise ValueError("output would overwrite the input file")
    _refuse_existing([target], overwrite)


def _video_fps(path: Path) -> float:
    """Frame rate reported by the container (OpenCV), or DEFAULT_FPS with a warning."""
    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if not fps > 0.0 or fps != fps or fps == float("inf"):
        print(
            f"enhance: warning: no frame rate in {path}; assuming {DEFAULT_FPS:g}", file=sys.stderr
        )
        return DEFAULT_FPS
    return fps


def _enhance_video(
    model: NTAZeroDCE,
    config: Config,
    device: torch.device,
    source: Path,
    target: Path,
) -> None:
    """Enhance a video frame by frame with one VideoEnhancer and write it at the source fps."""
    fps = _video_fps(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    enhancer = VideoEnhancer(model, config)
    coverage: list[float] = []
    cuts = 0
    started = time.perf_counter()

    def frames() -> Iterator[Tensor]:
        nonlocal cuts
        for frame in read_video(source):
            out = enhancer.enhance_frame(frame.to(device))
            cuts += int(enhancer.last_scene_cut)
            if enhancer.last_mask_coverage is not None:
                coverage.append(enhancer.last_mask_coverage)
            yield out["enhanced"].cpu()

    write_video(frames(), target, fps)
    elapsed = time.perf_counter() - started
    count = enhancer.frames_processed
    detail = "temporal off"
    if config.flags.temporal:
        mean = sum(coverage) / len(coverage) if coverage else 0.0
        detail = f"mean mask coverage {100.0 * mean:.0f}%, {cuts} scene cut(s)"
    print(f"{source} -> {target} ({count} frames at {fps:.2f} fps, {detail})")
    print(f"enhanced 1 video in {elapsed:.2f} s ({count / max(elapsed, 1e-9):.1f} frames/s)")


def run(args: argparse.Namespace) -> None:
    """Execute the enhance command after resolving validated settings.

    Args:
        args: Parsed CLI namespace; explicit overrides use None when absent.

    Returns:
        None. Prints one line per file and the elapsed time.

    Method:
        INTERFACES CLI: load checkpoint (stored config unless --config) -> image IO,
        VideoEnhancer or run_webcam.

    Raises:
        ValueError: Missing/conflicting options, existing outputs without --overwrite, an
            invalid config or a checkpoint that does not match it.
        FileNotFoundError: Missing checkpoint, config or input.
        RuntimeError: One or more images failed (all others were still written), or a
            camera/encoder problem.
    """
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required")
    if args.input is None and args.webcam is None:
        raise ValueError("provide --input (image, folder or video) or --webcam CAMERA_ID")
    if args.input is not None and args.webcam is not None:
        raise ValueError("--input and --webcam cannot be used together")
    if args.webcam is not None and args.output is not None:
        raise ValueError("--output is not used with --webcam")
    if args.input is not None and args.output is None:
        raise ValueError("--output is required with --input")
    if args.webcam is not None and args.webcam < 0:
        raise ValueError(f"--webcam camera index must be >= 0; got {args.webcam}")

    source = Path(args.input) if args.input is not None else None
    if source is not None and not source.exists():
        raise FileNotFoundError(f"input not found: {source}")
    output = Path(args.output) if args.output is not None else None
    plan: list[tuple[Path, Path]] = []
    is_video = False
    if source is not None and output is not None:
        is_video = source.is_file() and source.suffix.lower() in VIDEO_EXTENSIONS
        if is_video:
            _check_video_target(source, output, args.overwrite)
        elif source.is_file() and source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(
                f"unsupported input {source}; use an image {IMAGE_EXTENSIONS}, a folder or a "
                f"video {VIDEO_EXTENSIONS}"
            )
        else:
            plan = _plan_images(source, output)
            _refuse_existing([target for _, target in plan], args.overwrite)
    device_name = args.device if args.device is not None else "auto"
    model, config, device = load_inference_model(
        Path(args.checkpoint), args.config, overrides_from_args(args), device_name
    )
    if args.webcam is not None:
        run_webcam(model, config, args.webcam)
    elif is_video:
        assert source is not None and output is not None
        _enhance_video(model, config, device, source, output)
    else:
        _enhance_images(model, config, device, plan)


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and run, converting expected failures into exit codes.

    Args:
        argv: Argument list without the program name (None reads ``sys.argv``).

    Returns:
        0 on success, 2 for usage/config errors, 1 for runtime errors, 130 on Ctrl-C.
    """
    parser = build_parser()
    return main_guard(lambda: run(parser.parse_args(argv)), prog="enhance")


if __name__ == "__main__":
    raise SystemExit(main())
