"""Generate local synthetic noisy images or temporal pairs for tests, demos and smoke runs.

Without ``--input`` procedural low-light source images are generated with
``make_synthetic_images``; with ``--input`` the given image (or folder, searched recursively) is
used. Everything is deterministic in ``--seed`` (the common flag, ``training.seed``), the config
and the sources, and every sample carries a ``meta.json`` sidecar with the values needed to
reproduce or audit it.

Layout under ``--output`` (one directory per source, ``<index>_<name>``):

    images mode (default):  clean.png  noisy.png  meta.json
    --pairs:                image1.png image2.png meta.json

``--pairs`` uses ``make_pair`` (METHOD extension 5): an affine second frame from the shared
geometry, exposure flicker and independent heteroscedastic noise on both frames. The sidecar
stores the forward and inverse 2x3 affine matrices in pixel coordinates, the noise coefficients
(a, b) of both frames, the flicker factor, mask coverage, seed and sampler, so analytic flow can
be recomputed. Generated data are private local samples: do not commit or redistribute them.

See docs/ARCHITECTURE.md for the normative contract.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from nta_zerodce.cli import add_common_arguments, main_guard, resolve_config
from nta_zerodce.data.image_dataset import IMAGE_EXTENSIONS, index_files, make_synthetic_images
from nta_zerodce.data.noise_synth import sample_noise
from nta_zerodce.data.synthetic_pairs import make_pair
from nta_zerodce.data.transforms import item_generator
from nta_zerodce.inference.io import read_image, write_image
from nta_zerodce.training.config import Config

META_FORMAT = "lumina-synthetic-v1"
DEFAULT_COUNT = 8
DEFAULT_PROCEDURAL_SIZE = (64, 64)
MIN_SIDE = 16
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _parse_size(text: str) -> tuple[int, int]:
    """Parse ``N`` (square) or ``HxW`` into (height, width), each at least 16."""
    match = re.fullmatch(r"(\d+)(?:[xX](\d+))?", text.strip())
    if match is None:
        raise argparse.ArgumentTypeError(f"size must be N or HxW (for example 64 or 48x80): {text}")
    height = int(match.group(1))
    width = int(match.group(2)) if match.group(2) is not None else height
    if min(height, width) < MIN_SIDE:
        raise argparse.ArgumentTypeError(f"size must be at least {MIN_SIDE}x{MIN_SIDE}: {text}")
    return height, width


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser (shared flags plus data-generation options)."""
    parser = argparse.ArgumentParser(
        prog="make_synthetic_data",
        description=(
            "Write synthetic noisy images (default) or synthetic temporal pairs (--pairs) with "
            "JSON sidecars. Sources come from --input (image or folder, recursive) or, when "
            "omitted, from procedural low-light images. Deterministic given --seed, the config "
            "(noise and synthetic sections) and the sources. Refuses to overwrite without "
            "--overwrite and never writes outside --output."
        ),
        epilog=(
            "Output per source: <output>/<index>_<name>/{clean.png,noisy.png,meta.json} or, "
            "with --pairs, {image1.png,image2.png,meta.json}. --seed overrides training.seed; "
            "--device and the architecture flags are accepted but unused. Generated data are "
            "private samples for local use."
        ),
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--input", default=None, help="Image or folder of source images (default: procedural)"
    )
    parser.add_argument("--output", default=None, help="Output directory (required)")
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help=f"Number of sources (default: {DEFAULT_COUNT} procedural, all with --input)",
    )
    parser.add_argument(
        "--size",
        type=_parse_size,
        default=None,
        metavar="N|HxW",
        help=(
            f"Procedural image size (default {DEFAULT_PROCEDURAL_SIZE[0]}). With --input: a "
            "seeded random crop of this size; smaller images are skipped and counted"
        ),
    )
    parser.add_argument(
        "--pairs", action="store_true", help="Write temporal pairs (make_pair) instead of images"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing files (default: refuse)"
    )
    return parser


def _sample_id(index: int, source: Path | None, root: Path | None) -> str:
    """Filesystem-safe directory name ``<index>_<relative path with separators as __>``."""
    if source is None or root is None:
        return f"synthetic_{index:04d}"
    relative = source.relative_to(root) if root != source else Path(source.name)
    name = _SAFE.sub("_", "__".join(relative.with_suffix("").parts)).strip("._") or "image"
    return f"{index:04d}_{name}"


def _crop(image: Tensor, size: tuple[int, int] | None, seed: int, index: int) -> tuple[Tensor, Any]:
    """Seeded random crop of a (1,3,H,W) image; returns (crop, [top, left] or None)."""
    if size is None:
        return image, None
    height, width = int(image.shape[2]), int(image.shape[3])
    generator = item_generator(seed, 1, index)
    top = int(torch.randint(0, height - size[0] + 1, (1,), generator=generator))
    left = int(torch.randint(0, width - size[1] + 1, (1,), generator=generator))
    return image[:, :, top : top + size[0], left : left + size[1]].contiguous(), [top, left]


def _config_record(config: Config) -> dict[str, Any]:
    """The configuration sections that influence generation, for the sidecar."""
    return {"noise": asdict(config.noise), "synthetic": asdict(config.synthetic)}


def _noisy_image(
    clean: Tensor, config: Config, seed: int, index: int
) -> tuple[Tensor, dict[str, Any]]:
    """Add one heteroscedastic noise draw (a, b per image) to a clean (1,3,H,W) image."""
    generator = item_generator(seed, 0, index)
    noise = config.noise
    draws = torch.rand(3, generator=generator, dtype=torch.float64)
    a = float(noise.a_min + (noise.a_max - noise.a_min) * draws[0])
    b = float(noise.b_min + (noise.b_max - noise.b_min) * draws[1])
    applied = bool(draws[2] < noise.probability)
    if not applied:
        a = b = 0.0
    coefficient = torch.tensor([a, b], dtype=torch.float32)
    noisy, _ = sample_noise(
        clean, coefficient[0].view(1, 1, 1, 1), coefficient[1].view(1, 1, 1, 1), generator
    )
    return noisy, {
        "noise_a": float(coefficient[0]),
        "noise_b": float(coefficient[1]),
        "noise_applied": applied,
    }


def _pair(
    clean: Tensor, config: Config, seed: int, index: int
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Create one synthetic temporal pair and its sidecar fields."""
    pair = make_pair(clean, config, item_generator(seed, 0, index))
    meta: dict[str, Any] = {
        "affine": pair["affine"][0].tolist(),
        "affine_inverse": pair["affine_inverse"][0].tolist(),
        "flicker": float(pair["flicker"][0]),
        "noise_a": [float(v) for v in pair["noise_a"][0]],
        "noise_b": [float(v) for v in pair["noise_b"][0]],
        "mask_coverage": float(pair["mask"].mean()),
    }
    return pair, meta


def _inside(path: Path, root: Path) -> bool:
    """Whether ``path`` stays under ``root`` after resolving symlinks and ``..``."""
    return root.resolve() in path.resolve().parents


def run(args: argparse.Namespace) -> None:
    """Execute the make_synthetic_data command after resolving validated settings.

    Args:
        args: Parsed CLI namespace; explicit overrides use None when absent.

    Returns:
        None. Prints a one-line summary.

    Method:
        INTERFACES CLI: load_config -> source images (procedural or --input) -> make_pair or
        sample_noise -> local generated output with JSON sidecars.

    Raises:
        ValueError: Missing --output, bad --count, no usable sources, an output that exists
            without --overwrite, or an unsafe output location.
        FileNotFoundError: Missing --input or config.
    """
    config = resolve_config(args)
    seed = config.training.seed
    if args.output is None:
        raise ValueError("--output is required")
    if args.count is not None and args.count < 1:
        raise ValueError(f"--count must be at least 1; got {args.count}")
    output = Path(args.output)
    size: tuple[int, int] | None = args.size
    root: Path | None = None
    files: list[Path] = []
    if args.input is not None:
        source = Path(args.input)
        if not source.exists():
            raise FileNotFoundError(f"input not found: {source}")
        if source.is_dir():
            root = source
            files = index_files(source, IMAGE_EXTENSIONS)
        elif source.suffix.lower() in IMAGE_EXTENSIONS:
            root, files = source, [source]
        else:
            raise ValueError(f"unsupported input {source}; use an image or a folder of images")
        if output.resolve() == source.resolve() or (source.is_dir() and _inside(output, source)):
            raise ValueError(f"--output {output} must not be inside --input {source}")
        if not files:
            raise ValueError(f"no images ({', '.join(IMAGE_EXTENSIONS)}) found under {source}")
        if args.count is not None:
            files = files[: args.count]
    count = args.count if args.count is not None else DEFAULT_COUNT

    with tempfile.TemporaryDirectory(prefix="lumina_sources_") as scratch:
        if args.input is None:
            procedural = size if size is not None else DEFAULT_PROCEDURAL_SIZE
            files = make_synthetic_images(Path(scratch), count, procedural, seed)
            size = None  # procedural images already have the requested size
        plan: list[tuple[int, Path, Path, str]] = []  # index, source, sample directory, label
        for index, path in enumerate(files):
            ident = _sample_id(index, path if args.input is not None else None, root)
            label = (
                "procedural"
                if args.input is None
                else (
                    path.relative_to(root).as_posix()
                    if root is not None and root.is_dir()
                    else path.name
                )
            )
            plan.append((index, path, output / ident, label))
        names = (
            ("image1.png", "image2.png", "meta.json")
            if args.pairs
            else ("clean.png", "noisy.png", "meta.json")
        )
        existing = [d / n for _, _, d, _ in plan for n in names if (d / n).exists()]
        if existing and not args.overwrite:
            shown = ", ".join(str(p) for p in existing[:3])
            more = f" (and {len(existing) - 3} more)" if len(existing) > 3 else ""
            raise ValueError(f"output exists: {shown}{more}; pass --overwrite to replace")
        for _, _, directory, _ in plan:
            if not _inside(directory / names[0], output):
                raise ValueError(f"refusing to write outside --output: {directory}")

        written = skipped = 0
        for index, path, directory, label in plan:
            image = read_image(path)
            if min(image.shape[2:]) < MIN_SIDE or (
                size is not None and (image.shape[2] < size[0] or image.shape[3] < size[1])
            ):
                skipped += 1
                continue
            clean, offset = _crop(image, size, seed, index)
            directory.mkdir(parents=True, exist_ok=True)
            meta: dict[str, Any] = {
                "format": META_FORMAT,
                "mode": "pair" if args.pairs else "image",
                "source": label,
                "source_index": index,
                "seed": seed,
                "generator": f"nta_zerodce.data.transforms.item_generator({seed}, 0, {index})",
                "height": int(clean.shape[2]),
                "width": int(clean.shape[3]),
                "crop_top_left": offset,
                "config": _config_record(config),
            }
            if args.pairs:
                pair, fields = _pair(clean, config, seed, index)
                write_image(pair["image1"], directory / "image1.png")
                write_image(pair["image2"], directory / "image2.png")
            else:
                noisy, fields = _noisy_image(clean, config, seed, index)
                write_image(clean, directory / "clean.png")
                write_image(noisy, directory / "noisy.png")
            meta.update(fields)
            (directory / "meta.json").write_text(
                json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            written += 1
    if written == 0:
        raise ValueError("no usable source images (all were smaller than the requested size)")
    kind = "pair(s)" if args.pairs else "noisy image(s)"
    note = f"; skipped {skipped} source(s) that were too small" if skipped else ""
    print(f"wrote {written} synthetic {kind} to {output} (seed {seed}){note}")


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and run, converting expected failures into exit codes.

    Args:
        argv: Argument list without the program name (None reads ``sys.argv``).

    Returns:
        0 on success, 2 for usage/config errors, 1 for runtime errors, 130 on Ctrl-C.
    """
    parser = build_parser()
    return main_guard(lambda: run(parser.parse_args(argv)), prog="make_synthetic_data")


if __name__ == "__main__":
    raise SystemExit(main())
