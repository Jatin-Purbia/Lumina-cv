"""Checkpoint evaluation: paired, unpaired and held-out video metrics as long-format rows.

Used by ``scripts/evaluate.py`` and ``scripts/ablate.py``. Every metric value is one row of the
CSV schema of docs/EXPERIMENTS.md::

    run, seed, dataset, sample_id, metric, value, mask_coverage, input_clipping

Datasets (``dataset`` column):

* ``paired``: ``paths.eval_low`` + ``paths.eval_high`` matched by basename (LOL style), full
  size, no alignment or tone fitting. Metrics: ``psnr``, ``ssim``, ``angular_error``,
  ``clipping_fraction`` (``input_clipping`` holds the input's value), ``noise_estimate`` and
  ``noise_estimate_input``.
* ``unpaired`` or ``unpaired/<subdirectory>``: images under ``paths.unpaired`` (recursive).
  Metrics: ``clipping_fraction`` (with ``input_clipping``), ``noise_estimate``,
  ``noise_estimate_input``.
* ``video``: only when ``flags.temporal``; held-out clips, see ``heldout_video_dir``. Metric
  ``temporal_error`` (flow-aligned L1 of the enhanced frames with the input-derived
  reliability mask) with ``mask_coverage`` and ``input_clipping`` of the current frame.

The noise proxies use the Immerkaer estimator on a flat region chosen from the input only (the
same mask for input and output): pixels whose lightly blurred Sobel gradient magnitude is at most
``FLAT_GRADIENT_THRESHOLD``. The threshold is an a-priori evaluation constant, not tuned on any
result; ``mask_coverage`` of those rows reports the selected fraction. Clipping uses Y >= 0.99.
Metrics are proxies (docs/EXPERIMENTS.md caveats), not quality certificates.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from nta_zerodce.data.image_dataset import IMAGE_EXTENSIONS, ImageDataset, index_files
from nta_zerodce.data.video_pairs import VIDEO_EXTENSIONS
from nta_zerodce.flow.estimator import estimate_bidirectional
from nta_zerodce.flow.masks import reliability_mask
from nta_zerodce.inference.enhancer import enhance_image
from nta_zerodce.inference.io import read_image, read_video
from nta_zerodce.losses._ops import gaussian_blur_reflect, luminance, sobel_magnitude
from nta_zerodce.metrics.colour import angular_error
from nta_zerodce.metrics.exposure import clipping_fraction
from nta_zerodce.metrics.fidelity import psnr, ssim
from nta_zerodce.metrics.noise import noise_estimate
from nta_zerodce.metrics.temporal import warping_error
from nta_zerodce.models.nta_zerodce import NTAZeroDCE
from nta_zerodce.training.config import Config, resolve_path

EVAL_COLUMNS = (
    "run",
    "seed",
    "dataset",
    "sample_id",
    "metric",
    "value",
    "mask_coverage",
    "input_clipping",
)
FLAT_GRADIENT_THRESHOLD = 0.02
"""Largest blurred Sobel magnitude (RGB units per pixel) of a pixel counted as flat."""
MIN_SIDE = 16
MAX_VIDEO_PAIRS = 200
"""Cap on frame pairs per held-out video, so evaluation cost stays bounded."""

Row = dict[str, object]


@dataclass(frozen=True)
class EvalSets:
    """Evaluation locations found on disk (``None`` when absent) and notices for skipped sets."""

    paired: tuple[Path, Path] | None
    unpaired: Path | None
    video: Path | None
    notices: tuple[str, ...]
    expected: tuple[str, ...]

    @property
    def any(self) -> bool:
        """True when at least one evaluation set exists."""
        return self.paired is not None or self.unpaired is not None or self.video is not None


@dataclass(frozen=True)
class AggregateRow:
    """Mean of one metric over the finite samples of one dataset."""

    dataset: str
    metric: str
    mean: float
    count: int


def heldout_video_dir(video_dir: str | Path) -> Path | None:
    """Locate the held-out clip directory belonging to ``paths.video_dir``.

    docs/DATA.md splits videos into ``videos/train``, ``videos/val`` and ``videos/test`` by
    session. If ``video_dir`` itself is named ``val`` or ``test`` it is used as given;
    otherwise the sibling ``test`` and then ``val`` directory is used. Training clips are never
    evaluated.

    Args:
        video_dir: Configured ``paths.video_dir`` (relative paths resolve against the workspace).

    Returns:
        An existing directory, or ``None``.
    """
    root = resolve_path(video_dir)
    if root.name in ("val", "test"):
        return root if root.is_dir() else None
    for name in ("test", "val"):
        candidate = root.parent / name
        if candidate.is_dir():
            return candidate
    return None


def locate_eval_sets(config: Config) -> EvalSets:
    """Find which evaluation sets exist for ``config`` and describe the missing ones.

    Args:
        config: Validated Config (``paths.eval_low``, ``eval_high``, ``unpaired``,
            ``video_dir``; the video set is considered only when ``flags.temporal``).

    Returns:
        EvalSets with the existing locations, one notice per skipped optional set and the list
        of all expected paths (for an error message when nothing exists).
    """
    paths = config.paths
    low, high = resolve_path(paths.eval_low), resolve_path(paths.eval_high)
    unpaired = resolve_path(paths.unpaired)
    notices: list[str] = []
    expected = [
        f"paths.eval_low = {low} and paths.eval_high = {high} (paired, matched by file name)",
        f"paths.unpaired = {unpaired} (unpaired images, searched recursively)",
    ]
    paired = (low, high) if low.is_dir() and high.is_dir() else None
    if paired is None:
        notices.append(f"skipping paired evaluation: {low} and/or {high} not found")
    unpaired_dir = unpaired if unpaired.is_dir() else None
    if unpaired_dir is None:
        notices.append(f"skipping unpaired evaluation: {unpaired} not found")
    video: Path | None = None
    if config.flags.temporal:
        base = resolve_path(paths.video_dir)
        expected.append(
            f"held-out videos: {base} (if named val/test) or its sibling test/ or val/ directory"
        )
        video = heldout_video_dir(paths.video_dir)
        if video is None:
            notices.append(
                f"skipping temporal evaluation: no held-out clips (test/ or val/) next to {base}"
            )
    return EvalSets(paired, unpaired_dir, video, tuple(notices), tuple(expected))


def flat_mask(image: Tensor, threshold: float = FLAT_GRADIENT_THRESHOLD) -> Tensor:
    """Input-derived flat-region mask (B,1,H,W) of 0/1 values.

    Args:
        image: RGB (B,3,H,W) in [0,1].
        threshold: Largest blurred Sobel gradient magnitude counted as flat.

    Returns:
        Float mask; 1 where the lightly blurred luminance gradient is at most ``threshold``.
    """
    with torch.no_grad():
        smooth = gaussian_blur_reflect(luminance(image.detach()), 0.8, 5)
        return (sobel_magnitude(smooth) <= threshold).to(image.dtype)


def _scalar(value: Tensor) -> float:
    """First element of a per-sample metric tensor as a float."""
    return float(value.detach().reshape(-1)[0])


def _row(
    run: str,
    seed: int,
    dataset: str,
    sample_id: str,
    metric: str,
    value: float,
    mask_coverage: float | None = None,
    input_clipping: float | None = None,
) -> Row:
    """One CSV record (``None`` becomes an empty cell)."""
    return {
        "run": run,
        "seed": seed,
        "dataset": dataset,
        "sample_id": sample_id,
        "metric": metric,
        "value": value,
        "mask_coverage": mask_coverage,
        "input_clipping": input_clipping,
    }


def _image_rows(
    run: str,
    seed: int,
    dataset: str,
    sample_id: str,
    image: Tensor,
    output: Tensor,
    reference: Tensor | None,
) -> list[Row]:
    """All still-image metrics of one sample (tensors (1,3,H,W) on one device)."""
    rows: list[Row] = []
    if reference is not None:
        rows.append(_row(run, seed, dataset, sample_id, "psnr", _scalar(psnr(output, reference))))
        rows.append(_row(run, seed, dataset, sample_id, "ssim", _scalar(ssim(output, reference))))
        rows.append(
            _row(
                run,
                seed,
                dataset,
                sample_id,
                "angular_error",
                _scalar(angular_error(output, reference)),
            )
        )
    rows.append(
        _row(
            run,
            seed,
            dataset,
            sample_id,
            "clipping_fraction",
            _scalar(clipping_fraction(output)),
            input_clipping=_scalar(clipping_fraction(image)),
        )
    )
    if min(image.shape[-2:]) >= 3:
        mask = flat_mask(image)
        coverage = float(mask.mean())
        for metric, tensor in (("noise_estimate", output), ("noise_estimate_input", image)):
            rows.append(
                _row(
                    run,
                    seed,
                    dataset,
                    sample_id,
                    metric,
                    _scalar(noise_estimate(tensor, mask)),
                    mask_coverage=coverage,
                )
            )
    return rows


def _evaluate_paired(
    model: NTAZeroDCE,
    config: Config,
    device: torch.device,
    run: str,
    sets: tuple[Path, Path],
    log: Callable[[str], None],
) -> list[Row]:
    """Paired LOL-style evaluation with full-size images (batch size one)."""
    low, high = sets
    dataset = ImageDataset(low, config.training.crop_size, paired_root=high)
    rows: list[Row] = []
    for index, path in enumerate(dataset.paths):
        item = dataset[index]
        image = item["image"].unsqueeze(0).to(device)
        reference = item["reference"].unsqueeze(0).to(device)
        if min(image.shape[-2:]) < MIN_SIDE:
            log(f"skipping {path.name}: sides below {MIN_SIDE} px")
            continue
        output = enhance_image(model, image, config.model.curve_scale)["enhanced"]
        rows.extend(
            _image_rows(run, config.training.seed, "paired", path.stem, image, output, reference)
        )
    return rows


def _evaluate_unpaired(
    model: NTAZeroDCE,
    config: Config,
    device: torch.device,
    run: str,
    root: Path,
    log: Callable[[str], None],
) -> list[Row]:
    """No-reference evaluation of every image under ``root`` (recursive)."""
    rows: list[Row] = []
    skipped = 0
    for path in index_files(root, IMAGE_EXTENSIONS):
        relative = path.relative_to(root)
        parts = relative.parent.parts
        dataset = f"unpaired/{parts[0]}" if parts else "unpaired"
        image = read_image(path).to(device)
        if min(image.shape[-2:]) < MIN_SIDE:
            skipped += 1
            continue
        output = enhance_image(model, image, config.model.curve_scale)["enhanced"]
        rows.extend(
            _image_rows(
                run,
                config.training.seed,
                dataset,
                relative.with_suffix("").as_posix(),
                image,
                output,
                None,
            )
        )
    if skipped:
        log(f"unpaired: skipped {skipped} image(s) with a side below {MIN_SIDE} px")
    return rows


def _evaluate_video(
    model: NTAZeroDCE,
    config: Config,
    device: torch.device,
    run: str,
    root: Path,
    log: Callable[[str], None],
) -> list[Row]:
    """Flow-aligned temporal error of enhanced frame pairs with input-derived masks."""
    rows: list[Row] = []
    gap = config.temporal.frame_gap
    for path in index_files(root, VIDEO_EXTENSIONS):
        name = path.relative_to(root).with_suffix("").as_posix()
        window: list[Tensor] = []
        outputs: list[Tensor] = []
        pairs = 0
        try:
            for frame in read_video(path):
                if min(frame.shape[-2:]) < MIN_SIDE:
                    raise ValueError(f"frame sides below {MIN_SIDE} px")
                window.append(frame)
                outputs.append(
                    enhance_image(model, frame.to(device), config.model.curve_scale)["enhanced"]
                )
                if len(window) <= gap:
                    continue
                previous, current = window.pop(0), window[-1]
                prev_out, cur_out = outputs.pop(0), outputs[-1]
                forward, backward = estimate_bidirectional(previous, current, config.flow)
                mask = reliability_mask(
                    forward,
                    backward,
                    previous,
                    current,
                    alpha=config.flow.fb_alpha,
                    beta=config.flow.fb_beta,
                    photometric_threshold=config.flow.photometric_threshold,
                )
                error = warping_error(prev_out, cur_out, backward.to(device), mask.to(device))
                rows.append(
                    _row(
                        run,
                        config.training.seed,
                        "video",
                        f"{name}#{pairs}",
                        "temporal_error",
                        _scalar(error),
                        mask_coverage=float(mask.mean()),
                        input_clipping=_scalar(clipping_fraction(current.to(device))),
                    )
                )
                pairs += 1
                if pairs >= MAX_VIDEO_PAIRS:
                    break
        except (ValueError, RuntimeError) as err:
            log(f"skipping video {path}: {err}")
    return rows


def evaluate_model(
    model: NTAZeroDCE,
    config: Config,
    device: torch.device,
    *,
    run: str,
    sets: EvalSets | None = None,
    log: Callable[[str], None] | None = None,
) -> list[Row]:
    """Evaluate an eval-mode model on every available set.

    Args:
        model: Eval-mode model on ``device`` (see ``inference.loader.load_inference_model``).
        config: Resolved Config; ``model.curve_scale`` is the inference scale used throughout.
        device: Device of the model.
        run: Value of the ``run`` column.
        sets: Precomputed ``locate_eval_sets(config)``; computed when omitted.
        log: Receives notices about skipped sets or inputs (default: discarded).

    Returns:
        Long-format rows (see the module docstring), ordered paired, unpaired, video.

    Raises:
        FileNotFoundError: No evaluation set exists; the message lists every expected path.
        ValueError: The existing sets contain no usable sample.
    """
    notify = log if log is not None else (lambda _message: None)
    found = sets if sets is not None else locate_eval_sets(config)
    for message in found.notices:
        notify(message)
    if not found.any:
        raise FileNotFoundError(
            "no evaluation data found; provide at least one of: " + "; ".join(found.expected)
        )
    rows: list[Row] = []
    if found.paired is not None:
        rows.extend(_evaluate_paired(model, config, device, run, found.paired, notify))
    if found.unpaired is not None:
        rows.extend(_evaluate_unpaired(model, config, device, run, found.unpaired, notify))
    if found.video is not None:
        rows.extend(_evaluate_video(model, config, device, run, found.video, notify))
    if not rows:
        raise ValueError(
            "the evaluation sets exist but yielded no usable samples (empty directories, "
            f"images below {MIN_SIDE} px or unreadable clips)"
        )
    return rows


def _cell(value: object) -> str:
    """CSV text of one cell: empty for None, ``repr`` for floats (round-trips exactly)."""
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return str(value)


def write_rows(path: Path, rows: list[Row]) -> None:
    """Write evaluation rows with the fixed ``EVAL_COLUMNS`` header (overwrites ``path``).

    Args:
        path: Destination CSV; missing parent directories are created.
        rows: Rows from ``evaluate_model``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(EVAL_COLUMNS)
        for row in rows:
            writer.writerow([_cell(row[name]) for name in EVAL_COLUMNS])


def read_rows(path: Path) -> list[Row]:
    """Read a CSV written by ``write_rows`` (numbers parsed, empty cells become None).

    Raises:
        FileNotFoundError: Missing file.
        ValueError: The header differs from ``EVAL_COLUMNS``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"evaluation CSV not found: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != EVAL_COLUMNS:
            raise ValueError(f"{path} does not have the columns {list(EVAL_COLUMNS)}")
        rows: list[Row] = []
        for record in reader:
            rows.append(
                {
                    "run": record["run"],
                    "seed": int(record["seed"]),
                    "dataset": record["dataset"],
                    "sample_id": record["sample_id"],
                    "metric": record["metric"],
                    "value": float(record["value"]),
                    "mask_coverage": float(record["mask_coverage"])
                    if record["mask_coverage"]
                    else None,
                    "input_clipping": float(record["input_clipping"])
                    if record["input_clipping"]
                    else None,
                }
            )
    return rows


def aggregate(rows: list[Row]) -> list[AggregateRow]:
    """Mean per (dataset, metric) over finite values, in first-seen order.

    Args:
        rows: Evaluation rows.

    Returns:
        One AggregateRow per pair; ``count`` is the number of rows, ``mean`` ignores NaN and
        infinite samples (NaN when none are finite). This is a per-sample mean; the experiment
        protocol aggregates by scene and then by seed (docs/EXPERIMENTS.md).
    """
    groups: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["metric"]))
        value = row["value"]
        assert isinstance(value, float)
        groups.setdefault(key, []).append(value)
    out: list[AggregateRow] = []
    for (dataset, metric), values in groups.items():
        finite = [v for v in values if math.isfinite(v)]
        mean = sum(finite) / len(finite) if finite else math.nan
        out.append(AggregateRow(dataset, metric, mean, len(values)))
    return out


def format_aggregate(rows: list[AggregateRow]) -> str:
    """Fixed-width table of the aggregates for printing."""
    lines = [f"{'dataset':<24} {'metric':<22} {'mean':>12} {'n':>5}"]
    for item in rows:
        lines.append(f"{item.dataset:<24} {item.metric:<22} {item.mean:>12.6f} {item.count:>5d}")
    return "\n".join(lines)
