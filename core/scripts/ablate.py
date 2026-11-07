"""Run the ablation matrix: train every (run, seed) and evaluate it when data exists.

Examples (documented in README.md)::

    python scripts/ablate.py --matrix configs/ablation_matrix.py --output runs/ablations
    python scripts/ablate.py --dry-run                      # print the plan, train nothing
    python scripts/ablate.py --only baseline --only full    # a subset of the rows

Each cell ``<output>/<run>/seed_<seed>/`` holds the resolved config, ``metrics.csv``,
``val_metrics.csv``, the checkpoints, ``eval.csv`` (when evaluation data exists) and
``final.pt``. Finished cells (``final.pt`` present) are skipped unless ``--force``; an
interrupted cell resumes from its latest ``epoch_XXXX.pt``. ``<output>/summary.csv`` has one row
per (run, seed, dataset, metric) mean. Every run inherits the base hyperparameters and differs
only in flags and seed; ``--epochs`` changes the schedule for all runs and is meant for smoke
runs only. See docs/EXPERIMENTS.md.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path

from nta_zerodce.cli import add_common_arguments, main_guard
from nta_zerodce.inference.loader import load_inference_model
from nta_zerodce.training.ablation import RunSpec, expand_matrix, parse_matrix
from nta_zerodce.training.config import resolve_path
from nta_zerodce.training.evaluation import (
    aggregate,
    evaluate_model,
    locate_eval_sets,
    read_rows,
    write_rows,
)
from nta_zerodce.training.trainer import train

DEFAULT_MATRIX = "configs/ablation_matrix.py"
DEFAULT_OUTPUT = "runs/ablations"
SUMMARY_COLUMNS = ("run", "seed", "status", "dataset", "metric", "mean", "count")
_FORBIDDEN = ("noise", "denoise", "edge", "chroma", "adaptive_exp", "temporal", "light", "seed")

__all__ = ["RunSpec", "build_parser", "expand_matrix", "main", "run"]


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser (no data is touched).

    Returns:
        Parser with the shared flags (only --config, --curve-scale, --video-dir and --device are
        meaningful; per-run flags and seeds come from the matrix) plus ``--matrix``,
        ``--output``, ``--dry-run``, ``--only``, ``--epochs`` and ``--force``.
    """
    parser = argparse.ArgumentParser(description="Expand and run the ablation matrix.")
    add_common_arguments(parser)
    parser.set_defaults(config=None)
    parser.add_argument("--matrix", default=DEFAULT_MATRIX, help="Ablation matrix YAML")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output root directory")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the expanded plan and train nothing"
    )
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        metavar="NAME",
        help="Run only this row (repeatable)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training.epochs for every run (for smoke runs only)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Retrain cells that already have a final checkpoint"
    )
    return parser


def _overrides(args: argparse.Namespace) -> dict[str, object]:
    """Global overrides shared by all runs; per-run flags and seeds are rejected."""
    given = [
        f"--{name.replace('_', '-')}" for name in _FORBIDDEN if getattr(args, name) is not None
    ]
    if given:
        raise ValueError(
            f"{', '.join(given)} cannot be used with ablate: flags and seeds come from the matrix"
        )
    return {
        "model.curve_scale": args.curve_scale,
        "paths.video_dir": args.video_dir,
        "training.device": args.device,
        "training.epochs": args.epochs,
    }


def _select(specs: list[RunSpec], only: list[str] | None, matrix: Path) -> list[RunSpec]:
    """Keep the specs whose run name is in ``only`` (order preserved); unknown names fail."""
    if not only:
        return specs
    _, _, runs = parse_matrix(matrix)
    known = [name for name, _ in runs]
    unknown = [name for name in only if name not in known]
    if unknown:
        raise ValueError(f"--only names not in the matrix: {unknown}; available: {known}")
    return [spec for spec in specs if spec.name in only]


def _latest_epoch_checkpoint(leaf: Path) -> Path | None:
    """Highest-numbered ``epoch_XXXX.pt`` in ``leaf``, or None."""
    found = [
        (int(match.group(1)), path)
        for path in leaf.glob("epoch_*.pt")
        if (match := re.fullmatch(r"epoch_(\d+)\.pt", path.name))
    ]
    return max(found)[1] if found else None


def _plan_line(index: int, total: int, spec: RunSpec, leaf: Path, force: bool) -> str:
    """One line of the printed plan."""
    on = [name for name, value in vars(spec.flags).items() if value] or ["none"]
    if (leaf / "final.pt").is_file() and not force:
        status = "done (skip)"
    elif _latest_epoch_checkpoint(leaf) is not None:
        status = "resume"
    else:
        status = "train"
    return (
        f"[{index}/{total}] {spec.name} seed={spec.seed} flags={'+'.join(on)} -> {leaf} [{status}]"
    )


def run(args: argparse.Namespace) -> None:
    """Execute the ablate command.

    Args:
        args: Parsed namespace from ``build_parser``.

    Returns:
        None; prints the plan, trains and evaluates each selected cell and writes
        ``<output>/summary.csv`` (not written for ``--dry-run``).

    Raises:
        ValueError: Invalid matrix/config, unknown ``--only`` name, or per-run flags given.
        FileNotFoundError: Missing matrix, base config or training data.
        RuntimeError: A non-finite loss during training.
    """
    matrix = Path(args.matrix)
    specs = expand_matrix(matrix, _overrides(args), base_config=args.config)
    specs = _select(specs, args.only, matrix)
    output = resolve_path(args.output)
    print(f"ablation plan: {len(specs)} run(s) under {output}")
    for index, spec in enumerate(specs, start=1):
        print(_plan_line(index, len(specs), spec, spec.leaf(output), args.force))
    if args.dry_run:
        return
    summary: list[dict[str, object]] = []
    for index, spec in enumerate(specs, start=1):
        leaf = spec.leaf(output)
        final = leaf / "final.pt"
        status = "skipped"
        if args.force or not final.is_file():
            resume = None if args.force else _latest_epoch_checkpoint(leaf)
            print(f"== [{index}/{len(specs)}] training {spec.name} seed={spec.seed}")
            try:
                final = train(
                    spec.config,
                    resume,
                    run_name=f"seed_{spec.seed}",
                    run_dir=output / spec.name,
                    checkpoint_dir=output / spec.name,
                )
            except FloatingPointError as err:
                raise RuntimeError(str(err)) from err
            status = "trained"
        eval_csv = leaf / "eval.csv"
        if status == "trained":
            eval_csv.unlink(missing_ok=True)  # never report metrics of a replaced checkpoint
        sets = locate_eval_sets(spec.config)
        if sets.any and (status == "trained" or not eval_csv.is_file()):
            model, config, device = load_inference_model(final, None, None, args.device or "auto")
            rows = evaluate_model(
                model,
                config,
                device,
                run=spec.name,
                sets=sets,
                log=lambda message: print(f"notice: {message}"),
            )
            write_rows(eval_csv, rows)
        if eval_csv.is_file():
            for item in aggregate(read_rows(eval_csv)):
                summary.append(
                    {
                        "run": spec.name,
                        "seed": spec.seed,
                        "status": status,
                        "dataset": item.dataset,
                        "metric": item.metric,
                        "mean": item.mean,
                        "count": item.count,
                    }
                )
        else:
            print("notice: no evaluation data found; summary has no metric rows for this cell")
            summary.append(
                {
                    "run": spec.name,
                    "seed": spec.seed,
                    "status": status,
                    "dataset": "",
                    "metric": "",
                    "mean": "",
                    "count": "",
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SUMMARY_COLUMNS))
        writer.writeheader()
        writer.writerows(summary)
    print(f"wrote {len(summary)} summary rows to {output / 'summary.csv'}")


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and run, returning a process exit code (see ``cli.main_guard``)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    return main_guard(lambda: run(args), prog="ablate")


if __name__ == "__main__":
    raise SystemExit(main())
