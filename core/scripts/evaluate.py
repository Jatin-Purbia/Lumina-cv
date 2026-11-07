"""Evaluate a checkpoint on paired, unpaired and held-out video sets and write a CSV.

Examples (documented in README.md)::

    python scripts/evaluate.py --checkpoint checkpoints/full_s17/final.pt --output runs/eval.csv
    python scripts/evaluate.py --config configs/full.py --checkpoint checkpoints/full.pt \
        --output runs/eval.csv

Without ``--config`` the resolved config stored in the checkpoint is used; flag options then
override it (an architecture mismatch with the checkpoint is rejected). ``--curve-scale`` sets the
inference curve scale. Dataset directories come from ``paths.eval_low``/``eval_high``/
``unpaired``/``video_dir``; missing optional sets are skipped with a notice and the command fails
with exit code 2 only when none exists. CSV columns and metric definitions:
``nta_zerodce.training.evaluation`` and docs/ARCHITECTURE.md.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from nta_zerodce.cli import add_common_arguments, main_guard, overrides_from_args
from nta_zerodce.inference.loader import load_inference_model
from nta_zerodce.training.config import resolve_path
from nta_zerodce.training.evaluation import (
    aggregate,
    evaluate_model,
    format_aggregate,
    write_rows,
)

DEFAULT_OUTPUT = "runs/eval.csv"


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser (no data or checkpoint is touched).

    Returns:
        Parser with the shared flags plus ``--checkpoint``, ``--output`` and ``--run-name``.
        ``--config`` defaults to None, meaning "use the checkpoint's stored config".
    """
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint and write metrics as CSV.")
    add_common_arguments(parser)
    parser.set_defaults(config=None)
    parser.add_argument("--checkpoint", default=None, help="Checkpoint file (required)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument(
        "--run-name", default=None, help="Value of the CSV run column (default from checkpoint)"
    )
    return parser


def _default_run_name(checkpoint: Path) -> str:
    """Run name for the CSV: the checkpoint's run directory, or its stem for other names."""
    if checkpoint.name == "final.pt" or re.fullmatch(r"epoch_\d+\.pt", checkpoint.name):
        return checkpoint.parent.name
    return checkpoint.stem


def run(args: argparse.Namespace) -> None:
    """Execute the evaluate command.

    Args:
        args: Parsed namespace from ``build_parser``.

    Returns:
        None; writes the CSV and prints per-dataset/metric means.

    Raises:
        ValueError: No checkpoint given, or an invalid config/checkpoint (exit code 2).
        FileNotFoundError: Missing checkpoint/config, or no evaluation data at all (exit 2).
    """
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required to evaluate")
    checkpoint = Path(args.checkpoint)
    model, config, device = load_inference_model(
        checkpoint,
        args.config,
        overrides_from_args(args),
        args.device if args.device is not None else "auto",
    )
    run_name = args.run_name if args.run_name is not None else _default_run_name(checkpoint)
    rows = evaluate_model(model, config, device, run=run_name, log=lambda m: print(f"notice: {m}"))
    output = resolve_path(args.output)
    write_rows(output, rows)
    print(format_aggregate(aggregate(rows)))
    print(f"wrote {len(rows)} rows to {output}")


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and run, returning a process exit code (see ``cli.main_guard``)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return main_guard(lambda: run(args), prog="evaluate")


if __name__ == "__main__":
    raise SystemExit(main())
