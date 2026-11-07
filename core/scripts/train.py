"""Train a model from a YAML experiment and the shared feature flags.

Examples (documented in README.md)::

    python scripts/train.py --config configs/baseline.yaml
    python scripts/train.py --config configs/full.yaml --noise --denoise --edge --chroma \
        --adaptive-exp --temporal --video-dir data/videos/train
    python scripts/train.py --config configs/full.yaml --resume checkpoints/full_s17/epoch_0010.pt

Outputs go to ``<paths.run_dir>/<run name>/`` (config, metrics.csv, val_metrics.csv) and
``<paths.checkpoint_dir>/<run name>/`` (epoch_XXXX.pt, final.pt); the run name is derived from
the flags and the seed (``nta_zerodce.training.trainer.default_run_name``) unless ``--run-name``
is given. See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from nta_zerodce.cli import add_common_arguments, main_guard, resolve_config
from nta_zerodce.training.config import override_config
from nta_zerodce.training.trainer import train


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser (no data or checkpoint is touched).

    Returns:
        Parser with the shared flags plus ``--resume``, ``--epochs`` and ``--run-name``.
    """
    parser = argparse.ArgumentParser(
        description="Train from a YAML config; flags given here override the YAML."
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--resume", default=None, help="Checkpoint to resume from (exact flag compatibility)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override training.epochs (smoke runs)"
    )
    parser.add_argument(
        "--run-name", default=None, help="Run directory name (default derived from flags/seed)"
    )
    return parser


def run(args: argparse.Namespace) -> None:
    """Execute the train command: resolve the config, delegate to ``trainer.train``.

    Args:
        args: Parsed namespace from ``build_parser``; None means the option was not given.

    Returns:
        None; prints the final checkpoint path.

    Raises:
        ValueError: Invalid configuration or checkpoint (exit code 2 via ``main``).
        FileNotFoundError: Missing data roots or checkpoint (exit code 2).
        RuntimeError: A non-finite loss or unavailable device (exit code 1).
    """
    config = resolve_config(args)
    if args.epochs is not None:
        config = override_config(config, {"training.epochs": args.epochs})
    resume = Path(args.resume) if args.resume is not None else None
    try:
        final = train(config, resume, run_name=args.run_name)
    except FloatingPointError as err:
        raise RuntimeError(str(err)) from err
    print(f"final checkpoint: {final}")


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and run, returning a process exit code (see ``cli.main_guard``)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    return main_guard(lambda: run(args), prog="train")


if __name__ == "__main__":
    raise SystemExit(main())
