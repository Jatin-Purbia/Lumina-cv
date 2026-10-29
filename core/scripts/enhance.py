"""Argument parser for enhance; final commands documented in README.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

import argparse

from nta_zerodce.cli import dispatch


def run(args: argparse.Namespace) -> None:
    """Execute the enhance command after resolving validated settings.

    Args:
        args: Parsed CLI namespace; explicit overrides use None when absent.

    Returns:
        None; delegates to the documented local core workflow.

    Method:
        INTERFACES CLI: load_config/build_model/load_checkpoint -> image IO, VideoEnhancer or run_webcam.

    Raises:
        ValueError: Invalid inputs or incompatible configuration.
    """
    dispatch("enhance", args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="enhance: Noise- and Temporal-Aware Zero-DCE")
    parser.add_argument("--config", default="configs/baseline.yaml", help="Experiment YAML")
    parser.add_argument("--noise", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--denoise", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--edge", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--chroma", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--adaptive-exp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--temporal", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--light", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--curve-scale", type=float, default=None)
    parser.add_argument("--video-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--webcam", type=int, default=None, help="Camera index; incompatible with --input"
    )
    run(parser.parse_args())
