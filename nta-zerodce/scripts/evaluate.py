"""Argument parser for evaluate; final commands documented in README.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

import argparse


def run(args: argparse.Namespace) -> None:
    """Execute the evaluate command after resolving validated settings.

    Args:
        args: Parsed CLI namespace; explicit overrides use None when absent.

    Returns:
        None; planned delegation described below; all computation remains stubbed.

    Method:
        INTERFACES CLI: load_config/build_model/load_checkpoint -> paired/unpaired/video metrics -> CSV.

    Raises:
        NotImplementedError: Scaffold only; implement this contract before use.
    """
    raise NotImplementedError(
        "TODO: Execute the evaluate command after resolving validated settings"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="evaluate: interface scaffold; execution raises TODO."
    )
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
    parser.add_argument("--output", default=None)
    parser.add_argument("--input", default=None)
    run(parser.parse_args())
