"""Shared command-line layer for every script.

Adds the common INTERFACES flags to an argparse parser, maps them to dotted config overrides
(None means "not supplied", never false), resolves a validated Config and converts expected
failures into one-line messages with stable exit codes. Importing this module has no side effects.

See docs/ARCHITECTURE.md for the normative CLI contract.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path

from nta_zerodce.training.config import Config, FeatureFlags, parse_config

EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_USAGE = 2

ARG_TO_FIELD: dict[str, str] = {
    "noise": "flags.noise_aware",
    "denoise": "flags.denoise",
    "edge": "flags.edge",
    "chroma": "flags.chroma",
    "adaptive_exp": "flags.adaptive_exp",
    "temporal": "flags.temporal",
    "light": "flags.light",
    "curve_scale": "model.curve_scale",
    "video_dir": "paths.video_dir",
    "seed": "training.seed",
    "device": "training.device",
}
"""argparse destination name to dotted config field (INTERFACES CLI mapping)."""

TRACEBACK_ENV = "LUMINA_TRACEBACK"
"""Set to a truthy value to let main_guard re-raise instead of printing one line."""


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the flags shared by every script.

    Args:
        parser: Parser to extend in place.

    Returns:
        None. Adds --config (default configs/baseline.py), the boolean pairs --noise/--no-noise,
        --denoise, --edge, --chroma, --adaptive-exp, --temporal and --light (all default None, so
        absence preserves the YAML value), --curve-scale (float), --video-dir (str), --seed (int)
        and --device (str).

    Method:
        INTERFACES CLI contract using argparse.BooleanOptionalAction.

    Raises:
        argparse.ArgumentError: A flag name is already defined on the parser.
    """
    parser.add_argument(
        "--config",
        default="configs/baseline.py",
        help="Experiment config (.py with CONFIG, or .yaml)",
    )
    toggles = {
        "--noise": "noise-aware sigma branch (flags.noise_aware)",
        "--denoise": "residual denoiser; requires noise (flags.denoise)",
        "--edge": "edge-preservation loss (flags.edge)",
        "--chroma": "chroma-consistency loss (flags.chroma)",
        "--adaptive-exp": "adaptive exposure target (flags.adaptive_exp)",
        "--temporal": "temporal consistency (flags.temporal)",
        "--light": "lightweight network (flags.light)",
    }
    for flag, text in toggles.items():
        parser.add_argument(
            flag,
            action=argparse.BooleanOptionalAction,
            default=None,
            help=f"Enable or disable {text}; omitted keeps the YAML value",
        )
    parser.add_argument(
        "--curve-scale", type=float, default=None, help="Curve resolution scale in (0, 1]"
    )
    parser.add_argument("--video-dir", type=str, default=None, help="Override paths.video_dir")
    parser.add_argument("--seed", type=int, default=None, help="Override training.seed")
    parser.add_argument(
        "--device", type=str, default=None, help="auto, cpu, cuda or cuda:N (training.device)"
    )


def overrides_from_args(args: argparse.Namespace) -> dict[str, object]:
    """Map parsed shared flags to dotted config overrides.

    Args:
        args: Namespace produced by a parser extended with add_common_arguments.

    Returns:
        One entry per shared flag keyed by its dotted field (flags.noise_aware, model.curve_scale,
        paths.video_dir, training.seed, training.device, ...). A value of None means the flag
        was not supplied; an explicit --no-x is False and overrides YAML true.

    Method:
        INTERFACES: None is absent, not false.

    Raises:
        None.
    """
    return {field: getattr(args, dest, None) for dest, field in ARG_TO_FIELD.items()}


def flag_conflict(flags: FeatureFlags, args: argparse.Namespace | None = None) -> str | None:
    """Describe an invalid flag combination after CLI and YAML are merged.

    Args:
        flags: Resolved flags (not yet validated).
        args: Optional parsed namespace, used to name the responsible command-line option.

    Returns:
        None when consistent, else a one-line message. The only dependency is that denoise
        needs noise; the message says whether --no-noise or the YAML caused it.

    Method:
        denoise implies noise_aware (INTERFACES flags row).

    Raises:
        None.
    """
    if flags.denoise is True and flags.noise_aware is False:
        source = ""
        if args is not None and getattr(args, "noise", None) is False:
            source = " (--no-noise was given)"
        return (
            "denoise requires noise-awareness but noise is disabled" + source + "; "
            "drop --no-noise, add --no-denoise, or set flags.noise_aware: true in the YAML"
        )
    return None


def resolve_config(args: argparse.Namespace) -> Config:
    """Load the YAML named by --config, apply CLI overrides and validate.

    Args:
        args: Namespace from a parser extended with add_common_arguments.

    Returns:
        Validated immutable Config. Precedence: schema defaults < complete YAML < explicit CLI.

    Method:
        parse_config, then the flag-conflict explanation, then Config.validate.

    Raises:
        FileNotFoundError: The YAML does not exist.
        ValueError: Parse, unknown/missing key, flag conflict or bound violation.
    """
    config = parse_config(Path(args.config), overrides_from_args(args))
    message = flag_conflict(config.flags, args)
    if message is not None:
        raise ValueError(message)
    config.validate()
    return config


def main_guard(run: Callable[[], int | None], *, prog: str | None = None) -> int:
    """Run a script body and convert expected failures into an exit code.

    Args:
        run: Zero-argument callable returning an exit code or None (treated as 0).
        prog: Name used as the message prefix; defaults to the basename of sys.argv[0].

    Returns:
        0 on success. 2 for usage/config errors (ValueError, FileNotFoundError), 1 for runtime
        errors (NotImplementedError, RuntimeError, OSError), 130 on Ctrl-C. Each failure prints a
        single ``prog: error: message`` line to stderr and no traceback. Set the environment
        variable LUMINA_TRACEBACK=1 to re-raise for debugging. SystemExit (argparse) passes
        through; other exceptions are bugs and propagate.

    Method:
        Ordered except clauses; FileNotFoundError is matched before OSError.

    Raises:
        Exception: Unexpected exception types, and any exception when LUMINA_TRACEBACK is set.
    """
    name = prog if prog is not None else Path(sys.argv[0] or "lumina").name
    debug = os.environ.get(TRACEBACK_ENV, "").lower() in {"1", "true", "yes"}
    try:
        code = run()
    except KeyboardInterrupt:
        print(f"{name}: interrupted", file=sys.stderr)
        return 130
    except (ValueError, FileNotFoundError) as err:
        if debug:
            raise
        print(f"{name}: error: {_one_line(err)}", file=sys.stderr)
        return EXIT_USAGE
    except (NotImplementedError, RuntimeError, OSError) as err:
        if debug:
            raise
        print(f"{name}: error: {_one_line(err)}", file=sys.stderr)
        return EXIT_RUNTIME
    return EXIT_OK if code is None else int(code)


def _one_line(err: BaseException) -> str:
    """Collapse an exception message to a single line (with the type for empty messages)."""
    text = " ".join(str(err).split())
    return text or type(err).__name__
