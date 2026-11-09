"""Check candidate/reference evaluation CSVs on the same held-out ordinary photographs.

All three absolute regression limits must be chosen explicitly. This gate never trains,
selects a universal best checkpoint, or changes the deployed model. Exit codes: 0 passes,
1 regression found, 2 invalid/missing data. See docs/ARCHITECTURE.md.
"""

from __future__ import annotations

import argparse
import sys

from nta_zerodce.training.config import resolve_path
from nta_zerodce.training.evaluation import read_rows
from nta_zerodce.training.selection import photo_regressions


def main(argv: list[str] | None = None) -> int:
    """Compare one candidate to a reference using user-specified acceptance limits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="Reference checkpoint evaluation CSV")
    parser.add_argument("--candidate", required=True, help="Candidate checkpoint evaluation CSV")
    parser.add_argument("--dataset", required=True, help="Exact held-out CSV dataset name")
    parser.add_argument("--max-brightness-increase", type=float, required=True)
    parser.add_argument("--max-contrast-drop", type=float, required=True)
    parser.add_argument("--max-clipping-increase", type=float, required=True)
    args = parser.parse_args(argv)
    try:
        failures = photo_regressions(
            read_rows(resolve_path(args.reference)),
            read_rows(resolve_path(args.candidate)),
            dataset=args.dataset,
            max_brightness_increase=args.max_brightness_increase,
            max_contrast_drop=args.max_contrast_drop,
            max_clipping_increase=args.max_clipping_increase,
        )
    except (OSError, ValueError) as error:
        print(f"check_photo_regressions: {error}", file=sys.stderr)
        return 2
    for failure in failures:
        print(
            f"FAIL {failure.sample_id}: {failure.metric} "
            f"{failure.reference:.6f} -> {failure.candidate:.6f}; "
            f"change {failure.change:.6f} exceeds {failure.limit:.6f}"
        )
    if failures:
        print(f"Rejected: {len(failures)} measurement(s) exceed the supplied limits.")
        return 1
    print("Passed supplied regression limits. Visual/detail review is still required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
