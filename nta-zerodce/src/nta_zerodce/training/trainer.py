"""Training orchestration and reproducible runs, not model computation.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from pathlib import Path

from nta_zerodce.training.config import Config


def train(config: Config, resume: Path | None = None) -> Path:
    """Run the configured training schedule.

    Args:
        config: Validated Config with dataset paths and gated objectives.
        resume: Optional checkpoint; exact flag/schema compatibility required.

    Returns:
        Final checkpoint Path; persist resolved config, seed and metrics CSV.

    Method:
        EXPERIMENTS 100 epochs, Adam, fixed seeds; loop_step owns updates.

    Raises:
        NotImplementedError: Scaffold only; implement this contract before use.
    """
    raise NotImplementedError("TODO: Run the configured training schedule")
