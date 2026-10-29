"""Reproducible CPU fixtures."""

from __future__ import annotations

import pytest
import torch

from nta_zerodce.utils.seed import seed_everything


@pytest.fixture(autouse=True)
def cpu_seed() -> None:
    """Bound test runtime and isolate numerical fixtures."""
    torch.set_num_threads(2)
    seed_everything(17)
