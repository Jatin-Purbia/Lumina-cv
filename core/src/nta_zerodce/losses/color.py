"""Pure tensor losses/targets; no IO, OpenCV, optimizer or registry side effects.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor


def color_loss(output: Tensor) -> Tensor:
    """Penalize unequal global RGB channel means.

    Args:
        output: RGB float32 tensor (B,3,H,W) in [0,1], on the input device.

    Returns:
        Scalar float32 nonnegative differentiable Tensor on the input device.

    Method:
        Baseline L_col = mean_batch sqrt(d_RG^4+d_RB^4+d_GB^4). See docs/METHOD.md; baseline attribution [R1,R2], extensions are project proposals.

    Raises:
        ValueError: Invalid inputs or incompatible configuration.
    """
    means = output.mean((2, 3))
    diffs = torch.stack(
        (means[:, 0] - means[:, 1], means[:, 0] - means[:, 2], means[:, 1] - means[:, 2]), 1
    )
    # Vector norm avoids the undefined sqrt derivative at perfectly gray input.
    return cast(Tensor, torch.linalg.vector_norm(diffs.square(), dim=1)).mean()
