"""One training update; pseudocode follows.

1. Read unpaired RGB x, shared crop; synthesize independent n when noise-aware.
2. Forward E(x+n), obtain enhanced/raw/A/sigma; form detached gain and masks.
3. Compute spa/exp-or-adaptive/col/tv; gate edge/chroma/highlight by flags.
4. Under no_grad forward E(x): teacher enhanced for invariance, raw for fidelity.
5. Add synthetic sigma supervision, invariance and denoise fidelity where enabled.
6. If temporal, forward pair; obtain detached DIS or analytic bidirectional flow.
7. Build current-coordinate mask; add temporal loss on enhanced pair outputs.
8. Sum weighted scalar terms; zero_grad, backward, clip global norm to .1, step.
9. Return detached scalar logs; log zero-valued inactive terms explicitly.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from torch import Tensor
from torch.optim import Optimizer

from nta_zerodce.models.nta_zerodce import NTAZeroDCE
from nta_zerodce.training.config import Config


def train_step(
    model: NTAZeroDCE, batch: dict[str, Tensor], optimizer: Optimizer, config: Config
) -> dict[str, float]:
    """Perform exactly one gated optimization step.

    Args:
        model: Train-mode model on batch device.
        batch: image (B,3,H,W) in [0,1]; temporal fields as INTERFACES specifies.
        optimizer: Adam bound to model parameters.
        config: Validated flags/weights and gradient norm clip.

    Returns:
        Detached Python scalars: total and every loss name, finite for valid batches.

    Method:
        METHOD objective; teacher/masks/gain/targets detached, all active student paths differentiable.

    Raises:
        NotImplementedError: Scaffold only; implement this contract before use.
    """
    raise NotImplementedError("TODO: Perform exactly one gated optimization step")
