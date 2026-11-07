"""Load a trained model for inference from a checkpoint.

Shared by the enhance and evaluate CLIs and the HTTP API so checkpoint, config and device
handling live in one place.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch

from nta_zerodce.models.factory import build_model
from nta_zerodce.models.nta_zerodce import NTAZeroDCE
from nta_zerodce.training.checkpoint import checkpoint_config, load_checkpoint
from nta_zerodce.training.config import Config, load_config, override_config
from nta_zerodce.utils.device import resolve_device


def load_inference_model(
    checkpoint: Path,
    config_path: str | Path | None = None,
    overrides: Mapping[str, object] | None = None,
    device: str | torch.device = "auto",
) -> tuple[NTAZeroDCE, Config, torch.device]:
    """Build, load and freeze a model in eval mode on the requested device.

    Args:
        checkpoint: Trusted local checkpoint file.
        config_path: Optional experiment YAML. When omitted, the resolved config stored in the
            checkpoint is used, so no architecture guess is needed. When given, its
            architecture flags and shapes must match the checkpoint.
        overrides: Dotted overrides (``None`` values mean absent) applied on top of the config.
        device: ``auto``, ``cpu`` or ``cuda[:N]`` (or a ``torch.device``). The device stored in
            the config (``training.device``) is deliberately ignored: it describes the training
            machine, not the machine running inference.

    Returns:
        ``(model, config, torch_device)``; the model is in eval mode with gradients disabled.

    Raises:
        FileNotFoundError: Missing checkpoint or config file.
        ValueError: Invalid checkpoint, config, or an architecture mismatch.
        RuntimeError: The requested CUDA device is unavailable.
    """
    checkpoint = Path(checkpoint)
    if config_path is None:
        config = override_config(checkpoint_config(checkpoint), overrides)
    else:
        config = load_config(config_path, overrides)
    torch_device = resolve_device(str(device))
    model = build_model(config)
    load_checkpoint(checkpoint, model, optimizer=None, expected_config=config)
    model.to(torch_device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, config, torch_device
