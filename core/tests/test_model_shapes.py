"""Executable documented contracts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from nta_zerodce.inference.enhancer import enhance_image
from nta_zerodce.models.factory import build_model
from tests.support import small_config


@pytest.mark.parametrize(
    "full,light,scale", [(False, False, 1.0), (True, False, 0.5), (True, True, 0.25)]
)
def test_model_contract(tmp_path: Path, full: bool, light: bool, scale: float) -> None:
    """Full-resolution outputs, bounds, exact disabled branches and trainable heads."""
    config = small_config(tmp_path, full)
    config = replace(
        config,
        flags=replace(config.flags, light=light),
        model=replace(config.model, curve_scale=scale),
    )
    model = build_model(config)
    x = torch.rand(2, 3, 32, 48)
    output = model(x)
    assert set(output) == {"enhanced", "enhanced_raw", "A", "sigma"}
    assert output["A"].shape == (2, 24, 32, 48)
    assert output["sigma"].shape == (2, 1, 32, 48)
    assert output["enhanced"].shape == x.shape
    for value in output.values():
        assert torch.isfinite(value).all() and value.device == x.device
    assert output["A"].abs().max() <= 1
    assert 0 <= output["enhanced"].min() <= output["enhanced"].max() <= 1
    assert 0 <= output["sigma"].min() <= output["sigma"].max() <= config.model.sigma_max
    if not full:
        assert torch.count_nonzero(output["sigma"]) == 0
        assert torch.equal(output["enhanced"], output["enhanced_raw"])
    (output["enhanced"].mean() + output["sigma"].mean()).backward()
    for name, branch in (
        ("curves", model.estimator),
        ("noise", model.noise),
        ("denoise", model.denoiser),
    ):
        if branch is not None:
            assert any(
                p.grad is not None and p.grad.abs().sum() > 0 for p in branch.parameters()
            ), name
    predicted = enhance_image(model, x, 0.5)
    assert not predicted["enhanced"].requires_grad
    assert model.config == config
