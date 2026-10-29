"""Executable documented contracts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from nta_zerodce.data.synthetic_pairs import make_pair
from nta_zerodce.flow.estimator import estimate_bidirectional
from nta_zerodce.flow.masks import is_scene_cut, reliability_mask
from nta_zerodce.flow.warp import warp
from nta_zerodce.inference.enhancer import enhance_image
from nta_zerodce.inference.video_enhancer import VideoEnhancer
from nta_zerodce.models.factory import build_model
from nta_zerodce.training.config import Config
from tests.support import small_config


def test_translation_and_affine(tmp_path: Path) -> None:
    """Warp direction, pixel units, bounds and nontranslation inverse geometry."""
    y, x = torch.meshgrid(torch.arange(64), torch.arange(80), indexing="ij")
    image = torch.stack((x / 80, y / 64, (x + y) / 144)).float()[None]
    flow = torch.zeros(1, 2, 64, 80)
    flow[:, 0], flow[:, 1] = 3, -2
    aligned, bounds = warp(image, flow)
    assert torch.allclose(aligned[..., 2:, :-3], image[..., :-2, 3:], atol=1e-6)
    assert bounds.sum() == (64 - 2) * (80 - 3)
    config = small_config(tmp_path)
    config = replace(
        config,
        noise=replace(config.noise, a_max=0.0, b_max=0.0),
        synthetic=replace(config.synthetic, flicker_min=1.0, flicker_max=1.0),
    )
    pair = make_pair(image, config, torch.Generator().manual_seed(123))
    f, b, mask = pair["forward_flow"], pair["backward_flow"], pair["mask"]
    assert not torch.allclose(f, -b)
    warped_f, valid = warp(f, b)
    assert ((b + warped_f).abs() * mask * valid).max() < 2e-5
    reconstructed, _ = warp(image, b)
    assert torch.allclose(reconstructed, pair["image2"], atol=1e-6)
    assert not mask.requires_grad
    reliable = reliability_mask(flow, -flow, image, image, photometric_threshold=1.0)
    assert reliable[..., :3].sum() == 0


def test_dis_adapter() -> None:
    """Identical textured frames give zero CPU detached flow at full resolution."""
    image = torch.rand(1, 3, 48, 64, requires_grad=True)
    f, b = estimate_bidirectional(image, image, Config().flow)
    assert f.shape == b.shape == (1, 2, 48, 64)
    assert f.device.type == "cpu" and not f.requires_grad
    assert f.abs().max() < 1e-3 and b.abs().max() < 1e-3


def test_video_state(tmp_path: Path) -> None:
    """First/cut/shape resets and temporal-disabled equivalence to framewise execution."""
    config = small_config(tmp_path)
    image = torch.full((1, 3, 32, 32), 0.1)
    model = build_model(config)
    enhancer = VideoEnhancer(model, config)
    assert torch.equal(
        enhancer.enhance_frame(image)["enhanced"], enhance_image(model, image)["enhanced"]
    )
    assert enhancer.state.previous_image is None
    config = replace(config, flags=replace(config.flags, temporal=True))
    model = build_model(config)
    enhancer = VideoEnhancer(model, config)
    first = enhancer.enhance_frame(image)
    assert enhancer.state.previous_A is not None and not enhancer.state.previous_A.requires_grad
    second = enhancer.enhance_frame(image)
    assert torch.allclose(first["enhanced"], second["enhanced"], atol=1e-6)
    bright = image + 0.8
    assert is_scene_cut(image, bright)
    assert torch.equal(
        enhancer.enhance_frame(bright)["enhanced"], enhance_image(model, bright)["enhanced"]
    )
    changed = torch.full((1, 3, 32, 48), 0.1)
    assert enhancer.enhance_frame(changed)["A"].shape[-2:] == (32, 48)
    enhancer.reset()
    assert enhancer.state.previous_output is None
