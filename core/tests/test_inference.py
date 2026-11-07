"""Inference: curve-scale enhancement, checkpoint loader, VideoEnhancer and the webcam loop.

CPU only, seeded, tiny frames and a random-initialised model (no training, data or camera).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import torch
from torch import Tensor

from nta_zerodce.flow.estimator import estimate_bidirectional
from nta_zerodce.flow.masks import reliability_mask
from nta_zerodce.flow.warp import warp
from nta_zerodce.inference.enhancer import enhance_image, estimate_full_res_maps
from nta_zerodce.inference.loader import load_inference_model
from nta_zerodce.inference.video_enhancer import VideoEnhancer, VideoState
from nta_zerodce.inference.webcam import bgr_to_tensor, run_webcam, tensor_to_bgr
from nta_zerodce.models.factory import build_model
from nta_zerodce.models.nta_zerodce import ModelOutput, NTAZeroDCE
from nta_zerodce.training.checkpoint import save_checkpoint
from nta_zerodce.training.config import Config, override_config, save_config

SMALL: dict[str, object] = {
    "model.channels": 8,
    "model.noise_channels": 4,
    "model.denoise_channels": 4,
}
FULL: dict[str, object] = {
    **SMALL,
    "flags.noise_aware": True,
    "flags.denoise": True,
    "flags.temporal": True,
}
HEIGHT, WIDTH, SHIFT = 48, 64, 2


def make_config(**overrides: object) -> Config:
    """Small validated config; keyword keys use ``__`` instead of ``.`` (flags__temporal)."""
    merged = {**SMALL, **{key.replace("__", "."): value for key, value in overrides.items()}}
    return override_config(Config(), merged)


def make_model(config: Config, seed: int = 0, head_gain: float = 1.0) -> NTAZeroDCE:
    """Random-initialised, frozen, eval-mode model."""
    torch.manual_seed(seed)
    model = build_model(config)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if head_gain != 1.0:
        with torch.no_grad():
            model.dce.head.weight.mul_(head_gain)
    return model


def make_canvas(seed: int = 1, extra: int = 0) -> Tensor:
    """Smooth, textured, full-contrast canvas (1,3,H,W+extra); broad histogram avoids false cuts."""
    generator = torch.Generator().manual_seed(seed)
    wide = WIDTH + extra
    coarse = torch.rand(1, 3, 12, wide // 4 + 1, generator=generator)
    canvas = torch.nn.functional.interpolate(
        coarse, size=(HEIGHT, wide), mode="bilinear", align_corners=False
    )
    canvas = 0.5 * canvas.mean(1, keepdim=True) + 0.5 * canvas
    canvas = (canvas - canvas.min()) / (canvas.max() - canvas.min())
    fine = 0.02 * torch.rand(1, 3, HEIGHT, wide, generator=generator)
    return (0.05 + 0.55 * canvas + fine).clamp(0.0, 1.0)


def make_sequence(count: int, flicker: list[float] | None = None) -> list[Tensor]:
    """Frames of a canvas translating left by SHIFT px per frame (content moves, camera pans)."""
    canvas = make_canvas(extra=SHIFT * count)
    factors = flicker if flicker is not None else [1.0] * count
    return [
        (canvas[..., t * SHIFT : t * SHIFT + WIDTH] * factors[t]).clamp(0, 1).contiguous()
        for t in range(count)
    ]


def warp_aligned_error(outputs: list[Tensor]) -> float:
    """Mean |o_t - warp(o_{t-1}, b)| with the exact backward flow b=(+SHIFT, 0) on valid columns."""
    backward = torch.zeros(1, 2, HEIGHT, WIDTH)
    backward[:, 0] = SHIFT
    errors = []
    for previous, current in zip(outputs[:-1], outputs[1:], strict=True):
        warped, _ = warp(previous, backward)
        errors.append((warped - current)[..., : WIDTH - SHIFT - 1].abs().mean().item())
    return float(sum(errors) / len(errors))


# ----------------------------------------------------------------------- enhance_image


@pytest.mark.parametrize("overrides", [{}, FULL, {"flags.light": True}])
def test_enhance_image_scale_one_equals_forward(overrides: dict[str, object]) -> None:
    config = make_config(**overrides)
    model = make_model(config)
    image = make_canvas()
    result = enhance_image(model, image, 1.0)
    with torch.no_grad():
        expected = model(image)
    for key in ("enhanced", "enhanced_raw", "A", "sigma"):
        assert torch.equal(result[key], expected[key]), key


@pytest.mark.parametrize("scale", [1.0, 0.5, 0.25])
def test_enhance_image_full_resolution_range_and_no_grad(scale: float) -> None:
    model = make_model(make_config(**FULL))
    image = make_canvas()
    image = torch.cat([image, image[..., :16]], dim=3)  # 48x80, non-square
    result = enhance_image(model, image, scale)
    assert result["enhanced"].shape == image.shape
    assert result["A"].shape == (1, 24, 48, 80) and result["sigma"].shape == (1, 1, 48, 80)
    for tensor in (result["enhanced"], result["enhanced_raw"], result["A"], result["sigma"]):
        assert not tensor.requires_grad and tensor.grad_fn is None
        assert torch.isfinite(tensor).all()
    assert 0.0 <= float(result["enhanced"].min()) and float(result["enhanced"].max()) <= 1.0
    assert float(result["A"].abs().max()) <= 1.0 and float(result["sigma"].min()) >= 0.0


def test_reduced_scale_estimates_on_a_small_image_but_renders_full_resolution() -> None:
    model = make_model(make_config())
    seen: list[tuple[int, ...]] = []
    original = model.estimate_maps

    def spy(image: Tensor) -> tuple[Tensor, Tensor]:
        seen.append(tuple(image.shape[2:]))
        return original(image)

    model.estimate_maps = spy  # type: ignore[method-assign]
    image = make_canvas()
    enhance_image(model, image, 0.5)
    enhance_image(model, image, 0.25)
    assert seen == [(24, 32), (16, 16)]  # sides are max(16, round(side * scale))


def test_estimate_full_res_maps_shapes_and_sigma_zero_without_noise_branch() -> None:
    image = make_canvas()
    a_map, sigma = estimate_full_res_maps(make_model(make_config()), image, 0.5)
    assert a_map.shape == (1, 24, HEIGHT, WIDTH) and sigma.shape == (1, 1, HEIGHT, WIDTH)
    assert float(sigma.abs().max()) == 0.0
    noisy = make_model(make_config(flags__noise_aware=True))
    a_map, sigma = estimate_full_res_maps(noisy, image.expand(2, -1, -1, -1), 0.5)
    assert a_map.shape[0] == 2 and float(sigma.min()) >= 0.0 and float(sigma.max()) > 0.0
    assert not a_map.requires_grad and not sigma.requires_grad


def test_enhance_image_accepts_non_contiguous_input() -> None:
    model = make_model(make_config())
    image = make_canvas()
    view = image.transpose(2, 3).contiguous().transpose(2, 3)  # same values, strided layout
    assert not view.is_contiguous()
    assert torch.equal(
        enhance_image(model, view)["enhanced"], enhance_image(model, image)["enhanced"]
    )


@pytest.mark.parametrize(
    ("image", "scale"),
    [
        (torch.rand(3, 32, 32), 1.0),
        (torch.rand(1, 1, 32, 32), 1.0),
        (torch.rand(1, 3, 15, 32), 1.0),
        (torch.rand(1, 3, 32, 8), 0.5),
        (torch.rand(1, 3, 32, 32).double(), 1.0),
        ((torch.rand(1, 3, 32, 32) * 255).to(torch.uint8), 1.0),
        (torch.rand(0, 3, 32, 32), 1.0),
        (torch.rand(1, 3, 32, 32), 0.0),
        (torch.rand(1, 3, 32, 32), 1.5),
        (torch.rand(1, 3, 32, 32), -0.5),
        (torch.rand(1, 3, 32, 32), float("nan")),
        (torch.rand(1, 3, 32, 32), True),
    ],
)
def test_enhance_image_rejects_invalid_inputs(image: Tensor, scale: float) -> None:
    model = make_model(make_config())
    with pytest.raises(ValueError):
        enhance_image(model, image, scale)
    with pytest.raises(ValueError):
        estimate_full_res_maps(model, image, scale)


def test_enhance_image_rejects_non_tensor_and_minimum_size_is_accepted() -> None:
    model = make_model(make_config())
    with pytest.raises(ValueError):
        enhance_image(model, np.zeros((1, 3, 32, 32), dtype=np.float32), 1.0)
    out = enhance_image(model, torch.rand(1, 3, 16, 16), 0.25)
    assert out["enhanced"].shape == (1, 3, 16, 16)


# ----------------------------------------------------------------------- loader


def write_checkpoint(path: Path, config: Config, seed: int = 0) -> NTAZeroDCE:
    model = make_model(config, seed)
    save_checkpoint(path, model, torch.optim.Adam(model.parameters()), config, 0)
    return model


def test_loader_uses_stored_config_and_freezes_the_model(tmp_path: Path) -> None:
    config = make_config(**FULL)
    original = write_checkpoint(tmp_path / "a.pt", config)
    model, loaded, device = load_inference_model(tmp_path / "a.pt")
    assert loaded == config and device.type == "cpu"
    assert not model.training
    assert all(not p.requires_grad for p in model.parameters())
    image = make_canvas()
    assert torch.equal(
        enhance_image(model, image)["enhanced"], enhance_image(original, image)["enhanced"]
    )


def test_loader_explicit_config_path_and_overrides(tmp_path: Path) -> None:
    config = make_config(flags__noise_aware=True)
    write_checkpoint(tmp_path / "a.pt", config)
    save_config(config, tmp_path / "exp.yaml")
    model, loaded, _ = load_inference_model(
        tmp_path / "a.pt",
        tmp_path / "exp.yaml",
        {"model.curve_scale": 0.5, "flags.temporal": True, "training.seed": None},
    )
    assert loaded.model.curve_scale == 0.5 and loaded.flags.temporal is True
    assert loaded.training.seed == config.training.seed  # None means absent
    assert model.noise_aware
    _, stored, _ = load_inference_model(tmp_path / "a.pt", None, {"model.curve_scale": 0.25})
    assert stored.model.curve_scale == 0.25 and stored.flags == config.flags


def test_loader_accepts_a_torch_device_and_a_str_path(tmp_path: Path) -> None:
    write_checkpoint(tmp_path / "a.pt", make_config())
    _, _, device = load_inference_model(str(tmp_path / "a.pt"), device=torch.device("cpu"))  # type: ignore[arg-type]
    assert device == torch.device("cpu")


def test_loader_rejects_architecture_mismatch(tmp_path: Path) -> None:
    config = make_config()
    write_checkpoint(tmp_path / "a.pt", config)
    with pytest.raises(ValueError, match="light"):
        load_inference_model(tmp_path / "a.pt", None, {"flags.light": True})
    with pytest.raises(ValueError, match="noise_aware"):
        load_inference_model(tmp_path / "a.pt", None, {"flags.noise_aware": True})
    save_config(make_config(flags__noise_aware=True), tmp_path / "noise.yaml")
    with pytest.raises(ValueError, match="noise_aware"):
        load_inference_model(tmp_path / "a.pt", tmp_path / "noise.yaml")
    wide = override_config(config, {"model.channels": 16})
    save_config(wide, tmp_path / "wide.yaml")
    with pytest.raises(ValueError, match="channels"):
        load_inference_model(tmp_path / "a.pt", tmp_path / "wide.yaml")
    with pytest.raises(ValueError):  # denoise without noise fails config validation
        load_inference_model(tmp_path / "a.pt", None, {"flags.denoise": True})


def test_loader_reports_missing_and_corrupt_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_inference_model(tmp_path / "missing.pt")
    with pytest.raises(FileNotFoundError):
        load_inference_model(tmp_path)  # a directory is not a checkpoint
    (tmp_path / "bad.pt").write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError):
        load_inference_model(tmp_path / "bad.pt")
    write_checkpoint(tmp_path / "a.pt", make_config())
    with pytest.raises(FileNotFoundError):
        load_inference_model(tmp_path / "a.pt", tmp_path / "nope.yaml")
    with pytest.raises(ValueError):
        load_inference_model(tmp_path / "a.pt", device="tpu")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            load_inference_model(tmp_path / "a.pt", device="cuda")


# ----------------------------------------------------------------------- VideoEnhancer


def temporal_config(**overrides: object) -> Config:
    return make_config(flags__temporal=True, **overrides)


def test_temporal_off_equals_framewise_and_keeps_no_state() -> None:
    config = make_config(**{**FULL, "flags.temporal": False})
    model = make_model(config)
    enhancer = VideoEnhancer(model, config)
    for frame in make_sequence(3):
        result = enhancer.enhance_frame(frame)
        expected = enhance_image(model, frame, config.model.curve_scale)
        assert torch.equal(result["enhanced"], expected["enhanced"])
        assert torch.equal(result["A"], expected["A"])
        assert enhancer.state == VideoState(None, None, None)
    assert enhancer.frames_processed == 3
    assert enhancer.last_mask_coverage is None and enhancer.last_was_reset


def test_temporal_off_honours_curve_scale() -> None:
    config = make_config(model__curve_scale=0.5)
    model = make_model(config)
    frame = make_sequence(1)[0]
    result = VideoEnhancer(model, config).enhance_frame(frame)
    assert torch.equal(result["enhanced"], enhance_image(model, frame, 0.5)["enhanced"])


def test_first_frame_equals_framewise_and_state_is_stored_detached() -> None:
    config = temporal_config(flags__noise_aware=True, flags__denoise=True)
    model = make_model(config)
    enhancer = VideoEnhancer(model, config)
    frame = make_sequence(1)[0]
    result = enhancer.enhance_frame(frame)
    expected = enhance_image(model, frame)
    assert torch.equal(result["enhanced"], expected["enhanced"])
    assert enhancer.last_was_reset and enhancer.last_mask_coverage is None
    state = enhancer.state
    assert state.previous_image is not None and state.previous_A is not None
    assert state.previous_output is not None
    for tensor in (state.previous_image, state.previous_A, state.previous_output):
        assert not tensor.requires_grad and tensor.grad_fn is None
        assert tensor.device == frame.device
    assert state.previous_image.shape == (1, 3, HEIGHT, WIDTH)
    assert state.previous_A.shape == (1, 24, HEIGHT, WIDTH)
    assert torch.equal(state.previous_A, result["A"])  # stores the smoothed map
    frame.zero_()  # the state holds a copy of the frame
    assert float(state.previous_image.max()) > 0.0


def test_subsequent_frames_match_the_documented_blend() -> None:
    """A_bar = (1 - 0.7 m) A_t + 0.7 m warp(A_bar_prev, b), recomputed from public primitives."""
    config = temporal_config(flags__noise_aware=True)
    model = make_model(config, head_gain=10.0)
    frames = make_sequence(4)
    enhancer = VideoEnhancer(model, config)
    enhancer.enhance_frame(frames[0])
    for current in frames[1:]:
        previous_image = enhancer.state.previous_image
        previous_a = enhancer.state.previous_A
        assert previous_image is not None and previous_a is not None
        previous_a = previous_a.clone()
        result = enhancer.enhance_frame(current)
        forward, backward = estimate_bidirectional(previous_image, current, config.flow)
        mask = reliability_mask(
            forward,
            backward,
            previous_image,
            current,
            config.flow.fb_alpha,
            config.flow.fb_beta,
            config.flow.photometric_threshold,
        )
        warped, inside = warp(previous_a, backward)
        a_t, sigma_t = estimate_full_res_maps(model, current)
        weight = config.temporal.curve_blend * mask * inside
        expected_a = (1 - weight) * a_t + weight * warped
        assert torch.allclose(result["A"], expected_a, atol=1e-6)
        assert float(result["A"].min()) >= -1.0 and float(result["A"].max()) <= 1.0
        assert torch.equal(result["sigma"], sigma_t)
        rendered = model.apply_maps(current, expected_a, sigma_t)
        assert torch.allclose(result["enhanced"], rendered["enhanced"], atol=1e-6)
        assert enhancer.last_mask_coverage == pytest.approx(float(mask.mean()))
        assert 0.2 < float(mask.mean()) <= 1.0 and not enhancer.last_was_reset
        assert not torch.allclose(result["A"], a_t)  # smoothing actually happened
        assert enhancer.frames_processed >= 2


def test_blend_arithmetic_with_injected_maps_on_a_static_scene() -> None:
    """Static frames give zero flow and a full mask, so A_bar = 0.3 A_t + 0.7 A_prev exactly."""
    config = temporal_config()
    model = make_model(config)
    levels = iter([0.8, -0.4, 0.2])

    def constant_maps(image: Tensor) -> tuple[Tensor, Tensor]:
        shape = (image.shape[0], 24, image.shape[2], image.shape[3])
        return torch.full(shape, next(levels)), torch.zeros_like(image[:, :1])

    model.estimate_maps = constant_maps  # type: ignore[method-assign]
    frame = make_sequence(1)[0]
    enhancer = VideoEnhancer(model, config)
    enhancer.enhance_frame(frame)
    second = enhancer.enhance_frame(frame)
    assert enhancer.last_mask_coverage == pytest.approx(1.0, abs=0.02)
    inner = second["A"][..., 2:-2, 2:-2]
    assert torch.allclose(inner, torch.full_like(inner, 0.3 * -0.4 + 0.7 * 0.8), atol=1e-4)
    third = enhancer.enhance_frame(frame)
    inner = third["A"][..., 2:-2, 2:-2]
    assert torch.allclose(
        inner, torch.full_like(inner, 0.3 * 0.2 + 0.7 * (0.3 * -0.4 + 0.56)), atol=1e-3
    )


def test_zero_curve_blend_reduces_to_framewise_inference() -> None:
    config = temporal_config(temporal__curve_blend=0.0)
    model = make_model(config)
    enhancer = VideoEnhancer(model, config)
    for frame in make_sequence(3):
        result = enhancer.enhance_frame(frame)
        assert torch.equal(result["enhanced"], enhance_image(model, frame)["enhanced"])


def test_output_blend_matches_the_documented_formula() -> None:
    config = temporal_config(
        temporal__output_blend=0.5, flags__noise_aware=True, flags__denoise=True
    )
    model = make_model(config, head_gain=10.0)
    frames = make_sequence(3)
    enhancer = VideoEnhancer(model, config)
    enhancer.enhance_frame(frames[0])
    for current in frames[1:]:
        previous_image = enhancer.state.previous_image
        previous_output = enhancer.state.previous_output
        assert previous_image is not None and previous_output is not None
        previous_output = previous_output.clone()
        result = enhancer.enhance_frame(current)
        forward, backward = estimate_bidirectional(previous_image, current, config.flow)
        mask = reliability_mask(forward, backward, previous_image, current)
        warped, inside = warp(previous_output, backward)
        rendered = model.apply_maps(current, result["A"], result["sigma"])
        weight = 0.5 * mask * inside
        expected = (1 - weight) * rendered["enhanced"] + weight * warped
        assert torch.allclose(result["enhanced"], expected, atol=1e-6)
        assert torch.allclose(result["enhanced_raw"], rendered["enhanced_raw"], atol=1e-6)
        assert float(result["enhanced"].min()) >= 0.0 and float(result["enhanced"].max()) <= 1.0
        assert torch.equal(enhancer.state.previous_output, result["enhanced"])
    # blend 0 (default) leaves the output exactly the rendering of the smoothed curves
    plain = VideoEnhancer(model, temporal_config(flags__noise_aware=True, flags__denoise=True))
    plain.enhance_frame(frames[0])
    out = plain.enhance_frame(frames[1])
    rendered = model.apply_maps(frames[1], out["A"], out["sigma"])
    assert torch.equal(out["enhanced"], rendered["enhanced"])


def test_scene_cut_resets_state_and_the_new_scene_is_smoothed_again() -> None:
    config = temporal_config()
    model = make_model(config)
    frames = make_sequence(5)
    enhancer = VideoEnhancer(model, config)
    enhancer.enhance_frame(frames[0])
    enhancer.enhance_frame(frames[1])
    assert not enhancer.last_scene_cut and not enhancer.last_was_reset
    swapped = 1.0 - frames[2]  # brightness swapped: histogram mirrored
    result = enhancer.enhance_frame(swapped)
    assert enhancer.last_scene_cut and enhancer.last_was_reset
    assert enhancer.last_mask_coverage is None
    framewise = enhance_image(model, swapped)
    assert torch.equal(result["A"], framewise["A"]) and torch.equal(
        result["enhanced"], framewise["enhanced"]
    )
    assert enhancer.state.previous_image is not None
    assert torch.equal(enhancer.state.previous_image, swapped)  # new scene becomes the history
    enhancer.enhance_frame(1.0 - frames[3])  # same scene continues: history is used again
    assert not enhancer.last_scene_cut and not enhancer.last_was_reset
    assert enhancer.last_mask_coverage is not None and enhancer.last_mask_coverage > 0.2


def test_scene_cut_threshold_is_configurable() -> None:
    config = temporal_config(temporal__scene_cut_threshold=1.0)  # distance can never exceed 1
    model = make_model(config)
    frames = make_sequence(3)
    enhancer = VideoEnhancer(model, config)
    enhancer.enhance_frame(frames[0])
    enhancer.enhance_frame(1.0 - frames[1])
    assert not enhancer.last_scene_cut


def test_size_change_and_explicit_reset_clear_all_history() -> None:
    config = temporal_config()
    model = make_model(config)
    frames = make_sequence(3)
    enhancer = VideoEnhancer(model, config)
    enhancer.enhance_frame(frames[0])
    enhancer.enhance_frame(frames[1])
    assert not enhancer.last_was_reset
    smaller = frames[2][..., :40, :56].contiguous()
    result = enhancer.enhance_frame(smaller)
    assert enhancer.last_was_reset and not enhancer.last_scene_cut
    assert torch.equal(result["enhanced"], enhance_image(model, smaller)["enhanced"])
    assert enhancer.state.previous_A is not None and enhancer.state.previous_A.shape[2:] == (40, 56)
    enhancer.enhance_frame(smaller)
    assert not enhancer.last_was_reset  # continues at the new size
    enhancer.reset()
    assert enhancer.state == VideoState()
    assert enhancer.state.previous_image is None and enhancer.state.previous_output is None
    enhancer.enhance_frame(smaller)
    assert enhancer.last_was_reset and enhancer.last_mask_coverage is None


def test_streams_do_not_share_state() -> None:
    config = temporal_config()
    model = make_model(config)
    frames = make_sequence(3)
    first, second = VideoEnhancer(model, config), VideoEnhancer(model, config)
    first.enhance_frame(frames[0])
    first.enhance_frame(frames[1])
    second.enhance_frame(frames[2])
    assert second.last_was_reset and second.state.previous_image is not first.state.previous_image
    assert torch.equal(second.state.previous_image, frames[2])


def test_video_enhancer_rejects_bad_frames_and_keeps_state_on_error() -> None:
    config = temporal_config()
    model = make_model(config)
    frames = make_sequence(2)
    enhancer = VideoEnhancer(model, config)
    with pytest.raises(ValueError, match="one frame"):
        enhancer.enhance_frame(torch.cat(frames, dim=0))
    with pytest.raises(ValueError):
        enhancer.enhance_frame(frames[0][0])
    with pytest.raises(ValueError):
        enhancer.enhance_frame(torch.rand(1, 3, 8, 8))
    assert enhancer.frames_processed == 0 and enhancer.state == VideoState()
    enhancer.enhance_frame(frames[0])
    before = enhancer.state.previous_image
    with pytest.raises(ValueError):
        enhancer.enhance_frame(torch.rand(2, 3, HEIGHT, WIDTH))
    assert enhancer.state.previous_image is before and enhancer.frames_processed == 1


def test_tiny_frames_raise_the_flow_scale_instead_of_failing() -> None:
    config = temporal_config()  # flow.scale 0.5 would give 10 px: below the DIS minimum
    model = make_model(config)
    enhancer = VideoEnhancer(model, config)
    base = make_canvas()[..., :20, :24]
    enhancer.enhance_frame(base.contiguous())
    result = enhancer.enhance_frame(torch.roll(base, 1, dims=3).contiguous())
    assert result["enhanced"].shape == (1, 3, 20, 24) and not enhancer.last_was_reset


def test_temporal_inference_reduces_flicker_with_a_brightness_dependent_map() -> None:
    """Injected maps whose A follows the frame's mean brightness: smoothing must cut flicker.

    A random-initialised network barely reacts to a 10 percent exposure change (measured: the
    temporal output is then not reliably less flickery than framewise, because warped curve maps
    add flow interpolation error), so the maps are injected (A = 8 * (mean brightness - 0.3) everywhere) while flow, reliability mask, warp and
    blend run for real. Exposure alternates 1.0 / 0.9 (the training flicker range) on a panning
    sequence; error is measured against the exact analytic flow.
    """
    count = 8
    frames = make_sequence(count, flicker=[1.0, 0.9] * (count // 2))
    framewise_config = make_config()
    temporal = temporal_config()
    model = make_model(temporal)

    def brightness_maps(image: Tensor) -> tuple[Tensor, Tensor]:
        level = 0.4 * (image.mean(dim=(1, 2, 3), keepdim=True) - 0.3).clamp(-0.5, 0.5)
        return level.expand(-1, 24, image.shape[2], image.shape[3]).contiguous(), torch.zeros_like(
            image[:, :1]
        )

    model.estimate_maps = brightness_maps  # type: ignore[method-assign]
    framewise_outputs = [
        VideoEnhancer(model, framewise_config).enhance_frame(f)["enhanced"] for f in frames
    ]
    enhancer = VideoEnhancer(model, temporal)
    temporal_outputs = [enhancer.enhance_frame(f)["enhanced"] for f in frames]
    assert not enhancer.last_was_reset and enhancer.last_mask_coverage is not None
    flicker_framewise = warp_aligned_error(framewise_outputs)
    flicker_temporal = warp_aligned_error(temporal_outputs)
    # Declared margin: temporal error below 85 percent of framewise. Measured ratio is about 0.68;
    # the remainder is the exposure change of the input pixels, which curve smoothing cannot
    # remove (the curves are applied to the current frame by design).
    assert flicker_temporal < 0.85 * flicker_framewise, (flicker_temporal, flicker_framewise)


# ----------------------------------------------------------------------- webcam


class FakeCapture:
    """Minimal cv2.VideoCapture stand-in that replays BGR frames and records its lifecycle."""

    def __init__(self, frames: list[np.ndarray], opened: bool = True) -> None:
        self.frames = list(frames)
        self.opened = opened
        self.released = 0
        self.reads = 0

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        self.reads += 1
        if self.frames:
            return True, self.frames.pop(0)
        return False, None

    def release(self) -> None:
        self.released += 1


def bgr_frames(count: int, height: int = HEIGHT, width: int = WIDTH) -> list[np.ndarray]:
    frames = []
    for frame in make_sequence(count):
        array = tensor_to_bgr(frame)
        frames.append(array[:height, :width].copy())
    return frames


def factory_for(capture: FakeCapture, calls: list[int] | None = None) -> Callable[[int], Any]:
    def factory(camera_id: int) -> FakeCapture:
        if calls is not None:
            calls.append(camera_id)
        return capture

    return factory


def test_colour_conversion_round_trip_is_exact() -> None:
    rng = np.random.default_rng(0)
    bgr = rng.integers(0, 256, (12, 20, 3), dtype=np.uint8)
    tensor = bgr_to_tensor(bgr, torch.device("cpu"))
    assert tensor.shape == (1, 3, 12, 20) and tensor.dtype == torch.float32
    assert torch.equal(tensor[0, 0], torch.from_numpy(bgr[..., 2]).float() / 255.0)  # RGB order
    assert np.array_equal(tensor_to_bgr(tensor), bgr)
    assert np.array_equal(
        tensor_to_bgr(tensor * 2.0), np.where(bgr > 127, 255, np.round(bgr * 2.0)).astype(np.uint8)
    )


def test_webcam_loop_processes_frames_headless() -> None:
    config = temporal_config()
    model = make_model(config)
    big = [cv2.resize(f, (160, 120), interpolation=cv2.INTER_LINEAR) for f in bgr_frames(5)]
    capture = FakeCapture(list(big))
    calls: list[int] = []
    shown: list[np.ndarray] = []
    coverage: list[float | None] = []
    resets: list[bool] = []

    def on_frame(index: int, canvas: np.ndarray, enhancer: VideoEnhancer) -> None:
        assert index == len(shown)
        shown.append(canvas)
        coverage.append(enhancer.last_mask_coverage)
        resets.append(enhancer.last_was_reset)

    run_webcam(
        model,
        config,
        3,
        max_frames=4,
        capture_factory=factory_for(capture, calls),
        show=False,
        on_frame=on_frame,
    )
    assert calls == [3] and capture.released == 1 and capture.reads == 4
    assert len(shown) == 4 and all(s.shape == (120, 320, 3) for s in shown)  # side by side
    assert shown[0].dtype == np.uint8
    assert resets == [True, False, False, False]
    assert coverage[0] is None and all(c is not None and c > 0.2 for c in coverage[1:])
    below_text = slice(60, None)  # the two overlay lines end well above row 60
    assert np.array_equal(shown[0][below_text, :160], big[0][below_text])  # original on the left
    assert not np.array_equal(
        shown[0][below_text, 160:], big[0][below_text]
    )  # enhanced on the right
    assert not np.array_equal(shown[0][:50, :160], big[0][:50])  # the overlay was drawn


def test_webcam_enhanced_only_mode_and_stream_end() -> None:
    config = make_config()  # temporal off: overlay says so, frames are enhanced independently
    model = make_model(config)
    capture = FakeCapture(bgr_frames(3))
    widths: list[int] = []
    run_webcam(
        model,
        config,
        capture_factory=factory_for(capture),
        show=False,
        split=False,
        on_frame=lambda i, canvas, e: widths.append(canvas.shape[1]),
    )
    assert widths == [WIDTH] * 3  # ended by the stream, no max_frames
    assert capture.released == 1 and capture.reads == 3 + 3  # three consecutive failed reads


def test_webcam_resets_on_size_change() -> None:
    config = temporal_config()
    model = make_model(config)
    frames = bgr_frames(2) + bgr_frames(2, 40, 56)
    resets: list[bool] = []
    run_webcam(
        model,
        config,
        capture_factory=factory_for(FakeCapture(frames)),
        show=False,
        on_frame=lambda i, c, e: resets.append(e.last_was_reset),
    )
    assert resets == [True, False, True, False]


def test_webcam_state_is_never_shared_across_captures() -> None:
    config = temporal_config()
    model = make_model(config)
    seen: list[tuple[int, bool]] = []
    for _ in range(2):
        run_webcam(
            model,
            config,
            capture_factory=factory_for(FakeCapture(bgr_frames(2))),
            show=False,
            on_frame=lambda i, c, e: seen.append((id(e), e.last_was_reset)) if i == 0 else None,
        )
    assert [reset for _, reset in seen] == [True, True]


def test_webcam_failure_to_open_is_a_clear_error_and_releases() -> None:
    config = temporal_config()
    capture = FakeCapture([], opened=False)
    with pytest.raises(RuntimeError, match="cannot open camera 2"):
        run_webcam(make_model(config), config, 2, capture_factory=factory_for(capture), show=False)
    assert capture.released == 1
    empty = FakeCapture([])
    with pytest.raises(RuntimeError, match="no frames"):
        run_webcam(make_model(config), config, capture_factory=factory_for(empty), show=False)
    assert empty.released == 1


@pytest.mark.parametrize("failure", [RuntimeError("boom"), KeyboardInterrupt()])
def test_webcam_releases_capture_on_error_and_interrupt(failure: BaseException) -> None:
    config = temporal_config()
    capture = FakeCapture(bgr_frames(4))

    def explode(index: int, canvas: np.ndarray, enhancer: VideoEnhancer) -> None:
        if index == 1:
            raise failure

    with pytest.raises(type(failure)):
        run_webcam(
            make_model(config),
            config,
            capture_factory=factory_for(capture),
            show=False,
            on_frame=explode,
        )
    assert capture.released == 1 and capture.reads == 2


def test_webcam_rejects_invalid_arguments_and_frames() -> None:
    config = make_config()
    model = make_model(config)
    capture = FakeCapture(bgr_frames(1))
    with pytest.raises(ValueError, match="camera_id"):
        run_webcam(model, config, -1, capture_factory=factory_for(capture), show=False)
    with pytest.raises(ValueError, match="max_frames"):
        run_webcam(model, config, 0, max_frames=0, capture_factory=factory_for(capture), show=False)
    assert capture.reads == 0
    bad = FakeCapture([np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)])
    with pytest.raises(ValueError, match="uint8"):
        run_webcam(model, config, capture_factory=factory_for(bad), show=False)
    assert bad.released == 1
    gray = FakeCapture([np.full((HEIGHT, WIDTH), 40, dtype=np.uint8)])
    run_webcam(model, config, capture_factory=factory_for(gray), show=False, max_frames=1)
    assert gray.released == 1


def test_webcam_window_keys_toggle_split_and_quit(monkeypatch: pytest.MonkeyPatch) -> None:
    config = make_config()
    model = make_model(config)
    keys = iter([-1, ord("s"), ord("s"), ord("q")])
    widths: list[int] = []
    events: list[str] = []

    def fake_imshow(name: str, canvas: np.ndarray) -> None:
        widths.append(canvas.shape[1])

    monkeypatch.setattr(cv2, "imshow", fake_imshow)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: next(keys))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: events.append("destroyed"))
    capture = FakeCapture(bgr_frames(8))
    run_webcam(model, config, capture_factory=factory_for(capture), show=True)
    assert widths == [2 * WIDTH, 2 * WIDTH, WIDTH, 2 * WIDTH]  # toggle off, toggle on, then quit
    assert capture.released == 1 and events == ["destroyed"]


def test_webcam_escape_quits_and_missing_display_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config()
    model = make_model(config)
    monkeypatch.setattr(cv2, "imshow", lambda name, canvas: None)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: 27)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    capture = FakeCapture(bgr_frames(5))
    run_webcam(model, config, capture_factory=factory_for(capture), show=True)
    assert capture.reads == 1 and capture.released == 1

    def no_gui(name: str, canvas: np.ndarray) -> None:
        raise cv2.error("The function is not implemented")

    def failing_destroy() -> None:
        raise cv2.error("no GUI")

    monkeypatch.setattr(cv2, "imshow", no_gui)
    monkeypatch.setattr(cv2, "destroyAllWindows", failing_destroy)
    capture = FakeCapture(bgr_frames(2))
    with pytest.raises(RuntimeError, match="display"):
        run_webcam(model, config, capture_factory=factory_for(capture), show=True)
    assert capture.released == 1


def test_default_capture_factory_is_opencv(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[int] = []
    capture = FakeCapture([], opened=False)

    def fake_video_capture(camera_id: int) -> FakeCapture:
        opened.append(camera_id)
        return capture

    monkeypatch.setattr(cv2, "VideoCapture", fake_video_capture)
    config = make_config()
    with pytest.raises(RuntimeError, match="cannot open camera 5"):
        run_webcam(make_model(config), config, 5, show=False)
    assert opened == [5] and capture.released == 1


def test_output_types_are_model_outputs() -> None:
    result: ModelOutput = VideoEnhancer(make_model(make_config()), make_config()).enhance_frame(
        make_canvas()
    )
    assert set(result) == {"enhanced", "enhanced_raw", "A", "sigma"}
