"""Data pipeline: transforms, noise synthesis, synthetic pairs, image and video datasets."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from nta_zerodce.data.image_dataset import (
    ImageDataset,
    index_files,
    make_synthetic_images,
    scene_split,
)
from nta_zerodce.data.noise_synth import check_generator, sample_noise
from nta_zerodce.data.synthetic_pairs import make_pair
from nta_zerodce.data.transforms import item_generator, normalize_rgb, paired_crop, random_crop
from nta_zerodce.data.video_pairs import VideoPairDataset
from nta_zerodce.inference.io import read_image, write_video
from nta_zerodce.training.config import Config, NoiseConfig, SyntheticConfig


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _texture(batch: int = 2, size: int = 64, seed: int = 0) -> torch.Tensor:
    return torch.rand(batch, 3, size, size, generator=_gen(seed))


def _clean_config(**synthetic: float) -> Config:
    """Config with noise and flicker disabled unless overridden."""
    values = {"flicker_min": 1.0, "flicker_max": 1.0}
    values.update(synthetic)
    return Config(synthetic=SyntheticConfig(**values), noise=NoiseConfig(a_max=0.0, b_max=0.0))


# ------------------------------------------------------------------ transforms


def test_normalize_rgb_values_dtype_and_device() -> None:
    raw = torch.tensor([0, 1, 128, 255], dtype=torch.uint8).view(1, 1, 4).expand(3, 2, 4)
    out = normalize_rgb(raw)
    assert out.dtype == torch.float32 and out.shape == (3, 2, 4)
    assert out[0, 0].tolist() == pytest.approx([0.0, 1 / 255, 128 / 255, 1.0])
    assert out.device == raw.device


def test_normalize_rgb_rejects_bad_input() -> None:
    with pytest.raises(TypeError):
        normalize_rgb(torch.zeros(3, 4, 4))  # float, not uint8
    with pytest.raises(ValueError):
        normalize_rgb(torch.zeros(4, 4, 4, dtype=torch.uint8))
    with pytest.raises(ValueError):
        normalize_rgb(torch.zeros(1, 3, 4, 4, dtype=torch.uint8))


def test_paired_crop_shares_origin_and_is_deterministic() -> None:
    first = torch.rand(3, 48, 80, generator=_gen(1))
    second = first + 1.0  # constant offset reveals any origin mismatch
    a1, a2 = paired_crop(first, second, 32, _gen(5))
    b1, b2 = paired_crop(first, second, 32, _gen(5))
    assert a1.shape == (3, 32, 32) and a2.shape == (3, 32, 32)
    assert torch.equal(a1, b1) and torch.equal(a2, b2)
    assert torch.allclose(a2 - a1, torch.ones_like(a1))
    # The crop is an exact window of the source.
    found = any(
        torch.equal(first[:, t : t + 32, left : left + 32], a1)
        for t in range(48 - 32 + 1)
        for left in range(80 - 32 + 1)
    )
    assert found


def test_paired_crop_covers_all_origins_and_spares_global_rng() -> None:
    first = torch.arange(3 * 32 * 48, dtype=torch.float32).view(3, 32, 48)
    state = torch.get_rng_state()
    generator = _gen(0)
    origins = set()
    for _ in range(400):
        crop, _ = paired_crop(first, first, 32, generator)
        origins.add(int(crop[0, 0, 0]) % 48)  # left origin of the row-0 value
    assert origins == set(range(48 - 32 + 1))  # every valid left origin, none out of range
    assert torch.equal(torch.get_rng_state(), state)


def test_paired_crop_full_size_and_validation() -> None:
    image = torch.rand(3, 32, 32, generator=_gen(2))
    full, _ = paired_crop(image, image, 32, _gen(0))
    assert torch.equal(full, image)
    with pytest.raises(ValueError, match="exceeds"):
        paired_crop(image, image, 48, _gen(0))
    for bad in (0, -16, 20):
        with pytest.raises(ValueError, match="multiple of 16"):
            paired_crop(image, image, bad, _gen(0))
    with pytest.raises(ValueError, match="equal shapes"):
        paired_crop(image, image[:, :, :16], 16, _gen(0))
    with pytest.raises(ValueError, match="shape"):
        paired_crop(image[None], image[None], 16, _gen(0))
    with pytest.raises(TypeError):
        paired_crop(image, image, 16.0, _gen(0))  # type: ignore[arg-type]


def test_random_crop_matches_paired_crop_first_output() -> None:
    image = torch.rand(3, 64, 64, generator=_gen(3))
    assert torch.equal(random_crop(image, 32, _gen(9)), paired_crop(image, image, 32, _gen(9))[0])


def test_item_generator_depends_on_all_three_integers() -> None:
    def draw(seed: int, epoch: int, index: int) -> list[float]:
        values = torch.rand(4, generator=item_generator(seed, epoch, index))
        return [float(v) for v in values]

    assert draw(17, 0, 3) == draw(17, 0, 3)
    assert (
        len(
            {
                tuple(draw(17, 0, 3)),
                tuple(draw(18, 0, 3)),
                tuple(draw(17, 1, 3)),
                tuple(draw(17, 0, 4)),
            }
        )
        == 4
    )
    with pytest.raises(ValueError):
        item_generator(-1, 0, 0)


# ----------------------------------------------------------------- noise_synth


def test_sample_noise_statistics_match_model() -> None:
    # Constant levels far from clipping: x +/- 4 sigma stays inside [0,1].
    levels = torch.tensor([0.3, 0.5, 0.7])
    image = levels.view(3, 1, 1, 1).expand(3, 3, 256, 256).contiguous()
    a = torch.tensor([0.01, 0.005, 0.0]).view(3, 1, 1, 1)
    b = torch.tensor([0.02, 0.01, 0.03]).view(3, 1, 1, 1)
    noisy, sigma = sample_noise(image, a, b, _gen(11))
    noise = noisy - image
    expected = torch.sqrt(a * image + b**2)
    n = 3 * 256 * 256
    for i in range(3):
        measured = float(noise[i].std())
        # Relative error of a std estimate is ~1/sqrt(2n)=0.16%; allow 2% (> 12 sigma).
        assert measured == pytest.approx(float(expected[i, 0, 0, 0]), rel=0.02)
        assert abs(float(noise[i].mean())) < 5 * measured / n**0.5
        # Every channel individually.
        for c in range(3):
            assert float(noise[i, c].std()) == pytest.approx(float(expected[i, c, 0, 0]), rel=0.04)
    assert sigma.shape == (3, 1, 256, 256)
    assert torch.allclose(sigma, torch.sqrt(a * levels.view(3, 1, 1, 1) + b**2).expand_as(sigma))


def test_sample_noise_sigma_uses_rgb_mean_and_is_unclipped() -> None:
    image = torch.zeros(1, 3, 4, 4)
    image[:, 0] = 1.0  # Y = 1/3 per pixel
    a = torch.full((1, 1, 1, 1), 0.03)
    b = torch.full((1, 1, 1, 1), 0.04)
    noisy, sigma = sample_noise(image, a, b, _gen(0))
    assert torch.allclose(sigma, torch.full((1, 1, 4, 4), (0.03 / 3 + 0.04**2) ** 0.5))
    assert float(noisy.min()) >= 0.0 and float(noisy.max()) <= 1.0  # image clipped, target not


def test_sample_noise_zero_coefficients_is_identity_with_zero_sigma() -> None:
    image = _texture(2, 16)
    zero = torch.zeros(2, 1, 1, 1)
    noisy, sigma = sample_noise(image, zero, zero, _gen(1))
    assert torch.equal(noisy, image)
    assert torch.equal(sigma, torch.zeros(2, 1, 16, 16))


def test_sample_noise_determinism_and_generator_isolation() -> None:
    image = _texture(2, 16)
    a = torch.full((2, 1, 1, 1), 0.01)
    b = torch.full((2, 1, 1, 1), 0.02)
    first, _ = sample_noise(image, a, b, _gen(7))
    again, _ = sample_noise(image, a, b, _gen(7))
    other, _ = sample_noise(image, a, b, _gen(8))
    assert torch.equal(first, again) and not torch.equal(first, other)
    state = torch.get_rng_state()
    sample_noise(image, a, b, _gen(7))
    assert torch.equal(torch.get_rng_state(), state)  # global RNG untouched
    # Without a generator the global RNG is used and respects the caller's seeding.
    torch.manual_seed(5)
    x1, _ = sample_noise(image, a, b)
    torch.manual_seed(5)
    x2, _ = sample_noise(image, a, b)
    assert torch.equal(x1, x2)


def test_sample_noise_validation() -> None:
    image = _texture(2, 8)
    ok = torch.zeros(2, 1, 1, 1)
    with pytest.raises(ValueError, match="shape"):
        sample_noise(image[0], ok, ok)
    with pytest.raises(ValueError, match="float32"):
        sample_noise(image.double(), ok, ok)
    with pytest.raises(ValueError, match="within"):
        sample_noise(image * 2, ok, ok)
    with pytest.raises(ValueError, match="within"):
        sample_noise(torch.full((2, 3, 8, 8), float("nan")), ok, ok)
    with pytest.raises(ValueError, match="shape"):
        sample_noise(image, torch.zeros(2), ok)
    with pytest.raises(ValueError, match="nonnegative"):
        sample_noise(image, ok - 0.1, ok)
    with pytest.raises(ValueError, match="nonnegative"):
        sample_noise(image, ok, torch.full((2, 1, 1, 1), float("inf")))
    with pytest.raises(TypeError):
        sample_noise(image, 0.1, ok)


def test_check_generator_rejects_device_mismatch() -> None:
    check_generator(None, torch.device("cpu"))
    check_generator(_gen(0), torch.device("cpu"))
    fake = SimpleNamespace(device=torch.device("cuda"))
    with pytest.raises(ValueError, match="same device"):
        check_generator(fake, torch.device("cpu"))


# ------------------------------------------------------------- synthetic pairs


def _bilinear(field: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Reference bilinear sampler: field (C,H,W) at pixel coordinates (x,y); zero outside."""
    channels, height, width = field.shape
    field = field.double()
    x0, y0 = torch.floor(x), torch.floor(y)
    out = torch.zeros(channels, *x.shape, dtype=torch.float64)
    for dy in (0, 1):
        for dx in (0, 1):
            xi, yi = x0 + dx, y0 + dy
            weight = (1 - (x - xi).abs()).clamp(min=0) * (1 - (y - yi).abs()).clamp(min=0)
            valid = (xi >= 0) & (xi <= width - 1) & (yi >= 0) & (yi <= height - 1)
            xc = xi.clamp(0, width - 1).long()
            yc = yi.clamp(0, height - 1).long()
            out += field[:, yc, xc] * (weight * valid)
    return out


def _coords(height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=torch.float64),
        torch.arange(width, dtype=torch.float64),
        indexing="ij",
    )
    return xs, ys


def test_make_pair_output_contract() -> None:
    image = _texture(3, 48, seed=4)
    out = make_pair(image, Config(), _gen(0))
    for key in ("image1", "image2", "clean1", "clean2"):
        assert out[key].shape == (3, 3, 48, 48) and out[key].dtype == torch.float32
        assert 0.0 <= float(out[key].min()) and float(out[key].max()) <= 1.0
    for key in ("forward_flow", "backward_flow"):
        assert out[key].shape == (3, 2, 48, 48) and out[key].dtype == torch.float32
    assert out["mask"].shape == (3, 1, 48, 48)
    assert set(out["mask"].unique().tolist()) <= {0.0, 1.0}
    assert out["affine"].shape == (3, 2, 3) and out["affine_inverse"].shape == (3, 2, 3)
    assert out["flicker"].shape == (3,)
    assert out["noise_a"].shape == (3, 2) and out["noise_b"].shape == (3, 2)
    assert out["sigma_target1"].shape == (3, 1, 48, 48)
    assert torch.equal(out["clean1"], image)


def test_make_pair_parameters_respect_config_ranges() -> None:
    config = Config()
    out = make_pair(_texture(64, 16, seed=1), config, _gen(2))
    syn = config.synthetic
    assert float(out["flicker"].min()) >= syn.flicker_min
    assert float(out["flicker"].max()) <= syn.flicker_max
    matrix = out["affine"].double()
    scale = torch.sqrt(torch.abs(torch.linalg.det(matrix[:, :, :2])))
    assert float(scale.min()) >= syn.scale_min - 1e-6 and float(scale.max()) <= syn.scale_max + 1e-6
    angle = torch.rad2deg(torch.atan2(matrix[:, 1, 0], matrix[:, 0, 0]))
    assert float(angle.abs().max()) <= syn.rotation_degrees + 1e-4
    centre = torch.tensor([7.5, 7.5], dtype=torch.float64)
    shift = (matrix[:, :, :2] @ centre) + matrix[:, :, 2] - centre  # T(c) - c = t
    assert float(shift.abs().max()) <= syn.translation_pixels + 1e-4
    # Parameters actually vary across the batch.
    assert float(angle.std()) > 0.5 and float(out["flicker"].std()) > 0.02
    for key in ("noise_a", "noise_b"):
        assert float(out[key].min()) >= 0.0
    assert float(out["noise_a"].max()) <= config.noise.a_max
    assert float(out["noise_b"].max()) <= config.noise.b_max


def test_make_pair_backward_flow_warp_reproduces_transformed_frame() -> None:
    config = _clean_config(rotation_degrees=5.0, translation_pixels=6.0)
    image = _texture(4, 64, seed=5)
    out = make_pair(image, config, _gen(3))
    assert torch.equal(out["image2"], out["clean2"])  # noise and flicker disabled
    xs, ys = _coords(64, 64)
    for i in range(4):
        flow = out["backward_flow"][i].double()
        warped = _bilinear(image[i], xs + flow[0], ys + flow[1])
        # Away from borders: samples whose 2-pixel neighbourhood lies inside the source.
        sx, sy = xs + flow[0], ys + flow[1]
        interior = (sx >= 2) & (sx <= 61) & (sy >= 2) & (sy <= 61)
        assert int(interior.sum()) > 2000
        error = (warped - out["image2"][i].double()).abs()[:, interior]
        assert float(error.max()) < 1e-4


def test_make_pair_flows_compose_to_zero_on_mask() -> None:
    config = _clean_config(
        rotation_degrees=15.0, translation_pixels=5.0, scale_min=0.95, scale_max=1.05
    )
    out = make_pair(_texture(4, 64, seed=6), config, _gen(4))
    xs, ys = _coords(64, 64)
    for i in range(4):
        backward = out["backward_flow"][i].double()
        forward = out["forward_flow"][i]
        # Forward field evaluated at p = q + b(q) by bilinear interpolation.
        at_source = _bilinear(forward, xs + backward[0], ys + backward[1])
        residual = (backward + at_source).abs()
        mask = out["mask"][i, 0] > 0.5
        assert int(mask.sum()) > 2000
        assert float(residual[:, mask].max()) < 1e-4
        # The same identity must NOT hold outside the mask for the zero-padded sampler.
        assert float(residual[:, ~mask].max()) > 0.5


def test_make_pair_forward_negation_is_not_backward_for_rotation() -> None:
    out = make_pair(_texture(8, 64, seed=7), _clean_config(rotation_degrees=20.0), _gen(5))
    gap = (out["forward_flow"] + out["backward_flow"]).abs().amax(dim=(1, 2, 3))
    assert float(gap.max()) > 0.5  # a 20-degree rotation: negation is off by pixels
    # ... whereas a pure translation is the one case where negation is exact.
    pure = make_pair(
        _texture(4, 64, seed=7),
        _clean_config(rotation_degrees=0.0, scale_min=1.0, scale_max=1.0, translation_pixels=7.0),
        _gen(5),
    )
    assert torch.allclose(pure["forward_flow"], -pure["backward_flow"], atol=1e-5)
    assert float(pure["forward_flow"].abs().max()) > 1.0


def test_make_pair_mask_is_geometric_validity_in_current_coordinates() -> None:
    config = _clean_config(rotation_degrees=3.0, translation_pixels=8.0)
    image = torch.ones(4, 3, 64, 64)
    out = make_pair(image, config, _gen(8))
    xs, ys = _coords(64, 64)
    for i in range(4):
        sx = xs + out["backward_flow"][i, 0].double()
        sy = ys + out["backward_flow"][i, 1].double()
        inside = (sx >= 0) & (sx <= 63) & (sy >= 0) & (sy <= 63)
        # float32 flows may flip pixels that sit within ~1e-5 of the boundary; allow few.
        assert int((inside != (out["mask"][i, 0] > 0.5)).sum()) <= 4
        assert 0.5 < float(out["mask"][i].mean()) < 1.0  # some borders invalid, most valid
        outside_by_margin = (sx < -1) | (sx > 64) | (sy < -1) | (sy > 64)
        assert float(out["clean2"][i, :, outside_by_margin].abs().max()) == 0.0  # zero padding


def test_make_pair_identity_transform_has_zero_flow_and_full_mask() -> None:
    config = _clean_config(
        rotation_degrees=0.0, translation_pixels=0.0, scale_min=1.0, scale_max=1.0
    )
    image = _texture(2, 32, seed=8)
    out = make_pair(image, config, _gen(0))
    assert float(out["forward_flow"].abs().max()) == 0.0
    assert float(out["backward_flow"].abs().max()) == 0.0
    assert float(out["mask"].min()) == 1.0
    assert torch.allclose(out["image2"], image, atol=1e-5)


def test_make_pair_flicker_scales_second_frame_only() -> None:
    config = Config(
        synthetic=SyntheticConfig(
            rotation_degrees=0.0,
            translation_pixels=0.0,
            scale_min=1.0,
            scale_max=1.0,
            flicker_min=0.9,
            flicker_max=1.1,
        ),
        noise=NoiseConfig(a_max=0.0, b_max=0.0),
    )
    image = _texture(16, 16, seed=9) * 0.8  # keeps flicker*image <= 0.88 < 1
    out = make_pair(image, config, _gen(1))
    expected = out["flicker"].view(-1, 1, 1, 1) * image
    assert torch.allclose(out["image2"], expected, atol=1e-6)
    assert torch.equal(out["image1"], image)
    assert float(out["flicker"].min()) < 0.97 and float(out["flicker"].max()) > 1.03


def test_make_pair_determinism_and_generator_isolation() -> None:
    image = _texture(2, 32, seed=10)
    first = make_pair(image, Config(), _gen(21))
    again = make_pair(image, Config(), _gen(21))
    other = make_pair(image, Config(), _gen(22))
    assert first.keys() == again.keys()
    for key in first:
        assert torch.equal(first[key], again[key]), key
    assert not torch.equal(first["image2"], other["image2"])
    assert not torch.equal(first["affine"], other["affine"])
    state = torch.get_rng_state()
    make_pair(image, Config(), _gen(21))
    assert torch.equal(torch.get_rng_state(), state)


def test_make_pair_frames_get_independent_noise() -> None:
    config = Config(
        synthetic=SyntheticConfig(
            rotation_degrees=0.0,
            translation_pixels=0.0,
            scale_min=1.0,
            scale_max=1.0,
            flicker_min=1.0,
            flicker_max=1.0,
        ),
        noise=NoiseConfig(a_min=0.005, a_max=0.01, b_min=0.01, b_max=0.02),
    )
    image = torch.full((4, 3, 64, 64), 0.5)
    out = make_pair(image, config, _gen(30))
    noise1 = (out["image1"] - image).flatten(1)
    noise2 = (out["image2"] - out["clean2"]).flatten(1)
    for i in range(4):
        assert float(noise1[i].std()) > 0.01 and float(noise2[i].std()) > 0.01
        correlation = torch.corrcoef(torch.stack([noise1[i], noise2[i]]))[0, 1]
        assert abs(float(correlation)) < 0.05  # 12288 samples: sampling sd ~ 0.009
    # Coefficients are drawn separately for each frame and each image.
    assert not torch.equal(out["noise_a"][:, 0], out["noise_a"][:, 1])
    assert len(set(out["noise_b"][:, 0].tolist())) == 4
    # Targets follow METHOD with the recorded coefficients.
    a = out["noise_a"][:, 0].view(4, 1, 1, 1)
    b = out["noise_b"][:, 0].view(4, 1, 1, 1)
    assert torch.allclose(
        out["sigma_target1"], torch.sqrt(a * 0.5 + b**2).expand(4, 1, 64, 64), atol=1e-6
    )


def test_make_pair_noise_probability_gates_per_image() -> None:
    config = Config(
        noise=NoiseConfig(a_min=0.005, a_max=0.01, b_min=0.01, b_max=0.02, probability=0.5)
    )
    out = make_pair(_texture(64, 8, seed=11), config, _gen(2))
    applied = out["noise_b"] > 0
    assert 0.25 < float(applied.float().mean()) < 0.75
    skipped = ~applied[:, 0]
    assert torch.equal(out["image1"][skipped], out["clean1"][skipped])
    assert float(out["sigma_target1"][skipped].abs().max()) == 0.0
    off = make_pair(_texture(4, 8), Config(noise=NoiseConfig(probability=0.0)), _gen(2))
    assert torch.equal(off["image1"], off["clean1"]) and float(off["noise_a"].abs().max()) == 0.0


def test_make_pair_validation() -> None:
    image = _texture(2, 16)
    with pytest.raises(ValueError, match="shape"):
        make_pair(image[0], Config())
    with pytest.raises(ValueError, match="float32"):
        make_pair(image.double(), Config())
    with pytest.raises(ValueError, match="within"):
        make_pair(image + 1.0, Config())
    with pytest.raises(ValueError, match="H,W>=2"):
        make_pair(torch.zeros(1, 3, 1, 8), Config())
    with pytest.raises(TypeError):
        make_pair(image, "config")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="inverted"):
        make_pair(image, Config(synthetic=SyntheticConfig(scale_min=1.1, scale_max=0.9)))
    with pytest.raises(ValueError, match="inverted"):
        make_pair(image, Config(noise=NoiseConfig(a_min=0.1, a_max=0.01)))


# ------------------------------------------------------ synthetic test images


def test_make_synthetic_images_deterministic_dark_and_valid(tmp_path: Path) -> None:
    first = make_synthetic_images(tmp_path / "a", 5, 64, seed=3)
    again = make_synthetic_images(tmp_path / "b", 5, 64, seed=3)
    other = make_synthetic_images(tmp_path / "c", 5, 64, seed=4)
    assert [p.name for p in first] == [f"synthetic_{i:04d}.png" for i in range(5)]
    for x, y, z in zip(first, again, other):
        assert x.read_bytes() == y.read_bytes()
        assert x.read_bytes() != z.read_bytes()
    means = []
    for path in first:
        image = read_image(path)
        assert image.shape == (1, 3, 64, 64)
        means.append(float(image.mean()))
        assert float(image.std()) > 0.02  # textured, not flat
        assert float(image.max()) > 0.5  # a bright lamp spot exists
    assert all(0.05 <= m <= 0.25 for m in means), means
    assert len({p.read_bytes() for p in first}) == 5  # different images
    rect = make_synthetic_images(tmp_path / "d", 2, (32, 48), seed=1)
    assert read_image(rect[0]).shape == (1, 3, 32, 48)


def test_make_synthetic_images_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        make_synthetic_images(tmp_path, 0, 64)
    with pytest.raises(ValueError):
        make_synthetic_images(tmp_path, 1, 8)
    with pytest.raises(ValueError):
        make_synthetic_images(tmp_path, 1, (64, 8))


# --------------------------------------------------------------- ImageDataset


@pytest.fixture()
def image_root(tmp_path: Path) -> Path:
    root = tmp_path / "imgs"
    make_synthetic_images(root, 6, 64, seed=1)
    return root


def _find_crop(full: torch.Tensor, crop: torch.Tensor) -> tuple[int, int] | None:
    size = crop.shape[-1]
    for top in range(full.shape[-2] - size + 1):
        for left in range(full.shape[-1] - size + 1):
            if torch.equal(full[:, top : top + size, left : left + size], crop):
                return top, left
    return None


def test_image_dataset_training_items(image_root: Path) -> None:
    dataset = ImageDataset(image_root, crop_size=32)
    assert len(dataset) == 6
    assert dataset.paths == sorted(dataset.paths, key=lambda p: p.as_posix())
    item = dataset[0]
    assert set(item) == {"image"}
    assert item["image"].shape == (3, 32, 32) and item["image"].dtype == torch.float32
    assert 0.0 <= float(item["image"].min()) and float(item["image"].max()) <= 1.0
    assert _find_crop(read_image(dataset.paths[0])[0], item["image"]) is not None


def test_image_dataset_is_lazy(image_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import nta_zerodce.data.image_dataset as module

    def fail(path: Path) -> torch.Tensor:
        raise AssertionError("pixels decoded during indexing")

    monkeypatch.setattr(module, "read_image", fail)
    dataset = ImageDataset(image_root, crop_size=16)
    assert len(dataset) == 6


def test_image_dataset_crops_reproducible_across_epochs(image_root: Path) -> None:
    dataset = ImageDataset(image_root, crop_size=32, seed=17)
    epoch0 = [dataset[i]["image"] for i in range(len(dataset))]
    assert all(torch.equal(epoch0[i], dataset[i]["image"]) for i in range(len(dataset)))
    fresh = ImageDataset(image_root, crop_size=32, seed=17)
    assert all(torch.equal(epoch0[i], fresh[i]["image"]) for i in range(len(dataset)))
    dataset.set_epoch(1)
    epoch1 = [dataset[i]["image"] for i in range(len(dataset))]
    assert any(not torch.equal(a, b) for a, b in zip(epoch0, epoch1))  # crops move per epoch
    dataset.set_epoch(1)
    assert all(torch.equal(epoch1[i], dataset[i]["image"]) for i in range(len(dataset)))
    dataset.set_epoch(0)
    assert all(torch.equal(epoch0[i], dataset[i]["image"]) for i in range(len(dataset)))
    reseeded = ImageDataset(image_root, crop_size=32, seed=18)
    assert any(not torch.equal(epoch0[i], reseeded[i]["image"]) for i in range(len(dataset)))
    with pytest.raises(ValueError):
        dataset.set_epoch(-1)


def test_image_dataset_does_not_touch_global_rng(image_root: Path) -> None:
    dataset = ImageDataset(image_root, crop_size=32)
    state = torch.get_rng_state()
    dataset[0]
    assert torch.equal(torch.get_rng_state(), state)


def test_image_dataset_indexing_rules(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "scene_b").mkdir(parents=True)
    (root / "scene_a" / "deep").mkdir(parents=True)
    array = np.full((32, 32, 3), 40, dtype=np.uint8)
    for rel in (
        "scene_b/2.PNG",
        "scene_b/1.jpg",
        "scene_a/deep/x.JpEg",
        "scene_a/y.bmp",
        "top.png",
    ):
        Image.fromarray(array).save(
            root / rel,
            format={"PNG": "PNG", "JPG": "JPEG", "JPEG": "JPEG", "BMP": "BMP"}[
                Path(rel).suffix[1:].upper().replace("JPEG", "JPEG")
            ],
        )
    (root / "notes.txt").write_text("ignore me")
    (root / "scene_a" / "labels.json").write_text("{}")
    dataset = ImageDataset(root, crop_size=16)
    names = [p.relative_to(root).as_posix() for p in dataset.paths]
    assert names == sorted(names)
    assert names == [
        "scene_a/deep/x.JpEg",
        "scene_a/y.bmp",
        "scene_b/1.jpg",
        "scene_b/2.PNG",
        "top.png",
    ]
    assert [p.name for p in index_files(root, (".png",))] == ["2.PNG", "top.png"]


def test_image_dataset_error_paths(tmp_path: Path, image_root: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no images"):
        ImageDataset(empty, crop_size=16)
    with pytest.raises(NotADirectoryError):
        ImageDataset(tmp_path / "missing", crop_size=16)
    for bad in (0, -16, 20, 33):
        with pytest.raises(ValueError, match="multiple of 16"):
            ImageDataset(image_root, crop_size=bad)
    dataset = ImageDataset(image_root, crop_size=16)
    for bad_index in (-1, len(dataset), 10**6):
        with pytest.raises(IndexError):
            dataset[bad_index]
    with pytest.raises(FileNotFoundError):
        ImageDataset(image_root, crop_size=16, paths=[image_root / "nope.png"])
    with pytest.raises(ValueError, match="no images"):
        ImageDataset(image_root, crop_size=16, paths=[])


def test_image_dataset_skips_small_and_unreadable_with_logged_counts(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = tmp_path / "mixed"
    make_synthetic_images(root, 3, 64, seed=2)
    Image.fromarray(np.zeros((20, 64, 3), dtype=np.uint8)).save(root / "small_a.png")
    Image.fromarray(np.zeros((64, 24, 3), dtype=np.uint8)).save(root / "small_b.png")
    (root / "broken.png").write_bytes(b"not a png")
    with caplog.at_level(logging.WARNING, logger="nta_zerodce.data.image_dataset"):
        dataset = ImageDataset(root, crop_size=32)
    assert len(dataset) == 3  # nothing padded: only the three 64x64 images remain
    assert "skipped 2 image(s) smaller than crop 32 and 1 unreadable" in caplog.text
    only_small = tmp_path / "small"
    only_small.mkdir()
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(only_small / "s.png")
    with pytest.raises(ValueError, match="no usable images"):
        ImageDataset(only_small, crop_size=32)


def test_image_dataset_skip_uses_exif_oriented_size(tmp_path: Path) -> None:
    root = tmp_path / "exif"
    root.mkdir()
    exif = Image.Exif()
    exif[0x0112] = 6  # stored 20x64 (HxW) displays as 64x20
    Image.fromarray(np.zeros((20, 64, 3), dtype=np.uint8)).save(root / "a.jpg", exif=exif)
    Image.fromarray(np.zeros((64, 20, 3), dtype=np.uint8)).save(root / "b.jpg", exif=exif)
    # Displayed sizes: a -> 64 wide x 20... a is 20 high stored; after rotation H=64, W=20.
    with pytest.raises(ValueError, match="no usable images"):
        ImageDataset(root, crop_size=32)  # both have a displayed side of 20 < 32


def test_image_dataset_eval_mode_pairs_full_size(tmp_path: Path) -> None:
    low, high = tmp_path / "low", tmp_path / "high"
    low.mkdir()
    high.mkdir()
    rng = np.random.default_rng(0)
    for name, shape in (("2.png", (40, 56, 3)), ("1.png", (24, 32, 3))):
        Image.fromarray(rng.integers(0, 100, shape, dtype=np.uint8)).save(low / name)
        Image.fromarray(rng.integers(100, 255, shape, dtype=np.uint8)).save(high / name)
    dataset = ImageDataset(low, crop_size=256, paired_root=high)  # crop size ignored in eval
    assert len(dataset) == 2
    first, second = dataset[0], dataset[1]
    assert set(first) == {"image", "reference"}
    assert first["image"].shape == (3, 24, 32) == first["reference"].shape  # sorted by name
    assert second["image"].shape == (3, 40, 56)
    assert float(first["image"].max()) < 100 / 255 <= float(first["reference"].min())
    assert torch.equal(first["image"], dataset[0]["image"])  # no randomness


def test_image_dataset_eval_mode_rejects_bad_pairs(tmp_path: Path) -> None:
    low, high = tmp_path / "low", tmp_path / "high"
    low.mkdir()
    high.mkdir()
    pixel = np.zeros((16, 16, 3), dtype=np.uint8)
    for name in ("a.png", "b.png"):
        Image.fromarray(pixel).save(low / name)
    Image.fromarray(pixel).save(high / "a.png")
    with pytest.raises(ValueError, match="unmatched pairs.*b.png"):
        ImageDataset(low, paired_root=high)  # missing reference
    Image.fromarray(pixel).save(high / "b.png")
    Image.fromarray(pixel).save(high / "extra.png")
    with pytest.raises(ValueError, match="unmatched pairs.*extra.png"):
        ImageDataset(low, paired_root=high)  # missing input
    (high / "extra.png").unlink()
    (high / "sub").mkdir()
    Image.fromarray(pixel).save(high / "sub" / "a.png")
    with pytest.raises(ValueError, match="duplicate reference basename"):
        ImageDataset(low, paired_root=high)
    (high / "sub" / "a.png").unlink()
    Image.fromarray(np.zeros((16, 32, 3), dtype=np.uint8)).save(high / "b.png")
    with pytest.raises(ValueError, match="mismatched sizes"):
        ImageDataset(low, paired_root=high)
    (low / "dup").mkdir()
    Image.fromarray(pixel).save(low / "dup" / "a.png")
    with pytest.raises(ValueError, match="duplicate input basename"):
        ImageDataset(low, paired_root=high)


def test_image_dataset_dataloader_collation(image_root: Path) -> None:
    dataset = ImageDataset(image_root, crop_size=32)
    batch = next(iter(DataLoader(dataset, batch_size=3, shuffle=False, num_workers=0)))
    assert batch["image"].shape == (3, 3, 32, 32)


# ----------------------------------------------------------------- scene_split


def _scene_tree(root: Path, scenes: int = 10, files: int = 3) -> None:
    for s in range(scenes):
        directory = root / f"scene_{s:02d}"
        directory.mkdir(parents=True)
        for f in range(files):
            Image.fromarray(np.full((16, 16, 3), s * 10 + f, dtype=np.uint8)).save(
                directory / f"{f}.png"
            )


def test_scene_split_groups_by_scene_and_is_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "sice"
    _scene_tree(root)
    train, val = scene_split(root)
    assert (train, val) == scene_split(root)
    assert len(val) == 3 and len(train) == 27  # 10 scenes * 10% = 1 scene of 3 files
    assert not set(train) & set(val)
    assert sorted(train + val) == index_files(root, (".png",))
    assert {p.parent for p in train}.isdisjoint({p.parent for p in val})  # no scene straddles
    assert train == sorted(train, key=lambda p: p.as_posix())
    other = scene_split(root, seed=18)
    assert other[1] != val or other[0] != train
    big_val = scene_split(root, val_fraction=0.5)[1]
    assert len(big_val) == 15 and len({p.parent for p in big_val}) == 5
    none = scene_split(root, val_fraction=0.0)
    assert len(none[1]) == 0 and len(none[0]) == 30


def test_scene_split_flat_root_falls_back_to_per_file(tmp_path: Path) -> None:
    root = make_synthetic_images(tmp_path / "flat", 10, 16, seed=1)[0].parent
    train, val = scene_split(root, val_fraction=0.2)
    assert len(val) == 2 and len(train) == 8 and not set(train) & set(val)


def test_scene_split_edge_cases_and_dataset_integration(tmp_path: Path) -> None:
    root = tmp_path / "one"
    _scene_tree(root, scenes=1)
    train, val = scene_split(root)
    assert len(train) == 3 and val == []  # a single scene is never held out entirely
    with pytest.raises(ValueError):
        scene_split(root, val_fraction=1.0)
    with pytest.raises(ValueError):
        scene_split(root, val_fraction=-0.1)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no images"):
        scene_split(empty)
    many = tmp_path / "many"
    _scene_tree(many, scenes=5, files=2)
    train, val = scene_split(many, val_fraction=0.2)
    train_set = ImageDataset(many, crop_size=16, paths=train)
    val_set = ImageDataset(many, crop_size=16, paths=val)
    assert train_set.paths == train and val_set.paths == val
    assert len(train_set) == 8 and len(val_set) == 2


# ---------------------------------------------------------- VideoPairDataset


def _write_clip(path: Path, frames: int, height: int = 32, width: int = 48) -> None:
    """Smooth gradient clip where frame t equals gradient + 0.1*t (constant temporal step)."""
    xs = torch.linspace(0, 1, width).view(1, 1, 1, width).expand(1, 3, height, width)
    write_video(iter([torch.clamp(0.3 * xs + 0.1 * t, 0, 1) for t in range(frames)]), path, 10.0)


@pytest.fixture()
def video_root(tmp_path: Path) -> Path:
    root = tmp_path / "videos"
    (root / "session_b").mkdir(parents=True)
    (root / "session_a").mkdir()
    _write_clip(root / "session_b" / "clip2.AVI", 4)
    _write_clip(root / "session_a" / "clip1.avi", 5)
    return root


def test_video_pair_dataset_enumerates_adjacent_pairs(video_root: Path) -> None:
    dataset = VideoPairDataset(video_root, crop_size=32)
    assert len(dataset) == 4 + 3  # sorted: session_a (5 frames) then session_b (4 frames)
    item = dataset[0]
    assert set(item) == {"image1", "image2", "index"}
    assert item["image1"].shape == (3, 32, 32) == item["image2"].shape
    assert item["image1"].dtype == torch.float32
    assert item["index"].dtype == torch.int64 and item["index"].ndim == 0
    assert int(dataset[5]["index"]) == 5
    for i in range(len(dataset)):
        pair = dataset[i]
        # Shared origin: the temporal step 0.1 is constant over the whole crop (codec tolerance).
        step = pair["image2"] - pair["image1"]
        assert float(step.mean()) == pytest.approx(0.1, abs=0.03)
        assert float(step.std()) < 0.03


def test_video_pair_dataset_frame_gap(video_root: Path) -> None:
    dataset = VideoPairDataset(video_root, crop_size=32, frame_gap=2)
    assert len(dataset) == 3 + 2
    for i in range(len(dataset)):
        pair = dataset[i]
        assert float((pair["image2"] - pair["image1"]).mean()) == pytest.approx(0.2, abs=0.04)
    assert (
        len(VideoPairDataset(video_root, crop_size=32, frame_gap=4)) == 1
    )  # only the 5-frame clip


def test_video_pair_dataset_content_follows_frame_index(tmp_path: Path) -> None:
    root = tmp_path / "flat_clips"
    root.mkdir()
    for name, frames in (("a.avi", 5), ("b.avi", 4)):
        # Frame t is the constant 0.1*(t+1), so crops cannot change the mean.
        write_video(
            iter([torch.full((1, 3, 32, 48), 0.1 * (t + 1)) for t in range(frames)]),
            root / name,
            10.0,
        )
    dataset = VideoPairDataset(root, crop_size=32)
    first = [float(dataset[i]["image1"].mean()) for i in range(len(dataset))]
    second = [float(dataset[i]["image2"].mean()) for i in range(len(dataset))]
    assert first == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3], abs=0.03)
    assert second == pytest.approx([0.2, 0.3, 0.4, 0.5, 0.2, 0.3, 0.4], abs=0.03)


def test_video_pair_dataset_reproducible_crops(video_root: Path) -> None:
    dataset = VideoPairDataset(video_root, crop_size=16, seed=17)
    first = [dataset[i]["image1"] for i in range(len(dataset))]
    assert all(torch.equal(first[i], dataset[i]["image1"]) for i in range(len(dataset)))
    dataset.set_epoch(3)
    assert any(not torch.equal(first[i], dataset[i]["image1"]) for i in range(len(dataset)))
    dataset.set_epoch(0)
    assert all(torch.equal(first[i], dataset[i]["image1"]) for i in range(len(dataset)))
    with pytest.raises(ValueError):
        dataset.set_epoch(-1)


def test_video_pair_dataset_validation(video_root: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="paired_root"):
        VideoPairDataset(video_root, crop_size=32, paired_root=tmp_path)
    for gap in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="frame_gap"):
            VideoPairDataset(video_root, crop_size=32, frame_gap=gap)  # type: ignore[arg-type]
    for crop in (0, 20, -16):
        with pytest.raises(ValueError, match="multiple of 16"):
            VideoPairDataset(video_root, crop_size=crop)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no usable videos"):
        VideoPairDataset(empty, crop_size=16)
    with pytest.raises(NotADirectoryError):
        VideoPairDataset(tmp_path / "nope", crop_size=16)
    dataset = VideoPairDataset(video_root, crop_size=32)
    for bad_index in (-1, len(dataset), 999):
        with pytest.raises(IndexError):
            dataset[bad_index]


def test_video_pair_dataset_skips_corrupt_short_and_small_videos(
    video_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (video_root / "broken.mp4").write_bytes(b"definitely not a video" * 20)
    _write_clip(video_root / "single.avi", 1)  # no adjacent pair
    _write_clip(video_root / "tiny.avi", 3, height=16, width=16)  # smaller than the crop
    with caplog.at_level(logging.WARNING, logger="nta_zerodce.data.video_pairs"):
        dataset = VideoPairDataset(video_root, crop_size=32)
    assert len(dataset) == 7  # only the two good clips contribute
    assert "skipped 1 unreadable, 1 too short" in caplog.text
    assert "1 smaller-than-crop (32)" in caplog.text
    only_bad = video_root.parent / "bad_only"
    only_bad.mkdir()
    (only_bad / "x.mkv").write_bytes(b"garbage")
    with pytest.raises(ValueError, match="no usable videos"):
        VideoPairDataset(only_bad, crop_size=16)


def test_video_pair_dataset_dataloader_collation(video_root: Path) -> None:
    dataset = VideoPairDataset(video_root, crop_size=32)
    batch = next(iter(DataLoader(dataset, batch_size=4, num_workers=0)))
    assert batch["image1"].shape == (4, 3, 32, 32)
    assert batch["index"].tolist() == [0, 1, 2, 3] and batch["index"].dtype == torch.int64
