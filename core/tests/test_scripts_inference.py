"""enhance and make_synthetic_data command lines, exercised in-process through ``main(argv)``."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from nta_zerodce.data.image_dataset import make_synthetic_images
from nta_zerodce.data.synthetic_pairs import make_pair
from nta_zerodce.data.transforms import item_generator
from nta_zerodce.inference.io import read_image, read_video, write_video
from nta_zerodce.models.factory import build_model
from nta_zerodce.training.checkpoint import save_checkpoint
from nta_zerodce.training.config import Config, override_config, save_config

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_OVERRIDES: dict[str, object] = {
    "model.channels": 8,
    "model.noise_channels": 4,
    "model.denoise_channels": 4,
    "flags.noise_aware": True,
    "flags.temporal": True,
}


def load_script(name: str) -> ModuleType:
    """Import ``scripts/<name>.py`` (scripts are not a package)."""
    spec = importlib.util.spec_from_file_location(
        f"lumina_script_{name}", ROOT / "scripts" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


enhance = load_script("enhance")
synth = load_script("make_synthetic_data")


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A random-initialised (untrained) noise-aware, temporal checkpoint with tiny widths."""
    config = override_config(Config(), CHECKPOINT_OVERRIDES)
    torch.manual_seed(0)
    model = build_model(config)
    path = tmp_path_factory.mktemp("ckpt") / "tiny.pt"
    save_checkpoint(path, model, torch.optim.Adam(model.parameters()), config, 0)
    return path


def make_images(directory: Path, count: int = 2, size: tuple[int, int] = (48, 64)) -> list[Path]:
    return make_synthetic_images(directory, count, size, 3)


def pixels(path: Path) -> torch.Tensor:
    return read_image(path)


def run_enhance(*argv: str | Path) -> int:
    return int(enhance.main([str(a) for a in argv]))


def video_frames(count: int = 6, height: int = 48, width: int = 64) -> Iterator[torch.Tensor]:
    """Smooth textured frames panning 2 px per frame."""
    rng = np.random.default_rng(0)
    coarse = rng.random((12, 24 + count, 3)).astype(np.float32)
    canvas = cv2.resize(coarse, (width + 2 * count, height), interpolation=cv2.INTER_CUBIC)
    canvas = 0.05 + 0.55 * (canvas - canvas.min()) / (canvas.max() - canvas.min())
    for t in range(count):
        window = np.ascontiguousarray(canvas[:, 2 * t : 2 * t + width])
        yield torch.from_numpy(window).permute(2, 0, 1).unsqueeze(0).contiguous()


# ----------------------------------------------------------------------- enhance: images


def test_help_texts_document_the_contract(capsys: pytest.CaptureFixture[str]) -> None:
    for module, needles in (
        (
            enhance,
            ["--checkpoint", "--webcam", "PNG", "audio", "--overwrite", "stored in the checkpoint"],
        ),
        (
            synth,
            ["--pairs", "--count", "--size", "--overwrite", "meta.json", "never writes outside"],
        ),
    ):
        with pytest.raises(SystemExit) as exit_info:
            module.main(["--help"])
        assert exit_info.value.code == 0
        text = " ".join(capsys.readouterr().out.split())
        for needle in needles:
            assert needle in text, (module.__name__, needle)


def test_single_image_into_directory_and_explicit_png(
    checkpoint: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = make_images(tmp_path / "in", 1)[0]
    assert (
        run_enhance("--checkpoint", checkpoint, "--input", source, "--output", tmp_path / "out")
        == 0
    )
    result = tmp_path / "out" / "synthetic_0000.png"
    assert result.is_file()
    before, after = pixels(source), pixels(result)
    assert after.shape == before.shape and not torch.equal(before, after)
    text = capsys.readouterr().out
    assert "synthetic_0000.png" in text and "ms" in text and "enhanced 1 of 1 image(s) in" in text
    explicit = tmp_path / "deep" / "er" / "named.png"
    assert run_enhance("--checkpoint", checkpoint, "--input", source, "--output", explicit) == 0
    assert torch.equal(pixels(explicit), after)  # deterministic inference
    assert (
        run_enhance("--checkpoint", checkpoint, "--input", source, "--output", tmp_path / "x.jpg")
        == 2
    )
    assert "PNG" in capsys.readouterr().err


def test_directory_tree_keeps_layout_and_writes_png(
    checkpoint: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "in"
    make_images(root, 2)
    nested = make_images(root / "scene" / "night", 1, (40, 40))[0]
    Image.open(nested).convert("RGB").save(root / "scene" / "photo.jpg")
    nested.unlink()
    (root / "notes.txt").write_text("not an image")
    assert (
        run_enhance("--checkpoint", checkpoint, "--input", root, "--output", tmp_path / "out") == 0
    )
    produced = sorted(
        p.relative_to(tmp_path / "out").as_posix()
        for p in (tmp_path / "out").rglob("*")
        if p.is_file()
    )
    assert produced == ["scene/photo.png", "synthetic_0000.png", "synthetic_0001.png"]
    assert pixels(tmp_path / "out" / "scene" / "photo.png").shape == (1, 3, 40, 40)
    assert "enhanced 3 of 3 image(s)" in capsys.readouterr().out


def test_curve_scale_flag_and_config_precedence(checkpoint: Path, tmp_path: Path) -> None:
    source = make_images(tmp_path / "in", 1, (64, 64))[0]
    assert (
        run_enhance("--checkpoint", checkpoint, "--input", source, "--output", tmp_path / "a") == 0
    )
    assert (
        run_enhance(
            "--checkpoint",
            checkpoint,
            "--input",
            source,
            "--output",
            tmp_path / "b",
            "--curve-scale",
            "0.25",
            "--device",
            "cpu",
        )
        == 0
    )
    full, small = (
        pixels(tmp_path / "a" / "synthetic_0000.png"),
        pixels(tmp_path / "b" / "synthetic_0000.png"),
    )
    assert full.shape == small.shape and not torch.equal(full, small)
    config = override_config(Config(), CHECKPOINT_OVERRIDES)
    save_config(config, tmp_path / "exp.yaml")
    assert (
        run_enhance(
            "--checkpoint",
            checkpoint,
            "--config",
            tmp_path / "exp.yaml",
            "--input",
            source,
            "--output",
            tmp_path / "c",
        )
        == 0
    )
    assert torch.equal(pixels(tmp_path / "c" / "synthetic_0000.png"), full)  # same as stored config


def test_usage_errors_exit_two_with_one_line(
    checkpoint: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = make_images(tmp_path / "in", 1)[0]
    out = tmp_path / "out"
    cases: list[tuple[list[str | Path], str]] = [
        (["--input", source, "--output", out], "--checkpoint is required"),
        (["--checkpoint", checkpoint, "--output", out], "--input"),
        (["--checkpoint", checkpoint, "--input", source], "--output is required"),
        (["--checkpoint", checkpoint, "--webcam", "0", "--output", out], "--output is not used"),
        (["--checkpoint", checkpoint, "--webcam", "-1"], "camera index"),
        (
            ["--checkpoint", tmp_path / "none.pt", "--input", source, "--output", out],
            "checkpoint not found",
        ),
        (
            ["--checkpoint", checkpoint, "--input", tmp_path / "none.png", "--output", out],
            "input not found",
        ),
        (
            [
                "--checkpoint",
                checkpoint,
                "--input",
                tmp_path / "in",
                "--output",
                tmp_path / "in" / "o",
            ],
            "inside",
        ),
        (
            ["--checkpoint", checkpoint, "--input", source, "--output", out, "--curve-scale", "2"],
            "curve_scale",
        ),
        (["--checkpoint", checkpoint, "--input", source, "--output", out, "--light"], "light"),
        (
            [
                "--checkpoint",
                checkpoint,
                "--input",
                source,
                "--output",
                out,
                "--config",
                "configs/baseline.yaml",
            ],
            "noise_aware",
        ),
        (
            [
                "--checkpoint",
                checkpoint,
                "--input",
                source,
                "--output",
                out,
                "--no-noise",
                "--denoise",
            ],
            "denoise",
        ),
        (
            ["--checkpoint", checkpoint, "--input", source, "--output", out, "--device", "tpu"],
            "device",
        ),
    ]
    (tmp_path / "empty").mkdir()
    (tmp_path / "notes.txt").write_text("x")
    cases.append(
        (["--checkpoint", checkpoint, "--input", tmp_path / "empty", "--output", out], "no images")
    )
    cases.append(
        (
            ["--checkpoint", checkpoint, "--input", tmp_path / "notes.txt", "--output", out],
            "unsupported",
        )
    )
    for argv, message in cases:
        capsys.readouterr()
        assert run_enhance(*argv) == 2, argv
        err = capsys.readouterr().err.strip()
        assert err.startswith("enhance: error:") and message in err and "\n" not in err, (argv, err)
    assert not out.exists()  # nothing is created by a failed run


def test_input_and_webcam_conflict_is_an_argparse_error(
    checkpoint: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = make_images(tmp_path / "in", 1)[0]
    with pytest.raises(SystemExit) as exit_info:
        run_enhance(
            "--checkpoint",
            checkpoint,
            "--input",
            source,
            "--output",
            tmp_path / "o",
            "--webcam",
            "0",
        )
    assert exit_info.value.code == 2
    assert "not allowed with" in capsys.readouterr().err


def test_overwrite_protection(
    checkpoint: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "in"
    make_images(root, 2)
    out = tmp_path / "out"
    args: list[str | Path] = ["--checkpoint", checkpoint, "--input", root, "--output", out]
    assert run_enhance(*args) == 0
    target = out / "synthetic_0001.png"
    target.write_bytes(b"sentinel")
    capsys.readouterr()
    assert run_enhance(*args) == 2
    assert "exists" in capsys.readouterr().err and target.read_bytes() == b"sentinel"
    assert run_enhance(*args, "--overwrite") == 0
    assert target.read_bytes() != b"sentinel" and pixels(target).shape == (1, 3, 48, 64)
    assert (
        run_enhance(
            "--checkpoint",
            checkpoint,
            "--input",
            root / "synthetic_0000.png",
            "--output",
            root / "synthetic_0000.png",
            "--overwrite",
        )
        == 2
    )  # never the input
    assert "overwrite the input" in capsys.readouterr().err


def test_output_name_collisions_are_rejected(
    checkpoint: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "in"
    first = make_images(root, 1)[0]
    Image.open(first).save(root / "synthetic_0000.jpg")
    assert run_enhance("--checkpoint", checkpoint, "--input", root, "--output", tmp_path / "o") == 2
    assert "both be written" in capsys.readouterr().err and not (tmp_path / "o").exists()


def test_tiny_and_undecodable_files_fail_that_file_but_not_the_batch(
    checkpoint: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "in"
    make_images(root, 2)
    Image.fromarray(np.full((8, 8, 3), 100, dtype=np.uint8)).save(root / "tiny.png")
    (root / "broken.png").write_bytes(b"\x89PNG not really")
    assert (
        run_enhance("--checkpoint", checkpoint, "--input", root, "--output", tmp_path / "out") == 1
    )
    captured = capsys.readouterr()
    assert "tiny.png" in captured.err and ">= 16" in captured.err and "broken.png" in captured.err
    assert "2 of 4 image(s) failed" in captured.err and "enhanced 2 of 4" in captured.out
    produced = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert produced == ["synthetic_0000.png", "synthetic_0001.png"]
    assert (
        run_enhance(
            "--checkpoint", checkpoint, "--input", root / "tiny.png", "--output", tmp_path / "t"
        )
        == 1
    )
    assert "1 of 1 image(s) failed" in capsys.readouterr().err


# ----------------------------------------------------------------------- enhance: video


def test_video_round_trip_keeps_size_count_and_fps(
    checkpoint: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "clip.avi"
    write_video(video_frames(6), source, fps=10.0)
    target = tmp_path / "out" / "clip_enhanced.avi"
    assert run_enhance("--checkpoint", checkpoint, "--input", source, "--output", target) == 0
    original = list(read_video(source))
    enhanced = list(read_video(target))
    assert len(enhanced) == len(original) == 6
    assert enhanced[0].shape == original[0].shape == (1, 3, 48, 64)
    assert not torch.allclose(enhanced[3], original[3], atol=0.02)
    capture = cv2.VideoCapture(str(target))
    try:
        assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(10.0, abs=0.01)
    finally:
        capture.release()
    text = capsys.readouterr().out
    assert "6 frames at 10.00 fps" in text and "mean mask coverage" in text and "scene cut" in text
    assert run_enhance("--checkpoint", checkpoint, "--input", source, "--output", target) == 2
    assert "exists" in capsys.readouterr().err
    assert (
        run_enhance(
            "--checkpoint",
            checkpoint,
            "--input",
            source,
            "--output",
            target,
            "--overwrite",
            "--no-temporal",
        )
        == 0
    )
    assert "temporal off" in capsys.readouterr().out


def test_video_temporal_flag_changes_the_result_and_matches_framewise_when_off(
    checkpoint: Path, tmp_path: Path
) -> None:
    source = tmp_path / "clip.avi"
    write_video(video_frames(4), source, fps=8.0)
    assert (
        run_enhance("--checkpoint", checkpoint, "--input", source, "--output", tmp_path / "t.avi")
        == 0
    )
    assert (
        run_enhance(
            "--checkpoint",
            checkpoint,
            "--input",
            source,
            "--output",
            tmp_path / "f.avi",
            "--no-temporal",
        )
        == 0
    )
    temporal, framewise = list(read_video(tmp_path / "t.avi")), list(read_video(tmp_path / "f.avi"))
    assert torch.equal(temporal[0], framewise[0])  # the first frame has no history
    assert any(not torch.equal(a, b) for a, b in zip(temporal[1:], framewise[1:], strict=True))


def test_video_output_validation(
    checkpoint: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "clip.avi"
    write_video(video_frames(2), source, fps=5.0)
    assert (
        run_enhance("--checkpoint", checkpoint, "--input", source, "--output", tmp_path / "o.mkv")
        == 2
    )
    assert ".mp4 or .avi" in capsys.readouterr().err
    assert (
        run_enhance("--checkpoint", checkpoint, "--input", source, "--output", tmp_path / "o") == 2
    )
    assert (
        run_enhance(
            "--checkpoint", checkpoint, "--input", source, "--output", source, "--overwrite"
        )
        == 2
    )
    assert "overwrite the input" in capsys.readouterr().err
    tiny = tmp_path / "tiny.avi"
    write_video(video_frames(2, 12, 12), tiny, fps=5.0)
    assert (
        run_enhance("--checkpoint", checkpoint, "--input", tiny, "--output", tmp_path / "t.avi")
        == 2
    )
    assert (
        ">= 16" in capsys.readouterr().err and not (tmp_path / "t.avi").exists()
    )  # no partial file


def test_unreadable_video_is_a_clean_error(
    checkpoint: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"this is not a video")
    assert (
        run_enhance("--checkpoint", checkpoint, "--input", bad, "--output", tmp_path / "o.avi") == 2
    )
    assert "bad.mp4" in capsys.readouterr().err and not (tmp_path / "o.avi").exists()


# ----------------------------------------------------------------------- enhance: webcam


def test_webcam_mode_delegates_to_run_webcam(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[object, ...]] = []

    def fake_run_webcam(model: object, config: Config, camera_id: int = 0) -> None:
        calls.append(
            (type(model).__name__, config.model.curve_scale, config.flags.temporal, camera_id)
        )

    monkeypatch.setattr(enhance, "run_webcam", fake_run_webcam)
    assert (
        run_enhance(
            "--checkpoint", checkpoint, "--webcam", "2", "--curve-scale", "0.5", "--no-temporal"
        )
        == 0
    )
    assert calls == [("NTAZeroDCE", 0.5, False, 2)]
    assert run_enhance("--checkpoint", checkpoint, "--webcam", "0") == 0
    assert calls[-1] == ("NTAZeroDCE", 1.0, True, 0)  # stored config values


def test_webcam_failure_is_a_runtime_exit(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(model: object, config: Config, camera_id: int = 0) -> None:
        raise RuntimeError(f"cannot open camera {camera_id}")

    monkeypatch.setattr(enhance, "run_webcam", broken)
    assert run_enhance("--checkpoint", checkpoint, "--webcam", "7") == 1
    assert "cannot open camera 7" in capsys.readouterr().err


# ----------------------------------------------------------------------- make_synthetic_data


def run_synth(*argv: str | Path) -> int:
    return int(synth.main([str(a) for a in argv]))


def tree(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_synthetic_images_are_written_with_sidecars_and_are_reproducible(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_synth("--output", tmp_path / "a", "--count", "3", "--size", "32") == 0
    assert "wrote 3 synthetic noisy image(s)" in capsys.readouterr().out
    samples = sorted(p.name for p in (tmp_path / "a").iterdir())
    assert samples == ["synthetic_0000", "synthetic_0001", "synthetic_0002"]
    sample = tmp_path / "a" / "synthetic_0001"
    assert sorted(p.name for p in sample.iterdir()) == ["clean.png", "meta.json", "noisy.png"]
    clean, noisy = pixels(sample / "clean.png"), pixels(sample / "noisy.png")
    assert clean.shape == noisy.shape == (1, 3, 32, 32) and not torch.equal(clean, noisy)
    meta = json.loads((sample / "meta.json").read_text())
    assert meta["format"] == "lumina-synthetic-v1" and meta["mode"] == "image"
    assert meta["seed"] == 17 and meta["source"] == "procedural" and meta["source_index"] == 1
    assert (meta["height"], meta["width"]) == (32, 32) and meta["noise_applied"] is True
    assert 0.0 <= meta["noise_a"] <= 0.01 and 0.0 <= meta["noise_b"] <= 0.02
    assert set(meta["config"]) == {"noise", "synthetic"}
    assert run_synth("--output", tmp_path / "b", "--count", "3", "--size", "32") == 0
    assert tree(tmp_path / "a") == tree(tmp_path / "b")  # byte-identical for equal arguments
    assert run_synth("--output", tmp_path / "c", "--count", "3", "--size", "32", "--seed", "5") == 0
    assert tree(tmp_path / "c") != tree(tmp_path / "a")
    assert json.loads((tmp_path / "c" / "synthetic_0000" / "meta.json").read_text())["seed"] == 5


def test_synthetic_pairs_sidecar_reproduces_the_pair(tmp_path: Path) -> None:
    assert run_synth("--output", tmp_path / "p", "--pairs", "--count", "2", "--size", "48x64") == 0
    sample = tmp_path / "p" / "synthetic_0001"
    assert sorted(p.name for p in sample.iterdir()) == ["image1.png", "image2.png", "meta.json"]
    meta = json.loads((sample / "meta.json").read_text())
    assert meta["mode"] == "pair" and (meta["height"], meta["width"]) == (48, 64)
    assert np.array(meta["affine"]).shape == (2, 3) and np.array(meta["affine_inverse"]).shape == (
        2,
        3,
    )
    assert 0.9 <= meta["flicker"] <= 1.1 and len(meta["noise_a"]) == len(meta["noise_b"]) == 2
    assert 0.0 < meta["mask_coverage"] <= 1.0
    # Reproduce from the documented sampler: same source, same generator, same config.
    sources = make_synthetic_images(tmp_path / "src", 2, (48, 64), 17)
    pair = make_pair(read_image(sources[1]), Config(), item_generator(17, 0, 1))
    assert pair["affine"][0].tolist() == meta["affine"]
    assert float(pair["flicker"][0]) == meta["flicker"]
    assert [float(v) for v in pair["noise_a"][0]] == meta["noise_a"]
    forward = np.array(meta["affine"])
    inverse = np.array(meta["affine_inverse"])
    product = np.vstack([forward, [0, 0, 1]]) @ np.vstack([inverse, [0, 0, 1]])
    assert np.allclose(product, np.eye(3), atol=1e-4)  # matrices are consistent
    first = pixels(sample / "image1.png")
    expected = torch.round(255.0 * pair["image1"].clamp(0, 1)) / 255.0
    assert torch.allclose(first, expected, atol=1e-6)


def test_synthetic_from_input_folder_with_crop_count_and_skipping(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "src dir"
    make_synthetic_images(root / "scene one", 2, (64, 80), 1)
    make_synthetic_images(root / "tiny", 1, (24, 24), 2)
    out = tmp_path / "out"
    assert run_synth("--input", root, "--output", out, "--size", "32x48", "--count", "10") == 0
    text = capsys.readouterr().out
    assert "wrote 2 synthetic noisy image(s)" in text and "skipped 1" in text
    names = sorted(p.name for p in out.iterdir())
    assert names == ["0000_scene_one__synthetic_0000", "0001_scene_one__synthetic_0001"]
    meta = json.loads((out / names[0] / "meta.json").read_text())
    assert meta["source"] == "scene one/synthetic_0000.png"
    top, left = meta["crop_top_left"]
    assert 0 <= top <= 64 - 32 and 0 <= left <= 80 - 48
    source = read_image(root / "scene one" / "synthetic_0000.png")
    assert torch.equal(
        pixels(out / names[0] / "clean.png"), source[:, :, top : top + 32, left : left + 48]
    )
    assert all(out.resolve() in p.resolve().parents for p in out.rglob("*") if p.is_file())
    # --count limits the sources; a single file is a valid input
    out2 = tmp_path / "out2"
    assert run_synth("--input", root, "--output", out2, "--count", "1", "--pairs") == 0
    assert len(list(out2.iterdir())) == 1
    single = root / "scene one" / "synthetic_0001.png"
    assert run_synth("--input", single, "--output", tmp_path / "out3") == 0
    assert (tmp_path / "out3" / "0000_synthetic_0001" / "noisy.png").is_file()


def test_synthetic_refusals(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "out"
    assert run_synth("--output", out, "--count", "2", "--size", "32") == 0
    marker = out / "synthetic_0000" / "noisy.png"
    marker.write_bytes(b"sentinel")
    capsys.readouterr()
    assert run_synth("--output", out, "--count", "2", "--size", "32") == 2
    assert "exists" in capsys.readouterr().err and marker.read_bytes() == b"sentinel"
    assert run_synth("--output", out, "--count", "2", "--size", "32", "--overwrite") == 0
    assert marker.read_bytes() != b"sentinel"
    root = tmp_path / "src"
    make_synthetic_images(root, 1, 32, 1)
    cases: list[tuple[list[str | Path], str]] = [
        (["--count", "2"], "--output is required"),
        (["--output", tmp_path / "z", "--count", "0"], "--count"),
        (["--output", tmp_path / "z", "--input", tmp_path / "missing"], "input not found"),
        (["--output", root / "sub", "--input", root], "inside"),
        (["--output", tmp_path / "z", "--input", root, "--size", "64"], "no usable"),
    ]
    for argv, message in cases:
        capsys.readouterr()
        assert run_synth(*argv) == 2, argv
        assert message in capsys.readouterr().err
    (tmp_path / "t.txt").write_text("x")
    assert run_synth("--output", tmp_path / "z", "--input", tmp_path / "t.txt") == 2
    for bad_size in ("abc", "8", "32x8"):
        with pytest.raises(SystemExit) as exit_info:
            run_synth("--output", tmp_path / "z", "--size", bad_size)
        assert exit_info.value.code == 2
    assert not (tmp_path / "z").exists()
