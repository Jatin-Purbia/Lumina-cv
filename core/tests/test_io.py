"""Image and video IO: decoding policy, quantization, EXIF, round trips and resource release."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from nta_zerodce.inference import io as lumina_io
from nta_zerodce.inference.io import read_image, read_video, write_image, write_video


def _mp4_supported(tmp_path: Path) -> bool:
    """Whether this OpenCV build can open an mp4v writer."""
    probe = tmp_path / "probe.mp4"
    writer = cv2.VideoWriter(str(probe), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 32))
    opened = bool(writer.isOpened())
    writer.release()
    probe.unlink(missing_ok=True)
    return opened


def _rgb_array(height: int = 24, width: int = 40, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


def _frames(count: int = 4, height: int = 32, width: int = 48) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(3)
    base = torch.rand(1, 3, height, width, generator=generator)
    return [torch.clamp(base * (0.5 + 0.1 * i), 0, 1) for i in range(count)]


# ---------------------------------------------------------------- read_image


def test_read_image_png_exact_values(tmp_path: Path) -> None:
    array = _rgb_array()
    path = tmp_path / "a.png"
    Image.fromarray(array).save(path)
    tensor = read_image(path)
    assert tensor.shape == (1, 3, 24, 40)
    assert tensor.dtype == torch.float32 and tensor.device.type == "cpu"
    expected = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
    assert torch.equal(tensor[0], expected)
    assert 0.0 <= float(tensor.min()) and float(tensor.max()) <= 1.0


def test_read_image_grayscale_palette_alpha_to_rgb(tmp_path: Path) -> None:
    gray = np.arange(0, 256, 16, dtype=np.uint8).reshape(4, 4)
    Image.fromarray(gray, mode="L").save(tmp_path / "g.png")
    out = read_image(tmp_path / "g.png")
    assert out.shape == (1, 3, 4, 4)
    assert torch.equal(out[0, 0], out[0, 1]) and torch.equal(out[0, 1], out[0, 2])
    assert torch.allclose(out[0, 0], torch.from_numpy(gray).float() / 255.0)

    palette = Image.fromarray(gray, mode="L").convert("P")
    palette.save(tmp_path / "p.png")
    assert read_image(tmp_path / "p.png").shape == (1, 3, 4, 4)

    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[..., 0] = 200
    rgba[..., 3] = 0  # fully transparent -> black when composited on black
    rgba[0, :, 3] = 255  # first row opaque
    Image.fromarray(rgba, mode="RGBA").save(tmp_path / "a.png")
    composite = read_image(tmp_path / "a.png")[0]
    assert torch.allclose(composite[0, 0], torch.full((4,), 200 / 255))
    assert float(composite[:, 1:].abs().max()) == 0.0


def test_read_image_rejects_16_bit(tmp_path: Path) -> None:
    Image.fromarray(np.zeros((4, 4), dtype=np.uint16)).save(tmp_path / "s.png")
    with pytest.raises(ValueError, match="unsupported image mode"):
        read_image(tmp_path / "s.png")


def test_read_image_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_image(tmp_path / "missing.png")
    with pytest.raises(ValueError, match="not a file"):
        read_image(tmp_path)
    text = tmp_path / "fake.png"
    text.write_text("this is not an image")
    with pytest.raises(ValueError, match="cannot decode"):
        read_image(text)
    truncated = tmp_path / "t.png"
    Image.fromarray(_rgb_array(64, 64)).save(truncated)
    truncated.write_bytes(truncated.read_bytes()[:60])
    with pytest.raises(ValueError, match="cannot decode"):
        read_image(truncated)


def test_read_image_applies_exif_orientation(tmp_path: Path) -> None:
    array = np.zeros((20, 30, 3), dtype=np.uint8)
    array[:10, :15] = (250, 0, 0)  # red top-left quadrant
    exif = Image.Exif()
    exif[0x0112] = 6  # stored rotated: display needs 270 degree rotation -> 30x20 becomes 20x30
    path = tmp_path / "o.jpg"
    Image.fromarray(array).save(path, quality=100, exif=exif)
    tensor = read_image(path)
    # Orientation 6 rotates the stored image 90 degrees clockwise: HxW swap.
    assert tensor.shape == (1, 3, 30, 20)
    # Red top-left of the stored image ends up top-right after a clockwise rotation.
    top_right = tensor[0, :, :15, 10:]
    top_left = tensor[0, :, :15, :10]
    assert float(top_right[0].mean()) > 0.8 and float(top_right[1].mean()) < 0.1
    assert float(top_left[0].mean()) < 0.1


# --------------------------------------------------------------- write_image


def test_write_image_roundtrip_png_is_lossless(tmp_path: Path) -> None:
    array = _rgb_array()
    tensor = torch.from_numpy(array).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    path = tmp_path / "out.PNG"
    write_image(tensor, path)
    assert np.array_equal(np.asarray(Image.open(path)), array)
    assert torch.equal(read_image(path), tensor)


def test_write_image_quantization_policy(tmp_path: Path) -> None:
    values = torch.tensor([0.0, 0.5, 1.0, 1.2, -0.3, 0.002, 0.998]).view(1, 1, 1, 7)
    image = values.expand(1, 3, 1, 7).contiguous()
    write_image(image, tmp_path / "q.png")
    stored = np.asarray(Image.open(tmp_path / "q.png"))[0, :, 0]
    assert stored.tolist() == [0, 128, 255, 255, 0, 1, 254]  # round(255*x), clamped to [0,1]


def test_write_image_detaches_and_accepts_grad_tensor(tmp_path: Path) -> None:
    image = torch.full((1, 3, 4, 4), 0.25, requires_grad=True)
    write_image(image * 1.0, tmp_path / "g.png")
    assert image.grad is None
    assert np.asarray(Image.open(tmp_path / "g.png")).max() == 64  # round(63.75)


def test_write_image_jpeg_close(tmp_path: Path) -> None:
    tensor = torch.full((1, 3, 16, 16), 0.5)
    write_image(tensor, tmp_path / "x.jpeg")
    assert Image.open(tmp_path / "x.jpeg").format == "JPEG"
    assert torch.allclose(read_image(tmp_path / "x.jpeg"), tensor, atol=3 / 255)


def test_write_image_errors(tmp_path: Path) -> None:
    good = torch.zeros(1, 3, 4, 4)
    with pytest.raises(ValueError, match="shape"):
        write_image(torch.zeros(3, 4, 4), tmp_path / "a.png")
    with pytest.raises(ValueError, match="shape"):
        write_image(torch.zeros(2, 3, 4, 4), tmp_path / "a.png")
    with pytest.raises(ValueError, match="shape"):
        write_image(torch.zeros(1, 1, 4, 4), tmp_path / "a.png")
    with pytest.raises(TypeError):
        write_image(torch.zeros(1, 3, 4, 4, dtype=torch.uint8), tmp_path / "a.png")
    bad = good.clone()
    bad[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        write_image(bad, tmp_path / "a.png")
    with pytest.raises(ValueError, match="extension"):
        write_image(good, tmp_path / "a.bmp")
    with pytest.raises(FileNotFoundError):
        write_image(good, tmp_path / "no_such_dir" / "a.png")
    assert not (tmp_path / "no_such_dir").exists()  # parent directories are never created


# -------------------------------------------------------------------- video


def test_video_roundtrip_avi(tmp_path: Path) -> None:
    frames = _frames()
    path = tmp_path / "clip.avi"
    write_video(iter(frames), path, fps=10.0)
    assert path.exists()
    decoded = list(read_video(path))
    assert len(decoded) == len(frames)
    for original, got in zip(frames, decoded):
        assert got.shape == original.shape and got.dtype == torch.float32
        # MJPG is lossy; compare mean absolute error on 8-bit scale.
        assert float((got - original).abs().mean()) < 0.06


def test_video_roundtrip_mp4(tmp_path: Path) -> None:
    if not _mp4_supported(tmp_path):
        pytest.skip("this OpenCV build cannot open an mp4v VideoWriter")
    frames = _frames(count=3)
    path = tmp_path / "clip.mp4"
    write_video(iter(frames), path, fps=12.5)
    decoded = list(read_video(path))
    assert len(decoded) == 3
    assert float((decoded[0] - frames[0]).abs().mean()) < 0.08


def test_video_channel_order_is_rgb(tmp_path: Path) -> None:
    red = torch.zeros(1, 3, 32, 32)
    red[:, 0] = 1.0
    path = tmp_path / "red.avi"
    write_video(iter([red, red]), path, fps=5.0)
    frame = next(read_video(path))
    assert float(frame[0, 0].mean()) > 0.8 and float(frame[0, 2].mean()) < 0.2


def test_write_video_validation(tmp_path: Path) -> None:
    frames = _frames(count=2)
    for fps in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="fps"):
            write_video(iter(frames), tmp_path / "f.avi", fps=fps)
    with pytest.raises(ValueError, match="extension"):
        write_video(iter(frames), tmp_path / "f.webm", fps=10.0)
    with pytest.raises(FileNotFoundError):
        write_video(iter(frames), tmp_path / "missing" / "f.avi", fps=10.0)
    with pytest.raises(ValueError, match="no frames"):
        write_video(iter([]), tmp_path / "empty.avi", fps=10.0)
    mismatch = [frames[0], torch.zeros(1, 3, 16, 16)]
    with pytest.raises(ValueError, match="expected"):
        write_video(iter(mismatch), tmp_path / "m.avi", fps=10.0)
    assert not (tmp_path / "m.avi").exists()  # partial output removed on error
    with pytest.raises(ValueError, match="shape"):
        write_video(iter([torch.zeros(3, 8, 8)]), tmp_path / "s.avi", fps=10.0)


def test_write_video_releases_writer_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    released: list[bool] = []
    real_writer = cv2.VideoWriter

    class Spy:
        def __init__(self, *args: object) -> None:
            self.inner = real_writer(*args)  # type: ignore[call-overload]

        def isOpened(self) -> bool:  # noqa: N802 - OpenCV naming
            return bool(self.inner.isOpened())

        def write(self, frame: np.ndarray) -> None:
            self.inner.write(frame)

        def release(self) -> None:
            released.append(True)
            self.inner.release()

    monkeypatch.setattr(lumina_io.cv2, "VideoWriter", Spy)
    with pytest.raises(ValueError):
        write_video(iter([torch.zeros(1, 3, 32, 32), torch.zeros(1, 3, 16, 16)]), tmp_path / "e.avi", 5.0)
    assert released == [True]


def test_write_video_raises_when_writer_cannot_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Closed:
        def __init__(self, *args: object) -> None:
            self.released = False

        def isOpened(self) -> bool:  # noqa: N802
            return False

        def release(self) -> None:
            self.released = True

    monkeypatch.setattr(lumina_io.cv2, "VideoWriter", Closed)
    with pytest.raises(RuntimeError, match="cannot open video writer"):
        write_video(iter(_frames(1)), tmp_path / "c.avi", fps=10.0)


def test_read_video_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_video(tmp_path / "missing.avi")
    garbage = tmp_path / "garbage.avi"
    garbage.write_bytes(b"not a video at all" * 10)
    with pytest.raises(ValueError):
        list(read_video(garbage))


def test_read_video_releases_capture_on_close_exhaustion_and_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "clip.avi"
    write_video(iter(_frames(count=5)), path, fps=10.0)
    state = {"opened": 0, "released": 0}
    real_capture = cv2.VideoCapture

    class Spy:
        def __init__(self, *args: object) -> None:
            self.inner = real_capture(*args)  # type: ignore[call-overload]
            state["opened"] += 1

        def isOpened(self) -> bool:  # noqa: N802
            return bool(self.inner.isOpened())

        def read(self) -> tuple[bool, np.ndarray]:
            return self.inner.read()  # type: ignore[no-any-return]

        def release(self) -> None:
            state["released"] += 1
            self.inner.release()

    monkeypatch.setattr(lumina_io.cv2, "VideoCapture", Spy)

    stream = read_video(path)
    next(stream)
    assert state == {"opened": 1, "released": 0}
    stream.close()  # type: ignore[attr-defined]
    assert state == {"opened": 1, "released": 1}

    assert len(list(read_video(path))) == 5  # exhaustion
    assert state == {"opened": 2, "released": 2}

    garbage = tmp_path / "garbage.avi"
    garbage.write_bytes(b"x" * 100)
    with pytest.raises(ValueError):
        list(read_video(garbage))  # error path
    assert state["opened"] == state["released"] == 3
