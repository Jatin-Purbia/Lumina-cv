"""HTTP API: contract endpoints, JSON errors, upload limits, settings, startup and concurrency."""

from __future__ import annotations

import io
import logging
import re
import tempfile
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import NoReturn

import httpx
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from torch import Tensor

from nta_zerodce.api import Settings, SettingsError, StartupError, create_app
from nta_zerodce.api import app as app_module
from nta_zerodce.api.app import parse_curve_scale
from nta_zerodce.inference.enhancer import enhance_image
from nta_zerodce.inference.io import read_image
from nta_zerodce.inference.loader import load_inference_model
from nta_zerodce.models.factory import build_model
from nta_zerodce.models.nta_zerodce import ModelOutput, NTAZeroDCE
from nta_zerodce.training.checkpoint import save_checkpoint
from nta_zerodce.training.config import Config, load_config

ROOT = Path(__file__).resolve().parents[1]
BASE = "/api/v1"


# ------------------------------------------------------------------ helpers


def _pixels(height: int = 32, width: int = 48, seed: int = 3) -> np.ndarray:
    """Dim smooth-ish test content (low light) as uint8 HxWx3."""
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(0, 1, height), np.linspace(0, 1, width), indexing="ij")
    base = np.stack([xx, yy, 0.5 * (xx + yy)], axis=-1) * 60.0
    noise = rng.integers(0, 12, (height, width, 3))
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def _encode(image: Image.Image, fmt: str, **kwargs: object) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **kwargs)
    return buffer.getvalue()


def png_bytes(height: int = 32, width: int = 48) -> bytes:
    return _encode(Image.fromarray(_pixels(height, width)), "PNG")


def jpeg_bytes(height: int = 32, width: int = 48, **kwargs: object) -> bytes:
    return _encode(Image.fromarray(_pixels(height, width)), "JPEG", quality=95, **kwargs)


def post(
    client: TestClient,
    data: bytes,
    content_type: str | None = "image/png",
    filename: str = "upload.png",
    fields: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    files = {"file": (filename, data, content_type)} if content_type else {"file": data}
    response: httpx.Response = client.post(
        f"{BASE}/enhance", files=files, data=fields, headers=headers
    )
    return response


def decode(response_content: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(response_content)) as image:
        assert image.format == "PNG" and image.mode == "RGB"
        return np.asarray(image)


def assert_error(response: httpx.Response, status: int, code: str) -> None:
    assert response.status_code == status, response.text
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message"}
    assert body["error"]["code"] == code
    assert isinstance(body["error"]["message"], str) and body["error"]["message"]


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A random-init (untrained) full-feature checkpoint; the API only needs a loadable model."""
    torch.manual_seed(0)
    config = load_config(ROOT / "configs" / "full.py")
    model = build_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    path = tmp_path_factory.mktemp("ckpt") / "random_init.pt"
    save_checkpoint(path, model, optimizer, config, epoch=0)
    return path


@pytest.fixture(scope="module")
def settings(checkpoint: Path) -> Settings:
    return Settings(checkpoint=checkpoint, device="cpu")


@pytest.fixture(scope="module")
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def bare_client() -> Iterator[TestClient]:
    """Server started without a checkpoint."""
    with TestClient(create_app(Settings(device="cpu"))) as test_client:
        yield test_client


# ------------------------------------------------------------------ health and options


def test_health_with_model(client: TestClient) -> None:
    response = client.get(f"{BASE}/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["device"] == "cpu" and isinstance(body["version"], str) and body["version"]
    assert response.headers["x-content-type-options"] == "nosniff"


def test_health_without_model(bare_client: TestClient) -> None:
    body = bare_client.get(f"{BASE}/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is False
    assert body["device"] == "cpu"


def test_options_values(client: TestClient, settings: Settings) -> None:
    body = client.get(f"{BASE}/options").json()
    assert set(body) == {"flags", "curve_scale", "max_upload_bytes", "max_side", "accepted_types"}
    assert body["accepted_types"] == ["image/png", "image/jpeg"]
    assert body["max_upload_bytes"] == 20 * 1024 * 1024 and body["max_side"] == 4096
    assert body["curve_scale"] == 1.0
    full = load_config(ROOT / "configs" / "full.py")
    assert body["flags"]["noise_aware"] is True and body["flags"]["denoise"] is True
    assert set(body["flags"]) == set(vars(full.flags))


def test_options_without_model_and_overrides(checkpoint: Path) -> None:
    custom = Settings(max_upload_bytes=5000, max_side=100, default_curve_scale=0.5)
    with TestClient(create_app(custom)) as bare:
        body = bare.get(f"{BASE}/options").json()
    assert (body["max_upload_bytes"], body["max_side"], body["curve_scale"]) == (5000, 100, 0.5)
    assert body["flags"]["noise_aware"] is False  # library defaults without a checkpoint


# ------------------------------------------------------------------ enhance success


def test_enhance_png(client: TestClient) -> None:
    source = png_bytes(32, 48)
    response = post(client, source)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-lumina-size"] == "48x32"
    assert re.fullmatch(r"\d+", response.headers["x-lumina-elapsed-ms"])
    assert response.headers["x-content-type-options"] == "nosniff"
    out = decode(response.content)
    assert out.shape == (32, 48, 3)
    with Image.open(io.BytesIO(source)) as original:
        assert not np.array_equal(out, np.asarray(original.convert("RGB")))  # model applied


def test_enhance_jpeg_with_params_in_content_type(client: TestClient) -> None:
    response = post(client, jpeg_bytes(40, 24), "image/jpeg; charset=binary", "photo.jpg")
    assert response.status_code == 200
    assert response.headers["x-lumina-size"] == "24x40"
    assert decode(response.content).shape == (40, 24, 3)


def test_output_equals_enhance_image_exactly(
    client: TestClient, checkpoint: Path, tmp_path: Path
) -> None:
    source = png_bytes(40, 56)
    path = tmp_path / "in.png"
    path.write_bytes(source)
    model, _, device = load_inference_model(checkpoint, device="cpu")
    with torch.inference_mode():
        expected = enhance_image(model, read_image(path).to(device))["enhanced"]
    expected_u8 = torch.round(255.0 * expected.clamp(0, 1)).to(torch.uint8)[0].permute(1, 2, 0)
    response = post(client, source)
    assert response.status_code == 200
    assert np.array_equal(decode(response.content), expected_u8.numpy())


def test_exif_rotated_jpeg_gives_rotated_dimensions(client: TestClient) -> None:
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise to display
    response = post(client, jpeg_bytes(32, 48, exif=exif), "image/jpeg", "r.jpg")
    assert response.status_code == 200
    assert response.headers["x-lumina-size"] == "32x48"
    assert decode(response.content).shape == (48, 32, 3)


def test_rgba_png_is_composited_on_black(client: TestClient) -> None:
    rgba = np.concatenate([_pixels(), np.full((32, 48, 1), 255, np.uint8)], axis=-1)
    rgba[:, :8, 3] = 0  # transparent strip
    response = post(client, _encode(Image.fromarray(rgba, "RGBA"), "PNG"))
    assert response.status_code == 200
    assert decode(response.content).shape == (32, 48, 3)
    # The same pixels pre-composited on black must give the identical result.
    flat = rgba[..., :3].copy()
    flat[:, :8] = 0
    again = post(client, _encode(Image.fromarray(flat), "PNG"))
    assert np.array_equal(decode(response.content), decode(again.content))


def test_grayscale_png(client: TestClient) -> None:
    gray = Image.fromarray(_pixels()[..., 0], "L")
    response = post(client, _encode(gray, "PNG"))
    assert response.status_code == 200
    assert decode(response.content).shape == (32, 48, 3)


@pytest.mark.parametrize("value", ["1", "1.0", "0.5", "0.25", " 0.5 ", "5e-1", ".5"])
def test_curve_scale_variants(client: TestClient, value: str) -> None:
    response = post(client, png_bytes(), fields={"curve_scale": value})
    assert response.status_code == 200, response.text
    assert response.headers["x-lumina-size"] == "48x32"


def test_curve_scale_changes_result_and_default_applies(client: TestClient) -> None:
    source = png_bytes(64, 64)
    default = post(client, source).content
    assert default == post(client, source, fields={"curve_scale": "1"}).content
    assert default != post(client, source, fields={"curve_scale": "0.25"}).content


def test_default_curve_scale_from_settings(checkpoint: Path) -> None:
    source = png_bytes(64, 64)
    with TestClient(create_app(Settings(checkpoint=checkpoint, device="cpu"))) as full:
        reference = post(full, source, fields={"curve_scale": "0.25"}).content
    custom = Settings(checkpoint=checkpoint, device="cpu", default_curve_scale=0.25)
    with TestClient(create_app(custom)) as quarter:
        assert post(quarter, source).content == reference
        assert quarter.get(f"{BASE}/options").json()["curve_scale"] == 0.25


def test_unknown_form_fields_are_ignored(client: TestClient) -> None:
    response = post(client, png_bytes(), fields={"colour": "red", "x": "1"})
    assert response.status_code == 200


@pytest.mark.parametrize(
    "value", ["0", "0.0", "1.5", "abc", "nan", "inf", "-1", "-0.5", "", "1e400"]
)
def test_invalid_curve_scale(client: TestClient, value: str) -> None:
    assert_error(post(client, png_bytes(), fields={"curve_scale": value}), 422, "bad_option")


def test_duplicate_curve_scale_is_bad_option(client: TestClient) -> None:
    response = client.post(
        f"{BASE}/enhance",
        files=[
            ("file", ("a.png", png_bytes(), "image/png")),
            ("curve_scale", (None, "0.5")),
            ("curve_scale", (None, "0.5")),
        ],
    )
    assert_error(response, 422, "bad_option")


def test_parse_curve_scale_unit() -> None:
    assert parse_curve_scale("0.75") == 0.75
    for bad in ["", "0", "2", "1_0", "١", "0x1", "1,5", "+0.5", "--1"]:
        with pytest.raises(Exception, match="curve_scale"):
            parse_curve_scale(bad)


# ------------------------------------------------------------------ enhance errors


def test_corrupt_bytes_are_invalid_image(client: TestClient) -> None:
    corrupt = b"\x89PNG\r\n\x1a\n" + b"not really a png" * 8
    assert_error(post(client, corrupt), 400, "invalid_image")
    assert_error(
        post(client, b"\xff\xd8\xff\xe0" + b"junk" * 40, "image/jpeg"), 400, "invalid_image"
    )


def test_truncated_images_are_invalid_image(client: TestClient) -> None:
    whole = png_bytes(48, 64)
    assert_error(post(client, whole[: len(whole) // 2]), 400, "invalid_image")
    jpeg = jpeg_bytes(48, 64)
    assert_error(post(client, jpeg[: len(jpeg) // 2], "image/jpeg"), 400, "invalid_image")


def test_text_file_with_image_content_type(client: TestClient) -> None:
    assert_error(post(client, b"hello, I am not an image\n", "image/png"), 400, "invalid_image")


def test_declared_type_must_match_content(client: TestClient) -> None:
    assert_error(post(client, jpeg_bytes(), "image/png"), 400, "invalid_image")
    assert_error(post(client, png_bytes(), "image/jpeg"), 400, "invalid_image")


def test_empty_file_is_invalid_image(client: TestClient) -> None:
    assert_error(post(client, b""), 400, "invalid_image")


def test_unsupported_formats_are_415(client: TestClient) -> None:
    image = Image.fromarray(_pixels())
    gif = _encode(image.convert("P"), "GIF")
    webp = _encode(image, "WEBP")
    tiff = _encode(image, "TIFF")
    assert_error(post(client, gif, "image/gif", "a.gif"), 415, "unsupported_type")
    assert_error(post(client, gif, "image/png"), 415, "unsupported_type")  # real GIF, wrong label
    assert_error(post(client, webp, "image/webp", "a.webp"), 415, "unsupported_type")
    assert_error(post(client, tiff, "image/tiff", "a.tif"), 415, "unsupported_type")
    assert_error(post(client, tiff, "image/jpeg"), 415, "unsupported_type")
    assert_error(post(client, png_bytes(), "application/octet-stream"), 415, "unsupported_type")
    assert_error(post(client, png_bytes(), "text/plain"), 415, "unsupported_type")


def test_file_without_declared_type_is_415(client: TestClient) -> None:
    response = client.post(
        f"{BASE}/enhance", files={"file": ("a.png", png_bytes(), "")}
    )  # empty content type
    assert_error(response, 415, "unsupported_type")


def test_too_small_image(client: TestClient) -> None:
    small = _encode(Image.fromarray(_pixels(15, 40)), "PNG")
    assert_error(post(client, small), 400, "invalid_image")
    assert post(client, png_bytes(16, 16)).status_code == 200  # the boundary is accepted


def test_oversize_side_is_too_large(checkpoint: Path) -> None:
    custom = Settings(checkpoint=checkpoint, device="cpu", max_side=64)
    with TestClient(create_app(custom)) as small:
        assert post(small, png_bytes(64, 64)).status_code == 200
        assert_error(post(small, png_bytes(32, 65)), 413, "too_large")
        assert_error(post(small, jpeg_bytes(65, 32), "image/jpeg"), 413, "too_large")


def test_pixel_budget_is_enforced_independently_of_side(checkpoint: Path) -> None:
    custom = Settings(checkpoint=checkpoint, device="cpu", max_pixels=40 * 40)
    with TestClient(create_app(custom)) as small:
        assert post(small, png_bytes(40, 40)).status_code == 200
        assert_error(post(small, png_bytes(41, 40)), 413, "too_large")
        assert_error(post(small, jpeg_bytes(30, 60), "image/jpeg"), 413, "too_large")


def test_pixel_budget_setting_validation_and_env() -> None:
    with pytest.raises(SettingsError, match="LUMINA_MAX_PIXELS"):
        Settings(max_pixels=10)
    assert Settings().max_pixels == 2 * 1024 * 1024
    assert Settings.from_env({"LUMINA_MAX_PIXELS": "5000"}).max_pixels == 5000


def test_decompression_bomb_is_too_large(checkpoint: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 40 * 40)  # Pillow raises at twice this
    custom = Settings(checkpoint=checkpoint, device="cpu", max_side=40)
    with TestClient(create_app(custom)) as small:
        assert_error(post(small, png_bytes(64, 64)), 413, "too_large")  # between limit and 2x
        assert_error(post(small, png_bytes(100, 100)), 413, "too_large")  # beyond 2x: bomb error


def test_upload_larger_than_limit_after_parse(checkpoint: Path) -> None:
    custom = Settings(checkpoint=checkpoint, device="cpu", max_upload_bytes=1000)
    with TestClient(create_app(custom)) as small:
        data = png_bytes(48, 64)
        assert len(data) > 1000
        assert_error(post(small, data), 413, "too_large")


def test_upload_rejected_early_from_content_length(checkpoint: Path) -> None:
    custom = Settings(checkpoint=checkpoint, device="cpu", max_upload_bytes=1000)
    with TestClient(create_app(custom)) as small:
        response = post(small, b"\x89PNG\r\n\x1a\n" + b"0" * 200_000)
        assert_error(response, 413, "too_large")
        assert "1000 byte limit" in response.json()["error"]["message"]  # rejected up front


def test_upload_rejected_while_streaming_without_content_length(checkpoint: Path) -> None:
    custom = Settings(checkpoint=checkpoint, device="cpu", max_upload_bytes=1000)
    boundary = "lumina-test-boundary"
    prefix = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.png"\r\n'
        "Content-Type: image/png\r\n\r\n"
    ).encode()
    consumed = 0

    def body() -> Iterator[bytes]:
        nonlocal consumed
        yield prefix
        for _ in range(400):
            consumed += 1
            yield b"0" * 4096

    with TestClient(create_app(custom)) as small:
        response = small.post(
            f"{BASE}/enhance",
            content=body(),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
    assert_error(response, 413, "too_large")
    assert "maximum request size" in response.json()["error"]["message"]
    assert consumed > 0  # sent as a chunked body: there was no Content-Length to reject on


def test_large_uploads_never_touch_disk(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(self: object, *args: object) -> None:
        raise AssertionError("an upload was spooled to disk")

    monkeypatch.setattr(tempfile.SpooledTemporaryFile, "rollover", forbidden)
    junk = b"\x89PNG\r\n\x1a\n" + b"\x00" * (3 * 1024 * 1024)  # larger than Starlette's default
    assert_error(post(client, junk), 400, "invalid_image")


def test_missing_file_field_is_invalid_image(client: TestClient) -> None:
    response = client.post(f"{BASE}/enhance", data={"curve_scale": "0.5"})  # urlencoded form
    assert_error(response, 400, "invalid_image")
    response = client.post(f"{BASE}/enhance", files={"image": ("a.png", png_bytes(), "image/png")})
    assert_error(response, 400, "invalid_image")
    response = client.post(f"{BASE}/enhance", files={"file": (None, "just text")})
    assert_error(response, 400, "invalid_image")
    assert_error(client.post(f"{BASE}/enhance"), 400, "invalid_image")


def test_two_file_fields_are_rejected(client: TestClient) -> None:
    files = [
        ("file", ("a.png", png_bytes(), "image/png")),
        ("file", ("b.png", png_bytes(), "image/png")),
    ]
    assert_error(client.post(f"{BASE}/enhance", files=files), 400, "invalid_image")


def test_malformed_multipart_is_invalid_image(client: TestClient) -> None:
    response = client.post(
        f"{BASE}/enhance",
        content=b"garbage",
        headers={"Content-Type": "multipart/form-data; boundary=zzz"},
    )
    assert_error(response, 400, "invalid_image")
    response = client.post(
        f"{BASE}/enhance", content=b"x", headers={"Content-Type": "multipart/form-data"}
    )
    assert_error(response, 400, "invalid_image")


def test_model_not_loaded_returns_503(bare_client: TestClient) -> None:
    assert_error(post(bare_client, png_bytes()), 503, "model_not_loaded")
    assert_error(post(bare_client, b"not even an image"), 503, "model_not_loaded")


# ------------------------------------------------------------------ framework errors


def test_wrong_method_is_json_405(client: TestClient) -> None:
    for method, path in [("get", "/enhance"), ("post", "/health"), ("delete", "/options")]:
        response = getattr(client, method)(f"{BASE}{path}")
        assert_error(response, 405, "method_not_allowed")
        assert "allow" in response.headers


def test_unknown_path_is_json_404(client: TestClient) -> None:
    assert_error(client.get("/"), 404, "not_found")
    assert_error(client.get(f"{BASE}/nope"), 404, "not_found")
    assert_error(client.post("/api/v2/enhance"), 404, "not_found")
    assert_error(client.get("/docs"), 404, "not_found")


def test_unexpected_exception_is_json_500(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object, **kwargs: object) -> NoReturn:
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(app_module, "enhance_image", boom)
    with TestClient(
        create_app(Settings(checkpoint=checkpoint, device="cpu")), raise_server_exceptions=False
    ) as c:
        response = post(c, png_bytes())
    assert_error(response, 500, "internal_error")
    assert "secret" not in response.text
    assert response.headers["x-request-id"]


# ------------------------------------------------------------------ operability


def test_request_id_echo_and_generation(client: TestClient) -> None:
    echoed = client.get(f"{BASE}/health", headers={"X-Request-Id": "abc-123_X.y"})
    assert echoed.headers["x-request-id"] == "abc-123_X.y"
    generated = client.get(f"{BASE}/health")
    assert re.fullmatch(r"[0-9a-f]{32}", generated.headers["x-request-id"])
    replaced = client.get(f"{BASE}/health", headers={"X-Request-Id": "bad id with spaces"})
    assert re.fullmatch(r"[0-9a-f]{32}", replaced.headers["x-request-id"])
    assert (
        client.get("/missing", headers={"X-Request-Id": "r-404"}).headers["x-request-id"] == "r-404"
    )
    error = post(client, b"bad")
    assert error.headers["x-request-id"] and error.headers["cache-control"] == "no-store"


def test_access_log_line_has_no_payload_or_filename(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    response = post(
        client, png_bytes(), filename="my-secret-holiday.png", headers={"X-Request-Id": "log-1"}
    )
    assert response.status_code == 200
    lines = [r.getMessage() for r in caplog.records if r.name == "nta_zerodce.api.access"]
    lines = [line for line in lines if "request_id=log-1" in line]
    assert len(lines) == 1
    assert re.fullmatch(
        r"method=POST path=/api/v1/enhance status=200 elapsed_ms=\d+ request_id=log-1", lines[0]
    )
    assert "my-secret-holiday" not in caplog.text
    assert "my-secret-holiday" not in response.text


def test_error_messages_do_not_leak_filenames_or_paths(
    client: TestClient, checkpoint: Path
) -> None:
    response = post(client, b"not an image", "image/png", filename="/etc/passwd-leak.png")
    assert "passwd-leak" not in response.text and str(checkpoint) not in response.text


def test_cors_disabled_by_default_and_enabled_by_setting(
    checkpoint: Path, client: TestClient
) -> None:
    origin = {"Origin": "http://localhost:3000"}
    assert "access-control-allow-origin" not in client.get(f"{BASE}/health", headers=origin).headers
    custom = Settings(device="cpu", cors_origins=("http://localhost:3000",))
    with TestClient(create_app(custom)) as cors:
        allowed = cors.get(f"{BASE}/health", headers=origin)
        assert allowed.headers["access-control-allow-origin"] == "http://localhost:3000"
        assert "X-Lumina-Size" in allowed.headers["access-control-expose-headers"]
        other = cors.get(f"{BASE}/health", headers={"Origin": "http://evil.example"})
        assert "access-control-allow-origin" not in other.headers


# ------------------------------------------------------------------ startup


def test_bad_checkpoint_path_fails_startup(tmp_path: Path) -> None:
    app = create_app(Settings(checkpoint=tmp_path / "missing.pt", device="cpu"))
    with pytest.raises(StartupError, match="LUMINA_CHECKPOINT"):
        with TestClient(app):
            pass


def test_corrupt_checkpoint_fails_startup(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"this is not a checkpoint")
    with pytest.raises(StartupError, match="LUMINA_CHECKPOINT"):
        with TestClient(create_app(Settings(checkpoint=bad, device="cpu"))):
            pass


def test_incompatible_config_fails_startup(checkpoint: Path) -> None:
    custom = Settings(
        checkpoint=checkpoint, device="cpu", config_path=ROOT / "configs" / "baseline.py"
    )
    with pytest.raises(StartupError, match="LUMINA_CHECKPOINT"):
        with TestClient(create_app(custom)):
            pass


def test_bad_config_without_checkpoint_fails_startup(tmp_path: Path) -> None:
    with pytest.raises(StartupError, match="LUMINA_CONFIG"):
        with TestClient(create_app(Settings(config_path=tmp_path / "nope.yaml"))):
            pass


def test_config_without_checkpoint_reports_its_flags() -> None:
    custom = Settings(config_path=ROOT / "configs" / "full.py")
    with TestClient(create_app(custom)) as c:
        assert c.get(f"{BASE}/options").json()["flags"]["noise_aware"] is True
        assert c.get(f"{BASE}/health").json()["model_loaded"] is False


def test_model_is_loaded_once_not_per_request(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def counting(
        checkpoint: Path, config_path: Path | None = None, *, device: str = "auto"
    ) -> tuple[NTAZeroDCE, Config, torch.device]:
        nonlocal calls
        calls += 1
        return load_inference_model(checkpoint, config_path, device=device)

    monkeypatch.setattr(app_module, "load_inference_model", counting)
    with TestClient(create_app(Settings(checkpoint=checkpoint, device="cpu"))) as c:
        for _ in range(3):
            assert post(c, png_bytes()).status_code == 200
    assert calls == 1


# ------------------------------------------------------------------ settings


def test_settings_defaults() -> None:
    s = Settings.from_env({})
    assert s == Settings()
    assert (s.device, s.max_upload_bytes, s.max_side, s.max_concurrency) == (
        "auto",
        20971520,
        4096,
        1,
    )
    assert (s.checkpoint, s.config_path, s.default_curve_scale, s.cors_origins) == (
        None,
        None,
        None,
        (),
    )
    assert (s.host, s.port) == ("127.0.0.1", 8000)


def test_settings_from_env_values(tmp_path: Path) -> None:
    s = Settings.from_env(
        {
            "LUMINA_CHECKPOINT": str(tmp_path / "m.pt"),
            "LUMINA_CONFIG": str(tmp_path / "c.yaml"),
            "LUMINA_DEVICE": "cuda:1",
            "LUMINA_MAX_UPLOAD_BYTES": "1000",
            "LUMINA_MAX_SIDE": "128",
            "LUMINA_DEFAULT_CURVE_SCALE": "0.5",
            "LUMINA_MAX_CONCURRENCY": "3",
            "LUMINA_CORS_ORIGINS": "http://localhost:3000, https://demo.example.org ,",
            "LUMINA_HOST": "0.0.0.0",
            "LUMINA_PORT": "9001",
        }
    )
    assert s.checkpoint == tmp_path / "m.pt" and s.config_path == tmp_path / "c.yaml"
    assert (s.device, s.max_upload_bytes, s.max_side) == ("cuda:1", 1000, 128)
    assert (s.default_curve_scale, s.max_concurrency, s.host, s.port) == (0.5, 3, "0.0.0.0", 9001)
    assert s.cors_origins == ("http://localhost:3000", "https://demo.example.org")
    assert Settings.from_env({"LUMINA_CHECKPOINT": "  ", "LUMINA_DEVICE": ""}) == Settings()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("LUMINA_DEVICE", "gpu"),
        ("LUMINA_DEVICE", "CPU"),
        ("LUMINA_MAX_UPLOAD_BYTES", "0"),
        ("LUMINA_MAX_UPLOAD_BYTES", "-5"),
        ("LUMINA_MAX_UPLOAD_BYTES", "20MB"),
        ("LUMINA_MAX_SIDE", "15"),
        ("LUMINA_MAX_SIDE", "abc"),
        ("LUMINA_MAX_SIDE", "100000"),
        ("LUMINA_DEFAULT_CURVE_SCALE", "0"),
        ("LUMINA_DEFAULT_CURVE_SCALE", "1.5"),
        ("LUMINA_DEFAULT_CURVE_SCALE", "nan"),
        ("LUMINA_DEFAULT_CURVE_SCALE", "fast"),
        ("LUMINA_MAX_CONCURRENCY", "0"),
        ("LUMINA_MAX_CONCURRENCY", "1.5"),
        ("LUMINA_CORS_ORIGINS", "localhost:3000"),
        ("LUMINA_CORS_ORIGINS", "http://a.example/path"),
        ("LUMINA_PORT", "0"),
        ("LUMINA_PORT", "70000"),
        ("LUMINA_PORT", "http"),
    ],
)
def test_settings_validation_names_the_variable(name: str, value: str) -> None:
    with pytest.raises(SettingsError, match=name):
        Settings.from_env({name: value})


def test_settings_are_frozen() -> None:
    with pytest.raises(AttributeError):
        Settings().port = 1  # type: ignore[misc]
    assert replace(Settings(), port=9).port == 9
    with pytest.raises(SettingsError, match="LUMINA_PORT"):
        replace(Settings(), port=0)


# ------------------------------------------------------------------ concurrency


def _run_parallel(c: TestClient, jobs: int, source: bytes) -> list[int]:
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(post, c, source) for _ in range(jobs)]
        return [future.result(timeout=60).status_code for future in futures]


@pytest.mark.parametrize("limit", [1, 2])
def test_concurrency_limit_is_respected_and_does_not_deadlock(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch, limit: int
) -> None:
    active = 0
    peak = 0
    lock = threading.Lock()
    real = enhance_image

    def tracked(model: NTAZeroDCE, image: Tensor, curve_scale: float = 1.0) -> ModelOutput:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.05)
            return real(model, image, curve_scale)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(app_module, "enhance_image", tracked)
    custom = Settings(checkpoint=checkpoint, device="cpu", max_concurrency=limit)
    with TestClient(create_app(custom)) as c:
        statuses = _run_parallel(c, 8, png_bytes())
        assert c.get(f"{BASE}/health").status_code == 200
    assert statuses == [200] * 8
    assert 1 <= peak <= limit
