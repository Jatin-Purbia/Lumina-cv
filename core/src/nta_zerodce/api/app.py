"""FastAPI application factory for the Lumina CV enhancement API.

``create_app(settings)`` returns an app whose model is loaded once during the lifespan
startup. The HTTP contract (docs/PLAN.md, "HTTP contract") is served under ``/api/v1``:
``GET /health``, ``GET /options`` and ``POST /enhance``. See docs/API.md.
"""

from __future__ import annotations

import logging
import math
import re
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from importlib import metadata
from typing import Any

import anyio
import anyio.to_thread
import torch
from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.datastructures import Headers, MutableHeaders, UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.requests import ClientDisconnect
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from nta_zerodce.api.errors import (
    BAD_OPTION,
    INTERNAL_ERROR,
    INVALID_IMAGE,
    MODEL_NOT_LOADED,
    TOO_LARGE,
    ApiError,
    error_body,
    install_exception_handlers,
)
from nta_zerodce.api.imaging import (
    ACCEPTED_TYPES,
    check_upload_type,
    decode_image,
    encode_png,
)
from nta_zerodce.api.settings import Settings
from nta_zerodce.inference.enhancer import enhance_image
from nta_zerodce.inference.loader import load_inference_model
from nta_zerodce.models.nta_zerodce import NTAZeroDCE
from nta_zerodce.training.config import Config, load_config

API_PREFIX = "/api/v1"
ACCESS_LOGGER = logging.getLogger("nta_zerodce.api.access")
LOGGER = logging.getLogger("nta_zerodce.api")

_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_NON_PRINTABLE = re.compile(r"[^\x21-\x7e]")
_CURVE_SCALE = re.compile(r"^(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$")
_MAX_FORM_FILES = 4
_MAX_FORM_FIELDS = 16
_MAX_FIELD_BYTES = 1024
_EXPOSED_HEADERS = ["X-Lumina-Elapsed-Ms", "X-Lumina-Size", "X-Request-Id"]


class StartupError(RuntimeError):
    """The server cannot start (for example, the configured checkpoint cannot be loaded)."""


def package_version() -> str:
    """Installed ``nta_zerodce`` version, or ``unknown`` when not installed."""
    try:
        return metadata.version("nta_zerodce")
    except metadata.PackageNotFoundError:
        return "unknown"


@dataclass
class Runtime:
    """State created at startup and shared by all requests."""

    settings: Settings
    config: Config
    default_curve_scale: float
    device: str
    model: NTAZeroDCE | None
    torch_device: torch.device | None
    limiter: anyio.Semaphore
    version: str


def _load_runtime(settings: Settings) -> Runtime:
    """Load the configured model (if any); raise ``StartupError`` naming the variable on failure."""
    model: NTAZeroDCE | None = None
    torch_device: torch.device | None = None
    device = settings.device
    if settings.checkpoint is None:
        try:
            config = Config() if settings.config_path is None else load_config(settings.config_path)
        except Exception as error:
            raise StartupError(
                f"LUMINA_CONFIG could not be loaded ({type(error).__name__}: {error})"
            ) from error
        LOGGER.warning("LUMINA_CHECKPOINT is not set: starting without a model (503 on /enhance)")
    else:
        try:
            model, config, torch_device = load_inference_model(
                settings.checkpoint, settings.config_path, device=settings.device
            )
        except Exception as error:
            raise StartupError(
                f"LUMINA_CHECKPOINT {str(settings.checkpoint)!r} could not be loaded "
                f"({type(error).__name__}: {error})"
            ) from error
        device = str(torch_device)
        LOGGER.info("model loaded on %s", device)
    scale = settings.default_curve_scale
    return Runtime(
        settings=settings,
        config=config,
        default_curve_scale=config.model.curve_scale if scale is None else scale,
        device=device,
        model=model,
        torch_device=torch_device,
        limiter=anyio.Semaphore(settings.max_concurrency),
        version=package_version(),
    )


def parse_curve_scale(raw: str) -> float:
    """Parse the ``curve_scale`` form value strictly; the value must lie in (0, 1].

    Only plain decimal numbers (optionally with an exponent) are accepted: no signs, ``nan``,
    ``inf``, thousands separators or non-ASCII digits.

    Raises:
        ApiError: 422 ``bad_option`` for anything else.
    """
    text = raw.strip()
    error = ApiError(422, BAD_OPTION, "curve_scale must be a number in (0, 1].")
    if not _CURVE_SCALE.match(text):
        raise error
    value = float(text)
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise error
    return value


def _run_pipeline(
    runtime: Runtime, data: bytes, image_format: str, curve_scale: float
) -> tuple[bytes, int, int, int]:
    """Decode, enhance and encode one upload (blocking; runs in a worker thread).

    Returns:
        ``(png_bytes, width, height, elapsed_ms)``; the elapsed time covers tensor transfer,
        model inference and PNG encoding but not the decode.
    """
    model, device = runtime.model, runtime.torch_device
    assert model is not None and device is not None
    image = decode_image(data, image_format, runtime.settings.max_side)
    height, width = int(image.shape[2]), int(image.shape[3])
    started = time.perf_counter()
    with torch.inference_mode():
        output = enhance_image(model, image.to(device), curve_scale)
        try:
            png = encode_png(output["enhanced"])
        except ValueError as error:
            raise ApiError(500, INTERNAL_ERROR, "The model produced an invalid image.") from error
    elapsed_ms = round((time.perf_counter() - started) * 1000.0)
    return png, width, height, elapsed_ms


async def _limited_stream(request: Request, limit: int) -> AsyncGenerator[bytes, None]:
    """Yield request body chunks, raising 413 as soon as more than ``limit`` bytes arrive."""
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise ApiError(413, TOO_LARGE, "The upload exceeds the maximum request size.")
        yield chunk


def _too_large_upload(max_bytes: int) -> ApiError:
    return ApiError(413, TOO_LARGE, f"The upload exceeds the {max_bytes} byte limit.")


async def _read_upload(
    request: Request, settings: Settings
) -> tuple[bytes, str | None, str | None]:
    """Stream-parse the multipart body in memory and return ``(file bytes, type, curve_scale)``."""
    declared_length = request.headers.get("content-length")
    if declared_length is not None:
        if not declared_length.isascii() or not declared_length.isdigit():
            raise ApiError(400, INVALID_IMAGE, "The Content-Length header is invalid.")
        if int(declared_length) > settings.max_body_bytes:
            raise _too_large_upload(settings.max_upload_bytes)
    content_type = request.headers.get("content-type", "")
    if not content_type.lower().startswith("multipart/form-data"):
        raise ApiError(
            400,
            INVALID_IMAGE,
            "Expected multipart/form-data with a 'file' field containing an image.",
        )
    parser = MultiPartParser(
        Headers(scope=request.scope),
        _limited_stream(request, settings.max_body_bytes),
        max_files=_MAX_FORM_FILES,
        max_fields=_MAX_FORM_FIELDS,
        max_part_size=_MAX_FIELD_BYTES,
    )
    # Keep file parts in memory (never roll over to a temporary file on disk); the body limit
    # above bounds their total size.
    parser.spool_max_size = settings.max_body_bytes + 1
    try:
        form = await parser.parse()
    except MultiPartException as error:
        raise ApiError(400, INVALID_IMAGE, "The multipart body is malformed.") from error
    except ClientDisconnect as error:
        raise ApiError(400, INVALID_IMAGE, "The client disconnected during the upload.") from error
    try:
        files = form.getlist("file")
        if len(files) != 1 or not isinstance(files[0], UploadFile):
            raise ApiError(
                400, INVALID_IMAGE, "Expected exactly one multipart file field named 'file'."
            )
        upload = files[0]
        data = await upload.read()
        scales = form.getlist("curve_scale")
        if len(scales) > 1 or (scales and not isinstance(scales[0], str)):
            raise ApiError(422, BAD_OPTION, "curve_scale must be a number in (0, 1].")
        scale_text = scales[0] if scales and isinstance(scales[0], str) else None
        return data, upload.content_type, scale_text
    finally:
        await form.close()


class RequestContextMiddleware:
    """Request id, security headers and one-line access logging; also the last-resort 500."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = time.perf_counter()
        supplied = Headers(scope=scope).get("x-request-id", "")
        request_id = supplied if _REQUEST_ID.match(supplied) else uuid.uuid4().hex
        status = 500
        response_started = False

        async def send_wrapper(message: Message) -> None:
            nonlocal status, response_started
            if message["type"] == "http.response.start":
                response_started = True
                status = int(message["status"])
                headers = MutableHeaders(scope=message)
                headers["X-Request-Id"] = request_id
                headers.setdefault("X-Content-Type-Options", "nosniff")
                headers.setdefault("Cache-Control", "no-store")
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            LOGGER.error("unhandled error request_id=%s", request_id, exc_info=True)
            if response_started:
                raise
            response = JSONResponse(
                error_body(INTERNAL_ERROR, "Internal server error."), status_code=500
            )
            await response(scope, receive, send_wrapper)
        finally:
            elapsed_ms = round((time.perf_counter() - started) * 1000.0)
            path = _NON_PRINTABLE.sub("?", str(scope.get("path", "")))[:200]
            ACCESS_LOGGER.info(
                "method=%s path=%s status=%d elapsed_ms=%d request_id=%s",
                _NON_PRINTABLE.sub("?", str(scope.get("method", "")))[:16],
                path,
                status,
                elapsed_ms,
                request_id,
            )


def _runtime(request: Request) -> Runtime | None:
    runtime: Runtime | None = getattr(request.app.state, "runtime", None)
    return runtime


def _build_router() -> APIRouter:
    router = APIRouter(prefix=API_PREFIX)

    @router.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        runtime = _runtime(request)
        return {
            "status": "ok",
            "model_loaded": runtime is not None and runtime.model is not None,
            "device": runtime.device if runtime is not None else "unknown",
            "version": runtime.version if runtime is not None else package_version(),
        }

    @router.get("/options")
    async def options(request: Request) -> dict[str, Any]:
        runtime = _runtime(request)
        settings = runtime.settings if runtime is not None else request.app.state.settings
        config = runtime.config if runtime is not None else Config()
        scale = (
            runtime.default_curve_scale
            if runtime is not None
            else settings.default_curve_scale or config.model.curve_scale
        )
        return {
            "flags": asdict(config.flags),
            "curve_scale": scale,
            "max_upload_bytes": settings.max_upload_bytes,
            "max_side": settings.max_side,
            "accepted_types": list(ACCEPTED_TYPES),
        }

    @router.post("/enhance")
    async def enhance(request: Request) -> Response:
        runtime = _runtime(request)
        if runtime is None or runtime.model is None:
            raise ApiError(503, MODEL_NOT_LOADED, "No model checkpoint is loaded on the server.")
        settings = runtime.settings
        data, declared_type, scale_text = await _read_upload(request, settings)
        if not data:
            raise ApiError(400, INVALID_IMAGE, "The uploaded file is empty.")
        if len(data) > settings.max_upload_bytes:
            raise _too_large_upload(settings.max_upload_bytes)
        scale = runtime.default_curve_scale if scale_text is None else parse_curve_scale(scale_text)
        image_format = check_upload_type(declared_type, data)
        async with runtime.limiter:
            png, width, height, elapsed_ms = await anyio.to_thread.run_sync(
                _run_pipeline, runtime, data, image_format, scale
            )
        return Response(
            content=png,
            media_type="image/png",
            headers={
                "Cache-Control": "no-store",
                "X-Lumina-Elapsed-Ms": str(elapsed_ms),
                "X-Lumina-Size": f"{width}x{height}",
            },
        )

    return router


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the API application.

    Args:
        settings: Server settings; ``None`` reads them from the environment
            (``Settings.from_env()``), which also makes this usable as a uvicorn factory.

    Returns:
        A FastAPI app. The model named by ``settings.checkpoint`` is loaded once in the lifespan
        startup (a failure raises ``StartupError`` and the server does not start).
    """
    resolved = Settings.from_env() if settings is None else settings

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.runtime = _load_runtime(resolved)
        try:
            yield
        finally:
            app.state.runtime = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            LOGGER.info("shutdown complete")

    app = FastAPI(
        title="Lumina CV API",
        version=package_version(),
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.settings = resolved
    app.state.runtime = None
    install_exception_handlers(app)
    app.include_router(_build_router())
    if resolved.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(resolved.cors_origins),
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type", "X-Request-Id"],
            expose_headers=_EXPOSED_HEADERS,
            allow_credentials=False,
        )
    app.add_middleware(RequestContextMiddleware)
    return app
