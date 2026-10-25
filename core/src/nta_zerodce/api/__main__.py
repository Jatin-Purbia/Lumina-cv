"""Run the API with uvicorn: ``python -m nta_zerodce.api`` (configuration via LUMINA_* env vars)."""

from __future__ import annotations

import logging
import sys

import uvicorn

from nta_zerodce.api.app import create_app
from nta_zerodce.api.settings import Settings, SettingsError


def main() -> None:
    """Read settings from the environment and serve until interrupted."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        settings = Settings.from_env()
    except SettingsError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level="info",
        access_log=False,
        timeout_graceful_shutdown=10,
    )


if __name__ == "__main__":
    main()
