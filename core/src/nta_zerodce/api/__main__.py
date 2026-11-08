"""Run the API with uvicorn: ``python -m nta_zerodce.api`` (configuration via LUMINA_* env vars)."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import uvicorn

from nta_zerodce.api.app import create_app
from nta_zerodce.api.settings import Settings, SettingsError


def load_dotenv(path: Path) -> None:
    """Load ``KEY=value`` lines from ``path`` into ``os.environ`` without overriding existing vars."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def main() -> None:
    """Read settings from the environment (and ``.env``) and serve until interrupted."""
    load_dotenv(Path(".env"))
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
