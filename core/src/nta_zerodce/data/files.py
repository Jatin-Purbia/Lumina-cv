"""Stable local image and video manifests."""

from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def image_files(root: Path) -> list[Path]:
    """Return sorted image paths, accepting one file or a recursive directory."""
    return media_files(root, IMAGE_SUFFIXES)


def video_files(root: Path) -> list[Path]:
    """Return sorted video paths."""
    return media_files(root, VIDEO_SUFFIXES)


def media_files(root: Path, suffixes: set[str]) -> list[Path]:
    """Validate the root and enumerate supported local media."""
    if not root.exists():
        raise ValueError(f"Media path does not exist: {root}")
    paths = [root] if root.is_file() else list(root.rglob("*"))
    return sorted(p for p in paths if p.is_file() and p.suffix.lower() in suffixes)
