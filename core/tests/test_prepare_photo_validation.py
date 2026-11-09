"""Reproducible high-image extraction must respect the trainer's scene split."""

from __future__ import annotations

import importlib.util
import io
import json
import zipfile
from pathlib import Path
from types import ModuleType

import pytest
from PIL import Image

from nta_zerodce.data.image_dataset import scene_split


@pytest.fixture(scope="module")
def script() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_photo_validation.py"
    spec = importlib.util.spec_from_file_location("prepare_photo_validation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _png(value: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (value, value, value)).save(buffer, format="PNG")
    return buffer.getvalue()


def _dataset(root: Path, *, missing: str | None = None) -> tuple[Path, Path]:
    train = root / "train_low"
    archive = root / "lol.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for index in range(20):
            name = f"scene{index:02d}"
            low = train / name / "low.png"
            low.parent.mkdir(parents=True)
            low.write_bytes(_png(index))
            if name != missing:
                bundle.writestr(f"lol_dataset/our485/high/{name}.png", _png(index + 100))
        # Only exact whitelisted normal-light member names are ever read.
        bundle.writestr("../untrusted.txt", b"must never be extracted")
    return archive, train


def test_preparation_uses_only_heldout_scenes_and_is_reproducible(
    script: ModuleType, tmp_path: Path
) -> None:
    archive, train = _dataset(tmp_path)
    train_paths, heldout = scene_split(train)
    output = tmp_path / "validation"
    manifest = script.prepare_validation(archive, train, output)
    ids = {item["scene_id"] for item in manifest["images"]}
    assert ids == {p.parent.name for p in heldout}
    assert ids.isdisjoint({p.parent.name for p in train_paths})
    assert len(ids) == 2
    assert sorted(p.stem for p in output.glob("*.png")) == sorted(ids)
    assert not (tmp_path / "untrusted.txt").exists()
    assert script.prepare_validation(archive, train, output) == manifest
    assert json.loads((output / "manifest.json").read_text()) == manifest


def test_preparation_checks_all_members_before_creating_output(
    script: ModuleType, tmp_path: Path
) -> None:
    archive, train = _dataset(tmp_path)
    _, heldout = scene_split(train)
    missing = heldout[-1].parent.name
    # Rewrite without the last held-out image to catch partial-output bugs.
    with zipfile.ZipFile(archive) as source:
        records = [(info.filename, source.read(info)) for info in source.infolist()]
    with zipfile.ZipFile(archive, "w") as target:
        for name, data in records:
            if name != f"lol_dataset/our485/high/{missing}.png":
                target.writestr(name, data)
    output = tmp_path / "validation"
    with pytest.raises(ValueError, match="exactly one"):
        script.prepare_validation(archive, train, output)
    assert not output.exists()


def test_preparation_refuses_different_existing_content(script: ModuleType, tmp_path: Path) -> None:
    archive, train = _dataset(tmp_path)
    _, heldout = scene_split(train)
    output = tmp_path / "validation"
    output.mkdir()
    target = output / f"{heldout[0].parent.name}.png"
    target.write_bytes(b"existing file")
    with pytest.raises(ValueError, match="refusing to replace"):
        script.prepare_validation(archive, train, output)
    assert target.read_bytes() == b"existing file"


def test_preparation_refuses_unrelated_output_images(script: ModuleType, tmp_path: Path) -> None:
    archive, train = _dataset(tmp_path)
    output = tmp_path / "validation"
    output.mkdir()
    (output / "extra.png").write_bytes(_png(100))
    with pytest.raises(ValueError, match="unrelated images"):
        script.prepare_validation(archive, train, output)
