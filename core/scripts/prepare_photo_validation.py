"""Prepare normal-light preservation validation from the held-out LOL training scenes.

Reads the existing LOL archive; never downloads or extracts arbitrary archive paths.
Uses exactly the trainer's fixed scene split, excludes every training scene, and writes
a hashed manifest. These scenes test well-lit preservation within LOL, not real-world
generalization. Keep additional independent ordinary photographs for final validation.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path

from PIL import Image

from nta_zerodce.data.image_dataset import scene_split
from nta_zerodce.training.config import resolve_path
from nta_zerodce.training.trainer import SPLIT_SEED, SPLIT_VAL_FRACTION, split_hash


def prepare_validation(archive: Path, train_images: Path, output: Path) -> dict[str, object]:
    """Write only normal-light counterparts of the fixed validation scenes, plus manifest.

    All members and existing destinations are checked before writing any image. Existing
    identical images are reusable; differing images or manifests are never overwritten.
    """
    _, validation = scene_split(train_images, SPLIT_VAL_FRACTION, SPLIT_SEED)
    if not validation:
        raise ValueError("the fixed training split has no held-out validation scenes")
    records: list[dict[str, str]] = []
    images: list[tuple[Path, bytes]] = []
    scene_ids: set[str] = set()
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        for low in validation:
            scene_id = low.parent.relative_to(train_images).as_posix()
            if not re.fullmatch(r"[A-Za-z0-9_-]+", scene_id):
                raise ValueError("expected LOL layout train_low/<scene>/low.png")
            if scene_id in scene_ids:
                raise ValueError(f"multiple low images for scene {scene_id!r}; expected one")
            scene_ids.add(scene_id)
            member = f"lol_dataset/our485/high/{scene_id}.png"
            if names.count(member) != 1:
                raise ValueError(f"archive must contain exactly one {member}")
            info = bundle.getinfo(member)
            if info.file_size > 20 * 1024 * 1024:
                raise ValueError(f"oversized normal-light image: {member}")
            data = bundle.read(member)
            with Image.open(io.BytesIO(data), formats=["PNG"]) as image:
                image.verify()
            target = output / f"{scene_id}.png"
            if target.exists() and target.read_bytes() != data:
                raise ValueError(f"refusing to replace different existing image: {target}")
            records.append(
                {
                    "scene_id": scene_id,
                    "file": target.name,
                    "archive_member": member,
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
            images.append((target, data))
    manifest: dict[str, object] = {
        "schema_version": 1,
        "purpose": "normal-light preservation on held-out LOL training scenes",
        "split_seed": SPLIT_SEED,
        "split_val_fraction": SPLIT_VAL_FRACTION,
        "validation_split_sha256": split_hash(validation, train_images),
        "images": records,
    }
    manifest_path = output / "manifest.json"
    if output.is_dir():
        expected = {target.name for target, _ in images}
        existing = {
            p.name for p in output.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        }
        if existing - expected:
            raise ValueError(
                f"validation output contains unrelated images: {sorted(existing - expected)}"
            )
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError(f"refusing to replace a different validation manifest: {manifest_path}")
    output.mkdir(parents=True, exist_ok=True)
    for target, data in images:
        if not target.exists():
            target.write_bytes(data)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    """Prepare the deterministic preservation set from an existing local archive."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="data/_dl/lol.zip")
    parser.add_argument("--train-images", default="data/LOL/train_low")
    parser.add_argument("--output", default="runs/improvements/ordinary_validation")
    args = parser.parse_args(argv)
    try:
        manifest = prepare_validation(
            resolve_path(args.archive), resolve_path(args.train_images), resolve_path(args.output)
        )
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"prepare_photo_validation: {error}", file=sys.stderr)
        return 2
    images = manifest["images"]
    assert isinstance(images, list)
    print(f"Prepared {len(images)} held-out normal-light scenes in {resolve_path(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
