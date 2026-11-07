"""Download the LOL low-light dataset and lay it out where configs/full.py expects it.

Source: a public HuggingFace mirror of the LOL dataset (Wei et al., "Deep Retinex Decomposition
for Low-Light Enhancement", BMVC 2018). Check the dataset's own terms before use; it is meant for
research and education and is not redistributed by this repository.

Result (relative to core/)::

    data/LOL/train_low/<id>/low.png     485 low-light training images, one folder per scene
    data/LOL/eval15/low/<id>.png        15 held-out low-light images
    data/LOL/eval15/high/<id>.png       the matching normal-light references (evaluation only)

Run from core/::

    python scripts/download_lol.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

URL = "https://huggingface.co/datasets/geekyrakshit/LoL-Dataset/resolve/main/lol_dataset.zip"
ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    """Download, extract and arrange the dataset; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--output", default="data/LOL", help="Destination folder (default data/LOL)"
    )
    parser.add_argument("--url", default=URL, help="Zip archive URL")
    args = parser.parse_args(argv)

    out = Path(args.output)
    out = out if out.is_absolute() else ROOT / out
    if (out / "train_low").is_dir() and (out / "eval15" / "low").is_dir():
        print(f"dataset already present in {out}")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "lol.zip"
        print(f"downloading {args.url} (about 350 MB) ...")
        try:
            urllib.request.urlretrieve(args.url, archive)  # noqa: S310 - fixed https URL
        except OSError as err:
            print(f"download failed: {err}", file=sys.stderr)
            return 1
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(tmp)
        root = Path(tmp) / "lol_dataset"
        if not (root / "our485" / "low").is_dir():
            print("unexpected archive layout", file=sys.stderr)
            return 1
        for kind in ("low", "high"):
            target = out / "eval15" / kind
            target.mkdir(parents=True, exist_ok=True)
            for image in sorted((root / "eval15" / kind).glob("*.png")):
                shutil.copy(image, target / image.name)
        count = 0
        for image in sorted((root / "our485" / "low").glob("*.png")):
            scene = out / "train_low" / image.stem
            scene.mkdir(parents=True, exist_ok=True)
            shutil.copy(image, scene / "low.png")
            count += 1
    print(f"wrote {count} training scenes and 15 evaluation pairs to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
