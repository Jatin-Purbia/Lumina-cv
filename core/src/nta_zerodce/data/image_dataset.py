"""Unpaired SICE training and paired LOL evaluation; DATA split rules.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.

Public additions beyond INTERFACES: ``ImageDataset.set_epoch``, the keyword-only ``paths`` and
``seed`` constructor arguments, ``index_files``, ``scene_split`` and ``make_synthetic_images``.
"""

from __future__ import annotations

import logging
import operator
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from nta_zerodce.data.transforms import item_generator, random_crop
from nta_zerodce.inference.io import read_image

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")
_EXIF_ORIENTATION = 0x0112


def index_files(root: Path, extensions: Sequence[str]) -> list[Path]:
    """Recursively list files under ``root`` whose suffix (case-insensitive) is in ``extensions``.

    Args:
        root: Existing directory.
        extensions: Lower-case suffixes including the dot, e.g. ``(".png",)``.

    Returns:
        Paths sorted by their POSIX string, so the order is stable across platforms and runs.

    Raises:
        NotADirectoryError: If ``root`` is not an existing directory.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"dataset root is not a directory: {root}")
    wanted = {e.lower() for e in extensions}
    found = [p for p in root.rglob("*") if p.suffix.lower() in wanted and p.is_file()]
    return sorted(found, key=lambda p: p.as_posix())


def _image_size(path: Path) -> tuple[int, int]:
    """Return (H,W) after EXIF orientation by reading only the file header."""
    with Image.open(path) as image:
        width, height = image.size
        orientation = image.getexif().get(_EXIF_ORIENTATION, 1)
    if orientation in (5, 6, 7, 8):
        width, height = height, width
    return height, width


def scene_split(
    root: Path, val_fraction: float = 0.1, seed: int = 17
) -> tuple[list[Path], list[Path]]:
    """Split image files into train/validation lists by scene directory (DATA leakage rule).

    All files sharing one parent directory (relative to ``root``) form one scene and stay
    together. Scene ids are the sorted relative parent paths and are shuffled with a
    ``numpy`` generator seeded by ``seed``; the first ``round(val_fraction * n_scenes)``
    shuffled scenes (at least one when ``val_fraction > 0`` and there are two or more scenes,
    never all of them) form the validation set. Files that sit directly in ``root`` are each
    their own scene, so a flat root falls back to a per-file split. With a single scene
    everything goes to train. The result depends only on the sorted file list, ``val_fraction``
    and ``seed``, so keep one manifest for all training seeds.

    Args:
        root: Existing directory indexed recursively (png/jpg/jpeg/bmp).
        val_fraction: Fraction of scenes held out, in [0,1).
        seed: Split seed (default 17, DATA initial choice).

    Returns:
        ``(train_paths, val_paths)``, each sorted.

    Raises:
        ValueError: If ``val_fraction`` is outside [0,1) or the root has no images.
        NotADirectoryError: If ``root`` is not a directory.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0,1), got {val_fraction}")
    root = Path(root)
    files = index_files(root, IMAGE_EXTENSIONS)
    if not files:
        raise ValueError(f"no images found under {root}")
    scenes: dict[str, list[Path]] = {}
    for path in files:
        relative = path.relative_to(root)
        scene_id = relative.parent.as_posix() if relative.parent != Path(".") else relative.as_posix()
        scenes.setdefault(scene_id, []).append(path)
    ids = sorted(scenes)
    n_scenes = len(ids)
    n_val = int(round(val_fraction * n_scenes))
    if val_fraction > 0 and n_scenes >= 2:
        n_val = max(n_val, 1)
    n_val = min(n_val, n_scenes - 1)
    order = np.random.default_rng(seed).permutation(n_scenes)
    val_ids = {ids[i] for i in order[:n_val]}
    train = [p for s in ids if s not in val_ids for p in scenes[s]]
    val = [p for s in ids if s in val_ids for p in scenes[s]]
    return sorted(train, key=lambda p: p.as_posix()), sorted(val, key=lambda p: p.as_posix())


class ImageDataset(Dataset[dict[str, Tensor]]):
    """Unpaired SICE training and paired LOL evaluation; DATA split rules. Items use unbatched tensors; DATA manifest contract."""

    def __init__(
        self,
        root: Path,
        crop_size: int = 256,
        paired_root: Path | None = None,
        *,
        paths: Sequence[Path] | None = None,
        seed: int = 17,
    ) -> None:
        """Index the dataset without loading it eagerly.

        Args:
            root: Existing source directory; images or videos according to dataset type.
            crop_size: Positive multiple of 16; train only.
            paired_root: Optional aligned high-reference root for image evaluation only.
            paths: Optional explicit image list (for example one side of ``scene_split``)
                that replaces the recursive scan of ``root``; the order given is sorted.
            seed: Base seed of the per-item crop generators (training mode).

        Returns:
            No value. Training mode (``paired_root`` None) indexes png/jpg/jpeg/bmp files
            recursively in sorted order, reads only headers, and skips (with a logged count)
            images smaller than the crop or unreadable. Evaluation mode matches ``root`` and
            ``paired_root`` files by exact basename and keeps full-size images.

        Method:
            DATA layouts; image eval preserves size, video pairs use gap=1 by default. No
            pixels are decoded here.

        Raises:
            NotADirectoryError: If ``root`` (or ``paired_root``) is not a directory.
            FileNotFoundError: If an entry of ``paths`` does not exist.
            ValueError: On an invalid ``crop_size`` (training), an empty dataset, or an
                evaluation set with missing/duplicate/size-mismatched pairs.
        """
        root = Path(root)
        if not root.is_dir():
            raise NotADirectoryError(f"dataset root is not a directory: {root}")
        self.root = root
        self.paired_root = Path(paired_root) if paired_root is not None else None
        self.crop_size = crop_size
        self.seed = seed
        self._epoch = 0
        self._references: list[Path] = []

        if paths is None:
            candidates = index_files(root, IMAGE_EXTENSIONS)
        else:
            candidates = sorted((Path(p) for p in paths), key=lambda p: p.as_posix())
            for candidate in candidates:
                if not candidate.is_file():
                    raise FileNotFoundError(f"listed image does not exist: {candidate}")
        if not candidates:
            raise ValueError(f"no images found under {root}")

        if self.paired_root is None:
            self._index_training(candidates)
        else:
            self._index_evaluation(candidates)

    def _index_training(self, candidates: list[Path]) -> None:
        """Keep candidates that are readable and at least crop_size on both sides."""
        crop = self.crop_size
        if isinstance(crop, bool) or not isinstance(crop, int) or crop <= 0 or crop % 16 != 0:
            raise ValueError(f"crop_size must be a positive multiple of 16, got {crop}")
        kept: list[Path] = []
        too_small = 0
        unreadable = 0
        for path in candidates:
            try:
                height, width = _image_size(path)
            except (OSError, ValueError, SyntaxError) as error:
                logger.warning("skipping unreadable image %s: %s", path, error)
                unreadable += 1
                continue
            if min(height, width) < crop:
                too_small += 1
                continue
            kept.append(path)
        if too_small or unreadable:
            logger.warning(
                "ImageDataset skipped %d image(s) smaller than crop %d and %d unreadable "
                "image(s) out of %d",
                too_small,
                crop,
                unreadable,
                len(candidates),
            )
        if not kept:
            raise ValueError(
                f"no usable images under {self.root} (skipped {too_small} smaller than "
                f"{crop}, {unreadable} unreadable)"
            )
        self._paths = kept

    def _index_evaluation(self, candidates: list[Path]) -> None:
        """Match low and reference images by exact basename and check sizes."""
        assert self.paired_root is not None
        references = index_files(self.paired_root, IMAGE_EXTENSIONS)
        low_by_name = self._by_name(candidates, "input")
        high_by_name = self._by_name(references, "reference")
        missing_high = sorted(set(low_by_name) - set(high_by_name))
        missing_low = sorted(set(high_by_name) - set(low_by_name))
        if missing_high or missing_low:
            raise ValueError(
                "unmatched pairs: no reference for "
                f"{missing_high[:5]}{'...' if len(missing_high) > 5 else ''}, no input for "
                f"{missing_low[:5]}{'...' if len(missing_low) > 5 else ''}"
            )
        names = sorted(low_by_name)
        for name in names:
            try:
                low_size = _image_size(low_by_name[name])
                high_size = _image_size(high_by_name[name])
            except (OSError, ValueError, SyntaxError) as error:
                raise ValueError(f"unreadable image in pair {name!r}: {error}") from error
            if low_size != high_size:
                raise ValueError(
                    f"pair {name!r} has mismatched sizes {low_size} vs {high_size} (HxW)"
                )
        self._paths = [low_by_name[n] for n in names]
        self._references = [high_by_name[n] for n in names]

    @staticmethod
    def _by_name(paths: Sequence[Path], label: str) -> dict[str, Path]:
        """Map exact basename to path, rejecting duplicates."""
        mapping: dict[str, Path] = {}
        for path in paths:
            if path.name in mapping:
                raise ValueError(
                    f"duplicate {label} basename {path.name!r}: {mapping[path.name]} and {path}"
                )
            mapping[path.name] = path
        return mapping

    @property
    def paths(self) -> list[Path]:
        """Indexed input image paths in item order (copy)."""
        return list(self._paths)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch that, with ``seed`` and the item index, seeds each crop.

        Call before iterating a ``DataLoader`` epoch. Worker processes receive a copy of the
        dataset when iteration starts (unless ``persistent_workers=True``, where the value
        set at worker start is kept), so set it on the main-process object each epoch.

        Raises:
            ValueError: If ``epoch`` is negative.
        """
        if epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}")
        self._epoch = int(epoch)

    def __len__(self) -> int:
        """Count indexed samples.

        Args:
            self: Indexed dataset.

        Returns:
            Nonnegative sample count; empty training roots must fail during initialization.

        Method:
            DATA deterministic index, no sampling during length queries.
        """
        return len(self._paths)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Decode one indexed sample.

        Args:
            index: Integer in [0,len(self)); reject out of range.

        Returns:
            Training: ``{"image": (3,crop,crop)}`` using a random crop whose generator is
            seeded from ``(seed, epoch, index)``, so the same triple always gives the same
            crop. Evaluation: full-size ``{"image": (3,H,W), "reference": (3,H,W)}``. Tensors
            are CPU float32 RGB in [0,1] without a batch dimension.

        Method:
            INTERFACES dataset batch keys; RGB [0,1] CPU, shared pair geometry.

        Raises:
            IndexError: If ``index`` is negative or >= len(self).
            ValueError: If the image cannot be decoded.
        """
        position = operator.index(index)
        if position < 0 or position >= len(self._paths):
            raise IndexError(f"index {index} out of range for dataset of size {len(self._paths)}")
        image = read_image(self._paths[position])[0]
        if self.paired_root is not None:
            reference = read_image(self._references[position])[0]
            return {"image": image, "reference": reference}
        generator = item_generator(self.seed, self._epoch, position)
        return {"image": random_crop(image, self.crop_size, generator)}


def make_synthetic_images(
    output_dir: Path, count: int, size: int | tuple[int, int] = 64, seed: int = 17
) -> list[Path]:
    """Write deterministic procedural low-light test images as PNG files.

    Each image combines a smooth tinted gradient, low-frequency waves, soft blobs and a few
    sharp edges, scaled so the mean intensity is about 0.06-0.18, plus 1-3 small bright lamp
    spots and mild Gaussian noise (std 0.01). The overall mean is kept within 0.05-0.25. They
    are test and demo fixtures only, not a model of real camera data. Output depends only on
    ``(count, size, seed)``; image ``i`` uses ``numpy.random.default_rng([seed, i])``.

    Args:
        output_dir: Directory to write into; created (with parents) if missing.
        count: Number of images, at least 1.
        size: Side length for square images, or ``(height, width)``; each at least 16.
        seed: Base seed.

    Returns:
        The written paths ``synthetic_0000.png``, ... in order.

    Raises:
        ValueError: If ``count`` < 1 or ``size`` is smaller than 16.
    """
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}")
    height, width = (size, size) if isinstance(size, int) else (int(size[0]), int(size[1]))
    if height < 16 or width < 16:
        raise ValueError(f"size must be at least 16x16, got {height}x{width}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for i in range(count):
        array = _procedural_image(np.random.default_rng([seed, i]), height, width)
        path = output_dir / f"synthetic_{i:04d}.png"
        Image.fromarray(array, mode="RGB").save(path, format="PNG")
        written.append(path)
    return written


def _procedural_image(rng: np.random.Generator, height: int, width: int) -> np.ndarray:
    """Build one uint8 HxWx3 dark textured image (see ``make_synthetic_images``)."""
    yy, xx = np.meshgrid(np.linspace(0, 1, height), np.linspace(0, 1, width), indexing="ij")
    tint = 0.6 + 0.4 * rng.random(3)
    base = (0.3 + 0.5 * (rng.random() * xx + rng.random() * yy))[..., None] * tint
    for _ in range(2):
        fx, fy = rng.uniform(0.5, 2.5, 2)
        wave = 0.12 * np.sin(2 * np.pi * (fx * xx + fy * yy) + rng.uniform(0, 2 * np.pi))
        base = base + wave[..., None]
    for _ in range(int(rng.integers(3, 7))):
        cy, cx = rng.random(2)
        sigma = rng.uniform(0.05, 0.2)
        blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))
        base = base + (rng.uniform(0.2, 0.6) * blob)[..., None] * (0.5 + 0.5 * rng.random(3))
    for _ in range(int(rng.integers(2, 5))):
        angle = rng.uniform(0, np.pi)
        offset = rng.uniform(0.2, 0.8)
        distance = (xx - 0.5) * np.cos(angle) + (yy - 0.5) * np.sin(angle) - (offset - 0.5)
        step = 1.0 / (1.0 + np.exp(-300.0 * distance))
        base = base + (rng.uniform(-0.2, 0.2) * step)[..., None]
    base = np.clip(base, 0.0, None)
    base = base / max(float(base.max()), 1e-6)
    base = base * (rng.uniform(0.06, 0.18) / max(float(base.mean()), 1e-6))
    lamps = np.zeros((height, width, 3))
    for _ in range(int(rng.integers(1, 4))):
        cy, cx = rng.uniform(0.1, 0.9, 2)
        sigma = rng.uniform(0.01, 0.03)
        spot = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))
        lamps = lamps + (rng.uniform(0.8, 1.5) * spot)[..., None] * np.array([1.0, 0.9, 0.7])
    image = base + lamps + rng.normal(0.0, 0.01, (height, width, 3))
    image = np.clip(image, 0.0, 1.0)
    mean = float(image.mean())
    if mean > 0.24:
        image = image * (0.24 / mean)
    elif mean < 0.06:
        image = np.clip(image * (0.06 / mean), 0.0, 1.0)
    return np.round(255.0 * image).astype(np.uint8)
