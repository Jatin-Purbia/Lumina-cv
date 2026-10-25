"""Adjacent real-video pairs; flow is delegated exclusively to flow/estimator.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

import bisect
import logging
import operator
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from nta_zerodce.data.image_dataset import index_files
from nta_zerodce.data.transforms import item_generator, normalize_rgb, paired_crop

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def _decode(frame: np.ndarray) -> Tensor:
    """Convert an OpenCV BGR uint8 frame to RGB float32 (3,H,W) in [0,1]."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return normalize_rgb(torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1))


class VideoPairDataset(Dataset[dict[str, Tensor]]):
    """Adjacent real-video pairs; flow is delegated exclusively to flow/estimator. Items use unbatched tensors; DATA manifest contract."""

    def __init__(
        self,
        root: Path,
        crop_size: int = 256,
        paired_root: Path | None = None,
        frame_gap: int = 1,
        *,
        seed: int = 17,
    ) -> None:
        """Index the dataset without loading it eagerly.

        Args:
            root: Existing source directory; images or videos according to dataset type.
            crop_size: Positive multiple of 16; train only.
            paired_root: Must be None for this real-video dataset; no high-light references.
            frame_gap: Positive source-frame separation, default 1.
            seed: Base seed of the per-item crop generators; the crop of an item depends on
                ``(seed, epoch, index)`` as in ``ImageDataset`` (use ``set_epoch``).

        Returns:
            No value. mp4/avi/mov/mkv files are indexed recursively in sorted order; only the
            container header is read (frame count and size). Videos that cannot be opened,
            have fewer than ``frame_gap + 1`` frames, or are smaller than the crop are skipped
            with a logged count. Pair ``k`` of a video is frames ``(k, k + frame_gap)``.

        Method:
            DATA layouts; image eval preserves size, video pairs use gap=1 by default. The
            frame count comes from container metadata and can be wrong for some codecs; a
            failed frame read at access time raises ``RuntimeError``.

        Raises:
            ValueError: If ``paired_root`` is not None, ``crop_size`` is not a positive
                multiple of 16, ``frame_gap`` < 1, or no usable video remains.
            NotADirectoryError: If ``root`` is not a directory.
        """
        if paired_root is not None:
            raise ValueError(
                "VideoPairDataset has no high-light references; paired_root must be None"
            )
        if isinstance(frame_gap, bool) or not isinstance(frame_gap, int) or frame_gap < 1:
            raise ValueError(f"frame_gap must be an integer >= 1, got {frame_gap!r}")
        if (
            isinstance(crop_size, bool)
            or not isinstance(crop_size, int)
            or crop_size <= 0
            or crop_size % 16 != 0
        ):
            raise ValueError(f"crop_size must be a positive multiple of 16, got {crop_size}")
        self.root = Path(root)
        self.crop_size = crop_size
        self.frame_gap = frame_gap
        self.seed = seed
        self._epoch = 0

        candidates = index_files(self.root, VIDEO_EXTENSIONS)
        videos: list[Path] = []
        starts: list[int] = []
        total = 0
        unreadable = too_short = too_small = 0
        for path in candidates:
            capture = cv2.VideoCapture(str(path))
            try:
                if not capture.isOpened():
                    unreadable += 1
                    logger.warning("skipping unreadable video %s", path)
                    continue
                frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            finally:
                capture.release()
            if frames <= 0 or width <= 0 or height <= 0:
                unreadable += 1
                logger.warning("skipping video without readable metadata %s", path)
            elif frames <= frame_gap:
                too_short += 1
            elif min(height, width) < crop_size:
                too_small += 1
            else:
                videos.append(path)
                starts.append(total)
                total += frames - frame_gap
        if unreadable or too_short or too_small:
            logger.warning(
                "VideoPairDataset skipped %d unreadable, %d too short (<= gap %d frames) and "
                "%d smaller-than-crop (%d) video(s) out of %d",
                unreadable,
                too_short,
                frame_gap,
                too_small,
                crop_size,
                len(candidates),
            )
        if not videos:
            raise ValueError(f"no usable videos under {self.root}")
        self._videos = videos
        self._starts = starts
        self._total = total

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch that, with ``seed`` and the item index, seeds each crop.

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
        return self._total

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Decode one indexed sample.

        Args:
            index: Integer in [0,len(self)); reject out of range.

        Returns:
            ``{"image1", "image2"}`` CPU float32 RGB (3,crop,crop) in [0,1] cropped with one
            shared origin (seeded from ``(seed, epoch, index)``), and ``"index"``, a scalar
            int64 tensor equal to ``index``.

        Method:
            INTERFACES dataset batch keys; RGB [0,1] CPU, shared pair geometry. Frames are
            decoded from BGR to RGB at read time.

        Raises:
            IndexError: If ``index`` is negative or >= len(self).
            RuntimeError: If a frame cannot be read from the file.
        """
        position = operator.index(index)
        if position < 0 or position >= self._total:
            raise IndexError(f"index {index} out of range for dataset of size {self._total}")
        which = bisect.bisect_right(self._starts, position) - 1
        path = self._videos[which]
        frame_index = position - self._starts[which]
        first, second = self._read_pair(path, frame_index)
        generator = item_generator(self.seed, self._epoch, position)
        first, second = paired_crop(first, second, self.crop_size, generator)
        return {
            "image1": first,
            "image2": second,
            "index": torch.tensor(position, dtype=torch.int64),
        }

    def _read_pair(self, path: Path, frame_index: int) -> tuple[Tensor, Tensor]:
        """Read frames (t, t+gap) of one video through a fresh capture."""
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise RuntimeError(f"cannot open video {path}")
            if frame_index > 0 and not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
                raise RuntimeError(f"cannot seek to frame {frame_index} in {path}")
            ok, frame1 = capture.read()
            if not ok:
                raise RuntimeError(f"cannot read frame {frame_index} of {path}")
            for _ in range(self.frame_gap - 1):
                if not capture.grab():
                    raise RuntimeError(f"cannot skip to frame {frame_index + self.frame_gap}")
            ok, frame2 = capture.read()
            if not ok:
                raise RuntimeError(f"cannot read frame {frame_index + self.frame_gap} of {path}")
        finally:
            capture.release()
        return _decode(frame1), _decode(frame2)
