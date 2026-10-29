"""Adjacent real-video pairs; flow is delegated exclusively to flow/estimator.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import torch
from torch import Tensor
from torch.utils.data import Dataset

from nta_zerodce.data.files import video_files
from nta_zerodce.data.transforms import paired_crop


class VideoPairDataset(Dataset[dict[str, Tensor]]):
    """Adjacent real-video pairs; flow is delegated exclusively to flow/estimator. Items use unbatched tensors; DATA manifest contract."""

    def __init__(
        self, root: Path, crop_size: int = 256, paired_root: Path | None = None, frame_gap: int = 1
    ) -> None:
        """Index the dataset without loading it eagerly.

        Args:
            root: Existing source directory; images or videos according to dataset type.
            crop_size: Positive multiple of 16; train only.
            paired_root: Must be None for this real-video dataset; no high-light references.
            frame_gap: Positive source-frame separation, default 1.

        Returns:
            No value; records stable sorted paths.

        Method:
            DATA layouts; image eval preserves size, video pairs use gap=1 by default.

        Raises:
            ValueError: Invalid inputs or incompatible configuration.
        """
        if paired_root is not None or frame_gap < 1 or crop_size < 16 or crop_size % 16:
            raise ValueError("Invalid video pair settings")
        self.crop_size = crop_size
        self.frame_gap = frame_gap
        self.index: list[tuple[Path, int]] = []
        for path in video_files(root):
            capture = cv2.VideoCapture(str(path))
            try:
                if not capture.isOpened():
                    raise ValueError(f"Cannot open video: {path}")
                count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                self.index.extend((path, i) for i in range(max(0, count - frame_gap)))
            finally:
                capture.release()
        if not self.index:
            raise ValueError(f"No usable video pairs: {root}")

    def __len__(self) -> int:
        """Count indexed samples.

        Args:
            self: Indexed dataset.

        Returns:
            Nonnegative sample count; empty training roots must fail during initialization.

        Method:
            DATA deterministic index, no sampling during length queries.

        Raises:
            ValueError: Invalid inputs or incompatible configuration.
        """
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Decode one indexed sample.

        Args:
            index: Integer in [0,len(self)); reject out of range.

        Returns:
            Image: image (3,H,W), optional reference; video: image1/image2 (3,H,W), index scalar.

        Method:
            INTERFACES dataset batch keys; RGB [0,1] CPU, shared pair geometry.

        Raises:
            ValueError: Invalid inputs or incompatible configuration.
        """
        if not 0 <= index < len(self.index):
            raise IndexError(index)
        path, start = self.index[index]
        capture = cv2.VideoCapture(str(path))
        images = []
        try:
            for position in (start, start + self.frame_gap):
                capture.set(cv2.CAP_PROP_POS_FRAMES, position)
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"Cannot decode frame {position}: {path}")
                images.append(
                    torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    .permute(2, 0, 1)
                    .float()
                    / 255
                )
        finally:
            capture.release()
        first, second = paired_crop(images[0], images[1], self.crop_size)
        return {"image1": first, "image2": second, "index": torch.tensor(index)}
