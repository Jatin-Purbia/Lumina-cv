"""Unpaired SICE training and paired LOL evaluation; DATA split rules.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from pathlib import Path

from torch import Tensor
from torch.utils.data import Dataset

from nta_zerodce.data.files import image_files
from nta_zerodce.data.transforms import paired_crop
from nta_zerodce.inference.io import read_image


class ImageDataset(Dataset[dict[str, Tensor]]):
    """Unpaired SICE training and paired LOL evaluation; DATA split rules. Items use unbatched tensors; DATA manifest contract."""

    def __init__(self, root: Path, crop_size: int = 256, paired_root: Path | None = None) -> None:
        """Index the dataset without loading it eagerly.

        Args:
            root: Existing source directory; images or videos according to dataset type.
            crop_size: Positive multiple of 16; train only.
            paired_root: Optional aligned high-reference root for image evaluation only.

        Returns:
            No value; records stable sorted paths.

        Method:
            DATA layouts; image eval preserves size, video pairs use gap=1 by default.

        Raises:
            ValueError: Invalid inputs or incompatible configuration.
        """
        if not root.is_dir():
            raise ValueError(f"Image dataset root must be a directory: {root}")
        self.root = root
        self.files = image_files(root)
        if not self.files:
            raise ValueError(f"No images found: {root}")
        if crop_size < 0 or (paired_root is None and (crop_size < 16 or crop_size % 16)):
            raise ValueError("Training crop must be a positive multiple of 16")
        self.crop_size = crop_size
        self.paired_root = paired_root
        if paired_root is not None:
            expected = {p.relative_to(root) for p in self.files}
            actual = {p.relative_to(paired_root) for p in image_files(paired_root)}
            if expected != actual:
                raise ValueError("Paired image manifests must match exactly by relative filename")

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
        return len(self.files)

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
        if not 0 <= index < len(self.files):
            raise IndexError(index)
        path = self.files[index]
        image = read_image(path)[0]
        if self.paired_root is not None:
            reference = read_image(self.paired_root / path.relative_to(self.root))[0]
            if image.shape != reference.shape:
                raise ValueError(f"Pair size mismatch: {path}")
            return {"image": image, "reference": reference}
        image, _ = paired_crop(image, image, self.crop_size)
        return {"image": image}
