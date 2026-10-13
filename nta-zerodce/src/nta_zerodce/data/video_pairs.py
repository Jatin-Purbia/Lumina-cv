"""Adjacent real-video pairs; flow is delegated exclusively to flow/estimator.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from pathlib import Path

from torch import Tensor
from torch.utils.data import Dataset


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
            No value; future implementation records stable sorted paths and split manifests.

        Method:
            DATA layouts; image eval preserves size, video pairs use gap=1 by default.

        Raises:
            NotImplementedError: Scaffold only; implement this contract before use.
        """
        raise NotImplementedError("TODO: Index the dataset without loading it eagerly")

    def __len__(self) -> int:
        """Count indexed samples.

        Args:
            self: Indexed dataset.

        Returns:
            Nonnegative sample count; empty training roots must fail during initialization.

        Method:
            DATA deterministic index, no sampling during length queries.

        Raises:
            NotImplementedError: Scaffold only; implement this contract before use.
        """
        raise NotImplementedError("TODO: Count indexed samples")

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Decode one indexed sample.

        Args:
            index: Integer in [0,len(self)); reject out of range.

        Returns:
            Image: image (3,H,W), optional reference; video: image1/image2 (3,H,W), index scalar.

        Method:
            INTERFACES dataset batch keys; RGB [0,1] CPU, shared pair geometry.

        Raises:
            NotImplementedError: Scaffold only; implement this contract before use.
        """
        raise NotImplementedError("TODO: Decode one indexed sample")
