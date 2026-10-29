"""Sample-level paired, unpaired and temporal evaluation with explicit coverage."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from nta_zerodce.data.files import VIDEO_SUFFIXES, image_files, video_files
from nta_zerodce.data.image_dataset import ImageDataset
from nta_zerodce.flow.estimator import estimate_bidirectional
from nta_zerodce.flow.masks import is_scene_cut, reliability_mask
from nta_zerodce.inference.enhancer import enhance_image
from nta_zerodce.inference.io import read_image, read_video
from nta_zerodce.inference.video_enhancer import VideoEnhancer
from nta_zerodce.metrics.colour import angular_error
from nta_zerodce.metrics.exposure import clipping_fraction
from nta_zerodce.metrics.fidelity import psnr, ssim
from nta_zerodce.metrics.noise import noise_estimate
from nta_zerodce.metrics.temporal import warping_error
from nta_zerodce.models.nta_zerodce import NTAZeroDCE
from nta_zerodce.training.config import Config
from nta_zerodce.training.validation import core_path


def flat_mask(image: Tensor, threshold: float) -> Tensor:
    """Input-only intensity gradient threshold, shared across enhancement methods."""
    y = image.mean(1, keepdim=True)
    dx = F.pad((y[..., 1:] - y[..., :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((y[..., 1:, :] - y[..., :-1, :]).abs(), (0, 0, 0, 1))
    return (torch.maximum(dx, dy) < threshold).to(image.dtype)


def evaluate(
    model: NTAZeroDCE, config: Config, destination: Path, source: Path | None = None
) -> None:
    """Write long-form metrics. Explicit input selects unpaired images or videos."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    model.eval()
    with destination.open("w", newline="", encoding="utf-8") as handle, torch.no_grad():
        writer = csv.writer(handle)
        writer.writerow(
            [
                "run",
                "seed",
                "dataset",
                "sample_id",
                "metric",
                "value",
                "mask_coverage",
                "input_clipping",
            ]
        )

        def record(
            dataset: str,
            sample: str,
            image: Tensor,
            values: dict[str, Tensor],
            coverage: float = 1.0,
        ) -> None:
            for name, value in values.items():
                writer.writerow(
                    [
                        Path(config.paths.run_dir).name,
                        config.training.seed,
                        dataset,
                        sample,
                        name,
                        float(value[0]),
                        coverage,
                        float(clipping_fraction(image)[0]),
                    ]
                )

        def proxies(dataset: str, sample: str, image: Tensor, output: Tensor) -> None:
            record(
                dataset,
                sample,
                image,
                {"clipping": clipping_fraction(output), "mean_intensity": output.mean((1, 2, 3))},
            )
            mask = flat_mask(image, config.loss.reliability_low)
            record(
                dataset,
                sample,
                image,
                {"noise_proxy": noise_estimate(output, mask)},
                float((F.avg_pool2d(mask, 3, stride=1) >= 1).float().mean()),
            )

        if source is not None and (
            source.suffix.lower() in VIDEO_SUFFIXES or (source.is_dir() and not image_files(source))
        ):
            files = video_files(source)
            if not files:
                raise ValueError(f"No videos found: {source}")
            for path in files:
                enhancer = VideoEnhancer(model, config)
                previous = previous_output = None
                for index, frame in enumerate(read_video(path)):
                    frame = frame.to(device)
                    output = enhancer.enhance_frame(frame)["enhanced"]
                    sample = f"{path.name}:{index}"
                    proxies("video", sample, frame, output)
                    if (
                        previous is not None
                        and previous_output is not None
                        and not is_scene_cut(
                            previous,
                            frame,
                            config.temporal.scene_cut_threshold,
                            config.temporal.histogram_bins,
                        )
                    ):
                        f = config.flow
                        forward, backward = estimate_bidirectional(previous, frame, f)
                        forward, backward = forward.to(device), backward.to(device)
                        mask = reliability_mask(
                            forward,
                            backward,
                            previous,
                            frame,
                            f.fb_alpha,
                            f.fb_beta,
                            f.photometric_threshold,
                        )
                        record(
                            "video",
                            sample,
                            frame,
                            {
                                "warping_error": warping_error(
                                    previous_output, output, backward, mask
                                )
                            },
                            float(mask.mean()),
                        )
                    previous, previous_output = frame, output
            return
        if source is not None:
            files = image_files(source)
            if not files:
                raise ValueError(f"No images found: {source}")
            for path in files:
                image = read_image(path).to(device)
                output = enhance_image(model, image, config.model.curve_scale)["enhanced"]
                proxies(
                    "unpaired",
                    str(path.relative_to(source)) if source.is_dir() else path.name,
                    image,
                    output,
                )
            return
        paired = ImageDataset(
            core_path(config.paths.eval_low), paired_root=core_path(config.paths.eval_high)
        )
        for index, path in enumerate(paired.files):
            item = paired[index]
            image, reference = item["image"][None].to(device), item["reference"][None].to(device)
            output = enhance_image(model, image, config.model.curve_scale)["enhanced"]
            record(
                "paired",
                path.name,
                image,
                {
                    "psnr": psnr(output, reference),
                    "ssim": ssim(output, reference),
                    "angular_error": angular_error(output, reference),
                },
            )
            proxies("paired", path.name, image, output)
        unpaired = core_path(config.paths.unpaired)
        if unpaired.exists():
            for path in image_files(unpaired):
                image = read_image(path).to(device)
                output = enhance_image(model, image, config.model.curve_scale)["enhanced"]
                proxies("unpaired", str(path.relative_to(unpaired)), image, output)
