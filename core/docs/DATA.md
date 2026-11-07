# Data and acquisition

## Sources and permissions

| Dataset | Role | Primary access | Terms to record before use |
| --- | --- | --- | --- |
| SICE part 1 | unpaired training, no high-light GT | [SICE authors](https://github.com/csjcai/SICE#dataset), [Zero-DCE README training-data links](https://github.com/Li-Chongyi/Zero-DCE#train) | preserve downloaded archive terms and source date; no MIT data grant inferred |
| LOL original | paired evaluation only | [Retinex-Net project](https://daooshee.github.io/BMVC2018website/) dataset links | inspect archive/provider terms; cite authors; no redistribution assumed |
| DICM/LIME/MEF/NPE | unpaired qualitative/proxy evaluation | [Zero-DCE project](https://li-chongyi.github.io/Proj_Zero-DCE.html) test materials and [Retinex-Net testing images](https://daooshee.github.io/BMVC2018website/) | record individual upstream terms, attribution and scene provenance |
| Own videos | temporal train/held-out evaluation | private recording | obtain participant/location consent; document permitted use |

SICE is described in [R4](REFERENCES.md#r4); LOL/Retinex-Net in
[R5](REFERENCES.md#r5). Links are discovery pages, not a promise of perpetual
availability or an inferred open-data licence. Code licence CC BY-NC 4.0 for
Zero-DCE does not establish dataset licences. Do not distribute data in this repo.
Acquisition is manual; [download_data.md](../scripts/download_data.md) is a checklist.

## Expected local layout

```text
data/
  SICE/part1/<scene-id>/<exposure-id>.jpg
  LOL/eval15/low/<pair-id>.png
  LOL/eval15/high/<pair-id>.png
  unpaired/DICM/<image>.jpg
  unpaired/LIME/<image>.bmp
  unpaired/MEF/<image>.jpg
  unpaired/NPE/<image>.jpg
  videos/train/<session>/<clip>.mp4
  videos/val/<session>/<clip>.mp4
  videos/test/<session>/<clip>.mp4
  synthetic/<source-id>/<pair-id>/...
```

These are local conventions, not claims about archive naming. Keep original
archives separately; normalize layout explicitly. Baseline train_images recursively
indexes image files, never labels/reference composites. Match LOL basenames
exactly and reject missing/duplicate pairs. Store a private split manifest with
source hash, scene/session ID, exposure/frame index and acquisition terms.
Config paths resolve from the core workspace root; set them in the `paths` section of the config.

## Leakage and preprocessing

Split SICE by scene, keeping all exposures of one scene together; initial choice
is a deterministic 90/10 train/validation split with split seed 17 and sorted scene
IDs. Keep the manifest unchanged for all training seeds. Only low-light exposures
enter unpaired training: manually document exposure selection from scene metadata;
do not treat provided enhanced composites as input. LOL eval15 is held out from
all model/weight choices. Own video splits are by recording session/location,
never by adjacent frames. Check overlap between repackaged image sets.

Decode to RGB float32 [0,1], apply EXIF orientation, reject corrupted files with
an explicit log. No linearization, auto white balance or per-image min-max scaling.
Train with shared 256x256 crops on pairs; no extra augmentations are assumed. Do not resize paired evaluation differently. Avoid padding
small images silently; skip with a count or choose a documented resize policy
before M1. Pin that choice in a protocol change rather than improvising per run.

## Synthetic noise

[METHOD](METHOD.md) fixes the coefficients and variance. Draw a,b per image;
independent epsilon per RGB pixel, fixed RNG per sample/epoch for reproducibility.
Clip only the noisy image; preserve clean input and pre-clipping sigma target.
This is Gaussian-approximated shot/read noise in display RGB, not a raw sensor
simulation. Existing source noise and JPEG artifacts remain; calibrate on real
flat areas in M2 before interpreting predicted sigma as actual camera noise.

## Synthetic temporal pairs

Sample an affine about image centre (rotation +/-3 degrees, translation +/-8 px,
scale .98..1.02); use inverse mapping to create the second clean frame. Compute
analytic forward/inverse flows in original pixel units. Apply second-frame
exposure flicker .9..1.1 and independent noise to both frames. Use shared crop
geometry; recompute affine coordinates after crop. Mask invalid borders, do not
use the real-pair photometric gate on intended synthetic flicker. Save the source
ID, seed, affine matrix, a/b coefficients and flicker factor with generated data.
Never synthesize across evaluation source scenes. scripts/make_synthetic_data.py
writes noisy images (`clean.png`, `noisy.png`, `meta.json`) or, with `--pairs`, temporal pairs
(`image1.png`, `image2.png`, `meta.json`) under `<output>/<index>_<name>/`; the sidecar records
source, seed, noise coefficients and, for pairs, the affine matrix and flicker factor. Without
`--input` it renders procedural low-light scenes, so no download is needed .
No generated or real datasets are committed.

## Build a video set

Capture 720p or 1080p, native 25/30 fps, 10..30 second clips. Retain original
frame rate/time metadata; use adjacent gap-one pairs initially. Cover static
camera, slow pan, moving people/objects, lamps/screens, mixed lighting, severe
darkness and deliberate cuts. Keep a private per-clip camera/exposure/ISO/codec
log where available. Real pairs use CPU OpenCV DIS, medium preset, half-resolution
flow then correctly rescaled full-resolution vectors; cache only with frame/flow
config hashes. Avoid lossy re-encoding before metric extraction.

Classroom/hostel footage requires consent from identifiable participants and
permission from the location; exclude private screens, documents and rooms
without permission. Avoid recording minors without required guardian/institution
permission. Keep raw footage private and use consented/anonymized examples in the
report. Remove audio for experiments unless separately consented; demo IO does
not promise audio preservation. No recordings are made by this scaffold.
