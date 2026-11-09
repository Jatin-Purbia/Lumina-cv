# Checkpoints

Trained weights for `nta_zerodce` (about 1 MB each, `full.py` config: noise, denoise, edge, chroma
and adaptive exposure on, temporal off). Load them only if you trust them: they use PyTorch's
pickle-based format. Start the API with one of them, for example
`$env:LUMINA_CHECKPOINT = "checkpoints/lol/final.pt"`.

| Folder | Files | Trained on | Notes |
| --- | --- | --- | --- |
| `lol/` | `final.pt` | LOL low-light photos, 10 epochs | **Recommended.** Gentle, natural brightening; identical to `lol_dark_only/epoch_0010.pt`. |
| `lol_dark_only/` | `epoch_0005.pt` ... `epoch_0040.pt` (every 5) | LOL low-light photos | One run stopped at epoch 40. Epochs 5 and 10 look good on ordinary photos; from epoch 15 they over-brighten and by epoch 25 they wash out. Later epochs score higher on very dark LOL test images (PSNR 10.7 at epoch 10, 13.5 at 15, 16.6 at 40). Kept for comparison. |
| `photo_target05_20261109/` | `epoch_0001.pt` ... `epoch_0010.pt`, `final.pt` | LOL low-light photos, exposure target 0.5, 10 epochs, seed 17 | Experimental, not promoted. Better normal-input SSIM but worse low-light SSIM/detail than the shipped target-0.6 model. See `docs/ARCHITECTURE.md`. |

The first demo model, trained on 96 synthetic procedural images for 40 epochs, flattened real
photos to gray and has been removed rather than kept as a "do not use" artifact.

Evaluation scores and the over-brightening finding are in `docs/ARCHITECTURE.md`. All
original `lol`/`lol_dark_only` checkpoints were trained with `loss.exposure_target=0.6`.
The recommended photo recipe retains 0.6; the completed 0.5 experiment is preserved as
`configs/photo_target05.py` and was not selected as the default. Baseline/full research
configs still use 0.5, so their current settings do not reproduce the original weights.
