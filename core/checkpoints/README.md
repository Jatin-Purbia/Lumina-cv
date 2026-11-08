# Checkpoints

Trained weights for `nta_zerodce` (about 1 MB each, `full.py` config: noise, denoise, edge, chroma
and adaptive exposure on, temporal off). Load them only if you trust them: they use PyTorch's
pickle-based format. Start the API with one of them, for example
`$env:LUMINA_CHECKPOINT = "checkpoints/lol/final.pt"`.

| Folder | Files | Trained on | Notes |
| --- | --- | --- | --- |
| `lol/` | `final.pt` | LOL low-light photos, 10 epochs | **Recommended.** Gentle, natural brightening; identical to `lol_dark_only/epoch_0010.pt`. |
| `lol_dark_only/` | `epoch_0005.pt` ... `epoch_0040.pt` (every 5) | LOL low-light photos | One run stopped at epoch 40. Epochs 5 and 10 look good on ordinary photos; from epoch 15 they over-brighten and by epoch 25 they wash out. Later epochs score higher on very dark LOL test images (PSNR 10.7 at epoch 10, 13.5 at 15, 16.6 at 40). Kept for comparison. |
| `full/` | `epoch_0010.pt` ... `epoch_0040.pt`, `final.pt` | 96 synthetic procedural images, 40 epochs | First demo model. Flattens real photos to gray; not recommended. |

Evaluation scores and the over-brightening finding are in `docs/IMPLEMENTATION.md`. All
checkpoints above were trained with `loss.exposure_target=0.6`, the old schema default; the
configs now default to `0.5` as an unverified mitigation for the same regression (see
IMPLEMENTATION.md Limits), so retraining with the current configs will not reproduce these exact
weights.
