# Real-time design and measurement

Real-time is a goal, not an achieved result. The light estimator is a project
variant inspired by [R3](REFERENCES.md#r3); no published throughput applies to
our implementation/hardware. Use original full RGB for curve application; only
curve/sigma estimation is downscaled. Upsample A and sigma bilinearly to original
size with align_corners=False (map interpolation), then apply curves. Warping
coordinates use align_corners=True, a separate sampling contract.

## Target latency budget

The following is a design allocation for 30 fps at 720p (33.3 ms/frame), not a
measurement or prediction of Colab/webcam performance.

| Stage | Target ms | What to measure |
| --- | --- | --- |
| capture/decode and RGB conversion | 3 | camera/codec stalls separately |
| resize and device upload | 2 | transfer included |
| noise and curve estimation | 10 | baseline/light, curve_scale sweep |
| bidirectional DIS and mask | 8 | CPU latency and transfer, flow_scale=.5 initial |
| map warp/blend and eight curves | 4 | full-resolution costs |
| optional residual denoise | 3 | enabled/disabled |
| display or encode | 3 | synchronization and queueing |
| total allocation | 33 | remaining .3 ms headroom is small |

If this allocation cannot be met, report measured FPS and revise performance
settings explicitly; do not hide dropped frames or call a partial GPU stage FPS
end-to-end throughput. Temporal-off skips DIS/warp; denoise-off skips its head.

## Controls

curve_scale defaults to 1.0 and accepts (0,1]. Sweep 1,.5,.25 with fixed rounded
estimator dimensions >=16. Full-resolution curve application protects detail but
small maps can miss tiny lamps/edges. Light swaps hidden convolutions only; record
parameter counts and checkpoint compatibility after implementation. Flow scale
defaults .5; sweep .25 independently, rescale dx by W/W_small and dy by H/H_small.
Do not combine performance changes with main extension ablations unless labelled.
Noise maps must still be available to full-resolution denoising.

## Empty FPS table

| Device/software | RGB resolution | Light | Curve scale | Flow scale | Temporal/denoise | Median/p95 ms | FPS | Mask coverage | Quality CSV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | | | |

## Profiling recipe

1. Record device, CPU threads, torch/OpenCV versions, input codec/resolution/fps,
   all flags/scales and checkpoint commit. Use the same held-out 300-frame clip.
2. Eval/no_grad, batch one, 30 warm-up frames, then time 300 frames using
   perf_counter_ns. Synchronize CUDA around timed GPU boundaries; log stage times
   with torch.profiler for a short representative window rather than the full run.
3. Include transfers, CPU flow, display/encoding and frame queue behavior in
   end-to-end timing; separately report pure enhancement time.
4. Report median/p95 stage and end-to-end latency, throughput, dropped frames,
   VRAM peak, host RAM and temporal coverage. Throughput uses total wall time.
5. Compare quality on the same clip/images at all scale/light options. Show lamps,
   noise, motion and cuts; a faster but ghosting model is not automatically better.

```text
FPS = processed_frames / measured_wall_seconds
frame_budget_ms = 1000 / target_fps
```

Webcam lifetime owns one VideoEnhancer. Reset on reopening camera or size changes,
release resources on interruption, and never run multiple capture streams through
one state object. No capture or profiling implementation is present at M0.
