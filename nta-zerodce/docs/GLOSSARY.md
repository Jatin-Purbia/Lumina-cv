# Glossary

| Term | Meaning in this repository |
| --- | --- |
| Zero-reference | Enhancement training without high-light image targets for real inputs; synthetic auxiliary targets remain allowed |
| DCE-Net | Curve-map estimator introduced by [R1](REFERENCES.md#r1) |
| LE curve | Bounded fixed-parameter quadratic update, applied eight times |
| A | Iteration-major RGB curve parameters, BCHW with 24 channels, [-1,1] |
| enhanced_raw | RGB after curves and before residual denoise |
| enhanced | Final RGB after optional denoise; optional temporal output blend at inference |
| sigma | Per-pixel standard deviation in normalized RGB units, not variance |
| Heteroscedastic | Noise variance depends on signal; here a*x+b^2 |
| Gain | Detached, bounded local output/input mean-intensity ratio; not the exact curve derivative |
| Chromaticity | RGB divided by RGB sum plus epsilon, local proportion proxy |
| Luminance / Y | Mean of display RGB for compatibility here, not physical linear-light luminance |
| Smoothstep | Clamped cubic interpolation between two thresholds |
| Flow f(a,b) | Source-to-target displacement satisfying a(p) approximately b(p+f(p)) |
| Backward flow | Current->previous displacement used to sample previous-frame tensors |
| Warp | Bilinear backward sampling; warp(b,f(a,b)) places b in a's coordinates |
| FB mask | Current-domain forward/backward consistency and valid-boundary mask |
| DIS | Dense Inverse Search optical flow, [R6](REFERENCES.md#r6) |
| Scene cut | Histogram-change heuristic clearing temporal state |
| Curve-space blend | Convex mix of current and reliable warped previous A maps |
| curve_scale | Inference estimator resolution ratio; original RGB is enhanced at full resolution |
| light | Independent depthwise-separable model variant; not validated Zero-DCE++ parity |
| Ablation | Controlled removal/addition of a feature, with shared data/schedule/seeds |
| Proxy metric | Indirect statistic that may not agree with perceived quality |
| xfail | Expected pending test failure, not evidence that algorithm behavior works |
| M0 | Documentation/interface/tooling scaffold milestone |
