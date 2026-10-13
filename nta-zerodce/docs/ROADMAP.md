# Roadmap

Estimates are working days of engineering after datasets/compute are available,
not measured completion times. Owner is the repository maintainer (TBD by name).
All behavioral APIs are stubs at M0. Evidence must be committed/documented before
marking a later milestone complete; [TESTING](TESTING.md) defines checks.

| Milestone | Tasks | Owner | Estimate | Exit criteria / definition of done |
| --- | --- | --- | --- | --- |
| M0 scaffold and docs | exact layout, full mathematical/contracts docs, typed stubs, complete configs, tooling, initial commit | maintainer | 1..2 days | editable setup, lint/types/test/doc audits pass; all handwritten bodies TODO; no data/model implementation |
| M1 baseline | implement schema/IO/curves/7-conv model and four losses; registry/trainer/checkpoints; establish SICE split | maintainer | 4..6 days | shape/curve/baseline tests pass; tiny overfit and synthetic smoke pass; fixed held-out grid visually reproduces baseline behavior; reductions/sign explicitly logged, no unverified parity claim |
| M2 noise branch | implement synthetic noise, sigma head, fourth channel, invariance and bounded denoiser/fidelity | maintainer | 3..5 days | known-noise calibration test; no label leakage; branch-gradient tests; fixed flat/texture real-image comparison and limitations recorded |
| M3 edge and chroma | pure losses, detached gain/reliability, singles and interactions | maintainer | 2..3 days | zero-mask/gray/edge synthetic cases pass; edge/chroma single flags work without noise; fixed crop grid and loss scales recorded |
| M4 adaptive exposure | targets, replacement gate, highlight protection | maintainer | 2..3 days | numerical target-bound/threshold tests pass; bright-source crop evaluation; no simultaneous exp/adaptive double count |
| M5 temporal | DIS adapter, analytic pairs, warp/masks, pair training, curve state and reset; optional output blend | maintainer | 4..7 days | known affine conventions pass; backward current-domain mask; no history at cuts/new sizes; real and synthetic motion/flicker tests with coverage and ghosting examples |
| M6 ablation/report/demo | all 13 rows x 3 seeds, metrics CSVs/figures, light/curve-scale profiling, report and webcam demo | maintainer | 5..10 days plus GPU queue | reproducible artifacts/splits and CSV-backed tables; mean +/- std; held-out failure cases; measured latency; report sections complete; all relevant xfails removed |

Implement one contract at a time. Replace the corresponding TODO and strict xfail
with meaningful assertions together. Review formula/reduction/config changes in
the same Conventional Commit. Free-GPU limits may require checkpointed continuation;
do not quietly reduce epochs for one ablation. Schedule changes require a new
documented protocol shared across all affected rows.
