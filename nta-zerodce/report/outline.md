# Final project report outline

No Word report template was attached; this uses conventional research-report
sections, ready to map to the institution's headings later. It contains no results.

| Section | Content and questions | Source doc | Figure / table / evidence |
| --- | --- | --- | --- |
| Title and abstract | problem, proposed five extensions, evaluated scope; write quantitative claims only after M6 | README, EXPERIMENTS | final CSV-backed headline metrics |
| 1 Introduction | noise/colour/highlights/flicker in hostel/classroom/night capture; goals and research questions | README, DATA | motivating consented examples |
| 2 Background and related work | Zero-DCE/++, zero-reference losses, heteroscedastic noise, optical flow and temporal consistency | REFERENCES, METHOD, GLOSSARY | baseline vs project design table |
| 3 Requirements and research design | free GPU budget, independent flags, reproducible ablations and licence boundaries | ARCHITECTURE, EXPERIMENTS, NOTICE | extension/flag matrix |
| 4 Proposed methodology | curve equations, noise/denoise, edge/chroma, adaptive/highlight, temporal masks/blends | METHOD | F1 architecture, numbered formulas, weights table |
| 5 System architecture and implementation | package responsibilities, dependency/device/layout rules, schema, checkpoints, CLI | ARCHITECTURE, INTERFACES | training/inference graphs and API contract excerpts |
| 6 Data and experimental setup | sources/terms, scene/session splits, synthetic pairs/noise, seeds/schedule, evaluation metrics | DATA, EXPERIMENTS, TESTING | split manifest counts, 13-row ablation matrix |
| 7 Results and ablations | full vs baseline, singleton and leave-one-out, three-seed variability, image/video evidence | EXPERIMENTS | empty templates replaced from CSV, F2/F3/F4 |
| 8 Real-time demo and profiling | light/scales, stage timing, hardware, dropped frames and quality tradeoffs | REAL_TIME | measured latency/FPS table, F5 |
| 9 Discussion and threats | failed cases, noise-model mismatch, masks, proxy limitations, privacy/data overlap | EXPERIMENTS, DATA | failure grids, mask coverage and input clipping |
| 10 Conclusion and future work | only supported findings, unresolved decisions and limits | ROADMAP, EXPERIMENTS | completed milestones and bounded future scope |
| References | cited primary works and dataset/source attribution | REFERENCES, CITATION.cff | institution-formatted bibliography |
| Appendices | resolved YAMLs, environment/commit/split hashes, detailed tests, CLI recipes, consent process | INTERFACES, TESTING, CONTRIBUTING | reproducibility checklist and artifact locations |

F1..F5 follow the fixed qualitative plan in docs/EXPERIMENTS.md. Table cells remain
empty until a script-generated CSV and run metadata support them. Report M0 green
checks separately from tested M1..M6 behavior. Avoid presenting synthetic sanity
examples as empirical improvement, or reference-paper speed as our measurements.
