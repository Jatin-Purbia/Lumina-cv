# Contributing

Use Python 3.10+, src layout, full annotations and docstrings stating purpose,
arguments with shape/range/device, returns, formulas/references and failure modes.
RGB float32 BCHW is canonical; conversion occurs only at IO. Ruff formats at 100
columns and enforces imports/annotations; mypy strict checks all source/CLIs/tests.
Pure tensor losses cannot move devices or read configuration files. Respect the
dependency rules in [ARCHITECTURE](ARCHITECTURE.md). Do not copy the reference code;
licensing/provenance requirements are in [NOTICE](../NOTICE.md).

Every callable is implemented and tested. Run `make lint typecheck test docs-check` before
sending a change; `make smoke` exercises train and enhance end to end. New behaviour needs a
test in the same change, and documentation status tables must be updated with it.

## Commit convention

Conventional Commits: feat:, fix:, docs:, test:, refactor:, chore:. Optional scopes
such as feat(noise): or docs(method):. Initial commit is exactly
chore: initial scaffold. Work on main unless the maintainer asks for a branch.
Do not commit data, runs, checkpoints, personal footage or notebook outputs.
Run pre-commit and make lint typecheck test docs-check before proposing a change.

## Add one loss or extension

1. Specify mathematical objective, reductions, tensor inputs, detach policy,
   mask-zero behavior and initial weight in METHOD before implementing it.
2. Add a pure function in losses with full hints/docstrings; annotate the formula
   as a project proposal or link an entry in REFERENCES.
3. Add registry metadata (name, callable, flag, default weight, required_inputs)
   and register using register_loss once implemented; avoid loss-import side effects.
4. Add its FeatureFlags/schema fields, CLI positive/negative override and explicit
   baseline/full YAML fields; validate prerequisites and unknown keys.
5. Gate it in loop_step, supply only declared context inputs, record raw/weighted
   logs and checkpoint-resolved config/flags.
6. Add signature/gradient/mask tests and a singleton/leave-one-out matrix row.
7. Update README feature/status rows, ARCHITECTURE graph/matrix, INTERFACES contract,
   METHOD weights, EXPERIMENTS and ROADMAP. Run checks before review.

This is the extension procedure; it is not permission to add a sixth feature without
maintainer approval. Layout changes require maintainer approval. Keep implementation and protocol
changes reviewable together; no fabricated measurements in docs/report.
