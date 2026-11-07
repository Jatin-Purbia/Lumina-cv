PYTHON ?= python
.PHONY: setup lint typecheck test docs-check train-baseline train-full ablate demo
setup:
	$(PYTHON) -m pip install -e ".[dev]"
lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
typecheck:
	$(PYTHON) -m mypy
test:
	$(PYTHON) -m pytest
docs-check:
	$(PYTHON) -m pytest --collect-only -q tests/test_config_schema.py
train-baseline:
	$(PYTHON) scripts/train.py --config configs/baseline.yaml
train-full:
	$(PYTHON) scripts/train.py --config configs/full.yaml
ablate:
	$(PYTHON) scripts/ablate.py --matrix configs/ablation_matrix.yaml
demo:
	$(PYTHON) scripts/enhance.py --config configs/full.yaml --checkpoint checkpoints/full.pt --webcam 0
SMOKE ?= .smoke
.PHONY: smoke serve
# End-to-end check without datasets: synthetic images -> train -> enhance (CLI) -> verify output.
smoke:
	rm -rf $(SMOKE)
	$(PYTHON) scripts/make_synthetic_data.py --output $(SMOKE)/data/train --count 32 --size 96 --seed 1
	$(PYTHON) scripts/make_synthetic_data.py --output $(SMOKE)/data/test --count 4 --size 96 --seed 2
	$(PYTHON) scripts/train.py --config configs/smoke.yaml --run-name smoke
	$(PYTHON) scripts/enhance.py --checkpoint $(SMOKE)/checkpoints/smoke/final.pt --input $(SMOKE)/data/test --output $(SMOKE)/out --device cpu
	$(PYTHON) -c "import pathlib,sys; n=len(list(pathlib.Path('$(SMOKE)/out').rglob('*.png'))); print('enhanced', n, 'images'); sys.exit(0 if n else 1)"
# Start the HTTP API (needs LUMINA_CHECKPOINT); the web demo proxies to it.
serve:
	$(PYTHON) -m nta_zerodce.api
