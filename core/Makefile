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
