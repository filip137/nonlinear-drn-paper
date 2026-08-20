PYTHON ?= python

.PHONY: verify figures smoke-train test

verify:
	$(PYTHON) scripts/reproduce.py verify

figures:
	$(PYTHON) scripts/reproduce.py figures

smoke-train:
	$(PYTHON) scripts/reproduce.py train-smoke --device cpu

test:
	$(PYTHON) -m pytest
