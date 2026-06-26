PY ?= /home/ben/automated-day-trading-venv/bin/python3

.PHONY: qa qa-check autoflake isort ruff format mypy test bench universe live

qa: autoflake isort ruff format mypy
	@echo "QA complete."

qa-check:
	$(PY) -m autoflake --check --quiet --recursive quantzero tests
	$(PY) -m isort --check-only quantzero tests
	$(PY) -m ruff check quantzero tests
	$(PY) -m black --check quantzero tests
	$(PY) -m mypy quantzero

autoflake:
	$(PY) -m autoflake --in-place --remove-all-unused-imports --remove-unused-variables --recursive quantzero tests

isort:
	$(PY) -m isort quantzero tests

ruff:
	$(PY) -m ruff check --fix quantzero tests

format:
	$(PY) -m black quantzero tests

mypy:
	$(PY) -m mypy quantzero

test:
	$(PY) -m pytest

bench:
	$(PY) -m quantzero.bench

universe:
	$(PY) -m quantzero.universe build

live:
	$(PY) -m quantzero.run_live $(ARGS)
