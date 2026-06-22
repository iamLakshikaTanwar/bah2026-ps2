# Makefile for the cloudremoval project (BAH 2026 PS2).
# All targets run on CPU with synthetic data unless a different config is passed.
#
#   make setup      # create venv + install core (editable) + dev extras
#   make install    # editable install of the core package
#   make smoke      # end-to-end CPU smoke: simulate -> train -> eval -> infer
#   make train      # train with a config (default: cpu_smoke)
#   make eval       # evaluate a checkpoint
#   make benchmark  # leaderboard over all registered models
#   make infer      # tiled inference -> COG
#   make serve      # launch the FastAPI app
#   make test       # run the pytest suite
#   make lint       # ruff check
#   make format     # black + ruff --fix
#   make docker-build
#   make clean

PYTHON ?= python3
PIP    ?= $(PYTHON) -m pip
VENV   ?= .venv
CONFIG ?= configs/cpu_smoke.yaml
CLI    ?= $(PYTHON) -m cloudremoval.cli

.DEFAULT_GOAL := help
.PHONY: help setup install smoke train eval benchmark infer serve test lint format \
        typecheck docker-build docker-up clean

help: ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Create a virtualenv and install core + dev (CPU torch).
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
	$(VENV)/bin/python -m pip install -e ".[dev]"

install: ## Editable install of the core package.
	$(PIP) install -e .

smoke: ## End-to-end CPU smoke pipeline on synthetic data.
	$(CLI) simulate  --config $(CONFIG)
	$(CLI) train     --config $(CONFIG)
	$(CLI) eval      --config $(CONFIG)
	$(CLI) infer     --config $(CONFIG)

train: ## Train the configured model (CONFIG=...).
	$(CLI) train --config $(CONFIG)

eval: ## Evaluate (CONFIG=..., CKPT=...).
	$(CLI) eval --config $(CONFIG) $(if $(CKPT),--ckpt $(CKPT),)

benchmark: ## Benchmark all registered models -> leaderboard.
	$(CLI) benchmark --config $(CONFIG)

infer: ## Tiled inference -> COG (CONFIG=..., CKPT=..., INPUT=..., OUT=...).
	$(CLI) infer --config $(CONFIG) \
		$(if $(CKPT),--ckpt $(CKPT),) $(if $(INPUT),--input $(INPUT),) $(if $(OUT),--out $(OUT),)

serve: ## Launch the FastAPI serving app.
	$(CLI) serve --config $(CONFIG)

test: ## Run the test suite (CPU).
	$(PYTHON) -m pytest -q

lint: ## Lint with ruff and check formatting with black.
	$(PYTHON) -m ruff check src tests scripts
	$(PYTHON) -m black --check src tests scripts

format: ## Auto-format with black and ruff --fix.
	$(PYTHON) -m black src tests scripts
	$(PYTHON) -m ruff check --fix src tests scripts

typecheck: ## Static type check with mypy.
	$(PYTHON) -m mypy src

docker-build: ## Build the API Docker image.
	docker build -t cloudremoval:latest .

docker-up: ## Start api + redis via docker-compose.
	docker compose up --build

clean: ## Remove caches, build artifacts and outputs.
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf outputs mlruns
