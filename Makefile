.PHONY: all run ui cli test clean help install

# Default shell
SHELL := /usr/bin/env bash

# Python & UV configuration
VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
DESHIMMER_UI ?= $(VENV)/bin/deshimmer-ui
DESHIMMER_CLI ?= $(VENV)/bin/deshimmer-cli
UV := $(shell command -v uv 2>/dev/null)
MPLCONFIGDIR ?= $(CURDIR)/.matplotlib_cache

# Default target
all: help

## install: Set up virtual environment and install all dependencies
install:
	@mkdir -p $(MPLCONFIGDIR)
	@if [ ! -d "$(VENV)" ]; then \
		echo "==> Creating virtual environment..."; \
		if [ -n "$(UV)" ]; then \
			$(UV) venv $(VENV) && $(UV) pip install -e ".[dev]"; \
		else \
			python3 -m venv $(VENV) && $(PYTHON) -m pip install -e ".[dev]"; \
		fi \
	fi
	@if [ ! -f "$(DESHIMMER_UI)" ]; then \
		if [ -n "$(UV)" ]; then \
			$(UV) pip install -e ".[dev]"; \
		else \
			$(PYTHON) -m pip install -e ".[dev]"; \
		fi \
	fi

## run: Start the interactive Gradio Web UI (bootstrapping dependencies if needed)
run: install
	@echo "==> Launching Deshimmer Gradio Web UI at http://localhost:7860 ..."
	@MPLCONFIGDIR="$(MPLCONFIGDIR)" $(PYTHON) -m deshimmer.ui.app

## ui: Alias for make run
ui: run

## cli: Run the Deshimmer CLI (use ARGS="..." to pass flags)
cli: install
	@MPLCONFIGDIR="$(MPLCONFIGDIR)" $(PYTHON) -m deshimmer.cli $(ARGS)

## test: Run automated unit and regression tests
test: install
	@echo "==> Running test suite..."
	@MPLCONFIGDIR="$(MPLCONFIGDIR)" $(VENV)/bin/pytest tests

## clean: Remove build artifacts, bytecode, and cache directories
clean:
	@echo "==> Cleaning caches and artifacts..."
	@rm -rf build/ dist/ *.egg-info .pytest_cache .uv_cache
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.py[cod]" -delete 2>/dev/null || true

## help: Display this help message
help:
	@echo "Deshimmer - Audio Artifact Reconstruction & Spectral Repair"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@grep -E '^## ' $(MAKEFILE_LIST) | sed -e 's/## //g' | awk 'BEGIN {FS = ":"}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'
