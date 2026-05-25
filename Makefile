.PHONY: help install test test-integration test-all style style-check typecheck config deprecate run-models browse_models chat

help:
	@echo "Makefile for model-library"
	@echo "Usage:"
	@echo "  make install          Install dependencies"
	@echo "  make test             Run unit tests"
	@echo "  make test-integration Run integration tests (requires API keys)"
	@echo "  make test-all         Run all tests (unit + integration)"
	@echo "  make style            Lint & Format"
	@echo "  make style-check      Check style"
	@echo "  make typecheck        Typecheck"
	@echo "  make config           Generate all_models.json"
	@echo "  make deprecate        Deprecate a model"
	@echo "  make run-models       Run all models"
	@echo "  make browse_models    Interactively browse models and their configurations"
	@echo "  make chat             Run the browser chat UI"

PYTHON_VERSION ?= 3.11

install:
	uv venv --python $(PYTHON_VERSION)
	uv sync --dev --extra docent
	@echo "🎉 Done! Run 'source .venv/bin/activate' to activate the environment locally."

venv_check:
	@if [ ! -f .venv/bin/activate ]; then \
		echo "❌ Virtualenv not found! Run \`make install\` first."; \
		exit 1; \
	fi

DIR ?= tests/
test: venv_check
	@echo "Running unit tests..."
	@uv run pytest $(DIR) -m unit -v -n 4 --dist loadscope --model=$(MODEL)
test-integration: venv_check
	@echo "Running integration tests (requires API keys)..."
	@uv run pytest $(DIR) -m integration -v -n 4 --dist loadscope --model=$(MODEL)

format: venv_check
	@uv run ruff format .
lint: venv_check
	@uv run ruff check --fix .
style: format lint

style-check: venv_check
	@uv run ruff format --check .
	@uv run ruff check .

typecheck: venv_check
	@uv run basedpyright

config: venv_check
	@uv run python scripts/config.py

deprecate: venv_check
	@uv run python scripts/deprecate_model.py $(model)

run-models: venv_check
	@uv run python -m scripts.run_models

browse_models: venv_check
	@uv run python -m scripts.browse_models

chat: venv_check
	@uv run python -m model_library.web
