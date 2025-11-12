.PHONY: help install test test-integration test-all style style-check typecheck config run-models examples browse_models

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
	@echo "  make run-models       Run all models"
	@echo "  make examples         Run all examples"
	@echo "  make examples <model> Run all examples with specified model"
	@echo "  make browse_models    Interactively browse models and their configurations"

install:
	uv venv
	uv sync --dev
	@echo "🎉 Done! Run 'source .venv/bin/activate' to activate the environment locally."

venv_check:
	@if [ ! -f .venv/bin/activate ]; then \
		echo "❌ Virtualenv not found! Run \`make install\` first."; \
		exit 1; \
	fi

test: venv_check
	@echo "Running unit tests..."
	@uv run pytest tests/unit/ -m "not integration"
test-integration: venv_check
	@echo "Running integration tests (requires API keys)..."
	@uv run pytest tests/integration/ -m "not unit"
test-all: venv_check
	@echo "Running all tests..."
	@uv run pytest

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

run-models: venv_check
	@uv run python -m scripts.run_models

examples: venv_check
	@echo "Running examples with model: $(filter-out $@,$(MAKECMDGOALS))"
	@echo "\n=== Running basics.py ==="
	@uv run python -m examples.basics $(filter-out $@,$(MAKECMDGOALS)) || true
	@echo "\n=== Running images.py ==="
	@uv run python -m examples.images $(filter-out $@,$(MAKECMDGOALS)) || true
	@echo "\n=== Running files.py ==="
	@uv run python -m examples.files $(filter-out $@,$(MAKECMDGOALS)) || true
	@echo "\n=== Running tool_calls.py ==="
	@uv run python -m examples.tool_calls $(filter-out $@,$(MAKECMDGOALS)) || true
	@echo "\n=== Running advanced.structured_output ==="
	@uv run python -m examples.advanced.structured_output $(filter-out $@,$(MAKECMDGOALS)) || true
	@echo "\n=== Running advanced.batch ==="
	@uv run python -m examples.advanced.batch $(filter-out $@,$(MAKECMDGOALS)) || true
	@echo "\n=== Running advanced.custom_retrier ==="
	@uv run python -m examples.advanced.custom_retrier $(filter-out $@,$(MAKECMDGOALS)) || true
	@echo "\n=== Running advanced.stress ==="
	@uv run python -m examples.advanced.stress $(filter-out $@,$(MAKECMDGOALS)) || true
	
	@if [ -z "$(filter-out $@,$(MAKECMDGOALS))" ]; then \
		echo "\n=== Running embeddings.py ==="; \
		uv run python -m examples.embeddings; \

		echo "\n=== Running advanced.deep_research ==="; \
		uv run python -m examples.advanced.deep_research; \
	fi

	@echo "\n✅ All examples completed!"
	
browse_models: venv_check
	@uv run python -m scripts.browse_models

