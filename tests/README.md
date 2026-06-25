# Model Library Tests

Test suite for `model_library`.

## Structure

```text
tests/
├── unit/          # Tests with mocked dependencies
├── integration/   # Tests with real provider API calls
└── conftest.py    # Shared fixtures and configuration
```

## Unit vs integration tests

### Unit tests

- Mock external dependencies such as provider APIs and databases.
- Run without API keys.
- Cover isolated logic such as request construction, parameter handling, and response parsing.

### Integration tests

> **Warning:** Integration tests make real provider calls. Use sandbox or least-privilege keys, expect provider billing/rate limits, and do not send sensitive prompts unless intentional.

- Make real provider API calls.
- Require provider API keys and may incur cost/rate limits.

## Running tests

```bash
# Run unit tests
make test

# Run integration tests
make test-integration

# Run a specific test file
make test DIR=tests/unit/test_tools.py

# Run a specific test function
make test DIR=tests/unit/test_tools.py::test_build_body_includes_tools

# Run integration tests for one model through the Makefile
make test-integration MODEL=openai/gpt-5-nano-2025-08-07

# Run a raw pytest command for one model
uv run pytest tests/integration -m integration --model=openai/gpt-5-nano-2025-08-07

# Run the long-problem integration test for one model
make test-integration DIR=tests/integration/test_long_problem.py MODEL=openai/gpt-5-nano-2025-08-07
```

Use `MODEL=...` with Makefile targets. Use `--model=...` only when invoking `pytest` directly. Model parametrization decorators skip tests that do not match the selected model.

## Decorators

- `@parametrize_all_models` — pass all model keys, excluding deprecated models and including alternative keys.
- `@parametrize_models_for_provider` — pass all model keys for one provider, excluding deprecated models and including alternative keys.

## Test markers

- `@pytest.mark.unit` — unit tests.
- `@pytest.mark.integration` — integration tests.

## Environment setup

Integration tests load environment variables from the repo root `.env`.

In internal checkouts, integration-test setup can create `.env` from AWS Secrets Manager when `.env` is missing. Public users should create `.env` themselves or export the provider API keys required by the integration tests they run.

## Notes

By default, tests run asynchronously and use 4 xdist workers. Parametrizations of the same test function run on the same worker.
