# Model Library Tests

Test suite for the model_library.

## Structure

```
tests/
├── unit/                          # Tests with mocked dependencies
├── integration/                   # Tests with real API calls
├── conftest.py                    # Shared fixtures and configuration
```

## Unit vs Integration Tests

### Unit Tests

- Mock external dependencies (APIs, databases)
- Test isolated logic without network calls
- Fast, deterministic, run without API keys
- Focus on: request construction, parameter handling, response parsing
- Example: "Does the code add `thinking_config` when model is gemini-2.5?"

### Integration Tests

- Make real API calls to providers
- Test end-to-end functionality
- Slower, require API keys
- Focus on: actual API behavior, token counting, streaming
- Example: "Does Gemini API actually return reasoning tokens?"

## Running Tests

```bash
# Run unit tests
make test

# Run integration tests
make test-integration

# Run specific test file
make test DIR=tests/unit/test_tools.py

# Run specific test class or method
make test DIR=tests/unit/test_completion.py::TestCompletionUnit::test_temperature_parameter

# Run specific model
make test-integration MODEL=openai/gpt-5-nano-2025-08-07

# Run long problem test
make test-integration DIR=tests/integration/test_long_problem.py MODEL=openai/gpt-5-nano-2025-08-07

```

## Decorators

- `@parametrize_all_models` - Pass in all model keys, excluding deprecated, including alternative keys
- `@parametrize_models_for_provider` - Pass in all model keys for a specific provider, excluding deprecated, including alternative keys

When passing in --MODEL, these decorators will skip tests that don't match the model key.

## Test Markers

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests

## Environment Setup

Integration tests load environment variables from root `.env`

## Misc

By default, tests run asynchronously, and multithreaded with 4 workers.
Parameter to the same function run on the same worker.

