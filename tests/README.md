# Model Library Tests

Test suite for the model_library.

## Structure

```
tests/
├── unit/                           # Tests with mocked dependencies
│   ├── test_completion.py         # Test request construction, response parsing
│   └── providers/
│       └── test_google_provider.py # Provider-specific logic tests
│
├── integration/                    # Tests with real API calls
│   ├── test_completion.py         # Real completion API tests
│   ├── test_streaming.py          # Real streaming API tests
│   └── test_reasoning.py          # Reasoning/thinking mode tests
│
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
# Setup

# Run all tests
pytest tests/

# Run only unit tests (no API keys needed)
pytest tests/unit/ -m unit

# Run only integration tests (requires API keys)
pytest tests/integration/ -m integration

# Run specific test file
pytest tests/unit/test_completion.py

# Run with verbose output
pytest tests/ -v

# Run specific test class or method
pytest tests/unit/test_completion.py::TestCompletionUnit::test_temperature_parameter
```

## Test Markers

- `@pytest.mark.unit` - Unit tests with mocked dependencies
- `@pytest.mark.integration` - Integration tests requiring API access
- `@pytest.mark.slow` - Tests that may take longer to run
- `@requires_google_api` - Skip if Google/Gemini API key not available

## Environment Setup

The test suite loads environment variables from root `.env`

Required environment variables:

- `GEMINI_API_KEY` or `GOOGLE_API_KEY` - For Google Gemini tests
- `OPENAI_API_KEY` - For OpenAI compatibility mode tests (optional)

## Fixtures

Key fixtures from `conftest.py`:

- `mock_google_client` - Mocked Google client for unit tests
- `mock_google_response` - Factory for creating mock responses
- `create_test_input` - Helper to create TextInput/FileInput objects
- `assert_thinking_config` - Helper to validate thinking configuration
