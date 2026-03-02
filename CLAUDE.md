# CLAUDE.md

## Project overview

model-library is an open-source Python library by Vals AI that provides a unified interface for interacting with LLM providers (Anthropic, OpenAI, Google, Mistral, AWS Bedrock, and more).

## Development setup

```bash
make install          # Install dependencies with uv
source .venv/bin/activate
```

## Common commands

```bash
make test             # Run unit tests
make test-integration # Run integration tests (requires API keys)
make style            # Format (ruff format) and lint (ruff check --fix)
make style-check      # Check formatting and linting without fixing
make typecheck        # Run basedpyright (strict mode)
make config           # Regenerate all_models.json from YAML configs
```

## Code standards

- Python 3.11+
- Type checking: basedpyright in strict mode
- Formatting/linting: ruff
- Tests: pytest with pytest-asyncio; markers `@pytest.mark.unit` and `@pytest.mark.integration`
- Package management: uv with uv.lock
- Always run `make style` and `make typecheck` before submitting changes
- Unit tests should not require API keys or external services
