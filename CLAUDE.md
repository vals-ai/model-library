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
make style            # Format (ruff format) and lint (ruff check --fix)
make typecheck        # Run basedpyright (strict mode)
make config           # Regenerate all_models.json from YAML configs
```

## Code standards

- Python 3.11+
- Type checking: basedpyright in strict mode
- Formatting/linting: ruff
- Tests: pytest with pytest-asyncio; markers `@pytest.mark.unit` and `@pytest.mark.integration`
- Package management: uv with uv.lock

- During final or one-shot changes --- NOT DURING FEEDBACK ITERATIONS
    - Always run `make style` and `make typecheck`
    - Always run `make config` to regenerate `all_models.json`
    - Always update docs in `docs/` --- THESE SHOULD BE KEPT SYNCED, SIMPLE, AND SHORT
