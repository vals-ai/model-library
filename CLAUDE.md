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
make deprecate model=provider/model-key  # Move a model to deprecated
```

## PR checks

- `Run Models` intentionally fails on the first PR run with `Check must be run manually on PRs`; this is an approval gate, not a code failure.
- Manually rerun `Run Models` when live provider smoke coverage is needed, then investigate and report any provider/API failures from that rerun.

## Model config structure

- Active configs: `model_library/config/*.yaml` (one per provider)
- Deprecated configs: `model_library/config/deprecated/*.yaml` (not loaded by default)
- Generated: `model_library/config/all_models.json` (do not edit)
- YAML uses 3-level inheritance: provider base-config → block base-config → model entry
- `MODEL_LIBRARY_INCLUDE_DEPRECATED=True` to load deprecated models
- See `docs/config.md` for details

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
