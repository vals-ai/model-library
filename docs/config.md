# Model Configuration

YAML-based model registry with 3-level inheritance.

## Directory Structure

```
model_library/config/
├── all_models.json              # generated — do not edit
├── anthropic_models.yaml
├── openai_models.yaml
├── google_models.yaml
├── ...                          # one file per provider
└── deprecated/
    ├── anthropic_models.yaml
    ├── openai_models.yaml
    └── ...                      # deprecated models, same provider files
```

Active models live in `config/*.yaml`. Deprecated models live in `config/deprecated/*.yaml` and are **not loaded by default**.

Run `make config` to regenerate `all_models.json` from the active YAML files.

## YAML Inheritance

Three levels, each merged via `deep_update()`:

1. **Provider base-config** — top-level `base-config:` in the file, applies to all models in that provider
2. **Block base-config** — `base-config:` inside a named block, merged on top of provider base
3. **Model config** — individual model entry, merged on top of block config

```yaml
base-config:                          # level 1: provider
  company: Anthropic
  supports:
    images: true
  properties:
    reasoning_model: false

claude-sonnet-models:                 # named block
  base-config:                        # level 2: block
    supports:
      batch: true

  anthropic/claude-sonnet-4-6:        # level 3: model
    label: Claude Sonnet 4.6
    properties:
      context_window: 200_000
      max_tokens: 65_536
```

Result: `claude-sonnet-4-6` gets `company: Anthropic`, `images: true`, `batch: true`, `reasoning_model: false`, plus its own `context_window` and `max_tokens`.

## Deprecating Models

To deprecate a model, run:

```bash
make deprecate model=openai/gpt-4o-2024-05-13
```

This will:
1. Resolve the full model config from the registry (all inheritance applied)
2. Insert a self-contained entry at the top of `config/deprecated/<provider>_models.yaml`
3. Remove the entry from the active config file
4. Regenerate `all_models.json`

Alternative keys are handled automatically — they travel with the primary model entry.

## Loading Deprecated Models

Deprecated model configs live in `config/deprecated/` and are **not loaded by default**.

To include them:

```python
from model_library import model_library_settings

model_library_settings.set(MODEL_LIBRARY_INCLUDE_DEPRECATED=True)
```

Or via environment variable:

```bash
MODEL_LIBRARY_INCLUDE_DEPRECATED=True
```

## Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_LIBRARY_INCLUDE_DEPRECATED` | `False` | Load deprecated model configs from `config/deprecated/` |
| `MODEL_LIBRARY_CUSTOM_CONFIG` | — | Path or URL to additional YAML config to merge into registry |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `GOOGLE_API_KEY` | — | Google API key |
| `ARCEE_API_KEY` | — | Arcee AI API key |

Settings can be set via environment variables or programmatically:

```python
from model_library import model_library_settings

model_library_settings.set(OPENAI_API_KEY="sk-...")
```
