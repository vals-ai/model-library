# Model Configs

Each YAML file defines models for one provider. Configs use a three-level inheritance system: provider base → model-block base → individual model.

## Structure

```yaml
base-config:                        # inherited by all models in this file
  company: Anthropic
  open_source: false
  supports:
    images: true
    tools: true

claude-4-models:                    # model block
  base-config:                      # inherited by models in this block
    default_parameters:
      temperature: 1

  anthropic/claude-opus-4-6:        # individual model (overrides both bases)
    label: Claude Opus 4.6
    release_date: 2025-06-01
    properties:
      context_window: 200_000
      max_tokens: 32_000
      training_cutoff: "2025-03"
      reasoning_model: false
    supports:
      batch: true
      temperature: true
    costs_per_million_token:
      input: 15.0
      output: 75.0
    metadata:
      available_for_everyone: true
      available_as_evaluator: true
```

## Fields

| Field | Description |
|-------|-------------|
| `properties` | `context_window`, `max_tokens`, `training_cutoff`, `reasoning_model` |
| `supports` | Boolean flags: `images`, `videos`, `files`, `batch`, `temperature`, `tools`, `output_schema` |
| `costs_per_million_token` | `input`, `output`, optional `cache`, `batch`, `context` pricing. Set to `null` for models without known pricing |
| `metadata` | `deprecated`, `available_for_everyone`, `available_as_evaluator`, `ignored_for_cost`, `internal_only` |
| `default_parameters` | `temperature`, `top_p`, `top_k`, `reasoning_effort` |
| `provider_properties` | Provider-specific flags (e.g. `supports_auto_thinking`) |
| `provider_endpoint` | Override the model name sent to the provider API |
| `alternative_keys` | Alternative model identifiers/aliases |

## Configuration Inheritance

Configurations support hierarchical inheritance through `base-config` blocks:

1. Provider-level `base-config` applies to every model in the YAML file.
2. Model-block `base-config` applies to every model in that block.
3. Individual model fields override both base levels.

Nested dictionaries are merged recursively instead of replaced wholesale.

## Alternative Keys

Map additional identifiers to the same model, optionally with config overrides:

```yaml
alternative_keys:
  - anthropic/claude-opus-4-6-latest                 # simple alias
  - anthropic/claude-opus-4-6-thinking:              # alias with overrides
      properties:
        reasoning_model: true
```

## Custom Configs

Use `load_custom_model_configs` to load additional YAML files at runtime from a local path or URL. Models defined there override bundled defaults.

Use `load_latest_vals_model_configs` to pull bundled config YAML files from a branch of the public model-library repository and merge them into the runtime registry.

Set `MODEL_LIBRARY_CUSTOM_CONFIG` to load a local path or URL automatically during registry initialization.

## After Editing

Run `make config` to regenerate `all_models.json`. The generated file is used by the model registry at runtime; do not edit it manually.

## Schema Validation

The configuration is validated using Pydantic models defined in `register_models.py`:

- `Properties` - Model properties
- `Supports` - Feature support flags
- `Metadata` - Platform metadata
- `DefaultParameters` - Default parameter values
- `CostProperties` - Pricing information
- `ProviderProperties` - Provider-specific config, generated dynamically from provider config classes

## Migration Notes

### Previous Structure (Deprecated)

The old configuration used `class_properties` which mixed support flags and metadata:

```yaml
# OLD - Do not use
class_properties:
  supports_images: true
  supports_batch_requests: true
  deprecated: false
  available_for_everyone: true
properties:
  max_token_output: 32_000
```
