# Model Configs

Each YAML file defines one provider's models; speech-to-text-only providers live in `voice/`. Values merge from provider base → model-block base → model entry.

## Structure

```yaml
base-config:                        # inherited by all models in this file
  company: Anthropic
  country: United States
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
| `country` | Country of origin of `company`, declared wherever `company` is |
| `properties` | Required `context_window`, `max_tokens`, `reasoning_model`, optional `training_cutoff`; transcription-only models may omit the whole block |
| `supports` | Boolean flags: `images`, `audio`, `videos`, `files`, `batch`, `temperature`, `tools`, `output_schema`, `transcription` |
| `transcription_streaming` | Whether this entry uses a streaming transcription transport, including complete-file uploads with streamed transcript responses |
| `costs_per_million_token` | `input`, `output`, optional `cache`, `batch`, `context` pricing. Set to `null` for models without known pricing |
| `transcription_cost` | Duration STT pricing: `usd_per_minute`, `billing_basis`, and optional `minimum_billable_seconds` and `increment_seconds` |
| `metadata` | `deprecated`, `available_for_everyone`, `available_as_evaluator`, `ignored_for_cost`, `internal_only` |
| `default_parameters` | `temperature`, `top_p`, `top_k`, `reasoning_effort` |
| `rate_limit` | Optional rate-limit policy and static retry/admission capacity. `supports_live_monitoring` defaults to `false`; `cache_read_counts_toward_limit` defaults to `true`. `requests` is a list of `{limit, mode}` entries. `tokens` uses either `total` or `input` plus `output`, with optional `uncached_input`; each capacity is `{limit}`. Request mode defaults to `sliding_window`; token mode defaults to `token_bucket`. Policy-only blocks are valid when a policy field is explicit; empty blocks and `null` are invalid. Omit unknown capacities. |
| `provider_properties` | Provider-specific flags (e.g. `supports_auto_thinking`) |
| `provider_endpoint` | Override the model name sent to the provider API |
| `alternative_keys` | Alternative model identifiers/aliases |

## Configuration Inheritance

Values merge in this order:

1. Provider-level `base-config` applies to every model in the file.
2. Model-block `base-config` applies to every model in the block.
3. Model fields override both base levels.

Nested dictionaries merge recursively.

## Alternative Keys

Map additional identifiers to the same model, optionally with config overrides:

```yaml
alternative_keys:
  - anthropic/claude-opus-4-6-latest                 # simple alias
  - anthropic/claude-opus-4-6-thinking:              # alias with overrides
      properties:
        reasoning_model: true
```

## Contributor workflow

See [Model Configuration](../../docs/config.md) for deprecating or restoring models, loading custom configs, and Gateway registry behavior.

After editing active YAML, run `make config`. This regenerates the bundled `all_models.json` snapshot from local YAML; do not edit the snapshot directly.

Pydantic models in `model_library/register_models.py` validate the schema. `class_properties` is deprecated; use `properties`, `supports`, and `metadata`.
