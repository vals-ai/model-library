# Model Configuration

The model registry is defined in YAML and uses three levels of inheritance.

## Directory Structure

```text
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
base-config: # level 1: provider
  company: Anthropic
  country: United States
  supports:
    images: true
  properties:
    reasoning_model: false

claude-sonnet-models: # named block
  base-config: # level 2: block
    supports:
      batch: true

  anthropic/claude-sonnet-4-6: # level 3: model
    label: Claude Sonnet 4.6
    properties:
      context_window: 200_000
      max_tokens: 65_536
```

Result: `claude-sonnet-4-6` gets `company: Anthropic`, `country: United States`, `images: true`, `batch: true`, `reasoning_model: false`, plus its own `context_window` and `max_tokens`.

## Provider Properties

Provider-specific `provider_properties` are validated by each provider:

- **Anthropic**: Set `fallback_models` to an ordered list of up to three server-side fallback models for the Messages API. Fallback-served responses set `QueryResult.metadata.extra["fallback"]` to `true`, and their assistant turns are replayed verbatim, including the `fallback` boundary block.
- **Anthropic**: Set `task_budget_tokens` to send `output_config.task_budget` with the task-budgets beta, an advisory token budget the model paces its agentic loop against. It is not enforced; `max_tokens` remains the hard ceiling.
- **Anthropic**: Set `returns_thinking_truncated_turns: true` to return a turn that ran out of tokens inside a thinking block as a `max_tokens` result with its reasoning, instead of raising, so the client can continue the turn. On these keys only, replaying such a turn appends a short text block, since Anthropic rejects an assistant message whose final block is thinking.
- **OpenAI-compatible completions**: Set `stream_completions: false` to use non-streaming chat completions. The default is `true`.
- **OpenAI Responses**: Set `code_mode: true` to add the hosted Code Mode tool. When enabled, function tools without explicit `allowed_callers` are sent with `allowed_callers: ["code_mode", "direct"]`.
- **Meta**: Set `use_responses: true` on selected models to route the OpenAI-compatible delegate through the Responses API instead of Chat Completions.
- **OpenAI-compatible providers**: Set `prompt_cache_key: id` to derive an OpenAI prompt-cache key from the resolved `run_id` and `question_id`, or `prompt_cache_key: hash` to derive it from the stable prompt prefix, for Responses and Chat Completions.
- **Alibaba Qwen reasoning models**: Set `preserve_thinking: true` to preserve reasoning context across turns.
- **Alibaba**: Set `mainland: true` to route to the mainland China DashScope endpoint, authenticated with `DASHSCOPE_CN_API_KEY` instead of `DASHSCOPE_API_KEY`. Model Studio keys are region-scoped and are rejected by the other region's endpoint.

## Common model fields

Set `country` to the model creator's country, not the hosting provider's country. Declare it wherever `company` is defined.

Set `provider_endpoint` when the registry key differs from the model ID sent to the provider.

See the [field reference](../model_library/config/README.md#fields) for the `rate_limit` YAML shape. This field stores accounting policy and optional static retry/admission capacity. Live provider observations are separate and never rewrite YAML. Omit `rate_limit` when neither policy nor static capacity is known; public exports strip static capacity and retain policy.

Use `supports.files` only for non-image document or file inputs supported by the provider. For image- or video-only APIs, leave it `false` and set `supports.images` or `supports.videos` instead.

The validator still runs file examples when `supports.files` is `false`. If one succeeds, validation fails because the registry entry is stale.

Set `supports.audio: true` only when the provider accepts audio input, such as Gemini through `FileWithBytes`.

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

Alternative keys travel with the primary model entry.

### Restore a model

Move the entry from `config/deprecated/<provider>_models.yaml` back to the matching active provider file, then run `make config`.

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

`/registry?include_deprecated=true` adds deprecated entries to the Gateway response without changing the default registry. Active entries win when the same key exists in both registries.

`/registry?include_alt_keys=false` removes same-provider aliases. Cross-provider aliases remain in the response.

## Gateway registry loading

### Discovery and helper behavior

When `MODEL_GATEWAY_URL` is set before the registry is initialized:

- The first `get_model_registry()` call fetches the full `/registry` snapshot
  with `MODEL_GATEWAY_API_KEY`. `get_registry_config()` and
  `get_registry_model()` construction use that snapshot. No-argument calls
  retain it for the process lifetime.
- The snapshot omits fields older clients reject as unknown (currently
  `country`, `rate_limit`, and `supports.transcription`).
- `refresh_model_registry()` provides opt-in lazy refresh without changing the
  `get_model_registry()` singleton contract. It reloads whichever source the
  current settings select: Gateway, local YAML, or custom config. A successful
  refresh atomically replaces the shared snapshot; `timedelta(0)` attempts a
  refresh on every call.
- Refresh failures are strict by default and never switch registry sources.
  `allow_stale_on_error=True` returns a snapshot previously loaded by a
  successful refresh and restarts a positive-TTL refresh cooldown.
- Metadata helpers such as `get_model_cost()`,
  `get_model_input_context_window()`, and `get_model_names()` read the current
  snapshot, so in Gateway mode they return Gateway metadata and follow
  `refresh_model_registry()`. Missing Gateway keys raise instead of falling back
  to direct provider discovery; direct-provider mode keeps its existing lookup
  behavior.

```python
from datetime import timedelta

from model_library.register_models import get_model_registry, refresh_model_registry

refresh_model_registry(
    refresh_ttl=timedelta(seconds=30),
    allow_stale_on_error=True,
)
registry = get_model_registry()
```

### Request execution

Gateway execution is server-authoritative:

1. `get_registry_model()` constructs a `GatewayLLM` through the normal client
   registry path, with metadata and capabilities available immediately.
2. Requests send only explicit override config, not registry-derived defaults.
3. On a provider-model cache miss, the server merges overrides with its loaded
   registry config.
4. The server caches the provider model by model and override config, plus token
   retry parameters when active.

Registry changes are not hot-reloaded into existing cached provider models;
restart or explicit cache invalidation is required.

### Single-model metadata

- `get_registry_model()` results expose `model.metadata`, capability attributes,
  and the input context window immediately.
- The values come from the client registry snapshot used at construction.
- `refresh_model_registry()` affects models constructed afterward; existing
  model instances retain their construction snapshot.
- Gateway batch capability metadata is preserved, but client-side gateway batch
  calls raise until gateway batch endpoints exist.

## Custom config overrides

`MODEL_LIBRARY_CUSTOM_CONFIG` can point at a local YAML file or an `http(s)` URL
with the same block format as the bundled provider configs. The file is loaded
when the non-gateway registry is first initialized and merged after bundled
configs, so matching model keys override defaults and new model keys are added.

```bash
MODEL_LIBRARY_CUSTOM_CONFIG=/path/to/models.yaml
MODEL_LIBRARY_CUSTOM_CONFIG=https://example.com/models.yaml
```

Programmatic helpers are also exported:

```python
from model_library import load_custom_model_configs, load_latest_vals_model_configs

load_custom_model_configs("/path/to/models.yaml")
load_latest_vals_model_configs(branch="main")
```

`load_latest_vals_model_configs()` fetches every bundled provider YAML from the
public `vals-ai/model-library` repo for the requested branch and merges them into
the current registry. It discovers bundled YAML filenames dynamically so new
provider files are included automatically.

## Settings

| Variable                           | Default | Description                                                                                   |
| ---------------------------------- | ------- | --------------------------------------------------------------------------------------------- |
| `MODEL_GATEWAY_URL`                      | —       | Gateway server URL. When set, registry snapshots load from it and requests route through it    |
| `MODEL_GATEWAY_API_KEY`                  | —       | Bearer token used for gateway registry and request calls                                      |
| `MODEL_LIBRARY_INCLUDE_DEPRECATED` | `False` | Load deprecated model configs from `config/deprecated/`                                       |
| `MODEL_LIBRARY_CUSTOM_CONFIG`      | —       | Path or URL to additional YAML config to merge into non-gateway registry                      |
| `OPENAI_API_KEY`                   | —       | OpenAI API key                                                                                |
| `ANTHROPIC_API_KEY`                | —       | Anthropic pool 1; runtime discovers contiguous `_2` through `_N` keys (deployed through `_2`) |
| `META_API_KEY`                     | —       | Meta pool 1; runtime discovers contiguous `_2` through `_N` keys (deployed through `_4`)      |
| `GOOGLE_API_KEY`                   | —       | Google API key                                                                                |
| `ARCEE_API_KEY`                    | —       | Arcee AI API key                                                                              |
| `NVIDIA_API_KEY`                   | —       | NVIDIA API key                                                                                |
| `POOLSIDE_API_KEY`                 | —       | Poolside API key                                                                              |
| `BASETEN_API_KEY`                  | —       | Baseten API key                                                                               |
| `BASETEN_API_BASE_URL`             | —       | Full OpenAI-compatible Baseten `/v1` base URL                                                 |

Settings can be set via environment variables or programmatically:

```python
from model_library import model_library_settings

model_library_settings.set(OPENAI_API_KEY="sk-...")
```
