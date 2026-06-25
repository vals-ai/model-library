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
base-config: # level 1: provider
  company: Anthropic
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

Result: `claude-sonnet-4-6` gets `company: Anthropic`, `images: true`, `batch: true`, `reasoning_model: false`, plus its own `context_window` and `max_tokens`.

## Provider Properties

Provider-specific `provider_properties` are validated by each provider:

- **Anthropic**: Set `fallback_models` to an ordered list of up to three server-side fallback models for the Messages API. Fallback-served responses set `QueryResult.metadata.extra["fallback"]` to `true`.
- **OpenAI-compatible completions**: Set `stream_completions: false` to use non-streaming chat completions. The default is `true`.
- **OpenAI Responses**: Set `code_mode: true` to add the hosted Code Mode tool. When enabled, function tools without explicit `allowed_callers` are sent with `allowed_callers: ["code_mode", "direct"]`.
- **Meta**: Set `use_responses: true` on selected models to route the OpenAI-compatible delegate through the Responses API instead of Chat Completions.
- **Alibaba Qwen reasoning models**: Set `preserve_thinking: true` to preserve reasoning context across turns.

Models may also set `provider_endpoint` when the registry key should differ from the upstream provider model ID.

Configured provider default rate limits are not bundled in the public package; use live provider rate-limit headers when available.

Use `supports.files` only for provider-supported non-image document/file inputs. Providers with image/video-only multimodal APIs should keep `supports.files: false` and set `supports.images`/`supports.videos` instead, so callers can choose image or video fallbacks rather than provider file-upload paths.

To restore a deprecated model to the active registry, move its entry from
`config/deprecated/<provider>_models.yaml` back to the matching active provider
YAML file and run `make config`.

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

## Gateway registry loading

When `MODEL_GATEWAY_URL` is set, `get_model_registry()` fetches the full registry
snapshot from the gateway server's `/registry` endpoint using `MODEL_GATEWAY_API_KEY`.
This remote snapshot is for explicit bulk discovery; failures are strict and do
not silently fall back to local files.

Legacy local registry helpers such as `get_registry_config()`, `get_model_cost()`,
`get_model_input_context_window()`, and `get_model_names()` raise in gateway mode
instead of implicitly fetching or serving a cached full `/registry` snapshot.

Request execution is server-authoritative in gateway mode: `get_registry_model()`
constructs an unsynced `GatewayLLM` from the model string without fetching or
requiring client-side registry knowledge, and gateway requests send only explicit
override config. On provider-model cache miss, the gateway server merges those
overrides with its loaded registry config and caches the provider model by model
and override config, plus token retry params when token retry is active.
Registry/config changes are not hot-reloaded into existing cached provider
models; restart or cache invalidation is required to pick them up. Gateway
metadata attributes such as `supports_tools` and `supports_temperature` start
with local defaults until `await model.ensure_metadata_loaded()` loads current
gateway-side metadata onto the model instance. Repeated calls are no-ops after
the first successful load. After loading, `model.metadata` is the full server
`ModelConfig` registry entry for that model; local registry-backed models expose
the same property from the local registry immediately. The underlying
`/models/resolve` response returns both the server-built `effective_config` and
the full server `registry_config` for that one model, so callers migrating
registry-entry reads can use `model.metadata` without fetching the full
`/registry` snapshot. Gateway batch capability metadata is preserved, but
client-side gateway batch calls raise until gateway batch endpoints exist.

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
| `MODEL_GATEWAY_URL`                      | —       | Gateway server URL. When set, execution and single-model metadata resolve through the gateway |
| `MODEL_GATEWAY_API_KEY`                  | —       | Bearer token used for gateway registry and request calls                                      |
| `MODEL_LIBRARY_INCLUDE_DEPRECATED` | `False` | Load deprecated model configs from `config/deprecated/`                                       |
| `MODEL_LIBRARY_CUSTOM_CONFIG`      | —       | Path or URL to additional YAML config to merge into non-gateway registry                      |
| `OPENAI_API_KEY`                   | —       | OpenAI API key                                                                                |
| `ANTHROPIC_API_KEY`                | —       | Anthropic API key                                                                             |
| `GOOGLE_API_KEY`                   | —       | Google API key                                                                                |
| `ARCEE_API_KEY`                    | —       | Arcee AI API key                                                                              |
| `NVIDIA_API_KEY`                   | —       | NVIDIA API key                                                                                |
| `POOLSIDE_API_KEY`                 | —       | Poolside API key                                                                              |

Settings can be set via environment variables or programmatically:

```python
from model_library import model_library_settings

model_library_settings.set(OPENAI_API_KEY="sk-...")
```
