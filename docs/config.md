# Model Configuration

The model registry is defined in YAML and uses three levels of inheritance.

## Directory Structure

```text
model_library/config/
├── all_models.json              # generated — do not edit
├── anthropic_models.yaml
├── openai_models.yaml
├── google_models.yaml
├── ...                          # one file per general provider
├── voice/                       # speech-to-text-only provider configs
│   ├── assemblyai_models.yaml
│   └── ...
└── deprecated/
    ├── anthropic_models.yaml
    ├── openai_models.yaml
    └── ...                      # deprecated models, same provider files
```

Active models live in `config/*.yaml` and `config/voice/*.yaml`. Deprecated models live in `config/deprecated/*.yaml` and are **not loaded by default**.

Run `make config` to regenerate `all_models.json` (chat models only) from the active YAML files.

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
      tools: true

  anthropic/claude-sonnet-4-6: # level 3: model
    label: Claude Sonnet 4.6
    properties:
      context_window: 200_000
      max_tokens: 65_536
```

Result: `claude-sonnet-4-6` gets `company: Anthropic`, `country: United States`, `images: true`, `tools: true`, `reasoning_model: false`, plus its own `context_window` and `max_tokens`.

## Sampling

Chat adapters select `temperature`, `top_p`, and `top_k` from `LLMConfig`. Configured `None` values are omitted; zero remains a value. Configured non-`None` values disabled by `supports_temperature` or the adapter's field/mode rules are omitted with a warning, including inherited registry defaults.

For configured sampling, OpenAI Chat Completions and Google encode all three fields. OpenAI Responses (including xAI through its delegate), Bedrock, Mistral, AI21, and Vals encode `temperature` and `top_p`. Native Anthropic encodes `temperature` only without reasoning, and `top_p` only for MiniMax; it omits configured `top_k`. Delegated calls use their selected API adapter's rules.

Native query kwargs are custom overrides applied through each adapter's existing merge path, without these sampling capability checks. Use endpoint-native shapes, such as Bedrock `topP` or OpenAI Chat Completions `extra_body={"top_k": ...}`. Explicit `None` follows SDK behavior rather than a shared omission rule. Existing SDK translations and raw `extra_body` precedence remain unchanged; SDKs or providers may ignore or reject unsupported arguments.

Anthropic token counting reuses the request builder. Unsupported raw count kwargs can trigger its existing approximate fallback, including when estimating tokens for token-rate limits.

Gateway callers set sampling in `LLMConfig`, not query kwargs. Its request format omits `None` configuration values, so they do not clear inherited server defaults.

## Prompt Caching

Anthropic (and its delegates), Bedrock, and OpenRouter routers add cache breakpoints to every request by default, and Anthropic bills each cache write. For inputs that won't be reused, such as one-off grading, set `LLMConfig(prompt_caching=False)` to send no breakpoints, so the request writes no cache. OpenAI GPT-5.6+ caches implicitly and bills writes, so those models instead get `prompt_cache_options.mode: "explicit"` with no breakpoints.

## Provider Properties

OpenAI remains one provider and one `OpenAIModel`. Set `LLMConfig(provider_config=OpenAIConfig(use_completions=True))` in `get_registry_model(..., override_config=...)` to choose Chat Completions for an OpenAI model; the default remains Responses. An explicit constructor `use_completions` selector takes precedence, preserving delegate API choices. The separate API implementations remain internal; public model names do not change.

Provider-specific `provider_properties` are validated by each provider:

- **Anthropic**: Set `fallback_models` to an ordered list of up to three server-side fallback models for the Messages API. Fallback-served responses set `QueryResult.metadata.fallback` (`FallbackInfo`: requested/served model plus per-hop model, usage, and decline trigger/category), are billed at the fallback model's price, and their assistant turns preserve the `fallback` boundary block. Replaying any Anthropic assistant turn replaces empty text blocks in place with a short placeholder rather than dropping them, so thinking blocks keep their positions and signatures.
- **Anthropic**: Set `thinking_off_type: between_tools` on auto-thinking models that reject `{"type": "disabled"}`; a request with `reasoning: false` then sends `between_tools`, and since that setting only accepts low/medium/high effort, `xhigh`/`max` are sent as `high`. `fallback_models` entries on such a request carry `thinking: {"type": "disabled"}` since the API validates each fallback against its own model.
- **Anthropic**: Set `task_budget_tokens` to send `output_config.task_budget` with the task-budgets beta, an advisory token budget the model paces its agentic loop against. It is not enforced; `max_tokens` remains the hard ceiling.
- **Anthropic**: Set `returns_thinking_truncated_turns: true` to return a turn that ran out of tokens inside a thinking block as a `max_tokens` result with its reasoning, instead of raising, so the client can continue the turn. On these keys only, replaying such a turn appends a short text block, since Anthropic rejects an assistant message whose final block is thinking.
- **Google**: Set `uses_thinking_budget: true` on models that take `thinking_budget`; otherwise reasoning requests send `thinking_level`. Configured sampling follows `supports_temperature` independently of thinking mode. `automatic_continuation` defaults to false; set it to true per model to opt into SDK multi-hop generation. Native query kwargs override provider config.
- **OpenAI-compatible completions**: Set `stream_completions: false` to use non-streaming chat completions. The default is `true`.
- **OpenAI Responses**: Set `code_mode: true` to add the hosted Code Mode tool. When enabled, function tools without explicit `allowed_callers` are sent with `allowed_callers: ["code_mode", "direct"]`.
- **OpenAI Responses**: Set `access_programs` (for example `{cyber: daybreak_blue}`) to send OpenAI access programs. Set it in the registry entry, because a request `provider_config` replaces the entry's whole one. The gateway rejects it in a request unless the request has its own `custom_api_key`. The `-daybreak` keys set it; each pins one model. OpenAI returns a 400 on models the program does not cover, such as `gpt-6.1-sol`.
- **OpenAI / Google**: Set `service_tier: flex` to request Flex processing (sent as `service_tier`); responses are billed at the model's `batch` discount.
- **xAI**: Grok requests use the OpenAI-compatible Responses API (encrypted reasoning is requested via `include: ["reasoning.encrypted_content"]`). Every request carries a `prompt_cache_key` (`hash` by default, `id` to key on the resolved run/question IDs), which xAI uses for server routing ([maximizing cache hits](https://docs.x.ai/developers/advanced-api-usage/prompt-caching/maximizing-cache-hits)). The hash stays stable across turns with a `RawResponse` history boundary; replaying history only as `RawInput` dictionaries hashes the full input and may change the key each turn.
- **Meta**: Set `use_responses: true` on selected models to route the OpenAI-compatible delegate through the Responses API instead of Chat Completions.
- **OpenAI-compatible providers**: Set `prompt_cache_key: id` to derive an OpenAI prompt-cache key from the resolved `run_id` and `question_id`, or `prompt_cache_key: hash` to derive it from the prompt prefix (up to the first `RawResponse`, or the full input if none exists), for Responses and Chat Completions.
- **Alibaba Qwen reasoning models**: Set `preserve_thinking: true` to preserve reasoning context across turns.
- **OpenRouter**: Set `openrouter_provider_only` to the host slugs allowed to serve a non-router key (sent as `provider.only`). Without it OpenRouter load-balances across every host, including quantized ones, so a unit test requires it on every non-router `openrouter/*` key.
- **Alibaba**: Set `mainland: true` to route to the mainland China DashScope endpoint, authenticated with `DASHSCOPE_CN_API_KEY` instead of `DASHSCOPE_API_KEY`. Model Studio keys are region-scoped and are rejected by the other region's endpoint.

## Common model fields

Set `country` to the model creator's country, not the hosting provider's country. Declare it wherever `company` is defined.

Set `provider_endpoint` when the registry key differs from the model ID sent to the provider.

Company icons live in `model_library/config/icons/`. Name the light asset exactly `<company>.svg|png` and an optional dark asset `<company>-dark.svg|png`. Add a separate exact-name asset when two company labels use the same artwork; registry configs and `/registry` payloads contain only `company`, not icon filenames. Call `get_company_icon(company)` to resolve the bundled filenames; it returns `None` when no exact light asset exists and leaves `dark=None` when there is no dark asset. PNG and SVG files are bundled in the Python package.

See the [field reference](../model_library/config/README.md#fields) for the `rate_limit` YAML shape. This field stores accounting policy and optional static retry/admission capacity. Live provider observations are separate and never rewrite YAML. Omit `rate_limit` when neither policy nor static capacity is known; public exports strip static capacity and retain policy.

Use `supports.files` only for non-image document or file inputs supported by the provider. For image- or video-only APIs, leave it `false` and set `supports.images` or `supports.videos` instead.

Image inputs (`type="image"`) accept short MIME names (`png`) or full MIME types (`image/png`). All `FileBase` subclasses store and serialize the full form, including inputs decoded from gateway requests. Non-image MIME types and file contents stay unchanged. Providers use the stored MIME type; Bedrock reads `FileBase.image_format` for its required short format. The separate `upload_file()` API keeps its existing MIME requirements.

The validator still runs file examples when `supports.files` is `false`. If one succeeds, validation fails because the registry entry is stale.

OpenAI-compatible completions delegates accept base64 files only and reject `FileWithUrl`. OpenRouter is the exception: its PDF parser accepts a direct URL in `file_data`, so OpenRouter models that answer file probes declare `supports.files: true`.

Set `supports.audio: true` only when the provider accepts audio input, such as Gemini through `FileWithBytes`.

Expose reasoning-effort variants as `alternative_keys` entries that override `default_parameters.reasoning_effort` (for example a `-xhigh` alias). `alibaba/qwen3.8-max-low` and `-medium` pin Qwen 3.8 Max to `low` and `medium`; QwenCloud takes `reasoning_effort` alongside `enable_thinking` and rejects it together with `thinking_budget`.

xAI entries with `properties.reasoning_model: true` request `include: ["reasoning.encrypted_content"]`. The returned reasoning items live on the raw response kept in history, survive `serialize_input`/`deserialize_input`, and are replayed unchanged on later turns so reasoning continues across multi-turn and tool-call loops and the cached prompt prefix is preserved; xAI treats omitted reasoning content as the top cause of cache misses. Responses raw-response history remains supported; previously serialized native xAI protobuf history is incompatible after removal of `xai-sdk`.

## Transcription models

Transcription entries must set `supports.transcription`,
`transcription_streaming`, and `transcription_language`. They are the only
entries allowed to omit the whole `properties` block; runtime construction keeps
the normal `LLMConfig` defaults. When present, the resolved block must include
`context_window`, `max_tokens`, and `reasoning_model`.
Set `properties: null` to clear inherited chat properties.

Registry entries use `ModelConfig` for chat and `TranscriptionModelConfig` for
transcription-only models. `get_model_registry()` serves chat entries and
`get_transcription_registry()` serves transcription-only entries;
`get_registry_config()`/`get_registry_model()` resolve chat models and
`get_transcription_registry_config()`/`get_transcription_model()` resolve
transcription models.
Transcription-only entries require absent or null properties and
`supports.transcription: true`; complete properties select the chat schema.
`ModelConfig.properties` remains required.

Duration-billed models define `transcription_cost` with a `usd_per_minute` rate
and an `audio` or `session` `billing_basis`; optional `minimum_billable_seconds`
applies before optional `increment_seconds` rounding. Token-billed models use
`costs_per_million_token` instead. Transcription entries may omit token pricing.
`cost_usd` stays null when the price or the
billable measurement is missing.

`transcription_language` declares the spelling the provider expects, and is
usually set once in the provider's `base-config`:

```yaml
transcription_language:
  format: iso-639-1   # bare code, such as "en"; bcp-47 wants a locale, "en-US"
  default: en         # used when the caller names no language
```

`transcribe_audio` applies it before the provider sees the request, so `en-US`
reaches an ISO-639-1 provider as `en`, and a bare `en` reaches a BCP-47 provider
as that provider's default locale. Any other value passes through unchanged,
since a region cannot be invented for an arbitrary language.

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

- The first `get_model_registry()` or `get_transcription_registry()` call
  fetches the full `/registry` snapshot with `MODEL_GATEWAY_API_KEY`,
  `include_excluded_fields=true`, and `include_transcription=true`.
  Both registries are populated from that one snapshot.
  `get_registry_config()` and `get_registry_model()` construction use that
  snapshot. No-argument calls retain it for the process lifetime.
- The default `/registry` response preserves the legacy schema. It omits
  `country`, `rate_limit`, `metadata.router`, `supports.transcription`, and the
  transcription metadata fields. It also
  omits transcription-only entries because legacy clients require chat sizing
  properties. Clients requesting both `include_excluded_fields=true` and
  `include_transcription=true` receive the complete registry.
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
4. The server caches the provider model by model and override configuration,
   plus token retry parameters when active.
5. When live runtime snapshots are enabled, the cache key also contains the
   captured model and credential revisions.

Without live runtime snapshots, registry changes do not replace cached provider
models. A restart or explicit cache invalidation is required.

### Single-model metadata

- `get_registry_model()` results expose `model.metadata`, capability attributes,
  and the input context window immediately.
- The values come from the client registry snapshot used at construction.
- `refresh_model_registry()` affects models constructed afterward; existing
  model instances retain their construction snapshot.

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
| `RAMP_API_KEY`                     | —       | Ramp Router API key                                                                           |
| `REFLECTION_API_KEY`               | —       | Reflection API key                                                                            |
| `SAKANA_API_KEY`                   | —       | Sakana Fugu API key                                                                           |
| `DIGITAL_OCEAN_API_KEY`            | —       | DigitalOcean Inference model access key                                                       |
| `BASETEN_API_KEY`                  | —       | Baseten API key                                                                               |
| `BASETEN_API_BASE_URL`             | —       | Full OpenAI-compatible Baseten `/v1` base URL                                                 |

Settings can be set via environment variables or programmatically:

```python
from model_library import model_library_settings

model_library_settings.set(OPENAI_API_KEY="sk-...")
```
