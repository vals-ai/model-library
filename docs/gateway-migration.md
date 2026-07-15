# Migrating agents and providers to Model Gateway

This guide is for agents/providers that already use `model-library` model objects and want to route requests through a Model Gateway instead of calling providers directly.

## Short version

1. Use a `model-library` build that includes the gateway client (`GatewayLLM`).
2. Set these client environment variables in the process that runs the agent:

   ```env
   MODEL_GATEWAY_URL="https://<gateway-host>"
   MODEL_GATEWAY_API_KEY="<gateway-client-key>"
   ```

3. Construct models with `get_registry_model(model_key, override_config=...)`.
4. Keep calling `await model.query(...)` normally.
5. Move provider/request overrides into `LLMConfig`; do not pass provider-specific kwargs to `query()`.

When `MODEL_GATEWAY_URL` is set, `get_registry_model()` returns an unsynced `GatewayLLM` from the model string. It does not fetch or validate client-side registry state during construction. The same agent code can run direct-provider locally when the env var is unset, and through the gateway when it is set.

## Minimal model code

```python
from model_library.base import LLMConfig, TextInput
from model_library.registry_utils import get_registry_model

model = get_registry_model(
    "anthropic/claude-sonnet-4-5-20250929",
    override_config=LLMConfig(temperature=0.2, max_tokens=4096),
)

result = await model.query(
    [TextInput(text="Solve this task.")],
    run_id=run_id,
    question_id=task_id,
)
```

Do not instantiate provider classes or use `get_raw_model()` for gateway-routed agents. Those paths bypass registry defaults and do not switch to `GatewayLLM`.

## Overrides

Gateway requests send the model key plus the explicit `LLMConfig` override. The gateway server merges that override with its server-side registry entry before constructing the provider model.

| Need | Put it in | Constraint |
| --- | --- | --- |
| Agent/provider runtime parameter | Top-level `LLMConfig` fields | Pass the config to `get_registry_model(..., override_config=config)`. |
| Provider-specific parameter | `LLMConfig.provider_config` | Use the typed provider config. |
| Bring your own key or endpoint | `LLMConfig.custom_api_key` and `LLMConfig.custom_endpoint` | `custom_endpoint` requires `custom_api_key`; the gateway must not send server-held provider keys to caller-selected endpoints. |
| Query attribution | `query(...)` fields | Keep `identity`, `run_id`, `question_id`, and `query_id` out of `LLMConfig`. |

Agent/provider runtime parameters:

```python
from model_library.base import LLMConfig
from model_library.registry_utils import get_registry_model

config = LLMConfig(
    temperature=temperature,
    top_p=top_p,
    reasoning_effort=reasoning_effort,
    max_tokens=max_tokens,
)
model = get_registry_model(model_key, override_config=config)
```

Provider-specific overrides belong in `provider_config`:

```python
from model_library.base import LLMConfig
from model_library.providers.openai import OpenAIConfig
from model_library.registry_utils import get_registry_model

model = get_registry_model(
    "openai/gpt-5.4-nano-2026-03-17",
    override_config=LLMConfig(provider_config=OpenAIConfig(verbosity="low")),
)
```

Bring-your-own-key/custom-endpoint requests also go through `LLMConfig`:

```python
from pydantic import SecretStr

model = get_registry_model(
    "openai/gpt-4o",
    override_config=LLMConfig(
        custom_endpoint="https://provider.example/v1",
        custom_api_key=SecretStr(provider_key),
    ),
)
```

`custom_endpoint` requires `custom_api_key`. The gateway must not send server-held provider keys to caller-selected endpoints.

## Query kwargs

Gateway `query()` supports normal model-library inputs, history, tools, and output schemas:

```python
result = await model.query(
    input=new_items,
    history=history,
    tools=tools,
    output_schema=MySchema,
    run_id=run_id,
    question_id=task_id,
    in_agent=True,
)
```

Do not pass provider parameters such as `temperature`, `top_p`, `max_tokens`, `stop`, or provider SDK kwargs to `query()`. Put them in `LLMConfig` at model construction time. Gateway rejects unsupported extra query parameters before making the HTTP request.

`logger` is ignored client-side for compatibility. `system_prompt` is still normalized into a leading system input, but new code should prefer explicit `SystemInput` items.

## Metadata and registry access

In gateway mode, request execution is server-authoritative. `get_registry_model()` does not load capabilities, costs, or registry metadata during construction.

| Need | Use | Do not use |
| --- | --- | --- |
| One model's capabilities, costs, registry metadata, or context window | `await model.ensure_metadata_loaded()`, then `model.metadata` or `model.input_context_window` | Local-registry helpers |
| Full registry for intentional bulk discovery | `get_model_registry()` and the gateway `/registry` snapshot | Implicit full-registry loading during model construction |
| Request execution | The unsynced `GatewayLLM`; the server resolves authoritative config | Client-side registry state |

Preferred single-model metadata flow:

```python
model = get_registry_model(model_key)
await model.ensure_metadata_loaded()
metadata = model.metadata
context_window = model.input_context_window
```

Avoid these local-registry helper calls in gateway mode:

- `get_registry_config()`
- `get_model_cost()`
- `get_model_input_context_window()`
- `get_model_names()`

They are local-registry helpers and raise when `MODEL_GATEWAY_URL` is set. Use `await model.ensure_metadata_loaded()` for one model, or `get_model_registry()` only when you intentionally need the full gateway `/registry` snapshot for bulk discovery.

## Attribution and IDs

### Identity

| Source or use | Contract |
| --- | --- |
| `IDENTITY` setting/env var | JSON-encoded object string with caller-defined keys, for example `{"user_id":"user_123","benchmark_name":"swebench"}`. |
| Explicit `query(identity=...)` | Takes a Python mapping/object. A non-`None` value overrides `IDENTITY`; `None` falls back to `IDENTITY`. Invalid explicit values raise validation errors before the request is sent. |
| Storage and telemetry | `GatewayLLM` attaches identity to gateway query telemetry and usage ledger rows. Values are preserved as provided and emitted/stored as raw caller metadata for debugging and ledger attribution. |
| Security | `IDENTITY` persists in usage ledger and operational metadata. Do not include secrets, provider API keys, bearer tokens, or other credentials. PII is allowed when needed for attribution/analytics. |
| `benchmark_name` analytics | Recognized for rows written after the field/index exists when it trims to a non-empty value no larger than 512 UTF-8 bytes. Has a raw-ledger lookup path. |
| `agent_name` analytics | Recognized for rows written after the field/index exists when it trims to a non-empty value no larger than 512 UTF-8 bytes. Has a raw-ledger lookup path. |
| `email` analytics | Recognized when it is a valid email-shaped string no larger than 512 UTF-8 bytes. Stored as canonical lowercased `identity_email` for exact Redshift filtering. |
| Other fields | Stored on base usage-event rows, without dedicated ledger query indexes or Redshift dimensions unless a future query path is added. |

Invalid `IDENTITY` settings are omitted by the client when they are:

- non-object;
- non-finite;
- larger than 4 KiB;
- deeper than 8 levels.

### Request IDs

| Field | Resolution and validation |
| --- | --- |
| `run_id` | When explicit `query(run_id=...)` is absent or `None`, use `RUN_ID`, then a generated ID. Explicit blank or whitespace-only values are rejected. |
| `question_id` | When explicit `query(question_id=...)` is absent or `None`, use `QUESTION_ID`, then `TASK_ID` only when `QUESTION_ID` is unset or blank, then a generated ID. Explicit blank or whitespace-only values are rejected. |
| `query_id` | Generated per query unless passed explicitly. There is no `QUERY_ID` env/settings fallback. Direct blank values are treated as absent. |

- `RUN_ID`, `QUESTION_ID`, and `TASK_ID` are read through `model_library_settings` and apply to all `LLM.query()` calls.
- `RUN_ID` and `QUESTION_ID` stay separate top-level gateway fields.
- `TASK_ID` is an alias fallback for `QUESTION_ID`.
- Blank or whitespace-only `RUN_ID`, `QUESTION_ID`, and `TASK_ID` env values are ignored.

## Runtime model-config overrides

1. Expose typed runtime kwargs for model config.
2. Pass those kwargs into your agent command.
3. Convert them to `LLMConfig` before calling `get_registry_model()`.
4. Call `get_registry_model(model_key, override_config=config)`.

When the run has no model-config overrides, use plain `get_registry_model(model_key)`.

Keep attribution fields out of `LLMConfig`:

- `identity`
- `run_id`
- `question_id`
- `query_id`

```yaml
run_cmd: >-
  my_agent --task {problem_statement_path}
  --model {model}
  --temperature {temperature}
kwargs:
  temperature:
    type: float
    required: false
    default: 0.2
```

```python
config = LLMConfig(temperature=args.temperature)
model = get_registry_model(args.model, override_config=config)
```

## All migrations checklist

- [ ] Agent process receives `MODEL_GATEWAY_URL` and `MODEL_GATEWAY_API_KEY`.
- [ ] Model construction uses `get_registry_model(model_key)` when there are no model-config overrides.
- [ ] Model-config overrides use `get_registry_model(model_key, override_config=config)` when needed.
- [ ] No provider class or `get_raw_model()` construction remains in the gateway path.
- [ ] Runtime model-config overrides are represented as `LLMConfig`, including `provider_config` when needed.
- [ ] BYOK/custom-endpoint migrations explicitly account for `custom_api_key` and `custom_endpoint` as caller-supplied provider credentials.
- [ ] `query()` calls do not pass provider-specific kwargs.
- [ ] Code that reads metadata uses `await model.ensure_metadata_loaded()` or intentional `get_model_registry()` bulk discovery.
- [ ] Code does not rely on gateway-unsupported `get_rate_limit()`, batch, or custom retrier paths.

## Unsupported or special paths

| Path | Migration decision |
| --- | --- |
| Client-side custom retriers | Unsupported in gateway mode; retries run server-side. |
| Client-side batch calls | Raise until gateway batch endpoints exist. |
| `/tokens/count` | Supported through the gateway-side model implementation. |
| `/rate-limit` or direct `get_rate_limit()` | `/rate-limit` remains reserved today; validate agents that call `get_rate_limit()` directly before migration. |
| Raw provider responses/history | HMAC-signed by the gateway and echoed by the client; do not deserialize or mutate raw blobs client-side. |
