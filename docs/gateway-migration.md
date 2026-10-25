# Migrating agents and providers to Model Gateway

This guide is for agents and providers that already use `model-library` model objects and want to route requests through Model Gateway instead of calling providers directly.

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

When `MODEL_GATEWAY_URL` is set, `get_registry_model()` loads the gateway registry and returns a configured `GatewayLLM`. Metadata and capabilities are available immediately. The same code calls providers directly when the variable is unset and uses Model Gateway when it is set.

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

Gateway requests send the model key and explicit `LLMConfig` override. The server merges the override with its registry entry before constructing the provider model.

| Need | Put it in | Constraint |
| --- | --- | --- |
| Agent/provider runtime parameter | Top-level `LLMConfig` fields | Pass the config to `get_registry_model(..., override_config=config)`. |
| Provider-specific parameter | `LLMConfig.provider_config` | Use the typed provider config. |
| Bring your own key | `LLMConfig.custom_api_key` | The gateway rejects `custom_endpoint` and `registry_key`. |
| Query attribution | `query(...)` fields | Keep `identity`, `run_id`, `question_id`, `query_id`, and `attempt_id` out of `LLMConfig`. |

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

Bring-your-own-key requests also go through `LLMConfig`. The gateway rejects `custom_endpoint` and `registry_key`:

```python
from pydantic import SecretStr

model = get_registry_model(
    "openai/gpt-4o",
    override_config=LLMConfig(custom_api_key=SecretStr(provider_key)),
)
```

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

`logger` supplies the base logger for client-side query logs. `system_prompt` is still normalized into a leading system input, but new code should prefer explicit `SystemInput` items.

## Metadata and registry access

In Gateway mode, model construction uses the Gateway-backed client registry while request execution remains server-authoritative.

| Need | Use |
| --- | --- |
| One model's capabilities, registry metadata, or context window | Construct it with `get_registry_model()`, then read the model fields directly. |
| Full registry or explicit refresh | Use `get_model_registry()` or `refresh_model_registry()` followed by `get_model_registry()`. |
| Request execution | Query the configured `GatewayLLM`; the server merges explicit overrides with its current registry. |

Single-model metadata is available immediately:

```python
model = get_registry_model(model_key)
metadata = model.metadata
context_window = model.input_context_window
```

A registry refresh affects models constructed afterward. Existing model objects retain the metadata snapshot used during construction.

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
| `run_id` | Sent only when passed to `query(run_id=...)`. Explicit blank or whitespace-only values are rejected. |
| `question_id` | Sent only when passed to `query(question_id=...)`. Explicit blank or whitespace-only values are rejected. |
| `query_id` | Generated per query unless passed explicitly. There is no `QUERY_ID` env/settings fallback. Direct blank values are treated as absent. |
| `attempt_id` | Sent only when passed to `query(attempt_id=...)`. Explicit blank or whitespace-only values are rejected. |

- `RUN_ID`, `QUESTION_ID`, `TASK_ID`, and `ATTEMPT_ID` env/settings are not read; a run token carries run, task and attempt.
- The Gateway fills omitted IDs for its ledger. Token retry requires `run_id` and `question_id`.

Native `LLM.query()` resolves the same four IDs once per call. Provider, retry, and token-queue telemetry share the immutable internal group through retries and sleeps; query loggers include the attempt ID. Query kwargs and Gateway request fields remain flat. Standalone task-level retriers retain run/question attribution without inventing query or attempt IDs.

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
- `attempt_id`

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
- [ ] BYOK migrations explicitly account for `custom_api_key` as a caller-supplied provider credential; `custom_endpoint` paths stay off the gateway.
- [ ] `query()` calls do not pass provider-specific kwargs.
- [ ] Code that reads metadata constructs the model with `get_registry_model()` first.
- [ ] Code does not use custom retrier paths.

## Gateway feature support

| Feature | Gateway behavior |
| --- | --- |
| Custom retriers | Do not migrate client-side retriers. The gateway applies retries server-side. |
| `/tokens/count` | Supported through the gateway model. |
| `/rate-limit` / `get_rate_limit()` | Supported. `get_rate_limit()` returns `None` when the provider has no data. |
| Raw provider responses/history | The gateway signs them and the client returns them unchanged. Treat them as opaque. |
