# Gateway

Remote proxy server that routes model calls through a centralized FastAPI service. Clients send inputs over HTTP; the server holds provider API keys and returns results with HMAC-signed history blobs.

## Quick start

```env
# Server .env
MODEL_GATEWAY_API_KEYS="key1,key2"            # comma-separated client keys
OPENAI_API_KEY="sk-..."                  # provider keys as usual; configure any providers you serve
MODEL_GATEWAY_HMAC_SECRET="some-secret"        # required for startup and signed history blobs
```

```bash
make gateway  # localhost:8000
```

```env
# Client .env
MODEL_GATEWAY_URL="http://localhost:8000"
MODEL_GATEWAY_API_KEY="key1"
```

When `MODEL_GATEWAY_URL` is set, `get_model_registry()` fetches and caches the Gateway registry snapshot. `get_registry_model()` uses that snapshot to construct a fully configured `GatewayLLM`, while request execution remains authoritative on the Gateway server.

For application/agent migration steps, see [Migrating agents and providers to Model Gateway](gateway-migration.md).

## Environment variables

### Server

The gateway server uses server-side auth/signing config. Do not set `MODEL_GATEWAY_URL` on the gateway server task; that variable is client-only and `model_gateway.app` clears it at startup to prevent self-proxying.

#### Auth and provider credentials

- `MODEL_GATEWAY_API_KEYS` (**required**): comma-separated list of valid client API keys. Gateway startup fails if unset or empty.
- `MODEL_GATEWAY_HMAC_SECRET` (**required**): secret for HMAC-signing pickled fields in history blobs. Gateway startup fails without this so every task can safely return and accept raw history blobs.
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc. (**usually required**): provider API keys for providers served by this gateway. See [Provider API keys](api-keys.md). Per-request `custom_api_key` can supply caller credentials at call time. Per-request `custom_endpoint` is accepted only with `custom_api_key`; the gateway uses that caller-supplied key for the custom URL and never sends server-held provider keys to arbitrary endpoints.

#### Capacity and timeouts

| Contract | Behavior |
| --- | --- |
| Admission | Application code bounds active calls, queue size, and queue wait before request-body parsing. |
| Timeout ordering | The application timeout stays below target drain; caller and ALB timeouts stay above the gateway timeout; Uvicorn keepalive stays above the ALB idle timeout. |
| Provider timeout | Provider read timeouts may fire before the outer gateway cap. |
| Ownership | Deployment infrastructure supplies worker/process settings; keep those values consistent with the application limits above. |

#### Deployments

Deployment and drain behavior is infrastructure-specific. Drain is best-effort,
not durable work preservation: queue wait, cancellation cleanup, client
disconnects, task crashes, and provider behavior can interrupt or delay work.
Do not assume every accepted HTTP connection will return before a deployment
drain closes.

#### Startup canary

`GATEWAY_STARTUP_CANARY_ENABLED` enables one authenticated local `/query` and
keeps `/health/ready` at `503` until it passes.

#### Usage ledger

- `GATEWAY_USAGE_LEDGER_MODE` (**optional**): successful-query usage ledger mode: `disabled`, `shadow`, or `enforced`. Defaults to `shadow`. In current SQS mode, `/query` locally queues the usage event before returning and SQS send/retry runs in the background, every 10 seconds for up to 20 minutes. Synchronous SQS durable acceptance is intentionally not implemented; see the usage-ledger rationale below.
- `GATEWAY_USAGE_LEDGER_QUEUE_URL` (**required when gateway ledger is enabled**): SQS queue for successful completed `/query` usage events. When set, the gateway sends successful usage events through SQS and a consumer can write the ledger row asynchronously.
- `GATEWAY_USAGE_LEDGER_TABLE_NAME` (**required when DynamoDB writer is enabled**): DynamoDB table for successful completed `/query` usage events. Set this for DynamoDB writers; local/direct gateway DynamoDB mode can also set it when no queue URL is configured.
- `GATEWAY_USAGE_LEDGER_SHARDS` (**optional**): number of write shards for day-partitioned DynamoDB usage events and run/key/benchmark/agent GSIs. Readers and the Redshift export Lambda must use the configured shard count.
- `GATEWAY_USAGE_LEDGER_MAX_POOL_CONNECTIONS` (**optional**): aiobotocore HTTP connection pool size for the per-process async ledger client. Increase only with matching request concurrency and AWS account limits.

#### Redshift usage export

Set these variables for the scheduled Redshift export job. The job also uses
`GATEWAY_USAGE_LEDGER_TABLE_NAME` and `GATEWAY_USAGE_LEDGER_SHARDS` from the
usage-ledger section.

- `GATEWAY_USAGE_REDSHIFT_WORKGROUP_NAME`: shared Redshift Serverless workgroup for Data API COPY, MERGE, and aggregate refresh.
- `GATEWAY_USAGE_REDSHIFT_DATABASE_NAME`: shared Redshift database name.
- `GATEWAY_USAGE_REDSHIFT_SCHEMA_NAME`: destination schema, for example `gateway_usage`.
- `GATEWAY_USAGE_REDSHIFT_LANDING_BUCKET_NAME`: shared S3 bucket where the export Lambda writes Parquet parts and COPY manifests.
- `GATEWAY_USAGE_REDSHIFT_EXPORT_PREFIX`: S3 prefix for exported usage data, for example `gateway-usage/{stage}/raw`.
- `GATEWAY_USAGE_REDSHIFT_COPY_ROLE_ARN`: IAM role that Redshift assumes to read the landing bucket during COPY.

#### Telemetry

- `GATEWAY_OTEL_ENABLED` (**optional**): enables OpenTelemetry tracing when set to `true`, `1`, `yes`, or `on`. Disabled by default and safe to leave unset locally.
- `SENTRY_DSN` (**required when tracing is enabled**): Sentry project DSN used for gateway traces, logs, and handled exception events.
- `SENTRY_ENVIRONMENT` (**optional**): Sentry environment label.
- `SENTRY_RELEASE` (**optional**): release string attached to Sentry events/traces.
- `SENTRY_OTLP_COLLECTOR_URL` (**optional**): local/sidecar OTLP HTTP collector URL. When unset, Sentry's OTLP integration exports directly to the Sentry endpoint derived from `SENTRY_DSN`.
- `OTEL_SERVICE_NAME` (**optional**): OpenTelemetry service name. Defaults to `model-proxy-gateway` when tracing is enabled.

### Client

| Variable | Required | Description |
| ----------------- | -------- | ----------------------------------------------------------- |
| `MODEL_GATEWAY_URL` | Yes | URL of the gateway server (e.g. `http://localhost:8000`) |
| `MODEL_GATEWAY_API_KEY` | Yes | Client API key (must be in the server's `MODEL_GATEWAY_API_KEYS`) |
| `IDENTITY` | No | JSON object attached to gateway query telemetry and usage ledger rows. |
| `RUN_ID` | No | Default run ID for gateway queries when `query(run_id=...)` is absent or `None`. |
| `QUESTION_ID` | No | Default question/task ID for gateway queries when `query(question_id=...)` is absent or `None`. |
| `TASK_ID` | No | Alias fallback for `QUESTION_ID`; used only when `QUESTION_ID` is unset or blank. |

Attribution behavior:

- `IDENTITY` setting/env var must be a JSON-encoded object string, for example `{"user_id":"user_123","benchmark_name":"swebench"}`.
- Explicit `query(identity=...)` takes a Python mapping/object.
- Identity keys are caller-defined and values are preserved in the gateway request.
- The usage ledger promotes recognized `benchmark_name`, `agent_name`, and `email` dimensions while preserving the complete normalized identity in `details.request`.
- Identity can still enter operational telemetry, so do not put secrets, provider API keys, bearer tokens, or other credentials in it. PII is allowed when needed for attribution/analytics.
- `benchmark_name`, `agent_name`, and `email` are recognized usage-ledger analytics keys when their values are strings that trim to non-empty values no larger than 512 UTF-8 bytes.
- `email` must be email-shaped and is stored as canonical lowercased `identity_email` for exact Redshift filtering.
- Benchmark and agent raw-ledger lookup keys are exact-match and forward-only. Email analytics are available through Redshift; there is no raw-ledger email lookup or substring search path.
- Rows written before those fields, indexes, or mart columns existed are not included in those lookup or analytics paths.
- Invalid `IDENTITY` settings are omitted by the client:
  - non-object
  - non-finite
  - larger than 4 KiB
  - deeper than 8 levels
- Gateway accepts explicit `query(identity=...)`.
- Explicit non-`None` `query(identity=...)` overrides `IDENTITY` from `model_library_settings`.
- `query(identity=None)` falls back to `IDENTITY` from `model_library_settings`.
- Invalid explicit `query(identity=...)` values raise validation errors before the request is sent.
- `RUN_ID`, `QUESTION_ID`, and `TASK_ID` are read through `model_library_settings`.
- `RUN_ID`, `QUESTION_ID`, and `TASK_ID` apply to all `LLM.query()` calls.
- `QUESTION_ID` takes precedence over `TASK_ID`.
- There is no `QUERY_ID` env/settings fallback.
- `query_id` is generated per request unless passed explicitly.
- Blank or whitespace-only `RUN_ID`, `QUESTION_ID`, and `TASK_ID` values are ignored.
- Explicit blank or whitespace-only `run_id` and `question_id` values are rejected.
- Direct blank `query_id` values are treated as absent.

Gateway client HTTP retry policy:

- Gateway-mode `LLM.query()` does not use the normal model/provider retry wrapper; provider retries happen inside the gateway server.
- Provider-call failures from `/query`, `/tokens/count`, `/files/upload`, `/embeddings`, and `/moderation` return HTTP `200` with a top-level `error` envelope such as `{"error":{"type":"ProviderError","message":"...","exception_type":"RateLimitError"}}`.
- The client raises HTTP `200` error envelopes as `GatewayProviderError` without retrying.
- Provider-error envelopes include the provider-operation exception type and sanitized message. They include `code` and `status_code` only when the provider/library exception already exposes those fields; the gateway does not synthesize mapped provider codes or mapped provider status codes for the response body.
- Provider-error responses do not include `signed_history`, pickled exception objects, tracebacks, or raw provider exception transport.
- The client retries gateway HTTP transport failures, actual HTTP `429`, and actual HTTP `5xx` responses.
- The client does not retry actual HTTP `4xx` responses except `429`.

For deployed environments, keep server and client Secrets Manager entries separate:

- server config secret: `api_keys`, `hmac_secret`
- client config secret: `MODEL_GATEWAY_URL`, `MODEL_GATEWAY_API_KEY`

## Endpoints

| Method | Path                  | Auth | Description                                                                |
| ------ | --------------------- | ---- | -------------------------------------------------------------------------- |
| GET    | `/health/live`        | No   | Liveness check                                                             |
| GET    | `/health/ready`       | No   | Readiness check (verifies gateway auth/signing config and startup canary)  |
| GET    | `/models`             | Yes  | List available models with capability flags                                |
| GET    | `/registry`           | Yes  | Return the full model registry snapshot for client construction            |
| POST   | `/query`              | Yes  | Execute an LLM query                                                       |
| POST   | `/tokens/count`       | Yes  | Count input tokens by using the gateway-side model implementation          |
| POST   | `/rate-limit`         | Yes  | Reserved endpoint; currently rejects with `Gateway token retry use only`   |
| GET    | `/token-retry/status` | Yes  | Return 1s-cached Redis token retry and benchmark queue status              |
| POST   | `/files/upload`       | Yes  | Upload a provider file and return a `FileWithId`                           |
| POST   | `/embeddings`         | Yes  | Create an embedding vector                                                 |
| POST   | `/moderation`         | Yes  | Run content moderation                                                     |

## Architecture

```text
Client (get_model_registry / get_registry_model)
  │
  ├─ get_model_registry + MODEL_GATEWAY_URL ──► GET /registry
  │                                        └─ cache registry snapshot locally
  │
  ├─ get_registry_model + MODEL_GATEWAY_URL ──► construct registry-configured GatewayLLM
  │                                        ├─ metadata and capabilities available immediately
  │                                        ├─ send canonical model key + explicit overrides only
  │                                        ├─ gateway resolves current server config on cache miss
  │                                        ├─ POST supported calls with Bearer token
  │                                        └─ deserialize typed responses/history
  │
  └─ no MODEL_GATEWAY_URL ──► load local YAML registry and direct provider LLM
```

### Data flow

1. **Client loads the registry from gateway** — when `MODEL_GATEWAY_URL` is set, `get_model_registry()` fetches `/registry` with `MODEL_GATEWAY_API_KEY` and caches that snapshot instead of reading bundled YAML files. Provider properties remain available as raw client metadata without importing provider implementations. `refresh_model_registry()` replaces the snapshot for models constructed afterward. The Gateway remains request-time execution authority.

2. **Client constructs a registry-configured `GatewayLLM`**

   - In gateway mode, `get_registry_model()` follows the normal registry construction path:
     - resolves the model from the current client snapshot;
     - applies registry defaults and explicit overrides to local model fields;
     - exposes `model.metadata` and capability attributes immediately;
     - preserves only caller-supplied overrides for Gateway requests.
   - Gateway requests send:
     - top-level `model`
     - top-level `config` for model-config overrides
     - top-level attribution fields:
       - `identity`
       - `run_id`
       - `question_id`
       - `query_id`
       - `in_agent`
   - Provider-specific overrides live under `config.provider_config`, not as top-level fields.
   - `model.metadata` is a copy of the registry entry used at construction.
   - Existing model instances retain that construction snapshot; newly constructed models use the refreshed registry.
   - Gateway batch capability metadata is preserved.
   - Client-side gateway batch calls raise until gateway batch endpoints exist.
   - Client-side custom retriers are rejected in gateway mode because provider retries run on the gateway server.

3. **Client sends full input list**

   - `GatewayLLM.query()` joins history + new input into a single `inputs` array.
   - Normal items use `model_dump()`, such as:
     - `TextInput`
     - `ToolResult`
   - Raw items contain opaque blobs received from the server in a previous response:
     - `RawResponse`
     - `RawInput`
   - Raw blobs are base64-encoded and HMAC-signed.
   - The client echoes raw blobs as-is without deserializing them.
   - Gateway query requests reject arbitrary provider-specific query kwargs.
   - Use registry/default config or explicit `LLMConfig.provider_config` overrides instead.

4. **Server-authoritative execution config**

   - On provider-model cache miss, the gateway server merges:
     - loaded registry config
     - explicit override config supplied by the client
   - The server caches the resulting provider model instance by:
     - model
     - override config
     - token retry params, when token retry is active
   - Registry/config changes are not hot-reloaded into existing cached provider models.
   - Restart or cache invalidation is required to pick up registry/config changes.
   - Provider-call failures are surfaced as HTTP `200` `ProviderError` envelopes with provider/library exception information instead of gateway-mapped error codes.

5. **Server restores Raw fields**

   - On receiving `inputs`, the server calls `LLM.restore_raw_fields()`.
   - `LLM.restore_raw_fields()` finds `RawResponse`/`RawInput` items with serialized blobs.
   - The server requires and verifies HMAC before restoring raw fields in-place.
   - The full input list is then passed to `llm.query()`.

6. **Server returns signed history**

   - After each query, the server serializes the full `result.history` via `LLM.serialize_input(secret=hmac_secret)`.
   - `RawResponse` and `RawInput` fields are serialized, base64-encoded, and HMAC-signed inline.
   - The gateway returns this JSON string as `signed_history` in the response.
   - The client parses `signed_history` into `InputItem` objects.
   - Raw fields stay as blobs on the client.
   - The client uses parsed history as the base for the next turn's input.

### Security boundaries

- **Client → Server**: Bearer token auth with constant-time comparison (pre-hashed keys)
- **Server → Client**: Raw fields in history are HMAC-signed when returned
- **Server input validation**: Raw fields in client inputs are HMAC-verified before restoring — tampered or unsigned blobs are rejected
- **Client never deserializes**: The client only echoes opaque blobs; all deserialization happens server-side
- **custom_api_key**: Passed only in per-request provider config so clients can use BYOK without using it as the gateway bearer token
- **custom_endpoint**: Accepted only together with per-request `custom_api_key`; caller-selected endpoints never receive the gateway server's environment provider keys

## Running locally

```bash
make gateway  # uvicorn with --reload on port 8000
```
## Source files

| File | Role |
| ------------------------------- | ------------------------------------------------------ |
| `model_gateway/app.py` | FastAPI `create_app` factory, gateway endpoints, model caching |
| `model_gateway/auth.py` | Bearer token middleware |
| `model_gateway/cache.py` | LRU cache for LLM instances by (model, config) |
| `model_gateway/errors.py` | Exception → HTTP error response mapping |
| `model_gateway/history.py` | HMAC sign/verify wrappers around `LLM.serialize_input` |
| `model_gateway/types.py` | Gateway request/response Pydantic models |
| `model_gateway/usage_ledger/store.py` | Usage-event promoted spine and canonical details attachment |
| `model_gateway/usage_ledger/details.py` | Projected request/result details and explicit content reductions |
| `model_gateway/usage_ledger/message.py` | Shared direct/SQS event preparation and envelope codec |
| `model_gateway/usage_ledger/schema.py` | DynamoDB key/index, physical-field, and event-size contract |
| `model_gateway/usage_ledger/redshift_schema.py` | Redshift fact/performance/staging contract |
| `model_library/telemetry.py` | Optional OpenTelemetry helpers and attribute filtering |
| `model_library/base/gateway.py` | Client-side `GatewayLLM` class |
| `model_library/exceptions.py` | `GatewayMethodNotSupported` exception |
| `Dockerfile` | Multi-stage build with uv |
