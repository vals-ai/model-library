# Gateway

Remote proxy server that routes model calls through a centralized FastAPI service. Clients send inputs over HTTP; the server holds provider API keys and returns results with HMAC-signed history blobs.

## Client distribution

The Gateway server and native provider callers use the default, provider-full `model-library` distribution. For a separate Gateway-only client environment, install `model-library-client` from this checkout (`python -m pip install -e packages/client`) or a locally built wheel. The client distribution uses the same `model_library` import paths but does not require provider SDKs. It is an alternative, not an add-on: the two distributions install overlapping files and must not share an environment. A dependency on the full library elsewhere in the environment still brings its provider SDK requirements; the client does not uninstall SDKs required by other packages. For release-dependent availability of PyPI installation, see the [README installation instructions](../README.md).

The client supports ordinary Gateway entrypoints: registry-backed `GatewayLLM` construction, queries including tools and structured outputs, signed raw history, and Gateway utility routes. Native provider calls and public native SDK helpers require the full distribution. Gateway moderation returns a library-owned typed response preserving fields, nested attributes, aliases, and JSON output, but not the provider SDK response class; native moderation retains its SDK response. SDK exception aliases are no longer re-exported from `model_library.exceptions` in either distribution; native provider imports register SDK retry types when loaded, rather than making SDK-class retry matching available before provider load. Normal native provider calls and library serialization remain supported in the full distribution.

Native HTTPX2 client factories lazily register the existing retry categories for connection errors, read errors, and remote protocol errors; Gateway-only imports do not load HTTPX2.

## SDK dependency policy

The published full `model-library` declares compatible provider SDK ranges, including the voice and server extras. Gateway builds install exact versions from the root `uv.lock` with `uv sync --locked --extra server --extra voice`; the SDK-free client keeps its separate dependency manifest. The Gateway's boto3/botocore lock follows the server aiobotocore-compatible range, which can lag the latest boto3; the full-library boto3 requirement remains ranged.

Private Dependabot proactively checks only the configured provider SDKs daily at 08:00 UTC, with a one-day release-age buffer. It groups major, minor, and patch version updates together against `dev`; related lockfile dependencies may also move as needed. Review both `pyproject.toml` and `uv.lock` in each proposal; major releases may widen published ranges, so confirm compatibility across the declared range and migrate adapters if needed before merging. PRs show version changes and available release notes; none are auto-merged. The `Run Models` check requires a manual PR rerun.

Before merging a provider-SDK update PR, including Dependabot PRs, the live [Gateway history continuation example](../examples/README.md#gateway-history-continuation) must pass: save against the current Gateway revision, then follow up against the candidate revision with the same model and signing key. Review the original weather/note actions in the agent artifacts and both follow-up answers; exit zero alone does not establish a correct task or answers. Record both Gateway revisions, the tested model, and the manually reviewed semantic verdict in the PR. This is a manual review requirement, not an automated merge block; offline or synthetic checks do not satisfy it.

## Quick start

```env
# Server .env
MODEL_GATEWAY_API_KEYS='{"default":"key1","security-testing":"key2"}'  # name-to-key JSON map
OPENAI_API_KEY="sk-..."                  # provider keys as usual; configure any providers you serve
MODEL_GATEWAY_HMAC_SECRET="some-secret"        # required for startup and signed history blobs
```

```bash
make gateway
```

The gateway listens at `http://localhost:8001`.

```env
# Client .env
MODEL_GATEWAY_URL="http://localhost:8001"
MODEL_GATEWAY_API_KEY="key1"
```

When `MODEL_GATEWAY_URL` is set, `get_model_registry()` fetches and caches the Gateway registry snapshot. `get_registry_model()` uses that snapshot to construct a fully configured `GatewayLLM`, while request execution remains authoritative on the Gateway server.

For application/agent migration steps, see [Migrating agents and providers to Model Gateway](gateway-migration.md).

Task-scoped gateway credentials: [Run tokens](run-tokens.md).

## Environment variables

### Server

The gateway server uses server-side auth/signing config. Do not set `MODEL_GATEWAY_URL` on the gateway server task; that variable is client-only and `model_gateway.app` clears it at startup to prevent self-proxying.

#### Auth and provider credentials

- `MODEL_GATEWAY_API_KEYS` (**required**): JSON object mapping stable names to valid client API keys. Gateway startup fails if unset or empty.
- `MODEL_GATEWAY_HMAC_SECRET` (**required**): secret for HMAC-signing pickled fields in history blobs. Gateway startup fails without this so every task can safely return and accept raw history blobs.
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc. (**usually required**): provider API keys for providers served by this gateway. See [Provider API keys](api-keys.md). Per-request `custom_api_key` can supply caller credentials at call time. Requests that set `custom_endpoint` or `registry_key` are rejected.
- `GCP_CREDS`, `GCP_PROJECT_ID`, and `GCP_REGION` (**required for Google Vertex requests**): service-account credentials, project ID, and Vertex region. `config.provider_config.use_vertex=true` selects Vertex; other Google requests use `GOOGLE_API_KEY`.
#### Capacity and timeouts

| Contract | Behavior |
| --- | --- |
| Admission | Application code bounds active calls, queue size, and queue wait before request-body parsing. |
| Timeout ordering | `/query` reserves 200 seconds of its capacity-owned request budget, so the provider operation gets at most 3,340 seconds and less when pre-provider work consumes time. This stays below the 3,400-second per-attempt provider read timeout and 3,540-second gateway cap. |
| Provider timeout | Provider-operation expiry cancels the in-flight query and returns the normal HTTP `200` `ProviderError` envelope. |
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

- `GATEWAY_USAGE_LEDGER_MODE` (**optional**): usage-ledger mode for successful queries and structured-output failures with a result: `disabled`, `shadow`, or `enforced`. Defaults to `shadow`. In current SQS mode, `/query` locally queues the usage event before returning and SQS send/retry runs in the background, every 10 seconds for up to 20 minutes. Synchronous SQS durable acceptance is intentionally not implemented; see the usage-ledger rationale below.
- `GATEWAY_USAGE_LEDGER_QUEUE_URL` (**required when gateway ledger is enabled**): SQS queue for captured `/query` usage events. When set, the gateway sends usage events through SQS and a consumer writes the ledger row asynchronously.
- `GATEWAY_USAGE_LEDGER_TABLE_NAME` (**required when DynamoDB writer is enabled**): DynamoDB table for captured `/query` usage events. Set this for DynamoDB writers; local/direct gateway DynamoDB mode can also set it when no queue URL is configured.
- `GATEWAY_USAGE_LEDGER_SHARDS` (**optional**): number of write shards for day-partitioned DynamoDB usage events and run/key/benchmark/agent GSIs. Readers and the Redshift export Lambda must use the configured shard count.
- `GATEWAY_USAGE_LEDGER_MAX_POOL_CONNECTIONS` (**optional**): aiobotocore HTTP connection pool size for the per-process async ledger client. Increase only with matching request concurrency and AWS account limits.

#### Redshift usage export

Set these variables for the scheduled Redshift export job. The job also uses
`GATEWAY_USAGE_LEDGER_TABLE_NAME` and `GATEWAY_USAGE_LEDGER_SHARDS` from the
usage-ledger section.

- `GATEWAY_USAGE_REDSHIFT_WORKGROUP_NAME`: shared Redshift Serverless workgroup for Data API COPY, MERGE, and aggregate refresh.
- `GATEWAY_USAGE_REDSHIFT_DATABASE_NAME`: shared Redshift database name.
- `GATEWAY_USAGE_REDSHIFT_SECRET_ARN` (**required**): stage-specific loader credentials secret ARN for Redshift Data API batch submissions.
- `GATEWAY_USAGE_REDSHIFT_SCHEMA_NAME`: destination schema, for example `gateway_usage`.
- `GATEWAY_USAGE_REDSHIFT_LANDING_BUCKET_NAME`: shared S3 bucket where the export Lambda writes Parquet parts and COPY manifests.
- `GATEWAY_USAGE_REDSHIFT_EXPORT_PREFIX`: S3 prefix for exported usage data, for example `gateway-usage/{stage}/raw`.
- `GATEWAY_USAGE_REDSHIFT_COPY_ROLE_ARN`: IAM role that Redshift assumes to read the landing bucket during COPY.

#### Telemetry

- `GATEWAY_OTEL_ENABLED` (**optional**): enables OpenTelemetry tracing when set to `true`, `1`, `yes`, or `on`. Disabled by default and safe to leave unset locally.
- `SENTRY_DSN` (**required when tracing is enabled**): Sentry project DSN used for gateway traces, logs, and handled exception events.
- `SENTRY_ENVIRONMENT` (**optional**): Sentry environment label.
- `SENTRY_RELEASE` (**optional**): immutable release string attached to Sentry events/traces. Persistent dev/prod gateway releases set this to the full CodePipeline source Git SHA; manual preview releases leave it unset.
- `SENTRY_OTLP_COLLECTOR_URL` (**optional**): local/sidecar OTLP HTTP collector URL. When unset, Sentry's OTLP integration exports directly to the Sentry endpoint derived from `SENTRY_DSN`.
- `OTEL_SERVICE_NAME` (**optional**): OpenTelemetry service name. Defaults to `model-proxy-gateway` when tracing is enabled.
- `GATEWAY_KEY_ASSIGNMENTS_TABLE_NAME` (**deployed query runtime**): DynamoDB table that records which provider key each task attempt uses, one item per attempt and model (`pk` is the SHA-256 hex of the JSON pair `[["<attempt part>", ...],"<model>"]`). Assignments are kept like the usage ledger, with no TTL, so the key a request used can always be found. The first writer wins through a conditional put. When it is unset, the Gateway logs a warning and keeps the per-request pick.
- `GATEWAY_FAILED_EXCHANGE_CAPTURE_ENABLED` (**deployed query runtime**): deployed query runtimes set this to `true` so failed user-facing provider exchanges can be attached to Sentry. Control and rate-limit paths do not enable capture.
One capture spans the complete provider operation and all retries. It records HTTPX request/response entity bytes, aiohttp entity bytes as exposed by the configured client path, or Bedrock serialized request bodies and raw AWS event-stream bytes. Typed inline query and token-count image and file payloads are replaced with same-length, compressible omission markers while the rest of each serialized request is retained. If a registered payload cannot be removed, that attempt records only the request length. File-upload and transcription attempts always record request lengths without payloads. Responses are unchanged. These are diagnostic representations, not TLS or HTTP/2 framing. Successful captures are discarded without compression. Final failures and cancellations produce `provider-failed-exchange.ndjson.gz` with MIME `application/gzip` on the canonical Sentry event when Gateway telemetry and Sentry are enabled; otherwise the capture is discarded after metrics are emitted. Attachments may retain request data and unchanged provider response bodies, so Sentry access and retention must remain restricted.

Early closure of a started HTTPX2 response iterator records `closed` only after underlying cleanup succeeds; cleanup failures and cancellation remain errors.

Capture starts in a 1 MiB memory spool and rolls larger NDJSON to local disk. Retained NDJSON records and pending request buffers share a 128 MiB raw-capture budget across retries; this is not a maximum request size, and pending buffers can still use memory beyond the spool's rollover threshold. At the limit, capture adds a truncation marker and drops further records while payload counters continue. Compression runs in a worker thread, but capture recording still runs on the request path; cancellation cleanup waits for that worker before closing the spool. Gzip artifacts over the unchanged 20 MiB attachment limit are not attached. Metrics report attempts, payload/chunk and artifact bytes, compression time, capture errors, and transport. Changes to capture enablement take effect only after new query tasks deploy.

### Client

| Variable | Required | Description |
| ----------------- | -------- | ----------------------------------------------------------- |
| `MODEL_GATEWAY_URL` | Yes | URL of the gateway server (e.g. `http://localhost:8001`) |
| `MODEL_GATEWAY_API_KEY` | Yes | Client API key (must be in the server's `MODEL_GATEWAY_API_KEYS`) |
| `IDENTITY` | No | JSON object attached to gateway query telemetry and usage ledger rows. |
| `RUN_ID` | No | Default run ID for gateway queries when `query(run_id=...)` is absent or `None`. |
| `QUESTION_ID` | No | Default question/task ID for gateway queries when `query(question_id=...)` is absent or `None`. |
| `TASK_ID` | No | Alias fallback for `QUESTION_ID`; used only when `QUESTION_ID` is unset or blank. |
| `ATTEMPT_ID` | No | Default task-attempt ID for gateway queries when `query(attempt_id=...)` is absent or `None`. |

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
- Blank or whitespace-only `RUN_ID`, `QUESTION_ID`, `TASK_ID`, and `ATTEMPT_ID` values are ignored.
- Explicit blank or whitespace-only `run_id`, `question_id`, and `attempt_id` values are rejected.
- Direct blank `query_id` values are treated as absent.
- `attempt_id` names one execution of a task; a whole-task rerun gets a new one. It resolves like `run_id`: explicit value, then `ATTEMPT_ID`, then a generated ID.
- The client resolves the four IDs as one internal group and sends them as flat request fields and correlation headers. For raw HTTP `/query` requests with omitted IDs, gateway resolves one quartet for native execution, `gateway.query.start`, and usage records; managed credential selection still uses the original request IDs.
- The usage ledger stores `attempt_id` on the row beside `run_id`. It is not yet a Redshift column or a ledger index key.

Provider retry policy:

- Direct provider execution and gateway-server provider execution use the normal model retrier unless a custom retrier replaces it.
- Eight failed outer provider calls with an integer `status_code` of exactly `500` exhaust a query's HTTP-500 retry budget; seven such failures may still be followed by success on call eight. Other retryable errors retain the selected retrier's normal budget.
- Provider SDKs may retry internally within each outer provider call. Exceptions classified only by message text do not consume the structured HTTP-500 budget.
- When an exception is classified by its message text, a retryable status code (`429`, `500`, `502`, `503`, `504`, `529`) counts only as a whole number: `65000` or `1,500,000` in a validation message does not make the error retryable.

Local query deadlines:

- Direct and gateway-mode `LLM.query()` accept an optional absolute `deadline` from `asyncio.get_running_loop().time()`; omitted deadlines preserve the existing unlimited local behavior.
- The deadline bounds the complete local query lifecycle, including retry waits. Expiry raises non-retriable `QueryDeadlineExceededError`, which is also a `TimeoutError`.
- A `GatewayLLM` deadline cancels the active local HTTP request and stops local gateway retries, but it is not serialized to the server. The server continues to enforce its independent capacity-owned deadline.
- Custom `LLM.query()` overrides must accept and enforce `deadline` or delegate to the base implementation; provider `_query_impl()` methods do not receive it.
- Cancellation is best-effort downstream: asynchronous transports normally receive cancellation, while already-dispatched remote or threaded provider work may continue.

Gateway client HTTP retry policy:

- Gateway-mode `LLM.query()` does not use the normal model/provider retry wrapper; provider retries happen inside the gateway server.
- Provider-call failures from `/query`, `/tokens/count`, `/files/upload`, `/audio/transcriptions`, `/embeddings`, and `/moderation` return HTTP `200` with a top-level `error` envelope such as `{"error":{"type":"ProviderError","message":"...","exception_type":"RateLimitError"}}`.
- The client raises HTTP `200` error envelopes as `GatewayProviderError` without retrying, including provider-operation deadline errors, except normalized context-window and content-filter errors.
- An envelope becomes terminal `ContentFilterError` when `code`, `type`, `error_type`, or `exception_type` identifies `content_filter`, `ContentFilterError`, `cyber_policy`, `safety_policy`, `blocked_prompt`, or `content_policy_violation`, when the message contains `output blocked by content filtering policy`, or when the message is only the model's refusal sentence (`I'm sorry, but I can't help with that request.`, nothing more), as some Responses endpoints return in a code-less stream `error` event.
- Provider-error envelopes include the provider-operation exception type and message. They include `code` and `status_code` only when the provider/library exception already exposes those fields; the gateway does not synthesize mapped provider codes or mapped provider status codes for the response body.
- Error objects do not include pickled exception objects, tracebacks, or raw provider exception transport.
- Structured-output failures can include a sibling `result` containing the actual failed `QueryResult`, with `signed_history` nested in that result rather than a top-level error history. The client raises `GatewayProviderError` with `.query_result`; `.usage_metadata` derives from its metadata.
- Failed results are decoded without rerunning the requested output schema. Available output, reasoning, tools, and history can therefore be inspected even when parsing failed.
- Result availability depends on the error producer: many errors occur before a result exists. The optional shape supports future producers; it does not promise complete failed output or usage from earlier retry attempts.
- The client retries gateway HTTP transport failures, actual HTTP `429`, and actual HTTP `5xx` responses.
- The client does not retry actual HTTP `4xx` responses except `429`.

For deployed environments, keep server and client Secrets Manager entries separate:

- server config secret: `api_keys` (a JSON-encoded name-to-key object), `hmac_secret`
- client config secret: `MODEL_GATEWAY_URL`, `MODEL_GATEWAY_API_KEY`

## Endpoints

| Method | Path                  | Auth | Description                                                                |
| ------ | --------------------- | ---- | -------------------------------------------------------------------------- |
| GET    | `/health/live`        | No   | Liveness check                                                             |
| GET    | `/health/ready`       | No   | Readiness check (auth/signing config; control/combined Redis ping and monitor health; query startup canary when enabled)  |
| GET    | `/models`             | Yes  | List available models with capability flags                                |
| GET    | `/registry`           | Yes  | Return the model registry snapshot                                         |
| POST   | `/query`              | Yes  | Execute an LLM query                                                       |
| POST   | `/tokens/count`       | Yes  | Count input tokens by using the gateway-side model implementation          |
| POST   | `/rate-limit`         | Yes  | Probe the configured provider account's rate limits                        |
| GET    | `/rate-limit-monitor` | Yes  | List active and retained rate-limit snapshots                              |
| POST   | `/rate-limit-monitor/activate` | Yes | Start or extend monitoring for an eligible model key                |
| GET    | `/token-retry/status` | Yes  | Return 1s-cached Redis token retry and benchmark queue status              |
| POST   | `/benchmark-runs/acquire` | Yes | Request admission and resolve the effective token-retry limit           |
| POST   | `/benchmark-runs/wait`    | Yes | Long-poll until the benchmark run is admitted                           |
| POST   | `/benchmark-runs/renew`   | Yes | Renew the admitted run's heartbeat                                      |
| POST   | `/benchmark-runs/release` | Yes | Release admission with the terminal outcome                             |
| POST   | `/files/upload`       | Yes  | Upload a provider file and return a `FileWithId`                           |
| POST   | `/audio/transcriptions` | Yes | Transcribe one base64-encoded audio file                                  |
| POST   | `/embeddings`         | Yes  | Create an embedding vector                                                 |
| POST   | `/moderation`         | Yes  | Run content moderation                                                     |

### Registry snapshot

`GET /registry` omits transcription-only entries by default for compatibility
with older model-library clients. It also omits `country`, `rate_limit`,
`metadata.router`, `supports.transcription`, `transcription_cost`,
`transcription_language`, and `transcription_streaming`. Set
`include_excluded_fields=true` to include `country`, `rate_limit`,
`metadata.router`, and `supports.transcription`. The three transcription
metadata fields and transcription-only entries require both
`include_excluded_fields=true` and `include_transcription=true`; current clients
request both flags. Set `include_alt_keys=false` to omit same-provider alternative keys.

### One-off rate-limit probes

`POST /rate-limit` uses the model's registry/default provider endpoint and credentials. It does not probe caller-supplied custom endpoints or API keys.

The response contains:

- `rate_limit` when the provider reports data;
- `error` when the probe fails;
- neither field when no data is available.

It never contains both fields. A rate limit includes typed request and token capacity, optional remaining values and scope, and the observation timestamp. Unknown remaining values stay unset. Raw provider data, reset data, quota identifiers, and arbitrary windows are not returned.

The gateway returns HTTP 429 when probe admission is full. A probe timeout returns HTTP 200 with a sanitized `RateLimitResponse.error`.

### Shared rate-limit monitor

Monitor indexes, ownership leases, and snapshots are shared only within one `GATEWAY_STAGE`. Dev, Prod, and each preview use separate Redis keys. The stage defaults to `local` outside deployed runtimes. Old unscoped snapshots are not read; activate monitoring again after the upgrade.

Start or extend monitoring with `POST /rate-limit-monitor/activate` and `{"model": "<model-key>"}`. The key must be registered and have rate-limit monitoring enabled; alternative keys are valid.

Successful responses are:

- `GET /rate-limit-monitor`: `{"server_time": <timestamp>, "states": [<state>, ...]}`
- `POST /rate-limit-monitor/activate`: `{"server_time": <timestamp>, "state": <state>}`

Each state contains `model`, `active`, `active_until`, `retention_until`, `status`, and `sources`. Each source contains `source`, `status`, nullable `last_attempt_at`, `last_success_at`, `rate_limit`, and `error_code`. State and source statuses are `starting`, `ok`, `stale`, `unsupported`, or `error`. Provider poll failures use the generic `provider_error` code and are retried only by the next scheduled poll.

Live credential updates refresh the values used to poll activated source IDs. Adding or removing a credential pool does not change monitor membership or extend its active period. Activate the model again to update membership. A removed source stops provider polling and retains its last successful value as `stale`; a source without a previous successful value reports `error`. Other activated sources continue polling.

Invalid activation returns HTTP 400 with `{"code": "invalid_model", "message": "Invalid model activation request"}`. Redis, persisted-state, and monitor timeout failures use the gateway's normal server-error boundary rather than a monitor-specific response. `control` and `combined` runtimes require Redis and become unready if monitor supervision stops.

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
       - `attempt_id`
       - `in_agent`
   - Provider-specific overrides live under `config.provider_config`, not as top-level fields. For example, a `google/*` request selects Vertex with `config.provider_config.use_vertex=true`; the gateway supplies the server-side GCP credentials. An `openai/*` request selects Chat Completions with `config.provider_config.use_completions=true` (Responses is the default).
   - `model.metadata` is a copy of the registry entry used at construction.
   - Existing model instances retain that construction snapshot; newly constructed models use the refreshed registry.
   - Client-side custom retriers are rejected in gateway mode because provider retries run on the gateway server.

3. **Client sends full input list**

   - `GatewayLLM.query()` joins history + new input into a single `inputs` array.
   - Normal items use `model_dump()`, such as:
     - `TextInput`
     - `ToolResult`
   - Raw items received from the server in a previous response contain opaque blobs:
     - `RawResponse`
     - `RawInput`
   - Raw blobs are base64-encoded and HMAC-signed.
   - The client echoes raw blobs as-is without deserializing them.
   - A `RawInput` built by the caller, such as a provider message dict, is sent as plain JSON.
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
   - Existing cached provider models remain immutable. With live runtime snapshots,
     a revision change selects a new cache namespace for later requests while
     in-flight requests retain the object they captured. Without live runtime
     snapshots, a restart or explicit cache invalidation is required.
   - Provider-call failures are surfaced as HTTP `200` `ProviderError` envelopes with provider/library exception information instead of gateway-mapped error codes.

5. **Server restores Raw fields**

   - On receiving `inputs`, the server calls `LLM.restore_raw_fields()`.
   - `LLM.restore_raw_fields()` unpickles every string or dict `RawResponse` field, but a `RawInput` field only when it is a string or a dict with a `pickle` key. Other `RawInput` values, such as caller-built provider message dicts, pass to the provider unchanged.
   - The server requires and verifies HMAC before restoring raw fields in-place.
   - The full input list is then passed to `llm.query()`.

6. **Server returns signed history**

   - For a query with a result, the server serializes the full `result.history` via `LLM.serialize_input(secret=hmac_secret)`.
   - `RawResponse` and `RawInput` fields are serialized, base64-encoded, and HMAC-signed inline.
   - The gateway returns this JSON string as `signed_history` at the response top level on success, or inside the sibling `result` on a structured-output failure; the top-level `error` has no history.
   - The client parses `signed_history` into `InputItem` objects.
   - Raw fields stay as blobs on the client.
   - The client uses parsed history as the base for the next turn's input.

To save client-returned Gateway history locally without provider SDKs, use `LLM.serialize_input(result.history)` and later `LLM.deserialize_input(saved_json)`. Its raw fields remain opaque signed blobs inside the locally serialized history. This is not native SDK object unpickling, and the server's raw `signed_history` JSON is not the input to this local round trip. Deserialization uses pickle; only load trusted local data.

### Security boundaries

- **Client → Server**: Bearer token auth with constant-time comparison (pre-hashed keys)
- **Server → Client**: Raw fields in history are HMAC-signed when returned
- **Server input validation**: Serialized raw blobs in client inputs are HMAC-verified before restoring — tampered or unsigned blobs are rejected. Caller-built `RawInput` values that are not blobs are never unpickled
- **Client query path never deserializes**: It only echoes opaque blobs; server-side history restoration happens on the server. Explicit local `LLM.deserialize_input` is separate from the query path.
- **custom_api_key**: Passed only in per-request provider config so clients can use BYOK without using it as the gateway bearer token
- **provider_config.access_programs**: Caller-supplied overrides require a nonempty `custom_api_key`; requests using gateway-managed keys use the registry's access programs
- **custom_endpoint, registry_key**: Rejected on every request; they would serve the named model from another endpoint or registry entry

## Running locally

The [quick-start command](#quick-start) uses Docker Compose. It builds the gateway, loads `.env`, exposes Redis at `redis://127.0.0.1:6380`, and mounts `model_gateway/` and `model_library/` so Uvicorn reloads source changes.
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
