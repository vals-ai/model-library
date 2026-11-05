# Gateway

Remote proxy server that routes model calls through a centralized FastAPI service. Clients send inputs over HTTP; the server holds provider API keys and returns results with HMAC-signed history blobs.

## Client distribution

The Gateway server and native provider callers use the default, provider-full `model-library` distribution. For a separate Gateway-only client environment, install `model-library-client` from this checkout (using the [README installation command](../README.md)) or a locally built wheel. The client distribution uses the same `model_library` import paths but does not require provider SDKs. It is an alternative, not an add-on: the two distributions install overlapping files and must not share an environment. A dependency on the full library elsewhere in the environment still brings its provider SDK requirements; the client does not uninstall SDKs required by other packages. For release-dependent availability of PyPI installation, see the [README installation instructions](../README.md).

The client supports ordinary Gateway entrypoints: registry-backed `GatewayLLM` construction, queries including tools and structured outputs, signed raw history, and Gateway utility routes. Native provider calls and public native SDK helpers require the full distribution. Gateway moderation returns a library-owned typed response preserving fields, nested attributes, aliases, and JSON output, but not the provider SDK response class; native moderation retains its SDK response. SDK exception aliases are no longer re-exported from `model_library.exceptions` in either distribution; native provider imports register SDK retry types when loaded, rather than making SDK-class retry matching available before provider load. Normal native provider calls and library serialization remain supported in the full distribution.

Native HTTPX2 client factories lazily register the existing retry categories for connection errors, read errors, and remote protocol errors; Gateway-only imports do not load HTTPX2.

## SDK dependency policy

The published full `model-library` declares compatible provider SDK ranges, including the voice and server extras. Gateway builds install exact versions from the root `uv.lock` with `uv sync --locked --extra server --extra voice`; the SDK-free client keeps its separate dependency manifest. The Gateway's boto3/botocore lock follows the server aiobotocore-compatible range, which can lag the latest boto3; the full-library boto3 requirement remains ranged.

Private Renovate proactively checks only the configured provider SDKs, with a seven-day release-age buffer; its Dependency Dashboard issue lists pending updates. Minor and patch updates share one `provider-sdk-updates` PR against `dev` that is updated in place as new versions become eligible; major updates get a separate PR. Related lockfile dependencies may also move as needed. Review both `pyproject.toml` and `uv.lock` in each proposal; major releases may widen published ranges, so confirm compatibility across the declared range and migrate adapters if needed before merging. PRs show version changes and available release notes; none are auto-merged. The `Run Models` check requires a manual PR rerun.

Before merging a provider-SDK update PR, including Renovate PRs, the live [Gateway history continuation example](../examples/README.md#gateway-history-continuation) must pass: save against the current Gateway revision, then follow up against the candidate revision with the same model and signing key. Review the original weather/note actions in the agent artifacts and both follow-up answers; exit zero alone does not establish a correct task or answers. Record both Gateway revisions, the tested model, and the manually reviewed semantic verdict in the PR. This is a manual review requirement, not an automated merge block; offline or synthetic checks do not satisfy it.

## Quick start

```env
# Server .env
MODEL_GATEWAY_API_KEYS='{"legacy-client":"<existing-key>"}'  # named existing client keys
OPENAI_API_KEY="sk-..."                  # provider keys as usual; configure any providers you serve
MODEL_GATEWAY_HMAC_SECRET="some-secret"        # required for startup and signed history blobs
MODEL_GATEWAY_AUTH_NAMESPACE="local-db"   # Redis authority shared by runtimes using this auth database
```

Local startup also needs a migrated managed-auth PostgreSQL database for control, Redis for both roles, and an explicit `MODEL_GATEWAY_AUTH_NAMESPACE` shared by runtimes using that auth database. Control also needs its Descope settings. Deployed stacks provision the database and its credential through CDK; human-management and canary credentials still come from their authorized external owners

```bash
make gateway
```

The gateway listens at `http://localhost:8001`.

```env
# Client .env after explicitly minting a managed child token
MODEL_GATEWAY_URL="http://localhost:8001"
MODEL_GATEWAY_API_KEY="mgwt_<issued-child-token>"
```
When `MODEL_GATEWAY_URL` is set, `get_model_registry()` fetches and caches the Gateway registry snapshot. `get_registry_model()` uses that snapshot to construct a fully configured `GatewayLLM`, while request execution remains authoritative on the Gateway server.

For application/agent migration steps, see [Migrating agents and providers to Model Gateway](gateway-migration.md).

Task-scoped managed credentials, minting and revocation: [Run tokens](run-tokens.md).

## Managed authentication

Present a Descope human session bearer token to `/admin/managed-auth/*` on a control runtime. This branch checks the current Descope user, organization membership and permissions rather than trusting permission claims in the token

Restricted machine routes instead exchange a registered Descope access key and bind it to a configured managed service

- `GET /admin/managed-auth/capabilities` reports the verified human identity, platform-admin status, request eligibility and per-key actions; `GET /admin/managed-auth/catalog` lists models and run-token operations
- A human requests an additional key for an approved project at `POST /admin/managed-auth/requests`, for example `{"project_id":"<project UUID>","display_name":"...","description":"...","owner":{"kind":"personal","identity":{"issuer":"...","subject":"..."}},"key_limit":null}`. For service ownership, any authenticated human may look up the `service_id` at `GET /admin/managed-auth/services`, but only a platform-admin may create a service at `POST /admin/managed-auth/services` with `{"name":"..."}` (which returns a generated `id`) or request its key with `{"kind":"service","service_id":"<service UUID>"}` as owner
- `description` is required and nonempty for new human and machine requests, persists on the issued key as `KeyDTO.description`, and is stored in `RequestDTO.reason`. The first-stage migration backfills preexisting keys from exactly one approved create request with a nonempty reason and fails if that match is missing or ambiguous. During the overlap, old control writers may issue keys without a description; `KeyDTO.description` is then null. After those writers stop, a second release must backfill overlap keys and enforce `NOT NULL`; that constraint is not part of the first-stage upgrade
- A human may choose an individual cap in the request with `"key_limit":{"period":"monthly","limit_usd":"10"}` instead of null. The requested cap is issued with the key; registered machines request uncapped service keys through their separate route
- For a new project, Tracker in provision mode creates its first key request using its registered machine credential. A first request whose `requested_budget` is at or under `MODEL_GATEWAY_AUTH_AUTO_APPROVAL_THRESHOLD_USD` (default 500) is approved on submission with a monthly policy at that amount, audited by the `managed-auth-system` actor. Above that threshold, the request carries the amount and reason and remains pending until a `platform-admin` approves it at `POST /admin/managed-auth/requests/{request_id}/approve`, supplying `{"initial_policy":{"period":"monthly","base_limit_usd":"100"}}`; the period may be daily, weekly or monthly and the recurring USD base must be positive. A missing or zero base cannot approve the first key. Tracker's existing manual mode remains available, and this branch does not activate a Tracker rollout
- Once the project has an approved policy, any verified human may request an additional personal key directly, without a Tracker lookup or another approval; registered machines may also request additional service keys. Human service-key requests still require platform-admin permission
- The personal owner collects, rotates or revokes their `mgwk_` key with the matching revision; a platform-admin can revoke personal keys and collect, rotate or revoke service keys. For example, `POST /admin/managed-auth/keys/{key_id}/collect` takes `{"expected_revision":1}` (replace `1` with the current revision)
- `POST /admin/managed-auth/machine/requests` takes `{"project_id":"<project UUID>","project_organization_id":"<Tracker project org UUID>","project_environment":"dev","display_name":"...","description":"...","requested_budget":"300","budget_reason":"..."}`; the budget (above 0) and the non-blank reason are required, and binds the request to the registered machine's configured service. The machine may read that service's request at `GET /admin/managed-auth/machine/requests/{request_id}`; stage must be `dev`, `bench`, or `prod`
- Before collecting, the machine reads its service key at `GET /admin/managed-auth/machine/keys/{key_id}` for its status and revision. It then calls `POST /admin/managed-auth/machine/keys/{key_id}/collect` with `{"expected_revision":1}` (replace `1` with the returned revision), receiving a `SecretDTO` with `key` and `secret`; it cannot approve requests, update policies or manage another service
- `/service-auth` accepts explicit child model, embedding and operation scopes. Managed roots publish unrestricted model and operation lists; the child's minted embedding list governs `/embeddings`
- Managed roots still call `/query` directly, but direct `/embeddings` and pass-through calls are denied. Removing direct-root access is separately owned
- Root rotation preserves issued children. Root revocation or personal-owner disablement blocks managed requests after changed authority publishes
- Control publishes managed-key owner, project, optional key cap and shared project-limit authority from PostgreSQL to Redis on an asynchronous one-minute cycle. Query reads published Redis authority, not PostgreSQL per request; existing Redis authority records may still contain a nullable project limit during rollout
- A CRUD success confirms the database write, not immediate serving visibility; publication failures can extend delay beyond one minute. There is no per-key activation state or wait, and query-process initial snapshot readiness is a separate startup gate
- Known humans are reconciled against Descope every 60 seconds; disabling a person blocks personal keys but not service-owned keys. Lookup failure leaves the previous eligibility state until retry
- `MODEL_GATEWAY_API_KEYS` keeps existing named static keys valid for direct provider calls, `/service-auth` mint/revoke, `GET /registry`, `GET /models`, and existing operational routes during transition. They are not managed project-key records and have no spend cap
- The approved DEV migration for Proxy #1593 already assigned existing managed keys to Tracker projects without changing their secret values. This branch neither reassigns those keys nor repeats that migration
- The SDK uses separate `MODEL_GATEWAY_BENCHMARK_ADMISSION_API_KEY`, a configured static operational key, for `/benchmark-runs/*`. It overrides a scoped workload token for those calls

For managed keys, the project association and positive recurring shared project cap are required, while each root's own cap remains optional. Multiple independently managed keys in one project share the project policy; a root's own cap counts queries by that root and all its children, not by person or child

Project policies and recorded spend are separate for DEV and PROD; preview spending policy is not settled

- `GET /admin/managed-auth/projects/{project_id}/policy` returns `project`, required `period` and positive `base_limit_usd`, `revision`, `created_at`, and `updated_at`; policy-less projects still return 404. A `platform-admin` changes the shared recurring base for all keys in that project at `PUT /admin/managed-auth/projects/{project_id}/policy` with `{"expected_revision":1,"period":"monthly","base_limit_usd":"100"}`; the period must be present and the base positive. Historical project allowance snapshots retain nullable fields for earlier events
- A human can choose a root's initial cap in its request; afterward only a `platform-admin` changes or clears it at `PUT /admin/managed-auth/keys/{key_id}/limit` with `{"expected_revision":2,"key_limit":{"period":"daily","limit_usd":"10"}}` or `{"expected_revision":2,"key_limit":null}` (replace `2` with the current key revision). The returned `KeyDTO` includes nullable `key_limit`; the edit advances the revision without rotating the secret, changing the project or altering issued children
- Changing the project period uses recorded project spend for the newly selected period. A key's cap may use a different daily, weekly or monthly period, and already-recorded spend attributed to that root in the selected period counts when its cap is set or changed
- A project member (a `platform-admin`, or a person who owns an unrevoked personal key in the project) may submit a project increase request at `POST /admin/managed-auth/projects/{project_id}/limit-requests` with `{"kind":"temporary","increase_usd":"25","reason":"..."}` or `kind:"permanent"`. The `LimitRequestDTO` includes `id`, `project`, `kind`, `increase_usd`, nullable `period_start`/`period_end`, `reason`, `creator`, `status`, nullable `decided_by`/`decided_at`, and `created_at`
- A project member edits the total at `PUT /admin/managed-auth/projects/{project_id}/budget` with `{"expected_revision":3,"limit_usd":"150","reason":"..."}`:
  - a raise records its reason at once and files a pending request that remembers its target total and the policy revision; approval sets that target, and answers 409 if the budget changed since, so the member files it again;
  - a lower applies now, and answers 409 when the new total is below the spend in the last published snapshot (the admission check still refuses at the total if later spend arrives);
  - the same total with a new reason updates `budget_reason` at once; a new project starts with its request's reason.
- The project system archives a project its service holds a key in at `POST /admin/managed-auth/machine/projects/{project_id}/archive` and restores it at `/unarchive`. While archived, its pending requests leave `GET /admin/managed-auth/limit-requests` and cannot be approved; decided requests stay listed
- Pending requests do not change admission. A `platform-admin` approves at `POST /admin/managed-auth/limit-requests/{request_id}/approve` with `{"expected_current_policy_revision":1}` or denies at its `/deny` route
- An approved temporary increase lasts through its original current-period end even if the base period later changes. An approved permanent increase raises the recurring base; policy and request DTOs are separate, with no managed `/budgets` endpoint
- The exporter sums eligible recorded Redshift cost separately by `project_id` for every approved project and managed root key ID for capped keys, publishing both in the same managed spend snapshot to S3; control copies it to Redis for budgeted queries. Already-recorded project and root/child charges in the selected period count toward newly set limits, but historical email-only charges are assigned to neither
- Each project has one overall total (`total_limit_usd` on its policy). A new project starts with its base as its total; an approved increase, temporary or permanent, raises the total; an admin base change moves the total by the same amount. Migration `a4d7c2e9b315` sets each existing total from its recorded policy states (each state's base for every period it touched, the current period counted whole) plus every temporary increase ever approved, cancelled or not, since spend under it was already allowed, and logs one line per project
- The exporter publishes `total_spend_usd`, each project's spend since its stored `total_started_at` (the start of its first policy period, fixed once), and control copies the snapshot to Redis key `gateway:budgets:{stage}:managed-spend-v2`
- Managed `/query` and pass-through calls check each project's spend since its budget began against its total, and each root-key limit against its period. A refused project call answers 402 with the total, for example `Project total budget of $1000 is spent`. Spending at/above either limit returns HTTP 402 (`exhausted` for the shared project limit, `key_exhausted` for the root-key limit); a missing or unreadable managed spend snapshot returns HTTP 503 for a budgeted project or key. Previous-period or absent spend counts as $0, so a period rollover or new limit doesn't refuse calls before the next export
- The released Redis reader skips spend checking if a published authority has neither limit; an optional key limit still applies independently. Static keys and their digest-bound children have no email spend cap; identity email remains available in usage analytics
- Admission applies only to new requests, not provider-operation caps or in-flight work; snapshot delay can overshoot a limit. A published authority generation does not establish a current spend snapshot or prove all earlier usage arrived

Local verification does not establish a live rollout or end-to-end production billing completion.

Managed-auth rollout needs migrated PostgreSQL for control, Redis for both roles, Descope connectivity and current managed spend for capped projects or keys. This describes the current branch implementation, not a deployed DEV/PROD replacement or an approved existing-key migration; validate authentic integrations before admitting workloads

## Environment variables

### Server

The gateway server uses server-side auth/signing config. Do not set `MODEL_GATEWAY_URL` on the gateway server task; that variable is client-only and `model_gateway.app` clears it at startup to prevent self-proxying.

#### Auth and provider credentials

- `MODEL_GATEWAY_API_KEYS` (**required**): JSON object mapping stable names to existing client keys. Gateway startup fails if unset or empty
- `MODEL_GATEWAY_HMAC_SECRET` (**required**): secret for HMAC-signing pickled fields in history blobs. Gateway startup fails without this so every task can safely return and accept raw history blobs.
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc. (**usually required**): provider API keys for providers served by this gateway. See [Provider API keys](api-keys.md). Per-request `custom_api_key` can supply caller credentials at call time. Requests that set `custom_endpoint` or `registry_key` are rejected.
- `GCP_CREDS`, `GCP_PROJECT_ID`, and `GCP_REGION` (**required for Google Vertex requests**): service-account credentials, project ID, and Vertex region. `config.provider_config.use_vertex=true` selects Vertex; other Google requests use `GOOGLE_API_KEY`

Managed-auth runtime and exporter settings:

- Query and control: `REDIS_URL` and **required** `MODEL_GATEWAY_AUTH_NAMESPACE`, an explicit auth-database namespace (for example `dev-db` for DEV and auth-enabled previews sharing its database; `prod-db` for PROD), not the per-stage `GATEWAY_STAGE`
- Control only: `MODEL_GATEWAY_AUTH_DATABASE_SECRET` (the generated RDS credential JSON), `MODEL_GATEWAY_AUTH_DATABASE_NAME`, `MODEL_GATEWAY_AUTH_POOL_SIZE`, `DESCOPE_PROJECT_ID`, `DESCOPE_EXPECTED_ISSUER`, `DESCOPE_ORGANIZATION_ID`, `DESCOPE_MANAGEMENT_CREDENTIAL`, `MODEL_GATEWAY_AUTH_AUTO_APPROVAL_THRESHOLD_USD` (optional, default 500), `DESCOPE_MACHINE_ACCESS_KEY_ID` (registered Descope machine access-key ID), and `DESCOPE_MACHINE_SERVICE_OWNER_ID` (managed service UUID). The machine access-key ID comes from `ModelGatewayHumanManagement.descope_machine_access_key_id`; set `descope_machine_service_owner_id` to the generated `id` returned by `POST /admin/managed-auth/services` before a control task starts or restarts with that machine service and before Tracker submits a machine request. The machine caller supplies its access-key secret as a Bearer token, not as a Gateway environment value
- Scheduled exporter: `MODEL_GATEWAY_AUTH_DATABASE_SECRET_ARN` and `GATEWAY_MANAGED_BUDGET_SNAPSHOT_KEY`, alongside existing export settings; it reads the generated database credential from Secrets Manager
- Control snapshot reader: `GATEWAY_BUDGET_SNAPSHOT_BUCKET` and `GATEWAY_MANAGED_BUDGET_SNAPSHOT_KEY`; exporter and control require write/read access respectively to `gateway-usage/{stage}/spend/managed-current.json`

The stable environment CDK stack owns a private PostgreSQL instance, its generated `ModelGatewayAuthDatabase` credential secret, and database-client security group for control and the spend exporter. Query tasks receive Redis configuration but not managed-auth database credentials. The release coordinator applies managed-auth Alembic migrations before deploying a persistent-stage candidate. `ModelGatewayHumanManagement` still needs real Descope credentials provisioned before the first managed-auth release; normal query tasks also require `ModelGatewayStartupCanary` with a genuine managed `managed_root`. Only an explicitly bypassed bootstrap release omits the canary. Previews borrow the DEV database, control credential and auth namespace, do not migrate the shared schema, and record usage and publish spend snapshots under their own stage
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

`GATEWAY_STARTUP_CANARY_ENABLED` runs one managed-child local `/query`; `/health/ready` remains at `503` until the query process has observed its first complete published managed authority generation and, when enabled, the canary passes. `/health/live` and control management remain available during that initial wait; readiness does not recheck Redis on every request or wait for individual keys

- `GATEWAY_STARTUP_CANARY_CONTROL_URL`: control-facing Gateway URL for child mint and revoke
- `GATEWAY_STARTUP_CANARY_MANAGED_ROOT`: dedicated managed service root associated with a project, used to mint a child scoped to `query` on `openai/gpt-5.4-nano-2026-03-17`; the shared project and root-key caps can be unset, but a static operational key is not a substitute
- `GATEWAY_RELEASE_ID`: real release ID used in the canary run ID; each startup gets unique run, task and attempt IDs

The query runtime waits for its first published authority generation, then mints on control, queries localhost with the child and revokes its lease on control

For a first managed-auth release of a persistent stage without a managed root, use the one-release `SkipStartupCanary=true` pipeline variable described in [First managed-auth rollout](#first-managed-auth-rollout). For a new preview, use the manual release CLI's `--skip-startup-canary` flag as described there. Either bypass sets the query canary switch to `false` and removes only the query task's canary-root secret injection; it does not turn off query authentication, budget admission, or the usage ledger

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
| `MODEL_GATEWAY_API_KEY` | Yes for model calls | Existing configured key, managed `mgwk_` root, or explicitly issued `mgwt_` child token; the SDK does not mint children automatically |
| `IDENTITY` | No | JSON object attached to gateway query telemetry and usage ledger rows. |
| `MODEL_GATEWAY_BENCHMARK_ADMISSION_API_KEY` | For benchmark admission | Separate named static key configured on the server for operational `/benchmark-runs/*` calls; not the model child token |

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
- `run_id`, `question_id` and `attempt_id` come only from `query(...)` arguments or the run token; there is no env/settings fallback.
- `query_id` is generated per request unless passed explicitly.
- Explicit blank or whitespace-only `run_id`, `question_id`, and `attempt_id` values are rejected.
- Direct blank `query_id` values are treated as absent.
- `attempt_id` names one execution of a task; a whole-task rerun gets a new one.
- `model_library.base.gateway.gateway_token_scope(token)` sends Gateway requests in its `with` block with the child token as request-local `Authorization`, leaving the shared client and `MODEL_GATEWAY_API_KEY` untouched. The Gateway stamps its run, task, identity and attempt when supplied at mint; benchmark services use this scope around evaluations, while their admission client uses the separate static operational key
- The client resolves the four IDs as one internal group and sends them as flat request fields and correlation headers. For `/query` requests with omitted IDs, the gateway fills them for `gateway.query.start` and usage records; native execution and managed credential selection use only the request's IDs.
- The usage ledger stores `attempt_id` on the row beside `run_id` and exports it to the Redshift `usage_events.attempt_id` column, so analytics can tell repeated grading or generation of one task apart by attempt. It is not a ledger index key.

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

- server config secret: `api_keys` (a JSON-encoded name-to-key object for static operations), `hmac_secret`; managed root secrets are collected through human management and not added to this map
- client config secret: `MODEL_GATEWAY_URL`, `MODEL_GATEWAY_API_KEY` (issued child token); benchmark clients additionally set `MODEL_GATEWAY_BENCHMARK_ADMISSION_API_KEY` to a separate named operational key

## Endpoints

| Method | Path                  | Auth | Description                                                                |
| ------ | --------------------- | ---- | -------------------------------------------------------------------------- |
| GET    | `/health/live`        | No   | Liveness check                                                             |
| GET    | `/health/ready`       | No   | Readiness check (auth/signing config; control/combined Redis ping and monitor health; query initial published authority and startup canary when enabled) |
| GET    | `/models`             | Static operational key | List available models with capability flags on control runtime           |
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
| POST   | `/moderation`         | Child token | Run provider moderation |
| POST   | `/service-auth`       | Managed root | Explicitly mint a scoped child token on control runtime |
| POST   | `/service-auth/revoke` | Managed root | Revoke a child lease or the root's run leases on control runtime |
| Various | `/admin/managed-auth/*` | Descope human or restricted machine bearer | Manage services, project policies, key requests and roots on control runtime; machines can request a key for their configured service, read that service's request and key status/revision, and collect its issued root |

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
| `Dockerfile` | Multi-stage build with uv; the runtime stage upgrades distro `ca-certificates` so CA distrusts land without waiting for a base-image refresh |
