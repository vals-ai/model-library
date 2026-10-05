# Run tokens

- Opaque, task-scoped gateway credentials. The executor mints one with its static key and gives the sandbox only the token.
- A token works only on gateways that accept its minting key, so dev tokens miss on prod over the shared Redis.
- Static keys work as before.

## Mint

- `POST /service-auth`, control runtime, static key only:
  ```json
  {
    "run_id": "run-123",
    "task_id": "task_0",
    "allowed_models": ["openai/gpt-4o"],
    "identity": {"benchmark_name": "valsmith", "agent_name": "opencode"},
    "ttl_seconds": 9000
  }
  ```
  - Required:
    - `run_id`, `task_id`
    - `allowed_models`: registry keys, checked at mint
    - `ttl_seconds`: 1 to 604800 (7 days). There is no renew, so size it to outlast the task
  - Optional:
    - `allowed_embedding_models`: `/embeddings` `embedding_model` values
    - `allowed_operations`: narrows the provider routes (`query`, `tokens_count`, `files_upload`, `embeddings`, `moderation`, `audio_transcriptions`, `rate_limit`); all by default
    - `identity`
- Response: `{"token": "mgwt_…", "lease_id": "<sha256 of token>", "expires_at": <unix seconds>}`. Keep the `lease_id`.

## Use

- The sandbox sets `MODEL_GATEWAY_API_KEY` to the token.
- Run tokens may call only the provider routes in `allowed_operations` and `/registry`, which is filtered to `allowed_models`. Every other route returns 403.
- Provider requests must name a model in `allowed_models`, and an embedding model in `allowed_embedding_models`.
- Config overrides follow the static-key rules: any field except `custom_endpoint` and `registry_key`, which are refused for every caller.
- The gateway stamps `run_id`, `question_id` (= `task_id`) and `identity` over what the sandbox sends.
- Run tokens cannot set `token_retry_params`: they reconfigure provider budgets shared across runs.
- Run tokens cannot set `provider_config` on models with a YAML `openrouter_allowed_models` pool: it replaces the entry's whole config, which would drop the pool.

## Revoke

- `POST /service-auth/revoke`, static key only. Takes effect immediately.
- `{"lease_id": "…"}`: 200 `{"revoked": 1}`, 404 unknown or already revoked lease, 403 minted by another key.
- `{"run_id": "…"}`: revokes every token the key minted for the run; 200 `{"revoked": <count>}`. A token minted during the revoke cannot survive it.
