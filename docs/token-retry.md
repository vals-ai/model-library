# Token Retry & Benchmark Queue

Rate-limit-aware request scheduling and multi-run coordination via Redis.

## Overview

The system has two layers:

1. **Token Retry** (`TokenRetrier`) — Tracks available tokens in Redis, queues requests by priority, deducts estimated tokens before each API call, and refunds the difference after. Background loops refill tokens at the provider's rate and correct drift using response headers.

2. **Benchmark Queue** (`benchmark_queue`) — FIFO queue with a token-health concurrency window per model. One run starts first; additional FIFO runs become active heads as observed token health stays high. Supports early release (slot freed when all requests are dispatched, stragglers finish at high priority).

### Run identity (`run_id`)

Inflight tracking, dynamic estimates, and dispatched counting are all keyed by `run_id`. Pass `run_id` explicitly to `LLM.query()` or `Agent.run()`. If omitted, a short random UUID is generated per call.

For multi-process benchmarks where each question spawns a separate process (new Agent + new LLM), pass the same `run_id` to every `query()` / `agent.run()` call so all processes share a single run identity in Redis.

`TokenRetrier` detects whether a run is part of a benchmark queue by checking Redis (`LPOS`) on first use — no contextvar needed.

```text
App calls model.init_token_retry(params)
  → spawns background_loops (refill + correction + cleanup + heartbeat + watchdog)

App calls model.query(input, run_id=..., question_id=...)
  → run_id / question_id auto-generated if not provided
  → creates TokenRetrier(run_id, question_id) per request
  → _pre_function: lpos check (lazy, cached) to detect if queued
  → _pre_function: wait for tokens, deduct, enter inflight
  → _query_impl: call provider API
  → _post_function: refund difference, update dynamic estimate
  → cleanup: remove from inflight

Benchmark runs wrap all of the above:
  async with benchmark_queue(model_key, run_id, ...):
      → enqueues run_id in Redis LIST, blocks until admitted to active_heads
      → yield (app dispatches requests, passing run_id explicitly)
      → TokenRetrier detects queue membership via lpos
      → early release when dispatched count hits total_requests
```

### Source files

| File                                | Role                                                                                         |
| ----------------------------------- | -------------------------------------------------------------------------------------------- |
| `retriers/token/token.py`           | TokenRetrier, init_remaining_tokens, Lua scripts                                             |
| `retriers/token/background.py`      | Background loops (refill, correction, cleanup/reaper, heartbeat, watchdog), standby/takeover |
| `retriers/token/benchmark_queue.py` | FIFO queue, heartbeat, early release                                                         |
| `retriers/token/utils.py`           | Redis key schema, status models, `get_status()`                                              |
| `base/base.py`                      | `LLM.query()` integration, `TokenRetryParams`                                                |
| `agent/agent.py`                    | `Agent.run()` passes `question_id` to all turns                                              |

---

## Token Retry

### Initialization

`LLM.init_token_retry(params)` → `TokenRetrier.init_remaining_tokens(...)`:

1. Sets token count in Redis via `INIT_TOKENS_LUA` (only resets if limit changed or key missing)
2. Stores config hash (limit, tokens_per_second, burst_limit, initialized_at)
3. Checks `task:active`: unchanged duplicate config starts in standby; changed config replaces a same-process loop immediately, but a different process keeps ownership until it releases the key or its TTL expires
4. Spawns one `background_loops` coroutine (in `background.py`) managing a TaskGroup with:
   - **Refill loop** — every 1s, adds `limit / limit_refresh_seconds` tokens (capped at limit) and updates 2-minute and 15-second EWMAs of remaining-token ratio for queue admission
   - **Correction loop** — every 20s, reads provider rate-limit headers and corrects token count down if it's too high
   - **Cleanup/reaper loop** — every 30s, reaps stale entries and checks idle shutdown
   - **Heartbeat** — refreshes `task:active` key and per-worker alive markers only while required workers (`refill`, `reaper`) are alive
   - **Watchdog** — detects if another loop took the active key or heartbeat detected a dead required worker, raises `_Demoted`

**Standby/takeover**: No loop dies permanently. On `_Demoted` or idle shutdown,
the loop releases any ownership it still holds and enters standby. Standby polls
every 10 seconds (`LOOP_POLL_INTERVAL`) and atomically claims `task:active` only
when it is absent or already owned by the same loop ID. On takeover, it reloads
the latest config from Redis.

The cleanup loop reaps stale entries every 30s:

- Per-run inflight entries older than 2h (`INFLIGHT_MAX_AGE`), cleaning up associated metadata hashes; empty per-run ZSETs removed from active_runs
- Priority entries older than 5m (`PRIORITY_STALE_AGE`)

**Idle shutdown** is checked by the cleanup loop: tokens at limit with no active runs for 300s (`FULL_TOKENS_SHUTDOWN`) → raises `_Demoted` to kill the TaskGroup, loop enters standby.

### Priority Queue

Requests are scheduled by priority level (-5 to +5):

| Priority               | Meaning                                                   |
| ---------------------- | --------------------------------------------------------- |
| -5 (`MAX_PRIORITY`)    | Highest — assigned to stragglers from early-released runs |
| 0 (`INITIAL_PRIORITY`) | Default for new requests                                  |
| +5 (`MIN_PRIORITY`)    | Lowest — requests that have retried many times            |

On each retry, priority increments by 1 (toward MIN). A request waits if any lower-numbered priority queue has entries (checked via `HAS_LOWER_PRIORITY_LUA` in a single Redis round-trip).

### Request Lifecycle (\_pre_function → execute → \_post_function)

**\_pre_function** (token deduction):

1. Lazy burst limit read from config hash (once per instance)
2. Lazy queue check: `lpos(queue_key, run_id)` on first call, cached as `_is_queued`
3. If queued and straggler (run_id != queue head via `lindex`), set priority to MAX_PRIORITY
4. Remove from dispatched SET (`srem`) — agentic benchmarks re-enter `_pre_function` for each turn, so dispatched count reflects questions with current turn inflight
5. Register in priority ZSET at current level, store initial per-question metadata hash
6. Loop until tokens deducted (jittered wait: `uniform(TOKEN_WAIT_TIME * 0.5, TOKEN_WAIT_TIME * 1.5)`):
   - Check for lower-priority waiters → if found, sleep and retry
   - Attempt atomic deduction via `DEDUCT_TOKENS_LUA`. The same Redis operation checks the run metadata outcome, token count, and burst limit.
   - If the run outcome is `cancelled` or `failed`, return the terminal sentinel without deducting tokens and raise `BenchmarkRunTerminated` (`NoRetryException`).
   - On success: add to per-run inflight ZSET, register run in active_runs SET, add `question_id` to per-run dispatched SET, store per-question metadata hash
7. Finally (shielded): remove from priority ZSET; if no deduction occurred, delete metadata hash

Missing run metadata and non-terminal outcomes retain the normal deduction path. This keeps non-benchmark requests and completed runs compatible with token retry.

**execute** (wraps the actual API call):

- On success → `_post_function`
- On `RetryException` → increment priority, wait, re-enter `_pre_function`
- On `ImmediateRetryException` → retry immediately (no priority change)
- Finally (shielded): remove from per-run inflight ZSET; if empty, remove run from active_runs; delete per-question metadata hash

**\_post_function** (token adjustment):

1. Compute actual tokens: `total_input + total_output - cache_read`
2. Difference: `estimated - actual`
3. Apply the difference through `REFILL_TOKENS_LUA`; an underestimated request can make the increment negative, but the stored count is clamped to zero rather than preserving negative debt
4. Update dynamic estimate ratio via `ADJUST_RATIO_LUA`

### Dynamic Token Estimation

Learns actual token usage per run and adjusts future estimates via exponential moving average:

- **Ratio**: `actual_tokens / estimated_tokens`, EMA with alpha=0.3
- **Application**: `actual_estimate = ceil(base_estimate * ratio)`
- **Scoping**: keyed by `run_id` — all processes sharing the same `run_id` share one ratio
- **Each benchmark run starts fresh** at ratio 1.0

### question_id (Agentic Support)

For agentic runs where one question makes multiple sequential `model.query()` calls:

- `LLM.query()` accepts an optional `question_id` param (auto-generated 14-char hex UUID if not provided)
- `Agent.run()` accepts `question_id` and passes it to all turns
- `TokenRetrier` uses `question_id` (not `request_id`) for the per-run dispatched SET
- Agentic dispatched cycling: `srem` at start of `_pre_function`, `sadd` after deduction — dispatched count reflects questions with current turn inflight, not total queries made

---

## Benchmark Queue

### FIFO Queue Structure

Each model has a Redis LIST of run IDs plus an `active_heads` ZSET. Runs may dispatch only after they are admitted into `active_heads`. The active-head window starts at 1, scales up by 1 when current and 2-minute EWMA remaining-token ratios are both at least 25%, and scales down by 1 only after current and 15-second EWMA health stay below 15% for 15 seconds. Active runs are never revoked; downscale happens by attrition as active heads finish or early-release.

### Slot Lifecycle

```text
benchmark_queue(model_key, run_id, total_requests=N, early_release=True):
  1. Set alive_key with TTL=300s
  2. Store per-run metadata hash (total_requests, enqueued_at)
  3. Enqueue (idempotent — checks lpos first)
  4. Run active-head control; admitted runs receive notify_key
  5. On notification:
     - proceed only if run_id is present in active_heads
     - yield (app dispatches requests, passing run_id to query()/agent.run())
     - TokenRetrier detects queue membership via lpos on first _pre_function call
  6. Finally (shielded):
     - Cancel heartbeat task
     - Persist the terminal outcome before releasing queue ownership
     - Remove from queue and active_heads, delete alive key
     - Set popped_at if not already set (early release sets it earlier)
     - Set completed_at
     - Run active-head control to backfill if appropriate
```

### Terminal Outcome Boundary

Admission release writes `outcome` to the model-scoped run metadata hash before queue cleanup. That write is the terminal linearization point: a later `DEDUCT_TOKENS_LUA` call sees `cancelled` or `failed` and exits without changing the token or burst counters. The retrier then raises `BenchmarkRunTerminated`, so it does not continue waiting or call the provider, and its existing shielded cleanup removes priority and per-question metadata.

A deduction that completes immediately before the terminal write is already admitted and cannot be revoked by Redis; cancellation of already-running request tasks remains responsible for that boundary. Reacquiring the run initializes a new attempt and clears the previous `outcome`, so retained 24-hour history does not permanently block the run ID.

### Heartbeat (\_heartbeat)

Runs every 2s while in the queue:

- **Refresh**: renew alive_key TTL
- **Active-head control**: adjust the window and admit FIFO waiters up to the window
- **Dead entry cleanup**: remove dead active heads and dead queue entries encountered during admission
- **Self-promotion**: if we're active but missed notification, notify ourselves
- **Early release**: if `early_release=True` and `scard(dispatched) >= total_requests`, pop from queue and active_heads. Remaining inflight requests become stragglers.

### Straggler Detection

After early release, the popped run's requests are still inflight. `TokenRetrier` caches `_is_queued` after the first `lpos` check, so retrying requests from the same instance know they were queued even after the run is removed from the list. On each `_pre_function` call for a queued run:

- Check active-head membership via `zscore(active_heads, run_id)`
- If the run is no longer active → we're a straggler → priority set to MAX_PRIORITY (-5)

This gives stragglers highest priority so they finish quickly.

### `early_release` decision

| Value | Slot release | Retry/straggler behavior | Use for |
| --- | --- | --- | --- |
| `benchmark_queue(..., early_release=True)` (default) | Five seconds after all requests are dispatched | Fast failures can retry during the grace period; later retries become highest-priority stragglers after release | Non-agentic batches where dispatch completion closely tracks finished work |
| `benchmark_queue(..., early_release=False)` | When the context manager exits | The run retains its active-head slot until all managed work finishes | Agentic runs, where every question can dispatch its first query long before all turns finish |

---

## Redis Key Schema

Prefix: `model_library` (`KEY_PREFIX`). Identifiers: `{P}` = provider.model_name, `{K}` = sha256(api_key), `{N}` = priority (-5..5), `{RUN}` = run_id, `{Q}` = question_id.

### Token Retry

| Key                                              | Type   | TTL    | Purpose                                                             |
| ------------------------------------------------ | ------ | ------ | ------------------------------------------------------------------- |
| `{P}:{K}:tokens`                                 | STRING | —      | Remaining token count                                               |
| `{P}:{K}:tokens:limit`                           | STRING | —      | Token limit                                                         |
| `{P}:{K}:tokens:burst`                           | STRING | 1s TTL | Per-second burst counter (auto-expires)                             |
| `{P}:{K}:tokens:config`                          | HASH   | —      | Init config (limit, refresh window, tokens/second, burst limit, initialization time) |
| `{P}:{K}:tokens:last_header`                     | HASH   | 60s    | Most recent provider rate-limit header stored for status visibility |
| `{P}:{K}:tokens:task:active`                     | STRING | 30s    | Active loop owner (loop_id UUID), refreshed by heartbeat            |
| `{P}:{K}:tokens:task:refill`                     | STRING | 30s    | Refill loop alive marker                                            |
| `{P}:{K}:tokens:task:correction`                 | STRING | 30s    | Correction loop alive marker                                        |
| `{P}:{K}:tokens:task:reaper`                     | STRING | 30s    | Cleanup/reaper loop alive marker                                    |
| `{P}:{K}:tokens:dynamic_estimate:{RUN}`          | STRING | 24h    | EMA ratio for dynamic estimation                                    |
| `{P}:{K}:tokens:remaining_ratio_ewma_2m`         | STRING | 300s   | 2-minute EWMA used with current ratio for scale-up                  |
| `{P}:{K}:tokens:remaining_ratio_ewma_15s`        | STRING | 300s   | 15-second EWMA used with current ratio for scale-down               |
| `{P}:{K}:tokens:remaining_ratio_ewma_updated_at` | STRING | 300s   | Last EWMA update timestamp                                          |

### Inflight Tracking

| Key                                 | Type | TTL | Purpose                                                            |
| ----------------------------------- | ---- | --- | ------------------------------------------------------------------ |
| `{P}:{K}:tokens:active_runs`        | SET  | —   | Run IDs with active inflight requests                              |
| `{P}:{K}:tokens:run:{RUN}:inflight` | ZSET | —   | Per-run inflight questions (member=question_id, score=timestamp)   |
| `{P}:{K}:tokens:inflight:{Q}`       | HASH | 2h  | Per-question metadata (estimates, priority, run_id, dispatched_at) |

### Priority Queues

| Key                    | Type | TTL | Purpose                                                              |
| ---------------------- | ---- | --- | -------------------------------------------------------------------- |
| `{P}:{K}:priority:{N}` | ZSET | —   | Requests waiting at priority N (member=question_id, score=timestamp) |

### Benchmark Queue

| Key                                      | Type   | TTL  | Purpose                                                                                   |
| ---------------------------------------- | ------ | ---- | ----------------------------------------------------------------------------------------- |
| `{P}:{K}:benchmark:queue`                | LIST   | 24h  | FIFO run queue                                                                            |
| `{P}:{K}:benchmark:active_heads`         | ZSET   | —    | Admitted runs allowed to dispatch (member=run_id, score=admitted_at)                      |
| `{P}:{K}:benchmark:active_head_window`   | STRING | —    | Desired active-head concurrency window                                                    |
| `{P}:{K}:benchmark:last_scale_up_at`     | STRING | —    | Last scale-up timestamp for global scale-up throttle                                      |
| `{P}:{K}:benchmark:unhealthy_since`      | STRING | —    | Timestamp when token health first fell below scale-down threshold                         |
| `{P}:{K}:benchmark:queue:evict`          | LOCK   | 2s   | Dead head eviction lock; timeout equals `HEARTBEAT_INTERVAL`                              |
| `{P}:{K}:benchmark:alive:{RUN}`          | STRING | 300s | Run heartbeat                                                                             |
| `{P}:{K}:benchmark:notify:{RUN}`         | LIST   | 24h  | Notification channel polled by waiters                                                    |
| `{P}:{K}:benchmark:run:{RUN}`            | HASH   | 24h  | Per-run metadata (enqueued_at, slot_acquired_at, popped_at, completed_at, outcome, total_requests) |
| `{P}:{K}:benchmark:run:{RUN}:dispatched` | SET    | 24h  | Per-run dispatched question_ids                                                           |

---

## Lua Scripts

All scripts run atomically in Redis (no interleaving with other commands).

| Script                        | Keys                           | Args                               | Returns                                                              | Purpose                                                    |
| ----------------------------- | ------------------------------ | ---------------------------------- | -------------------------------------------------------------------- | ---------------------------------------------------------- |
| `DEDUCT_TOKENS_LUA`           | token_key, burst_key, optional run_meta_key | required_tokens, burst_limit | -1 terminal / 0 unavailable / 1 deducted | Atomically reject cancelled/failed runs or check-and-deduct tokens with the burst cap |
| `REFILL_TOKENS_LUA`           | token_key                      | amount, cap                        | new_count                                                            | Apply positive or negative adjustment, clamped to `[0, cap]` |
| `CORRECT_TOKENS_LUA`          | token_key                      | adjusted                           | [corrected, current, adjusted]                                       | Correct down from headers                                  |
| `ADJUST_RATIO_LUA`            | ratio_key                      | observed, alpha                    | [old, new]                                                           | EMA ratio update                                           |
| `HAS_LOWER_PRIORITY_LUA`      | —                              | base, current_p, min_p             | 1/0                                                                  | Check for lower-priority waiters                           |
| `INIT_TOKENS_LUA`             | token_key, limit_key           | new_limit                          | 1/0                                                                  | Init tokens (no version)                                   |
| `CONTROL_AND_ADMIT_HEADS_LUA` | queue/active/token/window keys | thresholds, intervals, notify data | [window, active_count, admitted, scale_up_health, scale_down_health] | Atomically scale active-head window and admit FIFO waiters |
| `EARLY_RELEASE_LUA`           | run_meta, queue, active_heads  | run_id, now                        | run_id/nil                                                           | Mark popped, remove run from queue and active_heads        |

---

## Configuration

### TokenRetryParams

`TokenRetryParams` is caller-owned configuration. Gateway or the local model
resolves it before token retry starts.

| Field                   | Type          | Default  | Purpose                                |
| ----------------------- | ------------- | -------- | -------------------------------------- |
| `input_modifier`        | float         | required | Scale factor for input token estimate  |
| `output_modifier`       | float         | required | Scale factor for output token estimate |
| `use_dynamic_estimate`  | bool          | True     | Enable EMA ratio learning              |
| `limit`                 | `int \| None` | None     | Provider token limit override          |
| `limit_refresh_seconds` | `60`          | 60       | Fixed provider limit refresh interval  |

`limit` may be omitted only when the target Gateway has configured provider
defaults. Direct or local initialization and Gateway deployments without those
defaults require an explicit limit.
`ResolvedTokenRetryParams` is internal. It contains the concrete `limit` and
fixed `limit_refresh_seconds=60`; only the retrier consumes it.

### Constants (token.py)

| Constant               | Value  | Purpose                                                              |
| ---------------------- | ------ | -------------------------------------------------------------------- |
| `TOKEN_WAIT_TIME`      | 10.0s  | Sleep between token deduction attempts (jittered ±50%)               |
| `RETRY_WAIT_TIME`      | 30.0s  | Sleep between actual retries (jittered)                              |
| `MAX_RETRIES`          | 10     | Max retry attempts                                                   |
| `BURST_FRACTION`       | 0.8    | Max 80% of token limit deducted per second (auto-expiring 1s window) |
| `PRIORITY_STALE_AGE`   | 300s   | Reap priority entries after this                                     |
| `INFLIGHT_MAX_AGE`     | 7200s  | Reap stale inflight entries after this                               |
| `REAP_INTERVAL`        | 30s    | How often cleanup/reaper loop runs                                   |
| `DYNAMIC_ESTIMATE_TTL` | 86400s | Expire dynamic estimate ratios for inactive runs                     |

### Constants (background.py)

| Constant               | Value | Purpose                                           |
| ---------------------- | ----- | ------------------------------------------------- |
| `FULL_TOKENS_SHUTDOWN` | 300s  | Idle shutdown threshold (checked by cleanup loop) |
| `REFILL_TASK_TTL`      | 30s   | Background loop alive marker / active key TTL     |
| `LOOP_POLL_INTERVAL`   | 10.0s | Heartbeat, watchdog, and standby poll interval    |

### Constants (benchmark_queue.py)

| Constant                           | Value  | Purpose                                         |
| ---------------------------------- | ------ | ----------------------------------------------- |
| `HEARTBEAT_INTERVAL`               | 2s     | Heartbeat and active-head control frequency     |
| `HEARTBEAT_TTL`                    | 300s   | Alive key expiry                                |
| `EARLY_RELEASE_GRACE_PERIOD`       | 5s     | Wait after all dispatched before releasing slot |
| `MAX_ACTIVE_HEADS`                 | 20     | Maximum active benchmark runs per model         |
| `ACTIVE_HEAD_SCALE_UP_THRESHOLD`   | 0.25   | Health threshold for `window += 1`              |
| `ACTIVE_HEAD_SCALE_UP_INTERVAL`    | 2s     | Minimum time between scale-up increments        |
| `ACTIVE_HEAD_QUEUE_SCAN_LIMIT`     | 100    | Max FIFO entries inspected by each control pass |
| `ACTIVE_HEAD_SCALE_DOWN_THRESHOLD` | 0.15   | Health threshold for sustained downscale        |
| `ACTIVE_HEAD_SCALE_DOWN_INTERVAL`  | 15s    | Time below threshold before `window -= 1`       |
| `HOURS_24`                         | 86400s | TTL for queue and metadata keys                 |

`get_status(queue_entry_limit=None, include_historical=True)` returns the full
queue and historical metadata for diagnostics. The Gateway
`/token-retry/status` endpoint calls it with those defaults, owns the Redis reads,
and caches the response. Clients poll that authenticated endpoint instead of
reading token-retry Redis state directly. The status payload exposes `active_heads`,
`active_head_window`, and per-entry `is_active_head` for displaying multiple
admitted benchmark heads.

---

## Integration

### Gateway client usage

```python
# 1. Resolve a GatewayLLM and attach token-retry parameters for its queries.
await model.init_token_retry(token_retry_params)

# 2. Send the same caller configuration to Gateway for admission.
async with gateway_benchmark_admission(
    model,
    run_id,
    token_retry_params=token_retry_params,
    enabled=use_queue,
    total_requests=len(tasks),
    is_cancelled=is_cancelled,
) as effective_token_limit:
    logger.info("Gateway token limit: %s tokens/minute", effective_token_limit)

    # 3. Query through GatewayLLM while the caller owns execution.
    results = await asyncio.gather(*tasks)
```

`gateway_benchmark_admission` uses the Gateway
`/benchmark-runs/acquire`, `/wait`, `/renew`, and `/release` endpoints. Gateway
resolves caller parameters to an effective limit, initializes the provider
model's token retrier with internal resolved parameters, persists the effective
limit with admission state, and returns it to
the client. Query and admission requests use the same resolver. Gateway owns all
token, queue, and background-loop Redis state; clients do not configure Redis or
inspect those keys.

### Agent integration

```python
# Agent.run() passes run_id and question_id to all turns
await agent.run(input, run_id=run_id, question_id=question_id)

# Internally, Agent passes both to llm.query() on every turn:
response = await self._llm.query(
    input=history,
    tools=self._tool_defs,
    run_id=run_id,
    question_id=question_id,  # same across all turns for this question
)
```

---

## Failure Modes

### Process crash during request execution

- Per-run inflight entry left in ZSET → reaped after 2h, metadata hash cleaned up
- Empty per-run ZSETs removed from active_runs by reaper
- Tokens not refunded → correction loop adjusts via headers
- Priority entry left → reaped after 5m

### Benchmark run crash (kill -9)

- Alive key expires (TTL=300s)
- Other runs' heartbeats detect missing alive key → evict dead entry → notify next
- If crash between lrem and notify: self-promotion in heartbeat detects we're at head

### Benchmark cancellation or failure while token waiters remain

- Admission release persists `outcome=cancelled|failed` before removing queue ownership.
- Waiting retriers reject their next atomic deduction without spending tokens, stop retrying, and run shielded cleanup.
- Requests deducted before that outcome write are already admitted; task cancellation handles those in-flight requests.

### Server restart

- `init_remaining_tokens` checks `task:active`; unchanged duplicate config starts
  in standby.
- Changed config replaces a loop in the same process after cancelling it and
  releasing its key. A different process cannot overwrite the current owner;
  takeover waits for owner release or TTL expiry and the next standby poll.
- Old loops that observe a different owner demote themselves to standby.
- Redis preserves the token count unless the configured limit changes.
- Queue insertion remains idempotent through the `LPOS` check before `RPUSH`.

### Underestimated token usage

- `_post_function` can pass a negative adjustment when actual usage exceeded the
  estimate.
- `REFILL_TOKENS_LUA` clamps the resulting token count to zero; negative debt is
  not stored.
- New requests wait until refill or header correction restores enough tokens.

### Stale dispatched data

- Dispatched SET is per-run (`benchmark:run:{RUN}:dispatched`), so no cross-run contamination
- Slot_acquired guard in heartbeat prevents waiting runs from reading stale counts
