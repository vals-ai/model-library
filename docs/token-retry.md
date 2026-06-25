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

```
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
      → enqueues run_id in Redis LIST, blocks until head
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
3. Checks `task:active` key — if an active loop exists with the same config, starts in standby; if config changed or no active loop, starts active
4. Spawns one `background_loops` coroutine (in `background.py`) managing a TaskGroup with:
   - **Refill loop** — every 1s, adds `limit / limit_refresh_seconds` tokens (capped at limit) and updates 2-minute and 15-second EWMAs of remaining-token ratio for queue admission
   - **Correction loop** — every 20s, reads provider rate-limit headers and corrects token count down if it's too high
   - **Cleanup/reaper loop** — every 30s, reaps stale entries and checks idle shutdown
   - **Heartbeat** — refreshes `task:active` key and per-worker alive markers only while required workers (`refill`, `reaper`) are alive
   - **Watchdog** — detects if another loop took the active key or heartbeat detected a dead required worker, raises `_Demoted`

**Standby/takeover**: No loop ever dies permanently. On `_Demoted` (watchdog or idle shutdown), loops go to standby. Standby polls every 10s (`LOOP_POLL_INTERVAL`) for `task:active` key expiry. On takeover, re-reads config from Redis.

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
   - Attempt atomic deduction via `DEDUCT_TOKENS_LUA` (checks both token count and burst limit)
   - On success: add to per-run inflight ZSET, register run in active_runs SET, add `question_id` to per-run dispatched SET, store per-question metadata hash
7. Finally (shielded): remove from priority ZSET; if no deduction occurred, delete metadata hash

**execute** (wraps the actual API call):

- On success → `_post_function`
- On `RetryException` → increment priority, wait, re-enter `_pre_function`
- On `ImmediateRetryException` → retry immediately (no priority change)
- Finally (shielded): remove from per-run inflight ZSET; if empty, remove run from active_runs; delete per-question metadata hash

**\_post_function** (token adjustment):

1. Compute actual tokens: `total_input + total_output - cache_read`
2. Difference: `estimated - actual`
3. Refill difference via `REFILL_TOKENS_LUA` (can be negative = debt)
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

```
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
     - Remove from queue and active_heads, delete alive key
     - Set popped_at if not already set (early release sets it earlier)
     - Set completed_at
     - Run active-head control to backfill if appropriate
```

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

### early_release Parameter

`benchmark_queue(..., early_release=True)` (default):

- Heartbeat checks dispatched count and releases slot early
- After all requests are dispatched, waits `EARLY_RELEASE_GRACE_PERIOD` (5s) before releasing — gives fast-failing requests time to retry at normal priority instead of becoming stragglers

`benchmark_queue(..., early_release=False)`:

- Slot held until the context manager exits (all work done)
- Use for agentic runs where dispatched count hits total_requests immediately (all questions start their first query near-simultaneously)

---

## Redis Key Schema

Prefix: `model_library` (`KEY_PREFIX`). Identifiers: `{P}` = provider.model_name, `{K}` = sha256(api_key), `{N}` = priority (-5..5), `{RUN}` = run_id, `{Q}` = question_id.

### Token Retry

| Key                                              | Type   | TTL    | Purpose                                                             |
| ------------------------------------------------ | ------ | ------ | ------------------------------------------------------------------- |
| `{P}:{K}:tokens`                                 | STRING | —      | Remaining token count                                               |
| `{P}:{K}:tokens:limit`                           | STRING | —      | Token limit                                                         |
| `{P}:{K}:tokens:burst`                           | STRING | 1s TTL | Per-second burst counter (auto-expires)                             |
| `{P}:{K}:tokens:config`                          | HASH   | —      | Init config (limit, tokens_per_second, burst_limit, initialized_at) |
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
| `{P}:{K}:benchmark:queue:evict`          | LOCK   | 5s   | Dead head eviction lock                                                                   |
| `{P}:{K}:benchmark:alive:{RUN}`          | STRING | 300s | Run heartbeat                                                                             |
| `{P}:{K}:benchmark:notify:{RUN}`         | LIST   | 24h  | Notification channel polled by waiters                                                    |
| `{P}:{K}:benchmark:run:{RUN}`            | HASH   | 24h  | Per-run metadata (enqueued_at, slot_acquired_at, popped_at, completed_at, total_requests) |
| `{P}:{K}:benchmark:run:{RUN}:dispatched` | SET    | 24h  | Per-run dispatched question_ids                                                           |

---

## Lua Scripts

All scripts run atomically in Redis (no interleaving with other commands).

| Script                        | Keys                           | Args                               | Returns                                                              | Purpose                                                    |
| ----------------------------- | ------------------------------ | ---------------------------------- | -------------------------------------------------------------------- | ---------------------------------------------------------- |
| `DEDUCT_TOKENS_LUA`           | token_key, burst_key           | required_tokens, burst_limit       | 1/0                                                                  | Check-and-deduct tokens with per-second burst cap          |
| `REFILL_TOKENS_LUA`           | token_key                      | amount, cap                        | new_count                                                            | Refill with cap                                            |
| `CORRECT_TOKENS_LUA`          | token_key                      | adjusted                           | [corrected, current, adjusted]                                       | Correct down from headers                                  |
| `ADJUST_RATIO_LUA`            | ratio_key                      | observed, alpha                    | [old, new]                                                           | EMA ratio update                                           |
| `HAS_LOWER_PRIORITY_LUA`      | —                              | base, current_p, min_p             | 1/0                                                                  | Check for lower-priority waiters                           |
| `INIT_TOKENS_LUA`             | token_key, limit_key           | new_limit                          | 1/0                                                                  | Init tokens (no version)                                   |
| `CONTROL_AND_ADMIT_HEADS_LUA` | queue/active/token/window keys | thresholds, intervals, notify data | [window, active_count, admitted, scale_up_health, scale_down_health] | Atomically scale active-head window and admit FIFO waiters |
| `EARLY_RELEASE_LUA`           | run_meta, queue, active_heads  | run_id, now                        | run_id/nil                                                           | Mark popped, remove run from queue and active_heads        |

---

## Configuration

### TokenRetryParams

| Field                   | Type  | Default  | Purpose                                |
| ----------------------- | ----- | -------- | -------------------------------------- |
| `limit`                 | int   | required | Total tokens per refresh window        |
| `limit_refresh_seconds` | int   | 60       | Refresh window (always 60 = 1 minute)  |
| `input_modifier`        | float | required | Scale factor for input token estimate  |
| `output_modifier`       | float | required | Scale factor for output token estimate |
| `use_dynamic_estimate`  | bool  | True     | Enable EMA ratio learning              |

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

### Constants (utils.py)

| Constant                           | Value | Purpose                                            |
| ---------------------------------- | ----- | -------------------------------------------------- |
| `DEFAULT_STATUS_QUEUE_ENTRY_LIMIT` | 100   | Queue entries fetched per model by platform status |

`get_status(queue_entry_limit=None, include_historical=True)` returns the full
queue and historical metadata for diagnostics. Platform polling uses
`queue_entry_limit=100` and `include_historical=True`, so Redis reads only `LLEN`
plus the visible queue prefix for queued runs, while also scanning benchmark run
metadata to show popped/completed history. If a live token-retry run is no longer
in the visible queue, status fetches only that run's exact benchmark metadata key
to distinguish popped queue runs from direct token-retry traffic. Platform status
exposes `active_heads`, `active_head_window`, and per-entry `is_active_head` so
the Requests UI can show multiple admitted benchmark heads instead of the legacy
single head.

---

## Integration

### App-level usage (question_answer_sets.py)

```python
# 1. Init token retry (once per model per run)
model, params = await fetch_model_run_info(run.parameters, user_info)
# ↑ calls model.init_token_retry(TokenRetryParams(...)) inside

run_id = str(run.id)

# 2. Enter benchmark queue (serializes runs per model)
async with benchmark_queue(
    model._client_registry_key_model_specific,
    run_id,
    logger,
    total_requests=len(tests),
    early_release=True,  # False for agentic runs
):
    # 3. Dispatch all questions concurrently
    # Pass run_id explicitly — TokenRetrier detects queue membership via lpos
    qa_pair_statuses = await asyncio.gather(*tasks)
    # Each task calls model.query(run_id=run_id, question_id=...) → TokenRetrier handles scheduling
```

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

### Server restart

- `init_remaining_tokens` checks `task:active` — if config unchanged, new loops start in standby; if config changed, new loops take over immediately
- Old loops detect new `task:active` owner via watchdog → go to standby
- Token count preserved in Redis (not reset unless limit changed)
- Queue entries idempotent (lpos check before rpush)

### Token debt (negative count)

- Allowed by design — represents overuse beyond estimate
- All requests wait until refill loop brings count positive
- Correction loop prevents sustained drift

### Stale dispatched data

- Dispatched SET is per-run (`benchmark:run:{RUN}:dispatched`), so no cross-run contamination
- Slot_acquired guard in heartbeat prevents waiting runs from reading stale counts
