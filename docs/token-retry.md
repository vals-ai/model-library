# Token Retry & Benchmark Queue

Rate-limit-aware request scheduling and multi-run coordination via Redis.

## Overview

The system has two layers:

1. **Token Retry** (`TokenRetrier`) — Tracks available tokens in Redis, queues requests by priority, deducts estimated tokens before each API call, and refunds the difference after. Background loops refill tokens at the provider's rate and correct drift using response headers.

2. **Benchmark Queue** (`benchmark_queue`) — FIFO queue that serializes benchmark runs per model. Only one run dispatches at a time to avoid TPM contention. Supports early release (slot freed when all requests are dispatched, stragglers finish at high priority).

```
App calls model.init_token_retry(params)
  → spawns refill loop + correction loop

App calls model.query(input, question_id=...)
  → creates TokenRetrier per request
  → _pre_function: wait for tokens, deduct, enter inflight
  → _query_impl: call provider API
  → _post_function: refund difference, update dynamic estimate
  → cleanup: remove from inflight

Benchmark runs wrap all of the above:
  async with benchmark_queue(model_key, run_id, ...):
      → sets RunContext(run_id, is_queued=True) via contextvar
      → all TokenRetrier instances read run_id from context
      → early release when dispatched count hits total_requests
```

### Source files

| File | Role |
|------|------|
| `retriers/token/token.py` | TokenRetrier, init_remaining_tokens, background loops, Lua scripts |
| `retriers/token/benchmark_queue.py` | FIFO queue, heartbeat, early release |
| `retriers/token/utils.py` | Redis key schema, status models, `get_status()` |
| `base/base.py` | `LLM.query()` integration, `TokenRetryParams` |
| `agent/agent.py` | `Agent.run()` passes `question_id` to all turns |

---

## Token Retry

### Initialization

`LLM.init_token_retry(params)` → `TokenRetrier.init_remaining_tokens(...)`:

1. Sets token count in Redis via `INIT_TOKENS_LUA` (only resets if limit changed or key missing)
2. Writes a new version UUID — old background loops detect the mismatch and exit
3. Stores config hash (limit, tokens_per_second, version, initialized_at)
4. Spawns two background tasks:
   - **Refill loop** — every 1s, adds `limit / limit_refresh_seconds` tokens (capped at limit)
   - **Correction loop** — every 20s, reads provider rate-limit headers and corrects token count down if it's too high

Both loops exit on:
- **Version mismatch** — another `init_remaining_tokens` call wrote a new UUID
- **Idle shutdown** — tokens at limit with no inflight for 300s (`FULL_TOKENS_SHUTDOWN`)

The refill loop also reaps stale entries every 30s:
- Inflight entries older than 2h (`INFLIGHT_MAX_AGE`)
- Priority entries older than 5m (`PRIORITY_STALE_AGE`)

### Priority Queue

Requests are scheduled by priority level (-5 to +5):

| Priority | Meaning |
|----------|---------|
| -5 (`MAX_PRIORITY`) | Highest — assigned to stragglers from early-released runs |
| 0 (`INITIAL_PRIORITY`) | Default for new requests |
| +5 (`MIN_PRIORITY`) | Lowest — requests that have retried many times |

On each retry, priority increments by 1 (toward MIN). A request waits if any lower-numbered priority queue has entries (checked via `HAS_LOWER_PRIORITY_LUA` in a single Redis round-trip).

### Request Lifecycle (_pre_function → execute → _post_function)

**_pre_function** (token deduction):
1. If straggler (detected in `validate()`), set priority to MAX_PRIORITY
2. Register in priority ZSET at current level
3. Loop until tokens deducted:
   - Check for lower-priority waiters → if found, sleep and retry
   - Attempt atomic deduction via `DEDUCT_TOKENS_LUA`
   - On success: add to inflight ZSET, add `question_id` to per-run dispatched SET, store per-request metadata hash
4. Finally: remove from priority ZSET

**execute** (wraps the actual API call):
- On success → `_post_function`
- On `RetryException` → increment priority, wait, re-enter `_pre_function`
- On `ImmediateRetryException` → retry immediately (no priority change)
- Finally: remove from inflight ZSET, delete per-request metadata hash

**_post_function** (token adjustment):
1. Compute actual tokens: `total_input + total_output - cache_read`
2. Difference: `estimated - actual`
3. Refill difference via `REFILL_TOKENS_LUA` (can be negative = debt)
4. Update dynamic estimate ratio via `ADJUST_RATIO_LUA`

### Dynamic Token Estimation

Learns actual token usage per run and adjusts future estimates via exponential moving average:

- **Ratio**: `actual_tokens / estimated_tokens`, EMA with alpha=0.3
- **Application**: `actual_estimate = ceil(base_estimate * ratio)`
- **Scoping**: Keyed by `run_id` (benchmark) or `instance_id` (non-benchmark)
- **Each benchmark run starts fresh** at ratio 1.0

### question_id (Agentic Support)

For agentic runs where one question makes multiple sequential `model.query()` calls:

- `LLM.query()` accepts an optional `question_id` param (defaults to `query_id`, i.e., 1:1 with requests)
- `Agent.run()` passes a stable `question_id` to all turns
- `TokenRetrier` uses `question_id` (not `request_id`) for the per-run dispatched SET → `sadd` is idempotent across turns
- Result: dispatched count reflects unique questions started, not total queries made

---

## Benchmark Queue

### FIFO Queue Structure

Each model has a Redis LIST of run IDs. Only the head of the queue dispatches requests. Runs behind it block on `blpop` waiting for a notification.

### Slot Lifecycle

```
benchmark_queue(model_key, run_id, total_requests=N, early_release=True):
  1. Set alive_key with TTL=300s
  2. Enqueue (idempotent — checks lpos first)
  3. If first in queue, self-notify; otherwise blpop(notify_key)
  4. On notification:
     - Set benchmark_run_key = run_id
     - Set RunContext contextvar
     - yield (app dispatches requests)
  5. Finally:
     - Cancel heartbeat task
     - Remove from queue
     - Set popped_at if not already set (early release sets it earlier)
     - Set completed_at
     - Atomic cleanup of benchmark_run key (only if still owner)
     - Notify next run
     - Reset contextvar
```

### Heartbeat (_heartbeat)

Runs every 5s while in the queue:

- **Refresh**: renew alive_key TTL
- **Dead head eviction**: if another run is at queue head but its alive_key is gone (crashed), acquire lock, evict, notify next
- **Self-promotion**: if we're at queue head but never got notified (previous head crashed between lrem and notify), notify ourselves
- **Early release**: if `early_release=True` and `scard(dispatched) >= total_requests`, pop from queue and notify next. Remaining inflight requests become stragglers.

### Straggler Detection

After early release, the popped run's requests are still inflight. On each new request's `validate()`:
- Read `benchmark_run_key` from Redis
- If it doesn't match our `run_id` → we're a straggler → priority set to MAX_PRIORITY (-5)

This gives stragglers highest priority so they finish quickly.

### early_release Parameter

`benchmark_queue(..., early_release=True)` (default):
- Heartbeat checks dispatched count and releases slot early

`benchmark_queue(..., early_release=False)`:
- Slot held until the context manager exits (all work done)
- Use for agentic runs where dispatched count hits total_requests immediately (all questions start their first query near-simultaneously)

---

## Redis Key Schema

Prefix: `model_library`. Identifiers: `{P}` = provider.model_name, `{K}` = sha256(api_key), `{R}` = request_id, `{N}` = priority (-5..5), `{RUN}` = run_id, `{Q}` = question_id.

### Token Retry

| Key | Type | TTL | Purpose |
|-----|------|-----|---------|
| `{P}:{K}:tokens` | STRING | — | Remaining token count |
| `{P}:{K}:tokens:limit` | STRING | — | Token limit |
| `{P}:{K}:tokens:version` | STRING | — | Loop version UUID |
| `{P}:{K}:tokens:config` | HASH | — | Init config (limit, tokens_per_second, version, initialized_at) |
| `{P}:{K}:tokens:task:refill` | STRING | 30s | Refill loop alive marker |
| `{P}:{K}:tokens:task:correction` | STRING | 30s | Correction loop alive marker |
| `{P}:{K}:tokens:dynamic_estimate:{RUN}` | STRING | — | EMA ratio for dynamic estimation |

### Inflight Tracking

| Key | Type | TTL | Purpose |
|-----|------|-----|---------|
| `{P}:{K}:tokens:inflight` | ZSET | — | Inflight requests (member=request_id, score=timestamp) |
| `{P}:{K}:tokens:inflight:{R}` | HASH | — | Per-request metadata (estimates, priority, run_id, dispatched_at) |
| `{P}:{K}:tokens:inflight:benchmark_run` | STRING | 24h | Active benchmark run_id |

### Priority Queues

| Key | Type | TTL | Purpose |
|-----|------|-----|---------|
| `{P}:{K}:priority:{N}` | ZSET | — | Requests waiting at priority N (member=request_id, score=timestamp) |

### Benchmark Queue

| Key | Type | TTL | Purpose |
|-----|------|-----|---------|
| `{P}:{K}:benchmark:queue` | LIST | 24h | FIFO run queue |
| `{P}:{K}:benchmark:queue:evict` | LOCK | 5s | Dead head eviction lock |
| `{P}:{K}:benchmark:alive:{RUN}` | STRING | 300s | Run heartbeat |
| `{P}:{K}:benchmark:notify:{RUN}` | LIST | 24h | Notification channel (blpop target) |
| `{P}:{K}:benchmark:run:{RUN}` | HASH | 24h | Per-run metadata (enqueued_at, slot_acquired_at, popped_at, completed_at, total_requests) |
| `{P}:{K}:benchmark:run:{RUN}:dispatched` | SET | 24h | Per-run dispatched question_ids |

---

## Lua Scripts

All scripts run atomically in Redis (no interleaving with other commands).

| Script | Keys | Args | Returns | Purpose |
|--------|------|------|---------|---------|
| `DEDUCT_TOKENS_LUA` | token_key | required_tokens | 1/0 | Check-and-deduct tokens |
| `REFILL_TOKENS_LUA` | token_key | amount, cap | new_count | Refill with cap |
| `CORRECT_TOKENS_LUA` | token_key | adjusted | [corrected, current, adjusted] | Correct down from headers |
| `ADJUST_RATIO_LUA` | ratio_key | observed, alpha | [old, new] | EMA ratio update |
| `HAS_LOWER_PRIORITY_LUA` | — | base, current_p, min_p | 1/0 | Check for lower-priority waiters |
| `INIT_TOKENS_LUA` | token, limit, version | new_limit, new_version | 1/0 | Init tokens + version |
| `CLEANUP_IF_OWNER_LUA` | benchmark_run | run_id | 1/0 | Atomic cleanup if still owner |

---

## Configuration

### TokenRetryParams

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `limit` | int | required | Total tokens per refresh window |
| `limit_refresh_seconds` | int | 60 | Refresh window (always 60 = 1 minute) |
| `input_modifier` | float | required | Scale factor for input token estimate |
| `output_modifier` | float | required | Scale factor for output token estimate |
| `use_dynamic_estimate` | bool | True | Enable EMA ratio learning |

### Constants (token.py)

| Constant | Value | Purpose |
|----------|-------|---------|
| `TOKEN_WAIT_TIME` | 5.0s | Sleep between token deduction attempts (jittered 4-6s) |
| `RETRY_WAIT_TIME` | 20.0s | Sleep between actual retries (jittered) |
| `MAX_RETRIES` | 10 | Max retry attempts |
| `REFILL_TASK_TTL` | 30s | Background loop alive marker TTL |
| `FULL_TOKENS_SHUTDOWN` | 300s | Idle shutdown threshold |
| `PRIORITY_STALE_AGE` | 300s | Reap priority entries after this |
| `INFLIGHT_MAX_AGE` | 7200s | Reap inflight entries after this |
| `REAP_INTERVAL` | 30s | How often reaper runs |
| `LOCK_TIMEOUT` | 10s | Redis lock timeout |

### Constants (benchmark_queue.py)

| Constant | Value | Purpose |
|----------|-------|---------|
| `HEARTBEAT_INTERVAL` | 5s | Heartbeat check frequency |
| `HEARTBEAT_TTL` | 300s | Alive key expiry |
| `HOURS_24` | 86400s | TTL for queue and metadata keys |

---

## Integration

### App-level usage (question_answer_sets.py)

```python
# 1. Init token retry (once per model per run)
model, params = await fetch_model_run_info(run.parameters, user_info)
# ↑ calls model.init_token_retry(TokenRetryParams(...)) inside

# 2. Enter benchmark queue (serializes runs per model)
async with benchmark_queue(
    model._client_registry_key_model_specific,
    str(run.id),
    logger,
    total_requests=len(tests),
    early_release=True,  # False for agentic runs
):
    # 3. Dispatch all questions concurrently
    qa_pair_statuses = await asyncio.gather(*tasks)
    # Each task calls model.query() → TokenRetrier handles scheduling
```

### Agent integration

```python
# Agent.run() passes a stable question_id to all turns
response = await self._llm.query(
    input=history,
    tools=self._tool_defs,
    question_id=question_id,  # same across all turns for this question
)
```

---

## Failure Modes

### Process crash during request execution
- Inflight entry left in ZSET → reaped after 2h
- Tokens not refunded → correction loop adjusts via headers
- Priority entry left → reaped after 5m

### Benchmark run crash (kill -9)
- Alive key expires (TTL=300s)
- Other runs' heartbeats detect missing alive key → evict dead entry → notify next
- If crash between lrem and notify: self-promotion in heartbeat detects we're at head

### Server restart
- `init_remaining_tokens` writes new version UUID → old loops exit
- Token count preserved in Redis (not reset unless limit changed)
- Queue entries idempotent (lpos check before rpush)

### Token debt (negative count)
- Allowed by design — represents overuse beyond estimate
- All requests wait until refill loop brings count positive
- Correction loop prevents sustained drift

### Stale dispatched data
- Dispatched SET is per-run (`benchmark:run:{RUN}:dispatched`), so no cross-run contamination
- Slot_acquired guard in heartbeat prevents waiting runs from reading stale counts
