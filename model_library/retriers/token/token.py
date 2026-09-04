import asyncio
import logging
import random
import time
import uuid
from contextlib import suppress
from math import ceil, floor
from typing import Any, Callable, Coroutine

import model_library.telemetry as telemetry
from model_library.base.base import QueryResult
from model_library.rate_limits import RateLimit
from model_library.exceptions import NoRetryException, exception_message
from model_library.retriers.base import BaseRetrier
from model_library.retriers.token import utils
from model_library.retriers.token.utils import KEY_PREFIX
from model_library.retriers.utils import jitter

# background loop settings
PRIORITY_STALE_AGE: int = (
    300  # 5 minutes — reap priority entries with no refresh (matches heartbeat TTL)
)
INFLIGHT_MAX_AGE: int = 7200  # 2 hours — reap stale inflight entries (this doesn't actually kill the task, just the entry)
REAP_INTERVAL: int = 30  # seconds between stale entry reap checks

RETRY_WAIT_TIME: float = 30.0
TOKEN_WAIT_TIME: float = 10.0

MAX_PRIORITY: int = -5
INITIAL_PRIORITY: int = 0
MIN_PRIORITY: int = 5

MAX_RETRIES: int = 10

DYNAMIC_ESTIMATE_TTL: int = (
    86400  # 24 hours — expire dynamic estimate ratios for inactive runs
)

BURST_FRACTION: float = 0.8  # max 80% of token limit deducted per second

# Retain admissions for one extra second as a conservative buffer between
# Redis admission and the expected provider dispatch.
REQUEST_WINDOW_SECONDS: int = 61
REQUEST_WINDOW_MILLISECONDS: int = REQUEST_WINDOW_SECONDS * 1000
REQUEST_LOG_TTL_MILLISECONDS: int = REQUEST_WINDOW_MILLISECONDS * 2

_BACKGROUND_LOOP_TASKS: dict[str, asyncio.Task[None]] = {}
_BACKGROUND_LOOP_LOCKS: dict[str, asyncio.Lock] = {}


class BenchmarkRunTerminated(NoRetryException):
    def __init__(self, run_id: str, outcome: str):
        self.run_id = run_id
        self.outcome = outcome
        super().__init__(f"Benchmark run {run_id} terminated with outcome {outcome}")


# Lua: atomically admit one provider request and, when required > 0,
# deduct its pessimistic token estimate. Redis TIME is authoritative across
# Gateway workers. Returns {admitted, blocked_reason, retry_after_ms, limit},
# where blocked_reason is 1=tokens, 2=requests, and 4=terminal benchmark run.
# A nonpositive configured limit leaves the rolling window untouched.
# KEYS[1] = token key, KEYS[2] = burst key, KEYS[3] = request ZSET,
# KEYS[4] = config hash, KEYS[5] = optional benchmark run metadata.
# ARGV[1] = required tokens, ARGV[2] = burst limit, ARGV[3] = unique member,
# ARGV[4] = rolling window ms, ARGV[5] = request-log TTL ms.
ADMIT_REQUEST_LUA = """
local run_meta_key = KEYS[5]
if run_meta_key and run_meta_key ~= '' then
    local outcome = redis.call('HGET', run_meta_key, 'outcome')
    if outcome == 'cancelled' or outcome == 'failed' then
        return {0, 4, 0, 0}
    end
end

-- RPM is checked first so a saturated window is reported as such rather than
-- being masked by a token shortage, and its slot is only recorded below once
-- TPM has also cleared. A request that passes RPM but fails TPM therefore
-- leaves the window untouched and is free to retry.
local request_limit = tonumber(
    redis.call('HGET', KEYS[4], 'requests_per_minute') or '0'
) or 0
local now_ms = 0

if request_limit > 0 then
    local now_parts = redis.call('TIME')
    now_ms = (tonumber(now_parts[1]) * 1000) + math.floor(tonumber(now_parts[2]) / 1000)
    local cutoff = now_ms - tonumber(ARGV[4])
    redis.call('ZREMRANGEBYSCORE', KEYS[3], '-inf', cutoff)

    -- a replayed permit already holds both budgets
    if redis.call('ZSCORE', KEYS[3], ARGV[3]) then
        return {1, 0, 0, request_limit}
    end

    local request_count = tonumber(redis.call('ZCARD', KEYS[3]))
    if request_count >= request_limit then
        local oldest = redis.call('ZRANGE', KEYS[3], 0, 0, 'WITHSCORES')
        local retry_after_ms = 1
        if oldest[2] then
            retry_after_ms = math.max(
                1,
                math.floor(tonumber(oldest[2]) + tonumber(ARGV[4]) - now_ms)
            )
        end
        return {0, 2, retry_after_ms, request_limit}
    end
end

local required = tonumber(ARGV[1])
if required > 0 then
    local remaining = tonumber(redis.call('GET', KEYS[1]))
    if remaining < required then
        return {0, 1, 0, request_limit}
    end

    local burst_limit = tonumber(ARGV[2])
    local used = tonumber(redis.call('GET', KEYS[2]) or '0')
    if used + required > burst_limit then
        return {0, 1, 0, request_limit}
    end
end

-- both budgets have room: commit them together
if required > 0 then
    redis.call('DECRBY', KEYS[1], required)
    redis.call('INCRBY', KEYS[2], required)
    if redis.call('TTL', KEYS[2]) == -1 then
        redis.call('PEXPIRE', KEYS[2], 1000)
    end
end

if request_limit > 0 then
    redis.call('ZADD', KEYS[3], now_ms, ARGV[3])
    redis.call('PEXPIRE', KEYS[3], ARGV[5])
end

return {1, 0, 0, request_limit}
"""

# Lua: atomic refill with cap. Returns new token count.
# KEYS[1] = token key, ARGV[1] = refill amount, ARGV[2] = cap limit
REFILL_TOKENS_LUA = """
local current = tonumber(redis.call('INCRBY', KEYS[1], ARGV[1]))
if current > tonumber(ARGV[2]) then
    redis.call('SET', KEYS[1], ARGV[2])
    return tonumber(ARGV[2])
end
if current < 0 then
    redis.call('SET', KEYS[1], 0)
    return 0
end
return current
"""

# Lua: atomic correct-down. Sets key to adjusted value only if it's lower than current.
# KEYS[1] = token key, ARGV[1] = adjusted value
# Returns: 1 if corrected, 0 if skipped, plus the current and adjusted values as a table
CORRECT_TOKENS_LUA = """
local current = tonumber(redis.call('GET', KEYS[1]))
local adjusted = tonumber(ARGV[1])
if adjusted < current then
    redis.call('SET', KEYS[1], adjusted)
    return {1, current, adjusted}
end
return {0, current, adjusted}
"""

# Lua: atomic EMA ratio update. GET old ratio, compute new, SET.
# KEYS[1] = ratio key, ARGV[1] = observed_ratio, ARGV[2] = alpha
ADJUST_RATIO_LUA = """
local current = tonumber(redis.call('GET', KEYS[1])) or 1.0
local observed = tonumber(ARGV[1])
local alpha = tonumber(ARGV[2])
local new_ratio = (observed * alpha) + (current * (1 - alpha))
redis.call('SET', KEYS[1], new_ratio)
return {tostring(current), tostring(new_ratio)}
"""


# Lua: atomic init. Checks old limit, sets version, conditionally sets tokens+limit.
# KEYS[1] = token key, KEYS[2] = limit key, KEYS[3] = version key
# ARGV[1] = new limit, ARGV[2] = new version
# Returns 1 if tokens were (re)set, 0 if limit unchanged and tokens existed.
HAS_LOWER_PRIORITY_LUA = """
local base = ARGV[1]
local current_priority = tonumber(ARGV[2])
local min_priority = tonumber(ARGV[3])
for p = min_priority, current_priority - 1 do
    if redis.call('ZCARD', base .. ':priority:' .. p) > 0 then
        return 1
    end
end
return 0
"""

INIT_TOKENS_LUA = """
local old_limit = tonumber(redis.call('GET', KEYS[2]) or 0)
if old_limit ~= tonumber(ARGV[1]) or redis.call('EXISTS', KEYS[1]) == 0 then
    redis.call('SET', KEYS[1], ARGV[1])
    redis.call('SET', KEYS[2], ARGV[1])
    return 1
end
return 0
"""


class TokenRetrier(BaseRetrier):
    """
    Token-based retry strategy to pessimistically fill TPM
    Predict the number of tokens required for a query, send requests to respect the rate limit,
    then adjusts the estimate based on actual usage
    """

    @staticmethod
    def get_token_key(client_registry_key: tuple[str, str]) -> str:
        """Get the key which stores remaining tokens"""
        return f"{KEY_PREFIX}:{client_registry_key[0]}:{client_registry_key[1]}:tokens"

    @staticmethod
    def get_priority_key(client_registry_key: tuple[str, str], priority: int) -> str:
        """Get the key which stores the amount of tasks waiting for a given priority"""
        return f"{KEY_PREFIX}:{client_registry_key[0]}:{client_registry_key[1]}:priority:{priority}"

    @staticmethod
    def get_request_key(client_registry_key: tuple[str, str]) -> str:
        """Get the rolling request-admission log key."""
        return (
            f"{KEY_PREFIX}:{client_registry_key[0]}:{client_registry_key[1]}:requests"
        )

    @staticmethod
    async def init_remaining_tokens(
        client_registry_key: tuple[str, str],
        limit: int | None,
        limit_refresh_seconds: int,
        logger: logging.Logger,
        get_rate_limit_func: Callable[[], Coroutine[Any, Any, RateLimit | None]],
        requests_per_minute: int | None = None,
    ) -> None:
        """
        Initialize remaining tokens in storage and start background refill process.

        A model without a configured TPM limit has no token bucket to initialize
        or refill: this writes only the RPM config (when configured) and returns
        without starting the background loop, which exists solely to refill and
        provider-correct the token bucket.
        """

        await utils.validate_redis_client()

        key = TokenRetrier.get_token_key(client_registry_key)

        config_mapping: dict[str, int | float] = {}
        if requests_per_minute is not None:
            config_mapping.update(
                {
                    "requests_per_minute": requests_per_minute,
                    "request_window_seconds": REQUEST_WINDOW_SECONDS,
                }
            )

        if limit is None:
            if config_mapping:
                await utils.redis_client.hset(f"{key}:config", mapping=config_mapping)
            return

        from model_library.retriers.token.background import LoopConfig, background_loops

        limit_key = f"{key}:limit"
        await utils.redis_client.eval(INIT_TOKENS_LUA, 2, key, limit_key, limit)

        tokens_per_second = floor(limit / limit_refresh_seconds)
        burst_limit = floor(limit * BURST_FRACTION)

        # if an active loop exists with the same config, start in standby
        # if config changed or no active loop, start active (take over immediately)
        active = await utils.redis_client.get(f"{key}:task:active")
        config_changed = False
        if active:
            existing = await utils.redis_client.hgetall(f"{key}:config")
            config_changed = (
                int(existing.get("limit", 0)) != limit
                or int(existing.get("tokens_per_second", 0)) != tokens_per_second
            )

        config_mapping.update(
            {
                "limit": limit,
                "limit_refresh_seconds": limit_refresh_seconds,
                "tokens_per_second": tokens_per_second,
                "burst_limit": burst_limit,
                "initialized_at": time.time(),
            }
        )
        await utils.redis_client.hset(f"{key}:config", mapping=config_mapping)

        lock = _BACKGROUND_LOOP_LOCKS.setdefault(key, asyncio.Lock())
        async with lock:
            existing_task = _BACKGROUND_LOOP_TASKS.get(key)
            if existing_task is not None:
                if existing_task.done():
                    del _BACKGROUND_LOOP_TASKS[key]
                elif not config_changed:
                    logger.debug(
                        f"Token retry background loop already running for {key}; skipping duplicate init"
                    )
                    return
                else:
                    existing_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await existing_task
                    if _BACKGROUND_LOOP_TASKS.get(key) is existing_task:
                        del _BACKGROUND_LOOP_TASKS[key]

            cfg = LoopConfig(key, limit, tokens_per_second)
            task = asyncio.create_task(
                background_loops(
                    cfg,
                    get_rate_limit_func,
                    logger,
                    standby=active is not None and not config_changed,
                )
            )
            _BACKGROUND_LOOP_TASKS[key] = task

            def _remove_completed_task(done_task: asyncio.Task[None]) -> None:
                if _BACKGROUND_LOOP_TASKS.get(key) is done_task:
                    del _BACKGROUND_LOOP_TASKS[key]
                    if not lock.locked():
                        _BACKGROUND_LOOP_LOCKS.pop(key, None)

            task.add_done_callback(_remove_completed_task)

    def __init__(
        self,
        logger: logging.Logger,
        max_tries: int | None = MAX_RETRIES,
        max_time: float | None = None,
        retry_callback: Callable[[int, Exception | None, float, float], None]
        | None = None,
        *,
        client_registry_key: tuple[str, str],
        run_id: str,
        question_id: str,
        estimate_input_tokens: int,
        estimate_output_tokens: int,
        use_dynamic_estimate: bool = True,
        manages_tokens: bool = True,
        cache_read_counts_toward_limit: bool = True,
    ):
        super().__init__(
            strategy="token",
            logger=logger,
            max_tries=max_tries,
            max_time=max_time,
            retry_callback=retry_callback,
        )

        self.client_registry_key = client_registry_key
        # RPM-only policy: there is no TPM bucket to deduct from, refill, or
        # adjust after the fact — only request-rate admission applies.
        self._manages_tokens = manages_tokens
        self._cache_read_counts_toward_limit = cache_read_counts_toward_limit

        self.estimate_input_tokens = estimate_input_tokens
        self.estimate_output_tokens = estimate_output_tokens
        self.estimate_total_tokens = estimate_input_tokens + estimate_output_tokens
        self.actual_estimate_total_tokens = (
            self.estimate_total_tokens
        )  # when multiplying base estimate_total_tokens by ratio

        self.priority = INITIAL_PRIORITY

        self._base_key = (
            f"{KEY_PREFIX}:{client_registry_key[0]}:{client_registry_key[1]}"
        )
        self.token_key = TokenRetrier.get_token_key(client_registry_key)
        self.request_key = TokenRetrier.get_request_key(client_registry_key)
        self._run_id = run_id
        self._telemetry_question_id = question_id
        self._question_id = f"{run_id}:{question_id}"
        self._is_queued: bool | None = None  # lazy: set on first _pre_function call
        self._burst_limit: int | None = None  # lazy: read from config

        # per-run inflight tracking
        self._active_runs_key = f"{self.token_key}:active_runs"
        self._run_inflight_key = f"{self.token_key}:run:{self._run_id}:inflight"
        # per-question metadata hash
        # NOTE: each query's metadata in a agentic question gets overwritten
        self._question_meta_key = f"{self.token_key}:inflight:{self._question_id}"

        # benchmark keys
        self._queue_key = f"{self._base_key}:benchmark:queue"
        self._active_heads_key = f"{self._base_key}:benchmark:active_heads"
        self._run_meta_key = f"{self._base_key}:benchmark:run:{self._run_id}"

        self.dynamic_estimate_key = (
            f"{self.token_key}:dynamic_estimate:{self._run_id}"
            if use_dynamic_estimate and manages_tokens
            else None
        )
        telemetry.set_attributes(
            {
                "retry_queue.mode": "enabled",
                "retry_queue.question_ref": self._question_id,
                "retry_queue.dynamic_estimate.mode": telemetry.mode_attribute(
                    use_dynamic_estimate
                ),
                "retry_queue.estimate_input_tokens": self.estimate_input_tokens,
                "retry_queue.estimate_output_tokens": self.estimate_output_tokens,
                "retry_queue.estimate_total_tokens": self.estimate_total_tokens,
            }
        )

    def _telemetry_ids(self) -> dict[str, str]:
        return {
            "run_id": self._run_id,
            "question_id": self._telemetry_question_id,
            "retry_queue.question_ref": self._question_id,
        }

    async def _calculate_wait_time(
        self, attempt: int, exception: Exception | None = None
    ) -> float:
        """Wait time between retries"""
        return jitter(RETRY_WAIT_TIME)

    async def _on_retry(
        self, exception: Exception | None, elapsed: float, wait_time: float
    ) -> None:
        """Log retry attempt and update priority/attempts only on actual exceptions"""

        self.priority = min(MIN_PRIORITY, self.priority + 1)

        logger_msg = (
            f"[Token Retry] | Attempt: {self.attempts}/{self.max_tries} | Elapsed: {elapsed:.1f}s | "
            f"Next wait: {wait_time:.1f}s | Priority: {self.priority} ({MAX_PRIORITY}-{MIN_PRIORITY}) | "
            f"Exception: {exception_message(exception)}"
        )

        self.logger.info(logger_msg)

        retry_attributes: dict[str, object | None] = {
            **self._telemetry_ids(),
            "retry_queue.attempt": self.attempts,
            "retry_queue.max_tries": self.max_tries,
            "retry_queue.elapsed_seconds": elapsed,
            "retry_queue.next_wait_seconds": wait_time,
            "retry_queue.priority": self.priority,
            "exception.type": type(exception).__name__ if exception else None,
        }
        telemetry.log_sentry_info(
            logger_msg,
            {"retry.strategy": self.strategy, **retry_attributes},
        )
        telemetry.add_event("retry_queue.provider_retry", retry_attributes)

        if self.retry_callback:
            self.retry_callback(self.attempts, exception, elapsed, wait_time)

    async def _has_lower_priority_waiting(self) -> bool:
        """Check if there are lower priority requests waiting"""
        result = await utils.redis_client.eval(
            HAS_LOWER_PRIORITY_LUA, 0, self._base_key, self.priority, MAX_PRIORITY
        )
        return bool(result)

    async def _pre_function(self) -> None:
        """
        Loop until sufficient tokens are available.
        Acquires priority semaphore, checks for lower priority requests, deducts tokens from Redis.
        Logs token waits but does not count as retry attempts.
        """

        # lazy: read burst limit once from config
        if self._burst_limit is None:
            config = await utils.redis_client.hgetall(f"{self.token_key}:config")
            bl = config.get("burst_limit")
            self._burst_limit = (
                int(bl) if bl else floor(self.estimate_total_tokens * 10)
            )

        # lazy: check once if this run_id is in the benchmark queue
        if self._is_queued is None:
            pos = await utils.redis_client.lpos(self._queue_key, self._run_id)
            self._is_queued = pos is not None

        # straggler: my benchmark is no longer an active head (early-released)
        if self._is_queued:
            is_active_head = await utils.redis_client.zscore(
                self._active_heads_key, self._run_id
            )
            if is_active_head is None:
                self.priority = MAX_PRIORITY

        # remove from dispatched so early release knows we're waiting again
        # (agentic benchmarks re-enter _pre_function for each turn)
        await utils.redis_client.srem(
            f"{self._run_meta_key}:dispatched", self._question_id
        )

        priority_key = TokenRetrier.get_priority_key(
            self.client_registry_key, self.priority
        )

        # let storage know we are waiting at this priority and expose metadata
        # for the status endpoint (pipelined — single round-trip).
        async with utils.redis_client.pipeline(transaction=False) as pipe:
            pipe.zadd(priority_key, {self._question_id: time.time()})
            pipe.hset(  # pyright: ignore[reportUnknownMemberType]
                self._question_meta_key,
                mapping={"run_id": self._run_id, "priority": str(self.priority)},
            )
            pipe.expire(self._question_meta_key, INFLIGHT_MAX_AGE)
            await pipe.execute()
        self.logger.debug(f"priority: {self.priority}, waiting: {priority_key}")
        wait_attrs = {
            **self._telemetry_ids(),
            "retry_queue.mode": "enabled",
            "retry_queue.priority": self.priority,
            "retry_queue.estimate_input_tokens": self.estimate_input_tokens,
            "retry_queue.estimate_output_tokens": self.estimate_output_tokens,
            "retry_queue.estimate_total_tokens": self.estimate_total_tokens,
            "retry_queue.dynamic_estimate.mode": telemetry.mode_attribute(
                self.dynamic_estimate_key is not None
            ),
            "retry_queue.benchmark_queue.mode": telemetry.mode_attribute(
                self._is_queued is True
            ),
        }
        telemetry.set_attributes(wait_attrs)
        telemetry.add_event("retry_queue.wait_start", wait_attrs)

        _deducted = False
        last_blocked_event_at = 0.0
        last_insufficient_event_at = 0.0
        last_request_event_at = 0.0
        try:
            while True:
                now = time.time()
                wait_time = random.uniform(TOKEN_WAIT_TIME * 0.5, TOKEN_WAIT_TIME * 1.5)

                # refresh timestamp so reaper knows we're alive
                await utils.redis_client.zadd(
                    priority_key, {self._question_id: time.time()}
                )

                # if there is a task with lower priority waiting, go back to waiting
                if await self._has_lower_priority_waiting():
                    self.logger.debug(
                        f"[Token Wait] Lower priority requests exist, waiting {wait_time:.1f}s | "
                        f"Priority: {self.priority}"
                    )
                    if now - last_blocked_event_at >= 30:
                        last_blocked_event_at = now
                        telemetry.add_event(
                            "retry_queue.lower_priority_blocked",
                            {
                                **self._telemetry_ids(),
                                "retry_queue.priority": self.priority,
                                "retry_queue.next_wait_seconds": wait_time,
                            },
                        )
                else:
                    # dynamically adjust actual estimate tokens based on past requests
                    if self.dynamic_estimate_key:
                        # NOTE: ok to not lock, don't need precise ratio
                        ratio = float(
                            await utils.redis_client.get(self.dynamic_estimate_key)
                            or 1.0
                        )
                        self.actual_estimate_total_tokens = ceil(
                            self.estimate_total_tokens * ratio
                        )
                        self.logger.debug(
                            f"Adjusted actual estimate tokens to {self.actual_estimate_total_tokens} using ratio {ratio}"
                        )

                    # One admission for both budgets. Models without a configured
                    # RPM skip the rolling window and pay only the token cost.
                    admission = await utils.redis_client.eval(
                        ADMIT_REQUEST_LUA,
                        5,
                        self.token_key,
                        f"{self.token_key}:burst",
                        self.request_key,
                        f"{self.token_key}:config",
                        self._run_meta_key,
                        self.actual_estimate_total_tokens,
                        self._burst_limit,
                        uuid.uuid4().hex,
                        REQUEST_WINDOW_MILLISECONDS,
                        REQUEST_LOG_TTL_MILLISECONDS,
                    )
                    (
                        admitted,
                        blocked_reason,
                        retry_after_ms,
                        request_limit,
                    ) = map(int, admission)

                    if blocked_reason == 4:
                        meta = await utils.redis_client.hgetall(self._run_meta_key)
                        outcome = meta.get("outcome", "failed")
                        telemetry.add_event(
                            "retry_queue.benchmark_run_terminated",
                            {
                                **self._telemetry_ids(),
                                "retry_queue.benchmark_outcome": outcome,
                            },
                        )
                        raise BenchmarkRunTerminated(self._run_id, outcome)
                    if admitted:
                        _deducted = True
                        # per-run inflight tracking (pipelined — single round-trip)
                        now = time.time()
                        dispatched_key = f"{self._run_meta_key}:dispatched"
                        async with utils.redis_client.pipeline(
                            transaction=False
                        ) as pipe:
                            pipe.zadd(self._run_inflight_key, {self._question_id: now})
                            pipe.sadd(self._active_runs_key, self._run_id)
                            pipe.hset(  # pyright: ignore[reportUnknownMemberType]
                                self._question_meta_key,
                                mapping={
                                    "estimate_input": self.estimate_input_tokens,
                                    "estimate_output": self.estimate_output_tokens,
                                    "estimate_total": self.actual_estimate_total_tokens,
                                    "priority": self.priority,
                                    "attempts": self.attempts,
                                    "run_id": self._run_id,
                                    "dispatched_at": now,
                                },
                            )
                            pipe.expire(self._question_meta_key, INFLIGHT_MAX_AGE)
                            pipe.sadd(dispatched_key, self._question_id)
                            pipe.expire(dispatched_key, INFLIGHT_MAX_AGE)
                            await pipe.execute()
                        self.logger.debug(
                            f"Deducted {self.actual_estimate_total_tokens} tokens from {self.token_key}"
                        )
                        telemetry.add_event(
                            "retry_queue.tokens_deducted",
                            {
                                **self._telemetry_ids(),
                                "retry_queue.priority": self.priority,
                                "retry_queue.estimated_tokens": self.actual_estimate_total_tokens,
                                "retry_queue.attempt": self.attempts,
                                "retry_queue.request_kind": "generation",
                                "retry_queue.requests_per_minute": request_limit
                                or None,
                            },
                        )
                        return

                    if blocked_reason == 2:
                        wait_time = (retry_after_ms / 1000) + random.uniform(0.01, 0.1)
                        self.logger.debug(
                            f"[Request Wait] generation at {request_limit} RPM, "
                            f"waiting {wait_time:.3f}s | Priority: {self.priority}"
                        )
                        if now - last_request_event_at >= 30:
                            last_request_event_at = now
                            telemetry.add_event(
                                "retry_queue.insufficient_requests",
                                {
                                    **self._telemetry_ids(),
                                    "retry_queue.priority": self.priority,
                                    "retry_queue.request_kind": "generation",
                                    "retry_queue.requests_per_minute": request_limit,
                                    "retry_queue.next_wait_seconds": wait_time,
                                },
                            )
                    else:
                        self.logger.debug(
                            f"[Token Wait] Insufficient tokens, waiting {wait_time:.1f}s | "
                            f"estimate_tokens: {self.actual_estimate_total_tokens} | "
                            f"Priority: {self.priority}"
                        )
                        if now - last_insufficient_event_at >= 30:
                            last_insufficient_event_at = now
                            telemetry.add_event(
                                "retry_queue.insufficient_tokens",
                                {
                                    **self._telemetry_ids(),
                                    "retry_queue.priority": self.priority,
                                    "retry_queue.estimated_tokens": self.actual_estimate_total_tokens,
                                    "retry_queue.next_wait_seconds": wait_time,
                                },
                            )

                # Zzz
                self.logger.debug(f"Sleeping for {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
        finally:

            async def _pre_cleanup() -> None:
                # let storage know we are done waiting at this priority
                await utils.redis_client.zrem(priority_key, self._question_id)
                if not _deducted:
                    await utils.redis_client.delete(self._question_meta_key)

            try:
                await asyncio.shield(_pre_cleanup())
            except Exception as exc:
                telemetry.record_exception(
                    exc,
                    {
                        **self._telemetry_ids(),
                        "retry_queue.cleanup.phase": "pre_function",
                    },
                )
                telemetry.add_event(
                    "retry_queue.wait_cleanup_failed",
                    self._telemetry_ids(),
                )
                raise
            telemetry.add_event(
                "retry_queue.wait_cleanup_done",
                {
                    **self._telemetry_ids(),
                    "retry_queue.deducted": _deducted,
                },
            )

    async def _adjust_dynamic_estimate_ratio(self, actual_tokens: int) -> None:
        if not self.dynamic_estimate_key:
            return

        observed_ratio = actual_tokens / self.estimate_total_tokens

        alpha = 0.3

        # atomic EMA update via Lua (no lock needed)
        result = await utils.redis_client.eval(
            ADJUST_RATIO_LUA, 1, self.dynamic_estimate_key, observed_ratio, alpha
        )
        current_ratio = float(result[0])
        new_ratio = float(result[1])

        await utils.redis_client.expire(self.dynamic_estimate_key, DYNAMIC_ESTIMATE_TTL)

        self.logger.info(
            f"[Token Ratio] {self.token_key} | Observed: {observed_ratio:.5f} | "
            f"Global Ratio: {current_ratio:.5f} -> {new_ratio:.5f}"
        )
        telemetry.add_event(
            "retry_queue.dynamic_estimate_adjusted",
            {
                **self._telemetry_ids(),
                "retry_queue.observed_ratio": observed_ratio,
                "retry_queue.previous_ratio": current_ratio,
                "retry_queue.new_ratio": new_ratio,
            },
        )

    async def _post_function(self, result: tuple[QueryResult, float]) -> None:
        """Adjust token estimate based on actual usage"""

        if not self._manages_tokens:
            # RPM-only: no token bucket was initialized, so there is nothing
            # to refill or correct.
            return

        metadata = result[0].metadata

        countable_input_tokens = metadata.total_input_tokens
        if not self._cache_read_counts_toward_limit:
            countable_input_tokens -= metadata.cache_read_tokens or 0
        countable_output_tokens = metadata.total_output_tokens
        actual_tokens = countable_input_tokens + countable_output_tokens

        difference = self.actual_estimate_total_tokens - actual_tokens
        self.logger.info(
            f"Adjusting {self.token_key} by {difference}. Estimated {self.actual_estimate_total_tokens}, actual {actual_tokens}"
        )

        await self._adjust_dynamic_estimate_ratio(actual_tokens)

        # NOTE: this can generate negative values, which represent `debt`
        # should not happen as we just hit rate limit instead
        # capped to prevent exceeding the token limit
        limit = await utils.redis_client.get(f"{self.token_key}:limit")
        assert limit
        limit = int(limit)

        await utils.redis_client.eval(
            REFILL_TOKENS_LUA, 1, self.token_key, difference, limit
        )
        telemetry.add_event(
            "retry_queue.tokens_released",
            {
                **self._telemetry_ids(),
                "retry_queue.estimated_tokens": self.actual_estimate_total_tokens,
                "retry_queue.actual_tokens": actual_tokens,
                "retry_queue.difference_tokens": difference,
            },
        )

        result[0].metadata.extra["token_metadata"] = {
            "estimated": self.estimate_total_tokens,
            "estimated_with_dynamic_ratio": self.actual_estimate_total_tokens,
            "actual": actual_tokens,
            "difference": difference,
            "ratio": actual_tokens / self.estimate_total_tokens,
            "dynamic_ratio_used": self.actual_estimate_total_tokens
            / self.estimate_total_tokens,
        }

    async def execute(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        try:
            return await super().execute(func, *args, **kwargs)
        finally:

            async def _exec_cleanup() -> None:
                # remove from per-run inflight ZSET; if now empty, remove run from active_runs
                await utils.redis_client.zrem(self._run_inflight_key, self._question_id)
                if await utils.redis_client.zcard(self._run_inflight_key) == 0:
                    await utils.redis_client.srem(self._active_runs_key, self._run_id)
                await utils.redis_client.delete(self._question_meta_key)

            try:
                await asyncio.shield(_exec_cleanup())
            except Exception as exc:
                telemetry.record_exception(
                    exc,
                    {
                        **self._telemetry_ids(),
                        "retry_queue.cleanup.phase": "execute",
                    },
                )
                telemetry.add_event(
                    "retry_queue.execute_cleanup_failed",
                    self._telemetry_ids(),
                )
                raise
            telemetry.add_event(
                "retry_queue.execute_cleanup_done",
                self._telemetry_ids(),
            )

    async def validate(self) -> None:
        # RPM-only never initializes a token bucket, so check the config key
        # that init_remaining_tokens does write instead.
        key = self.token_key if self._manages_tokens else f"{self.token_key}:config"
        await utils.validate_redis_client(key, "run `model.init_token_retry`")
