import asyncio
import logging
import random
import time
from math import ceil, floor
from typing import Any, Callable, Coroutine

from model_library.base.base import QueryResult, RateLimit
from model_library.exceptions import exception_message
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

BURST_FRACTION: float = 0.2  # max 20% of token limit per second

# Lua: atomic check-and-deduct with burst cap.
# KEYS[1] = token key, KEYS[2] = burst key
# ARGV[1] = required tokens, ARGV[2] = burst limit
DEDUCT_TOKENS_LUA = """
local required = tonumber(ARGV[1])
local remaining = tonumber(redis.call('GET', KEYS[1]))
if remaining < required then return 0 end
local burst = tonumber(redis.call('GET', KEYS[2]) or '0')
if burst > 0 and burst + required > tonumber(ARGV[2]) then return 0 end
redis.call('DECRBY', KEYS[1], required)
redis.call('INCRBY', KEYS[2], required)
return 1
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
    async def init_remaining_tokens(
        client_registry_key: tuple[str, str],
        limit: int,
        limit_refresh_seconds: int,
        logger: logging.Logger,
        get_rate_limit_func: Callable[[], Coroutine[Any, Any, RateLimit | None]],
    ) -> None:
        """
        Initialize remaining tokens in storage and start background refill process
        """

        from model_library.retriers.token.background import LoopConfig, background_loops

        await utils.validate_redis_client()

        key = TokenRetrier.get_token_key(client_registry_key)

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

        await utils.redis_client.hset(
            f"{key}:config",
            mapping={
                "limit": limit,
                "limit_refresh_seconds": limit_refresh_seconds,
                "tokens_per_second": tokens_per_second,
                "burst_limit": burst_limit,
                "initialized_at": time.time(),
            },
        )

        cfg = LoopConfig(key, limit, tokens_per_second)
        asyncio.create_task(
            background_loops(
                cfg,
                get_rate_limit_func,
                logger,
                standby=active is not None and not config_changed,
            )
        )

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
    ):
        super().__init__(
            strategy="token",
            logger=logger,
            max_tries=max_tries,
            max_time=max_time,
            retry_callback=retry_callback,
        )

        self.client_registry_key = client_registry_key

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
        self._run_id = run_id
        self._question_id = f"{run_id}:{question_id}"
        self._is_queued: bool | None = None  # lazy: set on first _pre_function call
        self._burst_limit: int | None = (
            None  # lazy: read from config on first _pre_function call
        )

        # per-run inflight tracking
        self._active_runs_key = f"{self.token_key}:active_runs"
        self._run_inflight_key = f"{self.token_key}:run:{self._run_id}:inflight"
        # per-question metadata hash
        # NOTE: each query's metadata in a agentic question gets overwritten
        self._question_meta_key = f"{self.token_key}:inflight:{self._question_id}"

        # benchmark keys
        self._queue_key = f"{self._base_key}:benchmark:queue"
        self._run_meta_key = f"{self._base_key}:benchmark:run:{self._run_id}"

        self.dynamic_estimate_key = (
            f"{self.token_key}:dynamic_estimate:{self._run_id}"
            if use_dynamic_estimate
            else None
        )

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

        self.logger.warning(logger_msg)

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

        # straggler: my benchmark is no longer the queue head (early-released)
        if self._is_queued:
            head = await utils.redis_client.lindex(self._queue_key, 0)
            if head != self._run_id:
                self.priority = MAX_PRIORITY

        # remove from dispatched so early release knows we're waiting again
        # (agentic benchmarks re-enter _pre_function for each turn)
        await utils.redis_client.srem(
            f"{self._run_meta_key}:dispatched", self._question_id
        )

        priority_key = TokenRetrier.get_priority_key(
            self.client_registry_key, self.priority
        )

        # let storage know we are waiting at this priority (sorted set: question_id → timestamp)
        await utils.redis_client.zadd(priority_key, {self._question_id: time.time()})
        # per-request hash so status endpoint can group queued requests by run
        await utils.redis_client.hset(
            self._question_meta_key,
            mapping={"run_id": self._run_id, "priority": str(self.priority)},
        )
        self.logger.debug(f"priority: {self.priority}, waiting: {priority_key}")

        _deducted = False
        try:
            while True:
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

                    # atomic check-and-deduct via Lua (no lock needed)
                    deducted = await utils.redis_client.eval(
                        DEDUCT_TOKENS_LUA,
                        2,
                        self.token_key,
                        f"{self.token_key}:burst",
                        self.actual_estimate_total_tokens,
                        self._burst_limit,
                    )
                    if deducted:
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
                        return

                    self.logger.debug(
                        f"[Token Wait] Insufficient tokens, waiting {wait_time:.1f}s | "
                        f"estimate_tokens: {self.actual_estimate_total_tokens} | "
                        f"Priority: {self.priority}"
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

            await asyncio.shield(_pre_cleanup())

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

    async def _post_function(self, result: tuple[QueryResult, float]) -> None:
        """Adjust token estimate based on actual usage"""

        metadata = result[0].metadata

        countable_input_tokens = metadata.total_input_tokens - (
            metadata.cache_read_tokens or 0
        )
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

            await asyncio.shield(_exec_cleanup())

    async def validate(self) -> None:
        await utils.validate_redis_client(
            self.token_key, "run `model.init_token_retry`"
        )
