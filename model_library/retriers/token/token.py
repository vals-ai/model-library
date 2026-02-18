import asyncio
import logging
import time
import uuid
from asyncio.tasks import Task
from math import ceil, floor
from typing import Any, Callable, Coroutine

from model_library.base.base import QueryResult, RateLimit
from model_library.exceptions import exception_message
from model_library.retriers.base import BaseRetrier
from model_library.retriers.token import utils
from model_library.retriers.utils import jitter

RETRY_WAIT_TIME: float = 20.0
TOKEN_WAIT_TIME: float = 5.0

MAX_PRIORITY: int = -5
INITIAL_PRIORITY: int = 0
MIN_PRIORITY: int = 5

MAX_RETRIES: int = 10

PRIORITY_STALE_AGE: int = (
    300  # 5 minutes — reap priority entries with no refresh (matches heartbeat TTL)
)
INFLIGHT_MAX_AGE: int = 7200  # 2 hours — reap stale inflight entries (this doesn't actually kill the task, just the entry)
REAP_INTERVAL: int = 30  # seconds between stale entry reap checks

LOCK_TIMEOUT: int = 10  # using 10 in case there is high compute load, don't want to error on lock releases

# Lua: atomic check-and-deduct. Returns 1 if deducted, 0 if insufficient.
# KEYS[1] = token key, ARGV[1] = required tokens
DEDUCT_TOKENS_LUA = """
local remaining = tonumber(redis.call('GET', KEYS[1]))
if remaining >= tonumber(ARGV[1]) then
    redis.call('DECRBY', KEYS[1], ARGV[1])
    return 1
end
return 0
"""

# Lua: atomic refill with cap. Returns new token count.
# KEYS[1] = token key, ARGV[1] = refill amount, ARGV[2] = cap limit
REFILL_TOKENS_LUA = """
local current = tonumber(redis.call('INCRBY', KEYS[1], ARGV[1]))
if current > tonumber(ARGV[2]) then
    redis.call('SET', KEYS[1], ARGV[2])
    return tonumber(ARGV[2])
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
redis.call('SET', KEYS[3], ARGV[2])
if old_limit ~= tonumber(ARGV[1]) or redis.call('EXISTS', KEYS[1]) == 0 then
    redis.call('SET', KEYS[1], ARGV[1])
    redis.call('SET', KEYS[2], ARGV[1])
    return 1
end
return 0
"""

refill_tasks: dict[str, tuple[dict[str, int | Any], Task[None]]] = {}


class TokenRetrier(BaseRetrier):
    """
    Token-based retry strategy
    Predicts the number of tokens required for a query, sends resquests to respect the rate limit,
    then adjusts the estimate based on actual usage.
    """

    @staticmethod
    def get_token_key(client_registry_key: tuple[str, str]) -> str:
        """Get the key which stores remaining tokens"""
        return f"{client_registry_key[0]}:{client_registry_key[1]}:tokens"

    @staticmethod
    def get_priority_key(client_registry_key: tuple[str, str], priority: int) -> str:
        """Get the key which stores the amount of tasks waiting for a given priority"""
        return f"{client_registry_key[0]}:{client_registry_key[1]}:priority:{priority}"

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

        async def _header_correction_loop(
            key: str,
            limit: int,
            tokens_per_second: int,
            get_rate_limit_func: Callable[[], Coroutine[Any, Any, RateLimit | None]],
            version: str,
        ) -> None:
            """
            Background loop that correct tokens based on provider headers
            Every 5 seconds
            """
            interval = 5.0

            while True:
                await asyncio.sleep(interval)
                current_version = await utils.redis_client.get("version:" + key)
                if current_version != version:
                    logger.debug(
                        f"version changed ({current_version} != {version}), exiting _header_correction_loop for {key}"
                    )
                    return

                rate_limit = await get_rate_limit_func()
                if rate_limit is None:
                    # kill the task as no headers are provided
                    logger.debug(
                        f"no rate limit headers, exiting _header_correction_loop for {key}"
                    )
                    return

                tokens_remaining = rate_limit.token_remaining_total

                # atomic correct-down via Lua (no lock needed)
                elapsed = time.time() - rate_limit.unix_timestamp
                adjusted = min(
                    limit, floor(tokens_remaining + (tokens_per_second * elapsed))
                )
                result = await utils.redis_client.eval(
                    CORRECT_TOKENS_LUA, 1, key, adjusted
                )
                corrected, current, adj = int(result[0]), int(result[1]), int(result[2])
                if corrected:
                    logger.info(
                        f"Corrected {key} from {current} to {adj} based on headers ({elapsed:.1f}s old)"
                    )
                else:
                    logger.debug(
                        f"Not correcting {key} from {current} to {adj} based on headers ({elapsed:.1f}s old) (higher value)"
                    )

        async def _token_refill_loop(
            key: str,
            limit: int,
            tokens_per_second: int,
            version: str,
        ) -> None:
            """
            Background loop that refills tokens
            Every second
            """
            interval: float = 1.0
            last_reap: float = 0.0

            while True:
                await asyncio.sleep(interval)
                current_version = await utils.redis_client.get("version:" + key)
                if current_version != version:
                    logger.debug(
                        f"version changed ({current_version} != {version}), exiting _token_refill_loop for {key}"
                    )
                    return

                # atomic refill with cap via Lua (no lock needed)
                current = int(
                    await utils.redis_client.eval(
                        REFILL_TOKENS_LUA, 1, key, tokens_per_second, limit
                    )
                )
                logger.debug(
                    f"[Token Refill] | {key} | Amount: {tokens_per_second} | Current: {current}"
                )
                if current == limit:
                    logger.debug(f"[Token Cap] | {key} | Limit: {limit}")

                # periodically reap stale inflight and priority entries
                now = time.time()
                if now - last_reap >= REAP_INTERVAL:
                    last_reap = now

                    inflight_key = f"{key}:inflight"
                    stale = await utils.redis_client.zrangebyscore(
                        inflight_key, "-inf", now - INFLIGHT_MAX_AGE
                    )
                    if stale:
                        await utils.redis_client.zrem(inflight_key, *stale)
                        logger.info(
                            f"[Reap] {key} | Removed {len(stale)} stale inflight entries"
                        )

                    base = key.removesuffix(":tokens")
                    for p in range(MAX_PRIORITY, MIN_PRIORITY + 1):
                        pkey = f"{base}:priority:{p}"
                        stale = await utils.redis_client.zrangebyscore(
                            pkey, "-inf", now - PRIORITY_STALE_AGE
                        )
                        if stale:
                            await utils.redis_client.zrem(pkey, *stale)
                            logger.info(
                                f"[Reap] {pkey} | Removed {len(stale)} stale priority entries"
                            )

        await utils.validate_redis_client()

        key = TokenRetrier.get_token_key(client_registry_key)

        # limit_key is only used to check if the limit has changed
        limit_key = f"{key}:limit"

        # atomic init: check old limit, set version, conditionally reset tokens
        version = str(uuid.uuid4())
        version_key = "version:" + key
        await utils.redis_client.eval(
            INIT_TOKENS_LUA, 3, key, limit_key, version_key, limit, version
        )

        tokens_per_second = floor(limit / limit_refresh_seconds)

        refill_task = asyncio.create_task(
            _token_refill_loop(key, limit, tokens_per_second, version)
        )
        correction_task = asyncio.create_task(
            _header_correction_loop(
                key, limit, tokens_per_second, get_rate_limit_func, version
            )
        )

        refill_tasks["refill:" + key] = (
            {
                "limit": limit,
                "limit_refresh_seconds": limit_refresh_seconds,
            },
            refill_task,
        )
        refill_tasks["correction:" + key] = (
            {
                "limit": limit,
                "limit_refresh_seconds": limit_refresh_seconds,
                "get_rate_limit_func": get_rate_limit_func,
            },
            correction_task,
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
        request_id: str,
        estimate_input_tokens: int,
        estimate_output_tokens: int,
        dynamic_estimate_instance_id: str | None = None,
        retry_wait_time: float = RETRY_WAIT_TIME,
        token_wait_time: float = TOKEN_WAIT_TIME,
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

        self.retry_wait_time = retry_wait_time
        self.token_wait_time = token_wait_time

        self.priority = INITIAL_PRIORITY

        self.token_key = TokenRetrier.get_token_key(client_registry_key)
        self._inflight_key = self.token_key + ":inflight"
        self._request_id = request_id
        self._token_key_lock = self.token_key + ":lock"
        self._init_key_lock = "init:" + self.token_key + ":lock"

        self.dynamic_estimate_key = (
            f"{self.token_key}:dynamic_estimate:{dynamic_estimate_instance_id}"
            if dynamic_estimate_instance_id
            else None
        )

        # set in validate() — benchmark run_id for straggler detection
        # if running benchmark_queue context_manager
        self._benchmark_run_id: str | None = None

    async def _calculate_wait_time(
        self, attempt: int, exception: Exception | None = None
    ) -> float:
        """Wait time between retries"""
        return jitter(self.retry_wait_time)

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
        """Check if there are lower priority requests waiting (single Redis round-trip)"""
        base = f"{self.client_registry_key[0]}:{self.client_registry_key[1]}"
        result = await utils.redis_client.eval(
            HAS_LOWER_PRIORITY_LUA, 0, base, self.priority, MAX_PRIORITY
        )
        return bool(result)

    async def _pre_function(self) -> None:
        """
        Loop until sufficient tokens are available.
        Acquires priority semaphore, checks for lower priority requests, deducts tokens from Redis.
        Logs token waits but does not count as retry attempts.
        """

        # straggler: my benchmark is no longer the queue head (early-released)
        if self._benchmark_run_id is not None:
            queue_key = f"{self.client_registry_key[0]}:{self.client_registry_key[1]}:benchmark:queue"
            head = await utils.redis_client.lindex(queue_key, 0)
            if head != self._benchmark_run_id:
                self.priority = MAX_PRIORITY

        priority_key = TokenRetrier.get_priority_key(
            self.client_registry_key, self.priority
        )

        # let storage know we are waiting at this priority (sorted set: request_id → timestamp)
        await utils.redis_client.zadd(priority_key, {self._request_id: time.time()})
        self.logger.debug(f"priority: {self.priority}, waiting: {priority_key}")

        try:
            while True:
                wait_time = jitter(self.token_wait_time)

                # refresh timestamp so reaper knows we're alive
                await utils.redis_client.zadd(
                    priority_key, {self._request_id: time.time()}
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
                        1,
                        self.token_key,
                        self.actual_estimate_total_tokens,
                    )
                    if deducted:
                        await utils.redis_client.zadd(
                            self._inflight_key,
                            {
                                self._request_id: time.time()
                            },  # tracks current inflight requests
                        )
                        await utils.redis_client.sadd(
                            f"{self._inflight_key}:dispatched",
                            self._request_id,  # tracks total requests dispatched so far
                        )
                        self.logger.debug(
                            f"Deducted {self.actual_estimate_total_tokens} tokens from {self.token_key}"
                        )
                        return

                    self.logger.warning(
                        f"[Token Wait] Insufficient tokens, waiting {wait_time:.1f}s | "
                        f"estimate_tokens: {self.actual_estimate_total_tokens} | "
                        f"Priority: {self.priority}"
                    )

                # Zzz
                self.logger.debug(f"Sleeping for {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
        finally:
            # let storage know we are done waiting at this priority
            await utils.redis_client.zrem(priority_key, self._request_id)

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
            await utils.redis_client.zrem(self._inflight_key, self._request_id)

    async def validate(self) -> None:
        await utils.validate_redis_client(
            self.token_key, "run `model.init_token_retry`"
        )
        self._benchmark_run_id = await utils.redis_client.get(
            f"{self._inflight_key}:benchmark_run"
        )
