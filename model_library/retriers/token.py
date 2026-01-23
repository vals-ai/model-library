import asyncio
import logging
import time
import uuid
from asyncio.tasks import Task
from math import ceil, floor
from typing import Any, Callable, Coroutine

from redis.asyncio import Redis

from model_library.base.base import QueryResult, RateLimit
from model_library.exceptions import exception_message
from model_library.retriers.base import BaseRetrier
from model_library.retriers.utils import jitter

RETRY_WAIT_TIME: float = 20.0
TOKEN_WAIT_TIME: float = 5.0

MAX_PRIORITY: int = 1
MIN_PRIORITY: int = 5

MAX_RETRIES: int = 10

redis_client: Redis
LOCK_TIMEOUT: int = 10  # using 10 in case there is high compute load, don't want to error on lock releases


def set_redis_client(client: Redis):
    global redis_client
    redis_client = client


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

            assert redis_client
            while True:
                await asyncio.sleep(interval)
                current_version = await redis_client.get("version:" + key)
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

                async with redis_client.lock(key + ":lock", timeout=LOCK_TIMEOUT):
                    current = int(await redis_client.get(key))

                    # increment
                    elapsed = time.time() - rate_limit.unix_timestamp
                    adjusted = floor(tokens_remaining + (tokens_per_second * elapsed))

                    # if the headers show a lower value, correct with that
                    if adjusted < current:
                        await redis_client.set(key, adjusted)
                        logger.info(
                            f"Corrected {key} from {current} to {adjusted} based on headers ({elapsed:.1f}s old)"
                        )
                    else:
                        logger.debug(
                            f"Not correcting {key} from {current} to {adjusted} based on headers ({elapsed:.1f}s old) (higher value)"
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

            assert redis_client
            while True:
                await asyncio.sleep(interval)
                current_version = await redis_client.get("version:" + key)
                if current_version != version:
                    logger.debug(
                        f"version changed ({current_version} != {version}), exiting _token_refill_loop for {key}"
                    )
                    return

                async with redis_client.lock(key + ":lock", timeout=LOCK_TIMEOUT):
                    # increment
                    current = await redis_client.incrby(key, tokens_per_second)
                    logger.debug(
                        f"[Token Refill] | {key} | Amount: {tokens_per_second} | Current: {current}"
                    )
                    # cap at limit
                    if current > limit:
                        logger.debug(f"[Token Cap] | {key} | Limit: {limit}")
                        await redis_client.set(key, limit)

        key = TokenRetrier.get_token_key(client_registry_key)

        # limit_key is only used to check if the limit has changed
        limit_key = f"{key}:limit"

        async with redis_client.lock("init:" + key + ":lock", timeout=LOCK_TIMEOUT):
            old_limit = int(await redis_client.get(limit_key) or 0)

            # keep track of version so we can clean up old tasks
            # even if the limit has not changed, reset background tasks just in case
            version = str(uuid.uuid4())
            await redis_client.set("version:" + key, version)

            if old_limit != limit or not await redis_client.exists(key):
                # if new limit if different, set it
                await redis_client.set(key, limit)
                await redis_client.set(limit_key, limit)

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

    async def _get_remaining_tokens(self) -> int:
        """Get remaining tokens"""
        tokens = await redis_client.get(self.token_key)
        return int(tokens)

    async def _deduct_remaining_tokens(self) -> None:
        """Deduct from remaining tokens"""
        # NOTE: decrby is atomic
        await redis_client.decrby(self.token_key, self.actual_estimate_total_tokens)

    def __init__(
        self,
        logger: logging.Logger,
        max_tries: int | None = MAX_RETRIES,
        max_time: float | None = None,
        retry_callback: Callable[[int, Exception | None, float, float], None]
        | None = None,
        *,
        client_registry_key: tuple[str, str],
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

        self.priority = MAX_PRIORITY

        self.token_key = TokenRetrier.get_token_key(client_registry_key)
        self._token_key_lock = self.token_key + ":lock"
        self._init_key_lock = "init:" + self.token_key + ":lock"

        self.dynamic_estimate_key = (
            f"{self.token_key}:dynamic_estimate:{dynamic_estimate_instance_id}"
            if dynamic_estimate_instance_id
            else None
        )

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
        """
        Check if there are lower priority requests waiting
        """

        # NOTE: no lock needed, stale counts are fine
        for priority in range(MAX_PRIORITY, self.priority):
            key = TokenRetrier.get_priority_key(self.client_registry_key, priority)
            count = await redis_client.get(key)
            self.logger.debug(f"priority: {priority}, count: {count}")
            if count and int(count) > 0:
                return True
        return False

    async def _pre_function(self) -> None:
        """
        Loop until sufficient tokens are available.
        Acquires priority semaphore, checks for lower priority requests, deducts tokens from Redis.
        Logs token waits but does not count as retry attempts.
        """

        priority_key = TokenRetrier.get_priority_key(
            self.client_registry_key, self.priority
        )

        # let storage know we are waiting at this priority
        await redis_client.incr(priority_key)
        self.logger.debug(f"priority: {self.priority}, waiting: {priority_key}")

        try:
            while True:
                wait_time = jitter(self.token_wait_time)

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
                            await redis_client.get(self.dynamic_estimate_key) or 1.0
                        )
                        self.actual_estimate_total_tokens = ceil(
                            self.estimate_total_tokens * ratio
                        )
                        self.logger.debug(
                            f"Adjusted actual estimate tokens to {self.actual_estimate_total_tokens} using ratio {ratio}"
                        )

                    # TODO: use luascript to avoid using locks

                    # NOTE: `async with` releases lock in all situations
                    async with redis_client.lock(
                        self._token_key_lock, timeout=LOCK_TIMEOUT
                    ):
                        tokens_remaining = await self._get_remaining_tokens()

                        # if we have enough tokens, deduct estimate tokens and make request
                        if tokens_remaining >= self.actual_estimate_total_tokens:
                            self.logger.debug(
                                f"Enough tokens {self.actual_estimate_total_tokens}/{tokens_remaining}, deducting"
                            )
                            await self._deduct_remaining_tokens()
                            return

                    self.logger.warning(
                        f"[Token Wait] Insufficient tokens, waiting {wait_time:.1f}s | "
                        f"estimate_tokens: {self.actual_estimate_total_tokens}/{tokens_remaining} | "
                        f"Priority: {self.priority}"
                    )

                # Zzz
                self.logger.debug(f"Sleeping for {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
        finally:
            # let storage know we are done waiting at this priority
            await redis_client.decr(priority_key)

    async def _adjust_dynamic_estimate_ratio(self, actual_tokens: int) -> None:
        if not self.dynamic_estimate_key:
            return

        observed_ratio = actual_tokens / self.estimate_total_tokens

        alpha = 0.3

        async with redis_client.lock(
            self.dynamic_estimate_key + ":lock", timeout=LOCK_TIMEOUT
        ):
            current_ratio = float(
                await redis_client.get(self.dynamic_estimate_key) or 1.0
            )

            new_ratio = (observed_ratio * alpha) + (current_ratio * (1 - alpha))

            # NOTE: for now, will not cap the ratio as estimates will likely be very off
            # the ratio between the tokenized estimate and the dynamic estimate should not be too far off
            # new_ratio = max(0.01, min(100.0, new_ratio))

            await redis_client.set(self.dynamic_estimate_key, new_ratio)

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
        async with redis_client.lock(self._token_key_lock, timeout=LOCK_TIMEOUT):
            await redis_client.incrby(self.token_key, difference)

        result[0].metadata.extra["token_metadata"] = {
            "estimated": self.estimate_total_tokens,
            "estimated_with_dynamic_ratio": self.actual_estimate_total_tokens,
            "actual": actual_tokens,
            "difference": difference,
            "ratio": actual_tokens / self.estimate_total_tokens,
            "dynamic_ratio_used": self.actual_estimate_total_tokens
            / self.estimate_total_tokens,
        }

    async def validate(self) -> None:
        try:
            assert redis_client
        except Exception as e:
            raise Exception(
                f"redis client not set, run `TokenRetrier.set_redis_client`. Exception: {e}"
            )
        if not await redis_client.exists(self.token_key):
            raise Exception(
                "remaining_tokens not intialized, run `model.init_token_retry`"
            )
