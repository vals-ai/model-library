import asyncio
import logging
import time
import uuid
from contextlib import suppress
from math import exp, floor
from typing import Any, Callable, Coroutine

from model_library.base.base import RateLimit
from model_library.retriers.token import utils
from model_library.retriers.token.token import (
    CORRECT_TOKENS_LUA,
    INFLIGHT_MAX_AGE,
    MAX_PRIORITY,
    MIN_PRIORITY,
    PRIORITY_STALE_AGE,
    REAP_INTERVAL,
    REFILL_TOKENS_LUA,
)

FULL_TOKENS_SHUTDOWN: int = (
    300  # 5 minutes, stop background loops after tokens sit at limit
)
REFILL_TASK_TTL: int = 30  # seconds — task keys expire if loop dies

LOOP_POLL_INTERVAL: float = 10.0  # seconds — poll interval
REQUIRED_ACTIVE_WORKERS = ("refill", "reaper")

COMPARE_DELETE_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""

REFRESH_ACTIVE_LUA = """
local owner = redis.call('GET', KEYS[1])
if not owner or owner == ARGV[1] then
  redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[2])
  return 1
end
return 0
"""


class _Demoted(Exception):
    """Raised by the watchdog to signal the active TaskGroup to shut down."""


class LoopConfig:
    """Mutable config for background loops, updated on standby takeover."""

    def __init__(self, key: str, limit: int, tokens_per_second: int) -> None:
        self.key = key
        self.limit = limit
        self.tokens_per_second = tokens_per_second


async def _delete_active_key_if_owned(active_key: str, loop_id: str) -> bool:
    deleted = await utils.redis_client.eval(COMPARE_DELETE_LUA, 1, active_key, loop_id)
    return int(deleted) == 1


async def _refresh_active_key_if_unowned_or_owned(
    active_key: str, loop_id: str, ttl: int
) -> bool:
    refreshed = await utils.redis_client.eval(
        REFRESH_ACTIVE_LUA, 1, active_key, loop_id, ttl
    )
    return int(refreshed) == 1


async def background_loops(
    cfg: LoopConfig,
    get_rate_limit_func: Callable[[], Coroutine[Any, Any, RateLimit | None]],
    logger: logging.Logger,
    standby: bool = False,
) -> None:
    """Manages all background loops with shared watchdog"""

    async def _update_remaining_ratio_ewma(current: int) -> None:
        """Update remaining-token ratio EWMAs for queue admission control."""
        if cfg.limit <= 0:
            return

        now = time.time()
        ratio = current / cfg.limit
        ewma_2m_key = f"{cfg.key}:remaining_ratio_ewma_2m"
        ewma_15s_key = f"{cfg.key}:remaining_ratio_ewma_15s"
        updated_key = f"{cfg.key}:remaining_ratio_ewma_updated_at"
        async with utils.redis_client.pipeline(transaction=False) as pipe:
            pipe.get(ewma_2m_key)
            pipe.get(ewma_15s_key)
            pipe.get(updated_key)
            (
                previous_2m_raw,
                previous_15s_raw,
                previous_updated_raw,
            ) = await pipe.execute()

        if not previous_updated_raw:
            ewma_2m = ratio
            ewma_15s = ratio
        else:
            elapsed = max(0.0, now - float(previous_updated_raw))
            alpha_2m = 1 - exp(-elapsed / 120.0)
            alpha_15s = 1 - exp(-elapsed / 15.0)
            previous_2m = float(previous_2m_raw) if previous_2m_raw else ratio
            previous_15s = float(previous_15s_raw) if previous_15s_raw else ratio
            ewma_2m = previous_2m + (alpha_2m * (ratio - previous_2m))
            ewma_15s = previous_15s + (alpha_15s * (ratio - previous_15s))

        async with utils.redis_client.pipeline(transaction=False) as pipe:
            pipe.set(ewma_2m_key, ewma_2m, ex=300)
            pipe.set(ewma_15s_key, ewma_15s, ex=300)
            pipe.set(updated_key, now, ex=300)
            await pipe.execute()

    async def _header_correction_loop() -> None:
        """
        Background loop that correct tokens based on provider headers
        Every 20.0 seconds
        """
        INTERVAL = 20.0
        while True:
            await asyncio.sleep(INTERVAL)

            try:
                rate_limit = await get_rate_limit_func()
                if rate_limit is None:
                    logger.debug(
                        f"no rate limit headers, exiting _header_correction_loop for {cfg.key}"
                    )
                    return

                # store last header to Redis for status visibility
                header_data = rate_limit.model_dump(exclude={"raw"})
                redis_mapping = {k: str(v) for k, v in header_data.items()}
                header_redis_key = f"{cfg.key}:last_header"
                async with utils.redis_client.pipeline(transaction=True) as pipe:
                    pipe.hset(  # pyright: ignore[reportUnknownMemberType]
                        header_redis_key, mapping=redis_mapping
                    )
                    pipe.expire(header_redis_key, 60)
                    await pipe.execute()

                header_limit = rate_limit.token_limit_total
                if not (0.8 * cfg.limit <= header_limit <= 1.2 * cfg.limit):
                    logger.debug(
                        f"header token limit ({header_limit}) outside 80-120% of configured limit ({cfg.limit}), exiting correction for {cfg.key}"
                    )
                    return

                tokens_remaining = rate_limit.token_remaining_total

                # atomic correct-down via Lua
                elapsed = time.time() - rate_limit.unix_timestamp
                adjusted = min(
                    cfg.limit,
                    floor(tokens_remaining + (cfg.tokens_per_second * elapsed)),
                )
                result = await utils.redis_client.eval(
                    CORRECT_TOKENS_LUA, 1, cfg.key, adjusted
                )
                corrected, current, adj = (
                    int(result[0]),
                    int(result[1]),
                    int(result[2]),
                )
                if corrected:
                    logger.info(
                        f"Corrected {cfg.key} from {current} to {adj} based on headers ({elapsed:.1f}s old)"
                    )
                else:
                    logger.debug(
                        f"Not correcting {cfg.key} from {current} to {adj} based on headers ({elapsed:.1f}s old) (higher value)"
                    )
            except Exception:
                logger.warning(
                    f"[Token Correction] {cfg.key} | error in correction loop, retrying",
                    exc_info=True,
                )

    async def _token_refill_loop() -> None:
        """
        Background loop that refills tokens
        Every 1.0 second
        """
        INTERVAL = 1.0
        last_refill: float = time.monotonic()

        while True:
            await asyncio.sleep(INTERVAL)

            try:
                # scale refill by actual elapsed time to compensate for loop drift
                mono_now = time.monotonic()
                refill_amount = max(
                    0, floor(cfg.tokens_per_second * (mono_now - last_refill))
                )
                last_refill = mono_now

                # atomic refill with cap via Lua
                current = int(
                    await utils.redis_client.eval(
                        REFILL_TOKENS_LUA, 1, cfg.key, refill_amount, cfg.limit
                    )
                )
                await _update_remaining_ratio_ewma(current)
                logger.debug(
                    f"[Token Refill] | {cfg.key} | Amount: {refill_amount} | Current: {current}"
                )
                if current == cfg.limit:
                    logger.debug(f"[Token Cap] | {cfg.key} | Limit: {cfg.limit}")
            except Exception:
                logger.warning(
                    f"[Token Refill] {cfg.key} | error in refill loop, retrying",
                    exc_info=True,
                )

    async def _cleanup_loop() -> None:
        """Reap stale inflight/priority entries and idle shutdown every 30s."""
        idle_since: float | None = None

        while True:
            await asyncio.sleep(REAP_INTERVAL)

            try:
                # idle shutdown: tokens at limit with no active runs for FULL_TOKENS_SHUTDOWN
                current = int(await utils.redis_client.get(cfg.key) or 0)
                active_runs = await utils.redis_client.scard(f"{cfg.key}:active_runs")
                if current >= cfg.limit and active_runs == 0:
                    if idle_since is None:
                        idle_since = time.time()
                    elif time.time() - idle_since >= FULL_TOKENS_SHUTDOWN:
                        logger.debug(
                            f"tokens at full for {FULL_TOKENS_SHUTDOWN}s, stopping background loops for {cfg.key}"
                        )
                        raise _Demoted()
                else:
                    idle_since = None

                # reap stale inflight/priority entries
                now = time.time()
                active_runs_key = f"{cfg.key}:active_runs"
                active_run_ids = await utils.redis_client.smembers(active_runs_key)
                for rid in active_run_ids:
                    run_inflight_key = f"{cfg.key}:run:{rid}:inflight"
                    stale = await utils.redis_client.zrangebyscore(
                        run_inflight_key, "-inf", now - INFLIGHT_MAX_AGE
                    )
                    if stale:
                        await utils.redis_client.zrem(run_inflight_key, *stale)
                        # clean up metadata hashes
                        for qid in stale:
                            await utils.redis_client.delete(f"{cfg.key}:inflight:{qid}")
                        logger.info(
                            f"[Reap] {run_inflight_key} | Removed {len(stale)} stale inflight entries"
                        )
                    # if run's inflight ZSET is now empty, remove from active_runs
                    if await utils.redis_client.zcard(run_inflight_key) == 0:
                        await utils.redis_client.srem(active_runs_key, rid)

                base = cfg.key.removesuffix(":tokens")
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
            except _Demoted:
                raise
            except Exception:
                logger.warning(
                    f"[Reaper] {cfg.key} | error in reaper loop, retrying",
                    exc_info=True,
                )

    active_key = f"{cfg.key}:task:active"
    loop_id = str(uuid.uuid4())
    is_active = False
    worker_tasks: dict[str, asyncio.Task[None]] = {}
    demotion_requested = asyncio.Event()
    required_workers_seen = False

    async def _heartbeat() -> None:
        nonlocal required_workers_seen

        while True:
            # always write task keys for alive workers
            for name, task in list(worker_tasks.items()):
                if not task.done():
                    await utils.redis_client.set(
                        f"{cfg.key}:task:{name}", "1", ex=REFILL_TASK_TTL
                    )

            # Only refresh active ownership if nobody else has taken over and
            # required active workers are alive.
            if is_active and not demotion_requested.is_set():
                missing_required = [
                    name for name in REQUIRED_ACTIVE_WORKERS if name not in worker_tasks
                ]
                dead_required = [
                    name
                    for name in REQUIRED_ACTIVE_WORKERS
                    if name in worker_tasks and worker_tasks[name].done()
                ]
                required_workers_ready = not missing_required
                required_workers_seen = required_workers_seen or required_workers_ready
                current_owner = await utils.redis_client.get(active_key)
                if current_owner and current_owner != loop_id:
                    demotion_requested.set()
                elif required_workers_seen and dead_required:
                    logger.warning(
                        f"[Background] {cfg.key} | active loop has dead required workers: {dead_required}; demoting"
                    )
                    demotion_requested.set()
                elif not required_workers_seen or required_workers_ready:
                    await _refresh_active_key_if_unowned_or_owned(
                        active_key, loop_id, REFILL_TASK_TTL
                    )

            await asyncio.sleep(LOOP_POLL_INTERVAL)

    heartbeat_task = asyncio.create_task(_heartbeat())

    async def _standby(takeover_loop_id: str) -> None:
        # standby: wait until active ownership can be claimed atomically, then take over
        while True:
            await asyncio.sleep(LOOP_POLL_INTERVAL)
            claimed = await _refresh_active_key_if_unowned_or_owned(
                active_key, takeover_loop_id, REFILL_TASK_TTL
            )
            if claimed:
                logger.info(f"[Background] {cfg.key} | standby taking over")
                config = await utils.redis_client.hgetall(f"{cfg.key}:config")
                cfg.limit = int(config["limit"])
                cfg.tokens_per_second = int(config["tokens_per_second"])
                return

    try:
        while True:
            loop_id = str(uuid.uuid4())
            if standby:
                await _standby(loop_id)
            else:
                claimed = await _refresh_active_key_if_unowned_or_owned(
                    active_key, loop_id, REFILL_TASK_TTL
                )
                if not claimed:
                    standby = True
                    continue

            is_active = True
            demotion_requested = asyncio.Event()
            required_workers_seen = False

            # watchdog raises _Demoted if another loop takes the active key or a
            # required worker task dies. The outer loop then releases active ownership
            # and restarts from standby instead of trying to restart one child task.
            async def _watchdog() -> None:
                while True:
                    await asyncio.sleep(LOOP_POLL_INTERVAL)
                    if demotion_requested.is_set():
                        raise _Demoted()
                    val = await utils.redis_client.get(active_key)
                    if val and val != loop_id:
                        raise _Demoted()

                    for name in REQUIRED_ACTIVE_WORKERS:
                        task = worker_tasks.get(name)
                        if task is not None and task.done():
                            logger.warning(
                                f"[Background] {cfg.key} | active loop required worker died: {name}; demoting"
                            )
                            raise _Demoted()

            try:
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(_watchdog())
                    worker_tasks["refill"] = tg.create_task(_token_refill_loop())
                    worker_tasks["correction"] = tg.create_task(
                        _header_correction_loop()
                    )
                    worker_tasks["reaper"] = tg.create_task(_cleanup_loop())
            except* _Demoted:
                is_active = False
            except* Exception:
                logger.warning(
                    f"[Background] {cfg.key} | loop group error", exc_info=True
                )

            is_active = False
            await _delete_active_key_if_owned(active_key, loop_id)
            worker_tasks.clear()
            standby = True
    finally:
        heartbeat_task.cancel()
        try:
            with suppress(asyncio.CancelledError):
                await heartbeat_task
        finally:
            await _delete_active_key_if_owned(active_key, loop_id)
