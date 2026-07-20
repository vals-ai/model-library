import asyncio
import logging
import os
import threading
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

from model_library.retriers.token import utils
from model_library.retriers.token.utils import (
    KEY_PREFIX,
    AsyncRedisClient,
)

HOURS_24 = 86400

BENCHMARK_NOTIFY_PROCEED = "go"
BENCHMARK_NOTIFY_CANCELLED = "cancelled"

HEARTBEAT_INTERVAL = 2
HEARTBEAT_TTL = 300  # 5 minutes
QUEUE_NOTIFY_POLL_INTERVAL = 0.5
QUEUE_WAITER_STALL_LOG_THRESHOLD = 5
HEARTBEAT_STALL_LOG_THRESHOLD = 5
BENCHMARK_QUEUE_DEBUG_ENABLED = False
EARLY_RELEASE_GRACE_PERIOD = 5  # seconds to wait after all dispatched before releasing

MAX_ACTIVE_HEADS = 20
ACTIVE_HEAD_SCALE_UP_THRESHOLD = 0.25
ACTIVE_HEAD_SCALE_DOWN_THRESHOLD = 0.15
ACTIVE_HEAD_SCALE_DOWN_INTERVAL = 15.0
ACTIVE_HEAD_SCALE_UP_INTERVAL = 2.0
DEFAULT_ACTIVE_HEAD_WINDOW = 1
ACTIVE_HEAD_QUEUE_SCAN_LIMIT = MAX_ACTIVE_HEADS * 5

EARLY_RELEASE_LUA = """
if redis.call('ZSCORE', KEYS[3], ARGV[1]) == false then
  return nil
end
redis.call('HSET', KEYS[1], 'popped_at', ARGV[2])
redis.call('LREM', KEYS[2], 1, ARGV[1])
redis.call('ZREM', KEYS[3], ARGV[1])
return ARGV[1]
"""

CONTROL_AND_ADMIT_HEADS_LUA = """
local queue_key = KEYS[1]
local active_heads_key = KEYS[2]
local token_key = KEYS[3]
local token_limit_key = KEYS[4]
local ewma_key = KEYS[5]
local short_ewma_key = KEYS[6]
local window_key = KEYS[7]
local unhealthy_since_key = KEYS[8]
local last_scale_up_key = KEYS[9]

local base_key = ARGV[1]
local now = tonumber(ARGV[2])
local max_heads = tonumber(ARGV[3])
local scale_up_threshold = tonumber(ARGV[4])
local scale_down_threshold = tonumber(ARGV[5])
local scale_down_interval = tonumber(ARGV[6])
local scale_up_interval = tonumber(ARGV[7])
local notify_value = ARGV[8]
local notify_ttl = tonumber(ARGV[9])
local queue_scan_limit = tonumber(ARGV[10])

local function alive_key(run_id)
  return base_key .. ':alive:' .. run_id
end

local function notify_key(run_id)
  return base_key .. ':notify:' .. run_id
end

local function run_meta_key(run_id)
  return base_key .. ':run:' .. run_id
end

local active_runs = redis.call('ZRANGE', active_heads_key, 0, -1)
for _, run_id in ipairs(active_runs) do
  if redis.call('LPOS', queue_key, run_id) == false or redis.call('EXISTS', alive_key(run_id)) == 0 then
    redis.call('ZREM', active_heads_key, run_id)
    redis.call('LREM', queue_key, 1, run_id)
  end
end

local queue_len = redis.call('LLEN', queue_key)
local active_count = redis.call('ZCARD', active_heads_key)
local loaded = queue_len > 0 or active_count > 0
local window = tonumber(redis.call('GET', window_key) or '1')
if window < 1 then window = 1 end
if window > max_heads then window = max_heads end

if not loaded then
  redis.call('SET', window_key, 1)
  redis.call('DEL', unhealthy_since_key)
  return {1, 0, 0, -1, -1}
end

local token_raw = redis.call('GET', token_key)
local limit_raw = redis.call('GET', token_limit_key)
local ewma_raw = redis.call('GET', ewma_key)
local short_ewma_raw = redis.call('GET', short_ewma_key)
local health = -1
local scale_down_health = -1
if token_raw and limit_raw and ewma_raw and tonumber(limit_raw) and tonumber(limit_raw) > 0 then
  local current_ratio = tonumber(token_raw) / tonumber(limit_raw)
  local ewma_ratio = tonumber(ewma_raw)
  if current_ratio < ewma_ratio then
    health = current_ratio
  else
    health = ewma_ratio
  end

  if short_ewma_raw then
    local short_ewma_ratio = tonumber(short_ewma_raw)
    if current_ratio > short_ewma_ratio then
      scale_down_health = current_ratio
    else
      scale_down_health = short_ewma_ratio
    end
  else
    scale_down_health = health
  end
end

local backlog_exists = queue_len > active_count
if active_count == 0 then
  window = math.max(window, 1)
elseif backlog_exists and health >= scale_up_threshold then
  redis.call('DEL', unhealthy_since_key)
  local last_scale_up = tonumber(redis.call('GET', last_scale_up_key) or '0')
  if now - last_scale_up >= scale_up_interval and window < max_heads then
    window = window + 1
    redis.call('SET', last_scale_up_key, now)
  end
elseif scale_down_health >= 0 and scale_down_health < scale_down_threshold then
  local unhealthy_since = tonumber(redis.call('GET', unhealthy_since_key) or '0')
  if unhealthy_since == 0 then
    redis.call('SET', unhealthy_since_key, now)
  elseif now - unhealthy_since >= scale_down_interval and window > 1 then
    window = window - 1
    redis.call('SET', unhealthy_since_key, now)
  end
elseif scale_down_health >= 0 then
  redis.call('DEL', unhealthy_since_key)
end

redis.call('SET', window_key, window)

local admitted = 0
local run_ids = redis.call('LRANGE', queue_key, 0, queue_scan_limit - 1)
for _, run_id in ipairs(run_ids) do
  active_count = redis.call('ZCARD', active_heads_key)
  if active_count >= window then
    break
  end

  if redis.call('ZSCORE', active_heads_key, run_id) == false then
    if redis.call('EXISTS', alive_key(run_id)) == 0 then
      redis.call('LREM', queue_key, 1, run_id)
    else
      redis.call('ZADD', active_heads_key, now, run_id)
      redis.call('HSET', run_meta_key(run_id), 'slot_acquired', 1, 'slot_acquired_at', now)
      redis.call('RPUSH', notify_key(run_id), notify_value)
      redis.call('EXPIRE', notify_key(run_id), notify_ttl)
      admitted = admitted + 1
    end
  end
end

return {window, redis.call('ZCARD', active_heads_key), admitted, health, scale_down_health}
"""


class BenchmarkQueueCancelled(Exception):
    """Raised when a queued benchmark run is cancelled before acquiring its slot."""


async def _lpop_long(
    redis_client: AsyncRedisClient,
    keys: list[str],
    logger: logging.Logger | None = None,
    run_id: str = "",
    timeout_seconds: float | None = None,
) -> list[str] | None:
    """Poll notification lists without holding blocked Redis connections.

    Benchmark load creates many waiters. BLPOP holds one Redis connection per
    waiter, which can starve queue release operations. LPOP polling trades up to
    QUEUE_NOTIFY_POLL_INTERVAL seconds of wakeup latency for zero blocked clients.
    """
    deadline = time.monotonic() + (
        HOURS_24 if timeout_seconds is None else timeout_seconds
    )
    wait_started = time.monotonic() if BENCHMARK_QUEUE_DEBUG_ENABLED else 0.0
    last_poll_started = wait_started
    poll_count = 0

    while time.monotonic() < deadline:
        if BENCHMARK_QUEUE_DEBUG_ENABLED:
            poll_started = time.monotonic()
            poll_gap = poll_started - last_poll_started
            if poll_gap > QUEUE_WAITER_STALL_LOG_THRESHOLD and logger:
                logger.warning(
                    "Benchmark queue: waiter poll delayed for %s by %.3fs "
                    "pid=%s thread=%s polls=%s",
                    run_id,
                    poll_gap,
                    os.getpid(),
                    threading.get_ident(),
                    poll_count,
                )
            last_poll_started = poll_started
            poll_count += 1

        for key in keys:
            value = await redis_client.lpop(key)
            if value is not None:
                if BENCHMARK_QUEUE_DEBUG_ENABLED and logger:
                    logger.info(
                        "Benchmark queue: waiter consumed notification for %s "
                        "after %.3fs pid=%s thread=%s polls=%s key=%s value=%r",
                        run_id,
                        time.monotonic() - wait_started,
                        os.getpid(),
                        threading.get_ident(),
                        poll_count,
                        key,
                        value,
                    )
                return [key, value]

        await asyncio.sleep(QUEUE_NOTIFY_POLL_INTERVAL)

    return None


def get_active_heads_key(base_key: str) -> str:
    """Return the benchmark active-head set key for a benchmark base key."""
    return f"{base_key}:active_heads"


async def _control_and_admit_heads(
    redis_client: AsyncRedisClient,
    run_queue_key: str,
    base_key: str,
    token_key: str,
    logger: logging.Logger,
) -> None:
    """Adjust active-head window and admit FIFO waiters up to that window."""
    result = await redis_client.eval(
        CONTROL_AND_ADMIT_HEADS_LUA,
        9,
        run_queue_key,
        get_active_heads_key(base_key),
        token_key,
        f"{token_key}:limit",
        f"{token_key}:remaining_ratio_ewma_2m",
        f"{token_key}:remaining_ratio_ewma_15s",
        f"{base_key}:active_head_window",
        f"{base_key}:unhealthy_since",
        f"{base_key}:last_scale_up_at",
        base_key,
        time.time(),
        MAX_ACTIVE_HEADS,
        ACTIVE_HEAD_SCALE_UP_THRESHOLD,
        ACTIVE_HEAD_SCALE_DOWN_THRESHOLD,
        ACTIVE_HEAD_SCALE_DOWN_INTERVAL,
        ACTIVE_HEAD_SCALE_UP_INTERVAL,
        BENCHMARK_NOTIFY_PROCEED,
        HOURS_24,
        ACTIVE_HEAD_QUEUE_SCAN_LIMIT,
    )
    logger.debug(
        "Benchmark queue: active-head control for %s returned %s",
        run_queue_key,
        result,
    )


@dataclass(frozen=True)
class BenchmarkQueueKeys:
    base: str
    token: str
    queue: str
    active_heads: str
    notify: str
    alive: str
    run_meta: str
    dispatched: str

    @classmethod
    def for_run(
        cls,
        model_registry_key: tuple[str, str],
        run_id: str,
    ) -> "BenchmarkQueueKeys":
        base = f"{KEY_PREFIX}:{model_registry_key[0]}:{model_registry_key[1]}:benchmark"
        return cls.for_base(base, run_id)

    @classmethod
    def for_base(cls, base: str, run_id: str) -> "BenchmarkQueueKeys":
        run_meta = f"{base}:run:{run_id}"
        return cls(
            base=base,
            token=f"{base.removesuffix(':benchmark')}:tokens",
            queue=f"{base}:queue",
            active_heads=get_active_heads_key(base),
            notify=f"{base}:notify:{run_id}",
            alive=f"{base}:alive:{run_id}",
            run_meta=run_meta,
            dispatched=f"{run_meta}:dispatched",
        )


async def clear_benchmark_notification(
    redis_client: AsyncRedisClient,
    keys: BenchmarkQueueKeys,
) -> None:
    """Clear stale notifications before checking a new local attempt."""
    await redis_client.delete(keys.notify)


async def initialize_benchmark_run(
    redis_client: AsyncRedisClient,
    keys: BenchmarkQueueKeys,
    run_id: str,
    total_requests: int | None,
    logger: logging.Logger,
) -> None:
    """Reset attempt state, enqueue once, and run admission control."""
    # initialize heartbeat with short TTL
    await redis_client.set(keys.alive, "1", ex=HEARTBEAT_TTL)

    # per-run metadata for debugging
    await redis_client.delete(keys.dispatched)
    await redis_client.hset(
        keys.run_meta,
        mapping={
            "total_requests": total_requests or 0,
            "slot_acquired": 0,
            "enqueued_at": time.time(),
        },
    )
    await redis_client.hdel(
        keys.run_meta,
        "popped_at",
        "completed_at",
        "slot_acquired_at",
        "outcome",
        "early_release_deadline",
    )
    await redis_client.expire(keys.run_meta, HOURS_24)

    # Idempotent enqueue (handles server restart where run is already in queue)
    # NOTE: lpos returns the index (0 for first element), so we must check `is None`
    if await redis_client.lpos(keys.queue, run_id) is None:
        await redis_client.rpush(keys.queue, run_id)
    await redis_client.expire(keys.queue, HOURS_24)

    await _control_and_admit_heads(
        redis_client,
        keys.queue,
        keys.base,
        keys.token,
        logger,
    )


async def wait_for_benchmark_slot(
    redis_client: AsyncRedisClient,
    keys: BenchmarkQueueKeys,
    run_id: str,
    logger: logging.Logger,
    is_cancelled: Callable[[], Awaitable[bool]] | None,
    *,
    timeout_seconds: float | None = None,
) -> bool:
    """Wait for an authoritative active-head notification."""
    # block until notified
    while True:
        result = await _lpop_long(
            redis_client,
            [keys.notify],
            logger=logger,
            run_id=run_id,
            timeout_seconds=timeout_seconds,
        )
        if result is None:
            return False

        notification = result[1]
        if notification == BENCHMARK_NOTIFY_CANCELLED:
            raise BenchmarkQueueCancelled(
                f"Run {run_id} cancelled while waiting in benchmark queue"
            )

        if notification != BENCHMARK_NOTIFY_PROCEED:
            logger.warning(
                f"Benchmark queue: ignoring unknown notification {notification!r} for {run_id}"
            )
            continue

        is_active_head = await redis_client.zscore(keys.active_heads, run_id)
        if is_active_head is None:
            logger.warning(
                f"Benchmark queue: ignoring stale proceed notification for {run_id}; not an active head"
            )
            continue

        if is_cancelled and await is_cancelled():
            raise BenchmarkQueueCancelled(
                f"Run {run_id} cancelled before acquiring benchmark queue slot"
            )

        # Clear duplicate proceed tokens after the slot is actually acquired.
        await redis_client.delete(keys.notify)
        return True


async def mark_benchmark_slot_acquired(
    redis_client: AsyncRedisClient,
    keys: BenchmarkQueueKeys,
) -> None:
    """Record that the caller consumed its authoritative admission."""
    await redis_client.hset(
        keys.run_meta,
        mapping={
            "slot_acquired": 1,
            "slot_acquired_at": time.time(),
        },
    )


async def release_benchmark_run(
    redis_client: AsyncRedisClient,
    keys: BenchmarkQueueKeys,
    run_id: str,
    logger: logging.Logger,
    *,
    notify_next: bool,
) -> None:
    """Release queue state and retain the existing 24-hour run history."""
    # remove ourselves from queue/active heads (idempotent if heartbeat already early-released)
    await redis_client.lrem(keys.queue, 1, run_id)
    await redis_client.zrem(keys.active_heads, run_id)
    await redis_client.delete(keys.alive)

    # mark completed and keep metadata alive for popped run visibility
    now = time.time()
    fields: dict[str, float] = {"completed_at": now}
    # set popped_at if not already set by early release in heartbeat
    if not await redis_client.hexists(keys.run_meta, "popped_at"):
        fields["popped_at"] = now
    await redis_client.hset(keys.run_meta, mapping=fields)
    await redis_client.expire(keys.run_meta, HOURS_24)
    await redis_client.expire(keys.dispatched, HOURS_24)

    if notify_next:
        await _control_and_admit_heads(
            redis_client,
            keys.queue,
            keys.base,
            keys.token,
            logger,
        )


async def refresh_benchmark_heartbeat(
    redis_client: AsyncRedisClient,
    keys: BenchmarkQueueKeys,
) -> None:
    """Refresh one run's liveness marker."""
    # refresh our heartbeat
    await redis_client.set(keys.alive, "1", ex=HEARTBEAT_TTL)


async def early_release_benchmark_run(
    redis_client: AsyncRedisClient,
    keys: BenchmarkQueueKeys,
    run_id: str,
    logger: logging.Logger,
) -> bool:
    """Atomically release an acquired queue slot before final run cleanup."""
    released_run = await redis_client.eval(
        EARLY_RELEASE_LUA,
        3,
        keys.run_meta,
        keys.queue,
        keys.active_heads,
        run_id,
        time.time(),
    )
    if not released_run:
        return False

    logger.debug("Benchmark queue: early released %s", released_run)
    await _control_and_admit_heads(
        redis_client,
        keys.queue,
        keys.base,
        keys.token,
        logger,
    )
    return True


async def control_benchmark_run(
    redis_client: AsyncRedisClient,
    keys: BenchmarkQueueKeys,
    run_id: str,
    logger: logging.Logger,
    *,
    self_promote: bool,
) -> None:
    """Run active-head control, notification recovery, and dead-head eviction."""
    await _control_and_admit_heads(
        redis_client,
        keys.queue,
        keys.base,
        keys.token,
        logger,
    )

    # self-promote: if we're active but missed notification, notify ourselves.
    if self_promote:
        is_active_head = await redis_client.zscore(keys.active_heads, run_id)
        if is_active_head is not None:
            await redis_client.rpush(keys.notify, BENCHMARK_NOTIFY_PROCEED)
            await redis_client.expire(keys.notify, HOURS_24)

    # check queue head heartbeat
    head = await redis_client.lindex(keys.queue, 0)
    if head and head != run_id:
        head_alive_key = f"{keys.base}:alive:{head}"
        if not await redis_client.exists(head_alive_key):
            async with redis_client.lock(
                f"{keys.queue}:evict",
                timeout=HEARTBEAT_INTERVAL,
            ):
                # re-check after acquiring lock
                head = await redis_client.lindex(keys.queue, 0)
                if (
                    head
                    and head != run_id
                    and not await redis_client.exists(f"{keys.base}:alive:{head}")
                ):
                    logger.info(f"Benchmark queue: evicting dead entry {head}")
                    await redis_client.lrem(keys.queue, 1, head)
                    await redis_client.zrem(keys.active_heads, head)
                    await _control_and_admit_heads(
                        redis_client,
                        keys.queue,
                        keys.base,
                        keys.token,
                        logger,
                    )


@asynccontextmanager
async def benchmark_queue(
    model_registry_key: tuple[str, str],
    run_id: str,
    logger: logging.Logger,
    enabled: bool = True,
    total_requests: int | None = None,
    early_release: bool = True,
    immediate_queue_release: bool = False,
    is_cancelled: Callable[[], Awaitable[bool]] | None = None,
):
    """
    FIFO queue with a token-health active-head window for a given model.

    When token retry is enabled, benchmark runs are admitted through a shared
    concurrency window so light workloads can overlap without guessing request
    sizes. The Redis token bucket still enforces hard TPM safety per request.

    - each run maintains a heartbeat key with a short TTL
    - a background task
        - keeps the heartbeat alive
        - runs active-head control and admits FIFO waiters up to the window
        - if total_requests is set, releases the active-head slot early once all
          requests have been dispatched (entered inflight)

    After early release, remaining inflight requests are "stragglers" — detected
    by TokenRetrier checking that its benchmark run_id is no longer active.

    Args:
        model_registry_key (tuple[str, str]): model registry key (or a unique model identifier shared across instance of the same model)
        run_id (str): unique run identifier (or a unique identifier for a benchmark run)
        total_requests (int | None): total number of requests to dispatch before early release
        immediate_queue_release (bool): release as soon as all requests are dispatched, without the grace period
    """

    if not enabled:
        yield
        return

    keys = BenchmarkQueueKeys.for_run(model_registry_key, run_id)

    heartbeat_task = None
    slot_acquired: asyncio.Event | None = None

    await utils.validate_redis_client()

    try:
        # Clear stale notifications from a previous attempt with the same run_id.
        await clear_benchmark_notification(utils.redis_client, keys)
        if is_cancelled and await is_cancelled():
            raise BenchmarkQueueCancelled(
                f"Run {run_id} cancelled before entering benchmark queue"
            )

        await initialize_benchmark_run(
            utils.redis_client,
            keys,
            run_id,
            total_requests,
            logger,
        )

        # signal heartbeat that slot has been acquired
        slot_acquired = asyncio.Event()

        # per-run dispatched key for early release counting
        run_dispatched_key = (
            keys.dispatched if total_requests and early_release else None
        )

        # start heartbeat (refreshes my alive key + active-head control + early release)
        heartbeat_task = asyncio.create_task(
            _heartbeat(
                utils.redis_client,
                keys,
                run_id,
                logger,
                dispatched_key=run_dispatched_key,
                total_requests=total_requests,
                slot_acquired=slot_acquired,
                immediate_queue_release=immediate_queue_release,
                is_cancelled=is_cancelled,
            )
        )

        logger.info(f"Benchmark queue: {run_id} waiting for slot ({keys.queue})")

        acquired = await wait_for_benchmark_slot(
            utils.redis_client,
            keys,
            run_id,
            logger,
            is_cancelled,
        )
        if not acquired:
            raise RuntimeError(f"Run {run_id} timed out waiting in benchmark queue")

        logger.info(
            "Benchmark queue: %s acquired slot pid=%s thread=%s",
            run_id,
            os.getpid(),
            threading.get_ident(),
        )

        slot_acquired.set()
        await mark_benchmark_slot_acquired(utils.redis_client, keys)

        yield

    finally:
        if heartbeat_task:
            heartbeat_task.cancel()

        async def _cleanup() -> None:
            should_notify_next = slot_acquired is not None and slot_acquired.is_set()
            if should_notify_next and is_cancelled:
                try:
                    should_notify_next = not await is_cancelled()
                except Exception:
                    logger.warning(
                        f"Benchmark queue: cancellation check failed during cleanup for {run_id}",
                        exc_info=True,
                    )
            await release_benchmark_run(
                utils.redis_client,
                keys,
                run_id,
                logger,
                notify_next=should_notify_next,
            )

        await asyncio.shield(_cleanup())

        logger.info(f"Benchmark queue: {run_id} released slot")


async def _heartbeat(
    redis_client: AsyncRedisClient,
    keys: BenchmarkQueueKeys,
    run_id: str,
    logger: logging.Logger,
    dispatched_key: str | None = None,
    total_requests: int | None = None,
    slot_acquired: asyncio.Event | None = None,
    immediate_queue_release: bool = False,
    is_cancelled: Callable[[], Awaitable[bool]] | None = None,
):
    """
    - keeps the heartbeat alive
    - runs active-head control and cleans up dead entries
    - if total_requests is set, releases active-head slot when all requests are dispatched
      (after EARLY_RELEASE_GRACE_PERIOD expires unless immediate_queue_release is set)
    """

    grace_deadline: float | None = None
    last_heartbeat_started = time.monotonic() if BENCHMARK_QUEUE_DEBUG_ENABLED else 0.0

    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL)
        if BENCHMARK_QUEUE_DEBUG_ENABLED:
            heartbeat_started = time.monotonic()
            heartbeat_gap = heartbeat_started - last_heartbeat_started
            if heartbeat_gap > HEARTBEAT_INTERVAL + HEARTBEAT_STALL_LOG_THRESHOLD:
                logger.warning(
                    "Benchmark queue: heartbeat delayed for %s by %.3fs "
                    "pid=%s thread=%s slot_acquired=%s",
                    run_id,
                    heartbeat_gap - HEARTBEAT_INTERVAL,
                    os.getpid(),
                    threading.get_ident(),
                    slot_acquired.is_set() if slot_acquired else None,
                )
            last_heartbeat_started = heartbeat_started
        try:
            await refresh_benchmark_heartbeat(redis_client, keys)

            # early release: all requests dispatched
            # only check after slot acquired
            if (
                dispatched_key
                and total_requests is not None
                and slot_acquired
                and slot_acquired.is_set()
            ):
                count = await redis_client.scard(dispatched_key)
                if count >= total_requests:
                    if is_cancelled and await is_cancelled():
                        dispatched_key = None
                        continue

                    if immediate_queue_release:
                        if BENCHMARK_QUEUE_DEBUG_ENABLED:
                            logger.info(
                                f"Benchmark queue: all {total_requests} dispatched, "
                                f"immediate early releasing {run_id}"
                            )
                    else:
                        if grace_deadline is None:
                            grace_deadline = (
                                time.monotonic() + EARLY_RELEASE_GRACE_PERIOD
                            )
                            if BENCHMARK_QUEUE_DEBUG_ENABLED:
                                logger.info(
                                    f"Benchmark queue: all {total_requests} dispatched, "
                                    f"grace period {EARLY_RELEASE_GRACE_PERIOD}s for {run_id}"
                                )

                        if time.monotonic() < grace_deadline:
                            continue

                        if BENCHMARK_QUEUE_DEBUG_ENABLED:
                            logger.info(
                                f"Benchmark queue: grace period expired, "
                                f"early releasing {run_id}"
                            )

                    await early_release_benchmark_run(
                        redis_client,
                        keys,
                        run_id,
                        logger,
                    )
                    dispatched_key = None  # stop checking

            await control_benchmark_run(
                redis_client,
                keys,
                run_id,
                logger,
                self_promote=bool(slot_acquired and not slot_acquired.is_set()),
            )
        except Exception:
            logger.warning(
                f"Benchmark queue heartbeat error for {run_id}", exc_info=True
            )
