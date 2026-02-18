import asyncio
import logging
from contextlib import asynccontextmanager

from model_library.retriers.token import utils
from model_library.retriers.token.utils import AsyncRedisClient

HOURS_24 = 86400

# atomic compare-and-delete: only delete keys if KEYS[1] still equals ARGV[1]
CLEANUP_IF_OWNER_LUA = """
local current = redis.call('GET', KEYS[1])
if current == ARGV[1] then
    redis.call('DEL', KEYS[1], KEYS[2])
    return 1
end
return 0
"""
HEARTBEAT_INTERVAL = 5
HEARTBEAT_TTL = 300  # 5 minutes


async def _notify_next(
    redis_client: AsyncRedisClient, run_queue_key: str, base_key: str
) -> None:
    """Notify the next run in queue (if any) to proceed"""
    next_run = await redis_client.lindex(run_queue_key, 0)
    if next_run:
        next_key = f"{base_key}:notify:{next_run}"
        await redis_client.rpush(next_key, "go")
        await redis_client.expire(next_key, HOURS_24)


@asynccontextmanager
async def benchmark_queue(
    model_registry_key: tuple[str, str],
    run_id: str,
    logger: logging.Logger,
    enabled: bool = True,
    total_requests: int | None = None,
):
    """
    FIFO queue that serializes benchmark runs for a given model

    When token retry is enabled, only one run should execute at a time per model
    to avoid competing for the same TPM. This context manager can coordinate
    across many applications, as long as the redis connection is shared.

    - each run maintains a heartbeat key with a short TTL
    - a background task
        - keeps the heartbeat alive
        - removes queue head if their heartbeat expires
        - if total_requests is set, releases the slot early once all requests
          have been dispatched (entered inflight)

    After early release, remaining inflight requests are "stragglers" — detected
    by TokenRetrier checking that its benchmark run_id is no longer the queue head.

    Args:
        model_registry_key (tuple[str, str]): model registry key (or a unique model identifier shared across instance of the same model)
        run_id (str): unique run identifier (or a unique identifier for a benchmark run)
        total_requests (int | None): total number of requests to dispatch before early release
    """

    if not enabled:
        yield
        return

    key = f"{model_registry_key[0]}:{model_registry_key[1]}:benchmark"
    run_queue_key = f"{key}:queue"
    my_notify_key = f"{key}:notify:{run_id}"
    alive_key = f"{key}:alive:{run_id}"
    inflight_key = f"{model_registry_key[0]}:{model_registry_key[1]}:tokens:inflight"
    dispatched_key = f"{inflight_key}:dispatched"
    benchmark_run_key = f"{inflight_key}:benchmark_run"

    heartbeat_task = None
    slot_acquired_flag = False

    await utils.validate_redis_client()

    try:
        # initialize heartbeat with short TTL
        await utils.redis_client.set(alive_key, "1", ex=HEARTBEAT_TTL)

        # Idempotent enqueue (handles server restart where run is already in queue)
        # NOTE: lpos returns the index (0 for first element), so we must check `is None`
        if await utils.redis_client.lpos(run_queue_key, run_id) is None:
            await utils.redis_client.rpush(run_queue_key, run_id)
        await utils.redis_client.expire(run_queue_key, HOURS_24)

        # if first in queue, notify myself to proceed
        first_run = await utils.redis_client.lindex(run_queue_key, 0)
        if first_run == run_id:
            await utils.redis_client.rpush(my_notify_key, "go")

        # signal heartbeat that slot has been acquired and dispatched set cleared
        slot_acquired = asyncio.Event()

        # start heartbeat (refreshes my alive key + evict dead head + early release)
        heartbeat_task = asyncio.create_task(
            _heartbeat(
                utils.redis_client,
                alive_key,
                run_queue_key,
                key,
                run_id,
                logger,
                dispatched_key=dispatched_key if total_requests else None,
                total_requests=total_requests,
                slot_acquired=slot_acquired,
            )
        )

        logger.info(f"Benchmark queue: {run_id} waiting for slot ({run_queue_key})")

        # block until notified
        result = await utils.redis_client.blpop([my_notify_key], timeout=HOURS_24)
        if result is None:
            raise RuntimeError(f"Run {run_id} timed out waiting in benchmark queue")

        await utils.redis_client.delete(my_notify_key)

        logger.info(f"Benchmark queue: {run_id} acquired slot")

        # register active run in Redis so TokenRetrier can detect stragglers
        await utils.redis_client.set(benchmark_run_key, run_id, ex=HOURS_24)

        if total_requests is not None:
            await utils.redis_client.delete(dispatched_key)

        slot_acquired_flag = True
        slot_acquired.set()

        yield

    finally:
        if heartbeat_task:
            heartbeat_task.cancel()

        # remove ourselves from queue (idempotent if heartbeat already early-released)
        await utils.redis_client.lrem(run_queue_key, 1, run_id)

        await utils.redis_client.delete(alive_key)

        # atomic cleanup: only delete shared keys if we're still the active run
        if slot_acquired_flag:
            await utils.redis_client.eval(
                CLEANUP_IF_OWNER_LUA, 2, benchmark_run_key, dispatched_key, run_id
            )

        await _notify_next(utils.redis_client, run_queue_key, key)

        logger.info(f"Benchmark queue: {run_id} released slot")


async def _heartbeat(
    redis_client: AsyncRedisClient,
    alive_key: str,
    run_queue_key: str,
    base_key: str,
    run_id: str,
    logger: logging.Logger,
    dispatched_key: str | None = None,
    total_requests: int | None = None,
    slot_acquired: asyncio.Event | None = None,
):
    """
    - keeps the heartbeat alive
    - removes queue head if their heartbeat expires
    - if total_requests is set, releases queue slot when all requests are dispatched
    """

    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL)
        try:
            # refresh our heartbeat
            await redis_client.set(alive_key, "1", ex=HEARTBEAT_TTL)

            # early release: all requests dispatched
            # only check after slot acquired (dispatched set is shared across runs)
            if (
                dispatched_key
                and total_requests is not None
                and slot_acquired
                and slot_acquired.is_set()
            ):
                count = await redis_client.scard(dispatched_key)
                if count >= total_requests:
                    logger.info(
                        f"Benchmark queue: all {total_requests} dispatched, "
                        f"early releasing {run_id}"
                    )
                    await redis_client.lrem(run_queue_key, 1, run_id)
                    await _notify_next(redis_client, run_queue_key, base_key)
                    dispatched_key = None  # stop checking

            # self-promote: if we're at head but missed notification (previous
            # head crashed between lrem and _notify_next), notify ourselves
            if slot_acquired and not slot_acquired.is_set():
                head = await redis_client.lindex(run_queue_key, 0)
                if head == run_id:
                    notify_key = f"{base_key}:notify:{run_id}"
                    await redis_client.rpush(notify_key, "go")
                    await redis_client.expire(notify_key, HOURS_24)

            # check queue head heartbeat
            head = await redis_client.lindex(run_queue_key, 0)
            if head and head != run_id:
                head_alive_key = f"{base_key}:alive:{head}"
                if not await redis_client.exists(head_alive_key):
                    async with redis_client.lock(
                        f"{run_queue_key}:evict", timeout=HEARTBEAT_INTERVAL
                    ):
                        # re-check after acquiring lock
                        head = await redis_client.lindex(run_queue_key, 0)
                        if (
                            head
                            and head != run_id
                            and not await redis_client.exists(
                                f"{base_key}:alive:{head}"
                            )
                        ):
                            logger.info(f"Benchmark queue: evicting dead entry {head}")
                            await redis_client.lrem(run_queue_key, 1, head)
                            await _notify_next(redis_client, run_queue_key, base_key)
        except Exception:
            logger.warning(
                f"Benchmark queue heartbeat error for {run_id}", exc_info=True
            )
