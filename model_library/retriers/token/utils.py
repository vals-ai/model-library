import time
from typing import Any, Protocol

from pydantic import BaseModel
from redis.asyncio import Redis
from redis.asyncio.lock import Lock


class AsyncRedisClient(Protocol):
    """Typed protocol for the subset of async Redis commands we use."""

    async def get(self, name: str) -> str | None: ...
    async def set(
        self, name: str, value: str | int | float, ex: int | None = None
    ) -> bool: ...
    async def exists(self, *names: str) -> int: ...
    async def delete(self, *names: str) -> int: ...
    async def expire(self, name: str, time: int) -> bool: ...
    async def incr(self, name: str) -> int: ...
    async def decr(self, name: str) -> int: ...
    async def incrby(self, name: str, amount: int) -> int: ...
    async def decrby(self, name: str, amount: int) -> int: ...
    async def lpos(self, name: str, value: str) -> int | None: ...
    async def rpush(self, name: str, *values: str) -> int: ...
    async def lrem(self, name: str, count: int, value: str) -> int: ...
    async def lindex(self, name: str, index: int) -> str | None: ...
    async def llen(self, name: str) -> int: ...
    async def lrange(self, name: str, start: int, end: int) -> list[str]: ...
    async def blpop(
        self, keys: list[str], timeout: int | float | None = 0
    ) -> list[str] | None: ...
    async def ttl(self, name: str) -> int: ...
    async def zadd(
        self, name: str, mapping: dict[str, float], nx: bool = False
    ) -> int: ...
    async def zrem(self, name: str, *members: str) -> int: ...
    async def zrangebyscore(
        self, name: str, min: float | str, max: float | str, withscores: bool = False
    ) -> list[str]: ...
    async def sadd(self, name: str, *values: str) -> int: ...
    async def scard(self, name: str) -> int: ...
    async def zcard(self, name: str) -> int: ...
    async def keys(self, pattern: str) -> list[str]: ...
    async def eval(
        self, script: str, numkeys: int, *keys_and_args: str | int | float
    ) -> Any: ...
    def lock(self, name: str, timeout: float | None = None) -> Lock: ...


redis_client: AsyncRedisClient = None  # pyright: ignore[reportAssignmentType]


def set_redis_client(client: Redis):
    global redis_client
    redis_client = client  # pyright: ignore[reportAssignmentType]


async def validate_redis_client(
    key: str | None = None, missing_key_message: str | None = None
):
    try:
        assert redis_client
    except Exception as e:
        raise Exception(f"redis client not set, run `set_redis_client`. Exception: {e}")

    if key:
        if not await redis_client.exists(key):
            raise Exception(f"{key} not intialized. {missing_key_message}")


async def get_token_keys() -> list[str]:
    """Return all valid token retry base keys"""

    keys: list[str] = []
    for key in await redis_client.keys("*:*:tokens"):
        val = await redis_client.get(key)
        try:
            int(val) if val else 0
        except ValueError:
            continue
        keys.append(key)
    return sorted(keys)


async def cleanup_all_keys() -> int:
    """Delete all token retry and benchmark queue keys"""

    to_delete: list[str] = []

    # token retry keys: find base token keys, then derive related keys
    for token_key in await get_token_keys():
        to_delete.append(token_key)
        # derived keys: limit, lock, version, dynamic estimates
        to_delete.extend(await redis_client.keys(f"{token_key}:*"))
        to_delete.extend(await redis_client.keys(f"version:{token_key}"))

    # priority keys (derived from token key prefix)
    to_delete.extend(await redis_client.keys("*:*:priority:*"))

    # benchmark queue keys
    to_delete.extend(await redis_client.keys("*:benchmark:*"))

    if not to_delete:
        return 0
    return await redis_client.delete(*to_delete)


# ── Status models ────────────────────────────────────────────────────


class InflightRequest(BaseModel):
    request_id: str
    elapsed_seconds: float


class TokenRetryStatus(BaseModel):
    token_key: str
    tokens_remaining: int
    token_limit: int
    inflight: list[InflightRequest]
    priorities: dict[str, int]
    local_refill_tasks_alive: int


class QueueEntry(BaseModel):
    run_id: str
    alive: bool
    heartbeat_ttl: int


class QueueStatus(BaseModel):
    queue_key: str
    length: int
    entries: list[QueueEntry]


class Status(BaseModel):
    token_retry: list[TokenRetryStatus]
    benchmark_queue: list[QueueStatus]


# ── Status function ──────────────────────────────────────────────────


async def get_status() -> Status:
    """Get combined token retry and benchmark queue status."""

    return Status(
        token_retry=await _get_token_retry_status(),
        benchmark_queue=await _get_queue_status(),
    )


async def _get_token_retry_status() -> list[TokenRetryStatus]:
    from model_library.retriers.token.token import (
        MAX_PRIORITY,
        MIN_PRIORITY,
        refill_tasks,
    )

    token_keys = await get_token_keys()

    statuses: list[TokenRetryStatus] = []
    for token_key in token_keys:
        limit_key = f"{token_key}:limit"

        tokens_raw = await redis_client.get(token_key)
        tokens_remaining = int(tokens_raw) if tokens_raw else 0

        limit_raw = await redis_client.get(limit_key)
        token_limit = int(limit_raw) if limit_raw else 0

        now = time.time()
        inflight_entries = await redis_client.zrangebyscore(
            f"{token_key}:inflight", "-inf", "+inf", withscores=True
        )
        inflight = sorted(
            [
                InflightRequest(
                    request_id=member, elapsed_seconds=round(now - float(score), 1)
                )
                for member, score in inflight_entries
            ],
            key=lambda r: r.elapsed_seconds,
            reverse=True,
        )

        # derive client_registry_key from token_key ("{provider}:{key}:tokens")
        parts = token_key.removesuffix(":tokens").rsplit(":", 1)
        client_registry_key = (parts[0], parts[1]) if len(parts) == 2 else ("", "")

        priorities: dict[str, int] = {}
        for priority in range(MAX_PRIORITY, MIN_PRIORITY + 1):
            key = (
                f"{client_registry_key[0]}:{client_registry_key[1]}:priority:{priority}"
            )
            count = await redis_client.zcard(key)
            priorities[str(priority)] = count

        alive = sum(
            1
            for k, (_, task) in refill_tasks.items()
            if k.endswith(token_key) and not task.done()
        )

        statuses.append(
            TokenRetryStatus(
                token_key=token_key,
                tokens_remaining=tokens_remaining,
                token_limit=token_limit,
                inflight=inflight,
                priorities=priorities,
                local_refill_tasks_alive=alive,
            )
        )

    return statuses


async def _get_queue_status() -> list[QueueStatus]:
    queue_keys = sorted(await redis_client.keys("*:benchmark:queue"))

    statuses: list[QueueStatus] = []
    for run_queue_key in queue_keys:
        base_key = run_queue_key.removesuffix(":queue")
        run_ids = await redis_client.lrange(run_queue_key, 0, -1)
        if not run_ids:
            continue

        entries: list[QueueEntry] = []
        for run_id in run_ids:
            alive_key = f"{base_key}:alive:{run_id}"
            alive = await redis_client.exists(alive_key) > 0
            ttl = await redis_client.ttl(alive_key) if alive else -1
            entries.append(QueueEntry(run_id=run_id, alive=alive, heartbeat_ttl=ttl))

        statuses.append(
            QueueStatus(
                queue_key=run_queue_key,
                length=len(entries),
                entries=entries,
            )
        )

    return statuses
