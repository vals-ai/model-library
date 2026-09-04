from itertools import count
import json
import time

from fakeredis import aioredis

from model_gateway.rate_limit_monitor.state import (
    ACTIVE_KEY,
    RETAINED_KEY,
    RETENTION_SECONDS,
    MonitorSourceUpdate,
    RateLimitMonitorStore,
    snapshot_key,
)
from model_gateway.rate_limit_monitor.types import MonitorSourceName
from model_library.rate_limits import (
    RateLimit,
    RateLimitCapacity,
    RequestRateLimit,
    TokenRateLimit,
)

MODEL = "openai/gpt-4o"

ANTHROPIC_MODEL = "anthropic/claude-sonnet-4-5"

_ATTEMPTED_AT_OFFSETS = count(1)


def _attempted_at() -> float:
    return time.time() - next(_ATTEMPTED_AT_OFFSETS)


async def _claim_generation(
    store: RateLimitMonitorStore,
    model: str,
    token: str,
) -> str:
    assert await store.claim_owner(model, token) is not None
    renewal = await store.renew_owner(model, token)
    assert renewal is not None
    return renewal[0]


def _success(source: MonitorSourceName, timestamp: float) -> MonitorSourceUpdate:
    return MonitorSourceUpdate(
        source=source,
        status="ok",
        rate_limit=RateLimit(
            requests=(RequestRateLimit(limit=10_000, remaining=9_000),),
            tokens=TokenRateLimit(
                total=RateLimitCapacity(limit=1_000_000, remaining=900_000)
            ),
            unix_timestamp=timestamp - 1,
        ),
    )


def _scoped_success(
    source: MonitorSourceName,
    timestamp: float,
) -> MonitorSourceUpdate:
    rate_limit = RateLimit(
        requests=(RequestRateLimit(limit=25, remaining=20, mode="concurrency"),),
        tokens=TokenRateLimit(
            input=RateLimitCapacity(limit=600, remaining=300),
            output=RateLimitCapacity(limit=400, remaining=100),
        ),
        scope="shared",
        unix_timestamp=timestamp - 1,
    )
    return MonitorSourceUpdate(source=source, status="ok", rate_limit=rate_limit)


def _error(source: MonitorSourceName) -> MonitorSourceUpdate:
    return MonitorSourceUpdate(source=source, status="error")


async def _expire_activation(redis: aioredis.FakeRedis, model: str) -> float:
    active_until = (int(time.time() * 1_000) - 1_000) / 1_000
    await redis.zadd(ACTIVE_KEY, {model: active_until})
    await redis.zadd(
        RETAINED_KEY,
        {model: active_until + RETENTION_SECONDS},
    )
    return active_until


async def _expire_retention(redis: aioredis.FakeRedis, model: str) -> None:
    retention_until = (int(time.time() * 1_000) - 1_000) / 1_000
    await redis.zadd(
        ACTIVE_KEY,
        {model: retention_until - RETENTION_SECONDS},
    )
    await redis.zadd(RETAINED_KEY, {model: retention_until})


async def _persist_invalid_snapshot(redis, raw: str) -> None:
    now = time.time()
    await redis.zadd(ACTIVE_KEY, {MODEL: now + 1_800})
    await redis.zadd(RETAINED_KEY, {MODEL: now + 88_200})
    await redis.set(snapshot_key(MODEL), raw)


def _invalid_source_set_snapshot() -> str:
    return json.dumps(
        {
            "generation": "a" * 32,
            "model": MODEL,
            "sources": [],
        }
    )
