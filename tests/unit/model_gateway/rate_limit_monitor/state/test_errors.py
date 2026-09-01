from typing import cast
from unittest.mock import AsyncMock

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from model_gateway.rate_limit_monitor.state import (
    MonitorRedis,
    MonitorSourceUpdate,
    RateLimitMonitorStore,
)

from tests.unit.model_gateway.rate_limit_monitor.state._support import (
    MODEL,
    ANTHROPIC_MODEL,
    _attempted_at,
    _claim_generation,
    _success,
    _error,
)


@pytest.mark.parametrize("operation", ["activate", "list", "publish"])
async def test_redis_transport_failures_are_not_remapped(operation):
    redis = AsyncMock()
    redis.eval.side_effect = RedisConnectionError("unavailable")
    store = RateLimitMonitorStore(cast(MonitorRedis, redis))

    with pytest.raises(RedisConnectionError):
        if operation == "activate":
            await store.activate(MODEL, ("default",))
        elif operation == "list":
            await store.list_states()
        else:
            await store.publish_source(
                MODEL,
                "owner",
                "a" * 32,
                _attempted_at(),
                _success("default", 1.0),
            )


async def test_owner_takeover_rejects_old_owner_publication(store):
    activated = await store.activate(MODEL, ("default",))
    generation = await _claim_generation(store, MODEL, "old-owner")
    await store.release_owner(MODEL, "old-owner")
    assert await _claim_generation(store, MODEL, "new-owner") == generation

    assert not await store.publish_source(
        MODEL,
        "old-owner",
        generation,
        _attempted_at(),
        _success("default", activated.server_time),
    )
    assert await store.publish_source(
        MODEL,
        "new-owner",
        generation,
        _attempted_at(),
        _success("default", activated.server_time),
    )


async def test_unsupported_and_error_statuses_match_python_derivation(store):
    await store.activate(ANTHROPIC_MODEL, ("pool_1", "pool_2"))
    deadline = await _claim_generation(store, ANTHROPIC_MODEL, "owner")
    unsupported = MonitorSourceUpdate(source="pool_1", status="unsupported")

    assert await store.publish_source(
        ANTHROPIC_MODEL,
        "owner",
        deadline,
        _attempted_at(),
        unsupported,
    )
    assert await store.publish_source(
        ANTHROPIC_MODEL,
        "owner",
        deadline,
        _attempted_at(),
        _error("pool_2"),
    )
    state = (await store.list_states()).states[0]

    assert state.status == "error"
    assert [source.status for source in state.sources] == ["unsupported", "error"]
