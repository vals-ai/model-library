import asyncio
import time
from typing import Literal

import pytest
from fakeredis import aioredis

from model_gateway.rate_limit_monitor.state import (
    ACTIVE_KEY,
    RETAINED_KEY,
    MonitorSourceUpdate,
    RateLimitMonitorStore,
    owner_key,
    snapshot_key,
)
from model_gateway.rate_limit_monitor.types import (
    MonitorSourceState,
)
from model_library.rate_limits import (
    RateLimit,
    RateLimitCapacity,
    RequestRateLimit,
    TokenRateLimit,
)

from tests.unit.model_gateway.rate_limit_monitor.state._support import (
    MODEL,
    ANTHROPIC_MODEL,
    _attempted_at,
    _claim_generation,
    _success,
    _scoped_success,
    _error,
    _expire_activation,
)


async def test_anthropic_publications_merge_without_lost_updates(store):
    activated = await store.activate(ANTHROPIC_MODEL, ("pool_1", "pool_2"))
    generation = await _claim_generation(store, ANTHROPIC_MODEL, "owner")

    accepted = await asyncio.gather(
        store.publish_source(
            ANTHROPIC_MODEL,
            "owner",
            generation,
            _attempted_at(),
            _success("pool_1", activated.server_time),
        ),
        store.publish_source(
            ANTHROPIC_MODEL,
            "owner",
            generation,
            _attempted_at(),
            _success("pool_2", activated.server_time + 1),
        ),
    )
    listed = await store.list_states()

    assert accepted == [True, True]
    assert listed.states[0].status == "ok"
    assert [source.status for source in listed.states[0].sources] == ["ok", "ok"]


async def test_renewal_redis_timestamp_becomes_publication_timestamp(store):
    activated = await store.activate(MODEL, ("default",))
    generation = await _claim_generation(store, MODEL, "owner")
    renewal = await store.renew_owner(MODEL, "owner")
    assert renewal is not None
    renewed_generation, attempted_at = renewal
    assert renewed_generation == generation

    assert await store.publish_source(
        MODEL,
        "owner",
        generation,
        attempted_at,
        _success("default", activated.server_time),
    )
    listed = await store.list_states()
    source = listed.states[0].sources[0]

    assert source.last_attempt_at == attempted_at
    assert source.last_success_at == attempted_at
    assert source.rate_limit is not None
    assert source.rate_limit.unix_timestamp == activated.server_time - 1


@pytest.mark.parametrize("replay_status", ["ok", "error"])
async def test_publication_replay_is_a_noop_and_later_observation_updates(
    store: RateLimitMonitorStore,
    redis: aioredis.FakeRedis,
    replay_status: str,
) -> None:
    activated = await store.activate(MODEL, ("default",))
    generation = await _claim_generation(store, MODEL, "owner")
    first_attempted_at = int((time.time() - 10) * 1_000) / 1_000
    success = _success("default", activated.server_time)
    assert await store.publish_source(
        MODEL,
        "owner",
        generation,
        first_attempted_at,
        success,
    )

    attempted_at = first_attempted_at
    update = success
    if replay_status == "error":
        attempted_at += 1
        update = _error("default")
        assert await store.publish_source(
            MODEL,
            "owner",
            generation,
            attempted_at,
            update,
        )

    raw_before = await redis.get(snapshot_key(MODEL))
    active_before = await redis.zscore(ACTIVE_KEY, MODEL)
    retained_before = await redis.zscore(RETAINED_KEY, MODEL)
    owner_ttl_before = await redis.pttl(owner_key(MODEL))
    snapshot_ttl_before = await redis.pttl(snapshot_key(MODEL))

    assert await store.publish_source(
        MODEL,
        "owner",
        generation,
        attempted_at,
        update,
    )

    assert await redis.get(snapshot_key(MODEL)) == raw_before
    assert await redis.zscore(ACTIVE_KEY, MODEL) == active_before
    assert await redis.zscore(RETAINED_KEY, MODEL) == retained_before
    assert await redis.get(owner_key(MODEL)) == "owner"
    assert 0 < await redis.pttl(owner_key(MODEL)) <= owner_ttl_before
    assert 0 < await redis.pttl(snapshot_key(MODEL)) <= snapshot_ttl_before

    distinct_attempted_at = attempted_at + 0.0001
    assert await store.publish_source(
        MODEL,
        "owner",
        generation,
        distinct_attempted_at,
        _success("default", activated.server_time + 1),
    )
    listed_source = (await store.list_states()).states[0].sources[0]
    assert listed_source.last_attempt_at == distinct_attempted_at
    assert await redis.get(snapshot_key(MODEL)) != raw_before


async def test_publication_replay_requires_current_owner(store, redis) -> None:
    activated = await store.activate(MODEL, ("default",))
    generation = await _claim_generation(store, MODEL, "owner")
    attempted_at = _attempted_at()
    update = _success("default", activated.server_time)
    assert await store.publish_source(MODEL, "owner", generation, attempted_at, update)

    raw_before = await redis.get(snapshot_key(MODEL))
    await redis.set(owner_key(MODEL), "other-owner", px=60_000)

    assert not await store.publish_source(
        MODEL, "owner", generation, attempted_at, update
    )
    assert await redis.get(snapshot_key(MODEL)) == raw_before


async def test_fixed_rate_limit_round_trips_through_redis(store, redis) -> None:
    activated = await store.activate(MODEL, ("default",))
    generation = await _claim_generation(store, MODEL, "owner")

    assert await store.publish_source(
        MODEL,
        "owner",
        generation,
        _attempted_at(),
        _scoped_success("default", activated.server_time),
    )
    source = (await store.list_states()).states[0].sources[0]

    assert source.rate_limit == RateLimit(
        requests=(RequestRateLimit(limit=25, remaining=20, mode="concurrency"),),
        tokens=TokenRateLimit(
            input=RateLimitCapacity(limit=600, remaining=300),
            output=RateLimitCapacity(limit=400, remaining=100),
        ),
        scope="shared",
        unix_timestamp=activated.server_time - 1,
    )
    assert "authorization" not in await redis.get(snapshot_key(MODEL))


async def test_one_anthropic_success_remains_starting_until_both_pools_succeed(store):
    activated = await store.activate(ANTHROPIC_MODEL, ("pool_1", "pool_2"))
    deadline = await _claim_generation(store, ANTHROPIC_MODEL, "owner")

    assert await store.publish_source(
        ANTHROPIC_MODEL,
        "owner",
        deadline,
        _attempted_at(),
        _success("pool_1", activated.server_time),
    )
    listed = await store.list_states()

    assert listed.states[0].status == "starting"
    assert [source.status for source in listed.states[0].sources] == [
        "ok",
        "starting",
    ]


async def test_failure_preserves_last_good_source_and_snapshot_expiry(
    store,
    redis,
):
    activated = await store.activate(MODEL, ("default",))
    deadline = await _claim_generation(store, MODEL, "owner")
    assert await store.publish_source(
        MODEL,
        "owner",
        deadline,
        _attempted_at(),
        _success("default", activated.server_time),
    )
    successful_source = (await store.list_states()).states[0].sources[0]

    assert await store.publish_source(
        MODEL,
        "owner",
        deadline,
        time.time(),
        _error("default"),
    )
    listed = await store.list_states()
    state = listed.states[0]
    expected_ttl_ms = round((state.retention_until - listed.server_time) * 1_000)

    assert (
        expected_ttl_ms - 250
        <= await redis.pttl(snapshot_key(MODEL))
        <= expected_ttl_ms
    )
    assert state.status == "stale"
    assert state.sources[0].status == "stale"
    assert state.sources[0].error_code == "provider_error"
    assert state.sources[0].last_attempt_at is not None
    assert state.sources[0].last_success_at == successful_source.last_success_at
    assert state.sources[0].last_success_at is not None
    assert state.sources[0].last_attempt_at >= state.sources[0].last_success_at
    assert state.sources[0].rate_limit is not None
    assert state.sources[0].rate_limit.requests[0].remaining == 9_000


async def test_publication_rejects_wrong_owner_and_generation(store, redis):
    await store.activate(MODEL, ("default",))
    first_generation = await _claim_generation(store, MODEL, "owner")
    await _expire_activation(redis, MODEL)
    await store.activate(MODEL, ("default",))
    second_generation = await _claim_generation(store, MODEL, "new-owner")
    assert second_generation != first_generation

    assert not await store.publish_source(
        MODEL,
        "wrong-owner",
        second_generation,
        _attempted_at(),
        _success("default", 1.0),
    )
    assert not await store.publish_source(
        MODEL,
        "new-owner",
        first_generation,
        _attempted_at(),
        _success("default", 1.0),
    )


async def test_inactive_reactivation_clears_prior_owner_and_keeps_last_good_stale(
    store, redis
):
    activated = await store.activate(MODEL, ("default",))
    deadline = await _claim_generation(store, MODEL, "old-owner")
    assert await store.publish_source(
        MODEL,
        "old-owner",
        deadline,
        _attempted_at(),
        _success("default", activated.server_time),
    )
    assert await store.publish_source(
        MODEL,
        "old-owner",
        deadline,
        time.time() - 0.01,
        _error("default"),
    )
    await _expire_activation(redis, MODEL)

    reactivated = await store.activate(MODEL, ("default",))

    assert await redis.get(owner_key(MODEL)) is None
    assert reactivated.state.active is True
    assert reactivated.state.status == "stale"
    assert reactivated.state.sources[0].status == "stale"
    assert reactivated.state.sources[0].rate_limit is not None
    assert reactivated.state.sources[0].error_code == "provider_error"
    assert await store.claim_owner(MODEL, "new-owner") is not None


async def test_same_attempt_timestamp_in_new_generation_is_not_a_replay(
    store,
    redis,
):
    activated = await store.activate(MODEL, ("default",))
    first_generation = await _claim_generation(store, MODEL, "old-owner")
    attempted_at = time.time() - 10
    assert await store.publish_source(
        MODEL,
        "old-owner",
        first_generation,
        attempted_at,
        _success("default", activated.server_time),
    )
    await _expire_activation(redis, MODEL)
    await store.activate(MODEL, ("default",))
    second_generation = await _claim_generation(store, MODEL, "new-owner")

    assert second_generation != first_generation
    assert await store.publish_source(
        MODEL,
        "new-owner",
        second_generation,
        attempted_at,
        _error("default"),
    )
    source = (await store.list_states()).states[0].sources[0]
    assert source.status == "stale"
    assert source.error_code == "provider_error"
    assert source.last_attempt_at == attempted_at
    assert source.last_success_at == attempted_at


@pytest.mark.parametrize("status", ["error", "unsupported"])
async def test_reactivation_resets_prior_failure_without_last_good_to_starting(
    store,
    redis,
    status: Literal["error", "unsupported"],
):
    await store.activate(MODEL, ("default",))
    generation = await _claim_generation(store, MODEL, "old-owner")
    assert await store.publish_source(
        MODEL,
        "old-owner",
        generation,
        _attempted_at(),
        MonitorSourceUpdate(source="default", status=status),
    )
    await _expire_activation(redis, MODEL)

    reactivated = await store.activate(MODEL, ("default",))

    assert reactivated.state.status == "starting"
    assert reactivated.state.sources == [
        MonitorSourceState(source="default", status="starting")
    ]


async def test_inactive_read_marks_last_good_values_stale(store, redis):
    activated = await store.activate(MODEL, ("default",))
    deadline = await _claim_generation(store, MODEL, "owner")
    assert await store.publish_source(
        MODEL,
        "owner",
        deadline,
        _attempted_at(),
        _success("default", activated.server_time),
    )
    expired_active_until = await _expire_activation(redis, MODEL)

    assert await store.claim_owner(MODEL, "new-owner") is None
    assert await redis.zscore(ACTIVE_KEY, MODEL) == expired_active_until
    state = (await store.list_states()).states[0]

    assert state.active is False
    assert state.status == "stale"
    assert state.sources[0].status == "stale"
