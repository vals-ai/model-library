import asyncio
import json
import time
from typing import cast
from unittest.mock import AsyncMock

import pytest
from fakeredis import aioredis
from pydantic import ValidationError

from model_gateway.rate_limit_monitor.state import (
    ACTIVE_KEY,
    MONITOR_KEY_PREFIX,
    RETAINED_KEY,
    RETENTION_SECONDS,
    MonitorRedis,
    MonitorSourceUpdate,
    MonitorStateCorrupt,
    RateLimitMonitorStore,
    owner_key,
    snapshot_key,
)
from model_gateway.rate_limit_monitor.types import (
    MonitorFacts,
    MonitorListResponse,
    MonitorSourceFacts,
    MonitorSourceState,
)
from model_library.rate_limits import (
    RateLimit,
    RateLimitCapacity,
    TokenRateLimit,
)

from tests.unit.model_gateway.rate_limit_monitor.state._support import (
    MODEL,
    ANTHROPIC_MODEL,
    _attempted_at,
    _claim_generation,
    _success,
    _expire_activation,
)


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), float("-inf")])
def test_monitor_contracts_reject_non_finite_timestamps(timestamp: float) -> None:
    with pytest.raises(ValidationError, match="finite number"):
        MonitorSourceState(
            source="default",
            status="error",
            last_attempt_at=timestamp,
            error_code="provider_error",
        )
    with pytest.raises(ValidationError, match="finite number"):
        MonitorSourceFacts(
            source="default",
            last_attempt_at=timestamp,
            last_attempt_generation="a" * 32,
            last_error_code="provider_error",
        )
    with pytest.raises(ValidationError, match="finite number"):
        MonitorListResponse(server_time=timestamp, states=[])


async def test_activation_creates_starting_state_and_millisecond_retention_ttl(
    store, redis
):
    response = await store.activate(MODEL, ("default",))

    assert response.state.active is True
    assert response.state.status == "starting"
    assert [source.source for source in response.state.sources] == ["default"]
    assert 599 <= response.state.active_until - response.server_time <= 601
    assert (
        86_399 <= response.state.retention_until - response.state.active_until <= 86_401
    )
    snapshot_ttl = await redis.pttl(snapshot_key(MODEL))
    raw_snapshot = json.loads(await redis.get(snapshot_key(MODEL)))
    expected_ttl_ms = round(
        (response.state.retention_until - response.server_time) * 1_000
    )
    assert expected_ttl_ms - 1_000 < snapshot_ttl <= expected_ttl_ms
    assert len(raw_snapshot["generation"]) == 32
    assert set(raw_snapshot) == {"generation", "model", "sources"}
    assert set(raw_snapshot["sources"][0]) == {
        "source",
        "last_attempt_at",
        "last_attempt_generation",
        "last_success_at",
        "rate_limit",
        "last_error_code",
    }
    assert "status" not in raw_snapshot["sources"][0]
    assert MONITOR_KEY_PREFIX == "model_gateway:rate_limit_monitor"
    assert "generation" not in response.model_dump_json()


async def test_anthropic_activation_persists_both_sources(store):
    response = await store.activate(
        ANTHROPIC_MODEL,
        ("pool_1", "pool_2"),
    )

    assert [source.source for source in response.state.sources] == [
        "pool_1",
        "pool_2",
    ]
    assert [source.status for source in response.state.sources] == [
        "starting",
        "starting",
    ]


async def test_activation_accepts_arbitrary_contiguous_managed_sources(store):
    expected_sources = tuple(f"pool_{index}" for index in range(1, 6))

    response = await store.activate(ANTHROPIC_MODEL, expected_sources)

    assert tuple(source.source for source in response.state.sources) == expected_sources
    assert all(source.status == "starting" for source in response.state.sources)


async def test_active_source_set_change_resets_facts_generation_and_owner(store, redis):
    first = await store.activate(ANTHROPIC_MODEL, ("pool_1", "pool_2"))
    first_generation = await _claim_generation(store, ANTHROPIC_MODEL, "old-owner")
    assert await store.publish_source(
        ANTHROPIC_MODEL,
        "old-owner",
        first_generation,
        _attempted_at(),
        _success("pool_1", first.server_time),
    )

    expected_sources = tuple(f"pool_{index}" for index in range(1, 6))
    expanded = await store.activate(ANTHROPIC_MODEL, expected_sources)

    assert tuple(source.source for source in expanded.state.sources) == expected_sources
    assert all(source.status == "starting" for source in expanded.state.sources)
    assert await redis.get(owner_key(ANTHROPIC_MODEL)) is None
    assert expanded.state.active_until >= first.state.active_until
    assert not await store.publish_source(
        ANTHROPIC_MODEL,
        "old-owner",
        first_generation,
        _attempted_at(),
        _success("pool_1", expanded.server_time),
    )

    assert await store.claim_owner(ANTHROPIC_MODEL, "new-owner") == expected_sources
    renewed = await store.renew_owner(ANTHROPIC_MODEL, "new-owner")
    assert renewed is not None
    expanded_generation, _ = renewed
    assert expanded_generation != first_generation


async def test_retained_source_set_change_discards_last_good_facts(store, redis):
    first = await store.activate(ANTHROPIC_MODEL, ("pool_1", "pool_2"))
    first_generation = await _claim_generation(store, ANTHROPIC_MODEL, "owner")
    assert await store.publish_source(
        ANTHROPIC_MODEL,
        "owner",
        first_generation,
        _attempted_at(),
        _success("pool_1", first.server_time),
    )
    await _expire_activation(redis, ANTHROPIC_MODEL)

    reactivated = await store.activate(
        ANTHROPIC_MODEL,
        ("pool_1", "pool_2", "pool_3"),
    )

    assert [source.status for source in reactivated.state.sources] == [
        "starting",
        "starting",
        "starting",
    ]
    assert all(source.rate_limit is None for source in reactivated.state.sources)


async def test_active_extension_uses_one_slot_and_refreshes_snapshot_expiry(
    store,
    redis,
):
    first = await store.activate(MODEL, ("default",))
    first_generation = await _claim_generation(store, MODEL, "owner")
    second = await store.activate(MODEL, ("default",))
    renewal = await store.renew_owner(MODEL, "owner")

    assert renewal is not None
    second_generation, _ = renewal
    assert second_generation == first_generation
    assert second.state.active_until >= first.state.active_until
    assert second.state.retention_until >= first.state.retention_until
    expected_ttl_ms = round((second.state.retention_until - second.server_time) * 1_000)
    assert (
        expected_ttl_ms - 1_000
        <= await redis.pttl(snapshot_key(MODEL))
        <= expected_ttl_ms
    )

    assert await store.publish_source(
        MODEL,
        "owner",
        first_generation,
        _attempted_at(),
        _success("default", second.server_time),
    )
    assert await store.publish_source(
        MODEL,
        "owner",
        second_generation,
        _attempted_at(),
        _success("default", second.server_time),
    )
    listed = await store.list_states()
    expected_ttl_ms = round(
        (listed.states[0].retention_until - listed.server_time) * 1_000
    )
    assert (
        expected_ttl_ms - 1_000
        <= await redis.pttl(snapshot_key(MODEL))
        <= expected_ttl_ms
    )


async def test_publish_token_only_rate_limit_preserves_empty_request_array(
    store, redis
):
    activated = await store.activate(MODEL, ("default",))
    generation = await _claim_generation(store, MODEL, "owner")
    token_only = MonitorSourceUpdate(
        source="default",
        status="ok",
        rate_limit=RateLimit(
            tokens=TokenRateLimit(
                total=RateLimitCapacity(limit=1_000_000, remaining=900_000)
            ),
            unix_timestamp=activated.server_time,
        ),
    )

    assert await store.publish_source(
        MODEL,
        "owner",
        generation,
        _attempted_at(),
        token_only,
    )

    reactivated = await store.activate(MODEL, ("default",))
    listed_source = (await store.list_states()).states[0].sources[0]
    assert reactivated.state.sources[0].rate_limit is not None
    assert listed_source.rate_limit is not None
    assert listed_source.rate_limit.requests == ()
    raw_snapshot = json.loads(await redis.get(snapshot_key(MODEL)))
    assert raw_snapshot["sources"][0]["rate_limit"]["requests"] == []


async def test_concurrent_activation_keeps_all_models_active(store):
    results = await asyncio.gather(
        *(store.activate(f"openai/model-{index}", ("default",)) for index in range(26))
    )
    listed = await store.list_states()

    assert len(results) == 26
    assert len(listed.states) == 26


async def test_activation_retries_when_active_score_expires_before_commit(
    store: RateLimitMonitorStore,
    redis: aioredis.FakeRedis,
) -> None:
    await store.activate(MODEL, ("default",))
    assert await store.claim_owner(MODEL, "old-owner") is not None
    renewal = await store.renew_owner(MODEL, "old-owner")
    assert renewal is not None
    old_generation, attempted_at = renewal
    assert await store.publish_source(
        MODEL,
        "old-owner",
        old_generation,
        attempted_at,
        _success("default", attempted_at),
    )

    expired_active_score = (int(time.time() * 1_000) - 1_000) / 1_000
    await redis.zadd(ACTIVE_KEY, {MODEL: expired_active_score})
    await redis.zadd(
        RETAINED_KEY,
        {MODEL: expired_active_score + RETENTION_SECONDS},
    )

    real_eval = cast(MonitorRedis, redis).eval
    first_read = True

    async def cross_activation_boundary(
        script: str,
        numkeys: int,
        *keys_and_args: str | int | float,
    ) -> list[str]:
        nonlocal first_read
        result = cast(
            list[str],
            await real_eval(script, numkeys, *keys_and_args),
        )
        if not first_read:
            return result
        first_read = False
        # Preserve the real raw/index values but make the read occur just
        # before the already-expired score. The commit still uses Redis TIME.
        return [str(expired_active_score - 1), *result[1:]]

    redis.eval = AsyncMock(side_effect=cross_activation_boundary)

    reactivated = await store.activate(MODEL, ("default",))
    stored = MonitorFacts.model_validate_json(await redis.get(snapshot_key(MODEL)))

    assert not first_read
    assert stored.generation != old_generation
    assert reactivated.state.sources[0].status == "stale"
    assert reactivated.state.sources[0].rate_limit is not None
    assert await redis.get(owner_key(MODEL)) is None


async def test_listing_retains_expired_active_score_until_retention_cleanup(
    store,
    redis,
):
    await store.activate(MODEL, ("default",))
    expired_active_until = await _expire_activation(redis, MODEL)
    retained_score = await redis.zscore(RETAINED_KEY, MODEL)
    await redis.zadd(ACTIVE_KEY, {"openai/expired": 0})
    await redis.zadd(RETAINED_KEY, {"openai/expired": 0})

    listed = await store.list_states()

    assert [state.model for state in listed.states] == [MODEL]
    assert listed.states[0].active is False
    assert listed.states[0].active_until == expired_active_until
    assert await redis.zscore(ACTIVE_KEY, MODEL) == expired_active_until
    assert await redis.zscore(RETAINED_KEY, MODEL) == retained_score
    assert await redis.zscore(ACTIVE_KEY, "openai/expired") is None
    assert await redis.zscore(RETAINED_KEY, "openai/expired") is None


async def test_claim_renew_and_release_compare_owner_tokens(
    store: RateLimitMonitorStore,
    redis: aioredis.FakeRedis,
):
    await store.activate(MODEL, ("default",))
    lease = await store.claim_owner(MODEL, "owner-a")

    assert lease == ("default",)
    renewal = await store.renew_owner(MODEL, "owner-a")
    assert renewal is not None
    generation = renewal[0]
    assert await store.claim_owner(MODEL, "owner-b") is None
    assert await store.renew_owner(MODEL, "owner-b") is None
    renewal = await store.renew_owner(MODEL, "owner-a")
    assert renewal is not None
    assert renewal[0] == generation
    await store.release_owner(MODEL, "owner-b")
    assert await redis.get(owner_key(MODEL)) == "owner-a"
    await store.release_owner(MODEL, "owner-a")
    assert await redis.get(owner_key(MODEL)) is None


async def test_claim_replay_preserves_same_token_lease_without_extension(
    store: RateLimitMonitorStore,
    redis: aioredis.FakeRedis,
):
    await store.activate(MODEL, ("default",))
    lease = await store.claim_owner(MODEL, "owner-a")
    assert lease == ("default",)
    await redis.pexpire(owner_key(MODEL), 5_000)
    ttl_before_replay = await redis.pttl(owner_key(MODEL))

    assert await store.claim_owner(MODEL, "owner-a") == lease
    ttl_after_replay = await redis.pttl(owner_key(MODEL))
    assert 0 < ttl_after_replay <= ttl_before_replay
    assert await store.claim_owner(MODEL, "owner-b") is None
    assert await redis.get(owner_key(MODEL)) == "owner-a"


@pytest.mark.parametrize("operation", ["claim", "renew"])
async def test_generation_is_validated_before_owner_lease_mutation(
    store,
    redis,
    operation,
):
    await store.activate(MODEL, ("default",))
    if operation == "renew":
        assert await store.claim_owner(MODEL, "owner") is not None

    raw = await redis.get(snapshot_key(MODEL))
    assert raw is not None
    snapshot = json.loads(raw)
    snapshot["generation"] = "g" * 32
    corrupted = json.dumps(snapshot)
    await redis.set(snapshot_key(MODEL), corrupted, px=60_000)
    if operation == "renew":
        await redis.pexpire(owner_key(MODEL), 5_000)
    owner_before = await redis.get(owner_key(MODEL))
    owner_ttl_before = await redis.pttl(owner_key(MODEL))
    snapshot_ttl_before = await redis.pttl(snapshot_key(MODEL))

    with pytest.raises(MonitorStateCorrupt):
        if operation == "claim":
            await store.claim_owner(MODEL, "owner")
        else:
            await store.renew_owner(MODEL, "owner")

    assert await redis.get(owner_key(MODEL)) == owner_before
    owner_ttl_after = await redis.pttl(owner_key(MODEL))
    if owner_before is None:
        assert owner_ttl_after == -2
    else:
        assert 0 < owner_ttl_after <= owner_ttl_before
    assert await redis.get(snapshot_key(MODEL)) == corrupted
    assert 0 < await redis.pttl(snapshot_key(MODEL)) <= snapshot_ttl_before


@pytest.mark.parametrize("operation", ["claim", "renew", "publish"])
async def test_missing_snapshot_fails_closed(store, redis, operation):
    await store.activate(MODEL, ("default",))
    generation = None
    if operation != "claim":
        generation = await _claim_generation(store, MODEL, "owner")
    await redis.delete(snapshot_key(MODEL))

    with pytest.raises(MonitorStateCorrupt):
        if operation == "claim":
            await store.claim_owner(MODEL, "owner")
        elif operation == "renew":
            await store.renew_owner(MODEL, "owner")
        else:
            assert generation is not None
            await store.publish_source(
                MODEL,
                "owner",
                generation,
                _attempted_at(),
                _success("default", 1.0),
            )
