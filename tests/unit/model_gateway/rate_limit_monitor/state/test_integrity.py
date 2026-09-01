import json
import time

import pytest
from fakeredis import aioredis

from model_gateway.rate_limit_monitor.state import (
    ACTIVE_KEY,
    RETAINED_KEY,
    MonitorStateCorrupt,
    RateLimitMonitorStore,
    owner_key,
    snapshot_key,
)
from model_gateway.rate_limit_monitor.types import (
    MonitorFacts,
    MonitorSourceFacts,
)

from tests.unit.model_gateway.rate_limit_monitor.state._support import (
    MODEL,
    ANTHROPIC_MODEL,
    _attempted_at,
    _claim_generation,
    _success,
    _expire_retention,
    _invalid_source_set_snapshot,
    _persist_invalid_snapshot,
)


async def test_discovery_returns_only_live_names_and_cleans_expired_retention(
    store,
    redis,
):
    expired_model = "openai/expired"
    await store.activate(MODEL, ("default",))
    await store.activate(expired_model, ("default",))
    await _expire_retention(redis, expired_model)
    assert await redis.get(snapshot_key(expired_model)) is not None

    active = await store.discover_active()

    assert active == {MODEL}
    assert await redis.zscore(ACTIVE_KEY, expired_model) is None
    assert await redis.zscore(RETAINED_KEY, expired_model) is None
    assert await redis.get(snapshot_key(expired_model)) is None


async def test_discovery_returns_active_names_despite_corrupt_snapshot(
    store,
    redis,
):
    await store.activate(MODEL, ("default",))
    await store.activate(ANTHROPIC_MODEL, ("pool_1", "pool_2"))
    raw = await redis.get(snapshot_key(MODEL))
    assert raw is not None
    corrupted = json.loads(raw)
    corrupted["model"] = "openai/different"
    await redis.set(snapshot_key(MODEL), json.dumps(corrupted), px=60_000)

    active = await store.discover_active()

    assert active == {MODEL, ANTHROPIC_MODEL}
    with pytest.raises(MonitorStateCorrupt):
        await store.claim_owner(MODEL, "owner")
    assert await redis.get(owner_key(MODEL)) is None
    assert await store.claim_owner(ANTHROPIC_MODEL, "owner") is not None


async def test_activation_maps_malformed_persisted_snapshot_to_corrupt(
    store,
    redis,
):
    await _persist_invalid_snapshot(redis, "{malformed")

    with pytest.raises(MonitorStateCorrupt):
        await store.activate(MODEL, ("default",))


@pytest.mark.parametrize("corruption", ["container", "order"])
async def test_activation_validates_source_layout_before_replacing_state(
    store,
    redis,
    corruption,
):
    await store.activate(ANTHROPIC_MODEL, ("pool_1", "pool_2"))
    assert await store.claim_owner(ANTHROPIC_MODEL, "owner") is not None
    raw = await redis.get(snapshot_key(ANTHROPIC_MODEL))
    assert raw is not None
    snapshot = json.loads(raw)
    if corruption == "container":
        snapshot["sources"] = "not-an-array"
    else:
        snapshot["sources"].reverse()
    corrupted = json.dumps(snapshot)
    await redis.set(snapshot_key(ANTHROPIC_MODEL), corrupted, px=60_000)
    owner_ttl_before = await redis.pttl(owner_key(ANTHROPIC_MODEL))
    snapshot_ttl_before = await redis.pttl(snapshot_key(ANTHROPIC_MODEL))

    with pytest.raises(MonitorStateCorrupt):
        await store.activate(ANTHROPIC_MODEL, ("pool_1", "pool_2"))

    assert await redis.get(owner_key(ANTHROPIC_MODEL)) == "owner"
    assert 0 < await redis.pttl(owner_key(ANTHROPIC_MODEL)) <= owner_ttl_before
    assert await redis.get(snapshot_key(ANTHROPIC_MODEL)) == corrupted
    assert 0 < await redis.pttl(snapshot_key(ANTHROPIC_MODEL)) <= snapshot_ttl_before


@pytest.mark.parametrize(
    "raw",
    ["{malformed", "{}", pytest.param(None, id="invalid-source-set")],
)
async def test_listing_maps_invalid_persisted_snapshot_to_corrupt(
    store,
    redis,
    raw,
):
    await _persist_invalid_snapshot(
        redis,
        _invalid_source_set_snapshot() if raw is None else raw,
    )

    with pytest.raises(MonitorStateCorrupt):
        await store.list_states()


async def test_listing_maps_overflowing_index_timestamp_to_corrupt(
    store,
    redis,
):
    deadline = 1e308
    snapshot = MonitorFacts(
        generation="a" * 32,
        model=MODEL,
        sources=[MonitorSourceFacts(source="default")],
    )
    await redis.zadd(ACTIVE_KEY, {MODEL: deadline})
    await redis.zadd(RETAINED_KEY, {MODEL: deadline})
    await redis.set(snapshot_key(MODEL), snapshot.model_dump_json())

    with pytest.raises(MonitorStateCorrupt):
        await store.list_states()


async def test_publication_maps_malformed_persisted_snapshot_to_corrupt(
    store,
    redis,
):
    activated = await store.activate(MODEL, ("default",))
    deadline = await _claim_generation(store, MODEL, "owner")
    await redis.set(snapshot_key(MODEL), "{malformed")

    with pytest.raises(MonitorStateCorrupt):
        await store.publish_source(
            MODEL,
            "owner",
            deadline,
            _attempted_at(),
            _success("default", activated.server_time),
        )


@pytest.mark.parametrize(
    "corruption",
    [
        "sources",
        "order",
        "unsupported-topology",
        "selected-source",
        "unselected-source",
        "attempt-generation",
    ],
)
async def test_publication_validates_topology_before_snapshot_mutation(
    store,
    redis,
    corruption,
):
    activated = await store.activate(ANTHROPIC_MODEL, ("pool_1", "pool_2"))
    generation = await _claim_generation(store, ANTHROPIC_MODEL, "owner")
    raw = await redis.get(snapshot_key(ANTHROPIC_MODEL))
    assert raw is not None
    snapshot = json.loads(raw)
    if corruption == "sources":
        snapshot["sources"] = "not-an-array"
    elif corruption == "order":
        snapshot["sources"].reverse()
    elif corruption == "unsupported-topology":
        snapshot["sources"] = snapshot["sources"][:1]
    elif corruption == "selected-source":
        snapshot["sources"][0]["source"] = "default"
    elif corruption == "unselected-source":
        snapshot["sources"][1]["source"] = "default"
    else:
        snapshot["sources"][0]["last_attempt_generation"] = "not-a-generation"
    corrupted = json.dumps(snapshot)
    await redis.set(snapshot_key(ANTHROPIC_MODEL), corrupted, px=60_000)
    owner_ttl_before = await redis.pttl(owner_key(ANTHROPIC_MODEL))
    snapshot_ttl_before = await redis.pttl(snapshot_key(ANTHROPIC_MODEL))

    with pytest.raises(MonitorStateCorrupt):
        await store.publish_source(
            ANTHROPIC_MODEL,
            "owner",
            generation,
            _attempted_at(),
            _success("pool_1", activated.server_time),
        )

    assert await redis.get(owner_key(ANTHROPIC_MODEL)) == "owner"
    assert 0 < await redis.pttl(owner_key(ANTHROPIC_MODEL)) <= owner_ttl_before
    assert await redis.get(snapshot_key(ANTHROPIC_MODEL)) == corrupted
    assert 0 < await redis.pttl(snapshot_key(ANTHROPIC_MODEL)) <= snapshot_ttl_before


@pytest.mark.parametrize(
    "operation",
    ["activate", "list", "claim", "renew", "publish"],
)
@pytest.mark.parametrize("corruption", ["model", "rate-limit"])
async def test_semantic_snapshot_corruption_fails_closed_before_mutation(
    store: RateLimitMonitorStore,
    redis: aioredis.FakeRedis,
    operation: str,
    corruption: str,
) -> None:
    activated = await store.activate(MODEL, ("default",))
    generation = await _claim_generation(store, MODEL, "owner")
    assert await store.publish_source(
        MODEL,
        "owner",
        generation,
        time.time() - 10,
        _success("default", activated.server_time),
    )
    raw = await redis.get(snapshot_key(MODEL))
    assert raw is not None
    snapshot = json.loads(raw)
    if corruption == "model":
        snapshot["model"] = "openai/different"
    else:
        snapshot["sources"][0]["rate_limit"]["request_remaining"] = "invalid"
    corrupted = json.dumps(snapshot)
    await redis.set(snapshot_key(MODEL), corrupted, px=60_000)
    active_before = await redis.zscore(ACTIVE_KEY, MODEL)
    retained_before = await redis.zscore(RETAINED_KEY, MODEL)
    owner_ttl_before = await redis.pttl(owner_key(MODEL))
    snapshot_ttl_before = await redis.pttl(snapshot_key(MODEL))

    with pytest.raises(MonitorStateCorrupt):
        if operation == "activate":
            await store.activate(MODEL, ("default",))
        elif operation == "list":
            await store.list_states()
        elif operation == "claim":
            await store.claim_owner(MODEL, "owner")
        elif operation == "renew":
            await store.renew_owner(MODEL, "owner")
        else:
            await store.publish_source(
                MODEL,
                "owner",
                generation,
                time.time() - 1,
                _success("default", activated.server_time + 1),
            )

    assert await redis.get(snapshot_key(MODEL)) == corrupted
    assert await redis.zscore(ACTIVE_KEY, MODEL) == active_before
    assert await redis.zscore(RETAINED_KEY, MODEL) == retained_before
    assert await redis.get(owner_key(MODEL)) == "owner"
    assert 0 < await redis.pttl(owner_key(MODEL)) <= owner_ttl_before
    assert 0 < await redis.pttl(snapshot_key(MODEL)) <= snapshot_ttl_before


@pytest.mark.parametrize(
    "operation",
    ["activate", "list", "claim", "renew", "publish"],
)
@pytest.mark.parametrize("index_name", ["active", "retained"])
async def test_index_deadline_mismatch_fails_closed_before_mutation(
    store: RateLimitMonitorStore,
    redis: aioredis.FakeRedis,
    operation: str,
    index_name: str,
) -> None:
    activated = await store.activate(MODEL, ("default",))
    generation = await _claim_generation(store, MODEL, "owner")
    index_key = ACTIVE_KEY if index_name == "active" else RETAINED_KEY
    score = await redis.zscore(index_key, MODEL)
    assert score is not None
    await redis.zadd(index_key, {MODEL: score + 60})
    active_before = await redis.zscore(ACTIVE_KEY, MODEL)
    retained_before = await redis.zscore(RETAINED_KEY, MODEL)
    raw_before = await redis.get(snapshot_key(MODEL))
    owner_ttl_before = await redis.pttl(owner_key(MODEL))
    snapshot_ttl_before = await redis.pttl(snapshot_key(MODEL))

    with pytest.raises(MonitorStateCorrupt):
        if operation == "activate":
            await store.activate(MODEL, ("default",))
        elif operation == "list":
            await store.list_states()
        elif operation == "claim":
            await store.claim_owner(MODEL, "owner")
        elif operation == "renew":
            await store.renew_owner(MODEL, "owner")
        else:
            await store.publish_source(
                MODEL,
                "owner",
                generation,
                time.time() - 1,
                _success("default", activated.server_time),
            )

    assert await redis.zscore(ACTIVE_KEY, MODEL) == active_before
    assert await redis.zscore(RETAINED_KEY, MODEL) == retained_before
    assert await redis.get(snapshot_key(MODEL)) == raw_before
    assert await redis.get(owner_key(MODEL)) == "owner"
    assert 0 < await redis.pttl(owner_key(MODEL)) <= owner_ttl_before
    assert 0 < await redis.pttl(snapshot_key(MODEL)) <= snapshot_ttl_before


async def test_fractional_retention_deadline_fails_closed(
    store: RateLimitMonitorStore,
    redis: aioredis.FakeRedis,
) -> None:
    await store.activate(MODEL, ("default",))
    retained_until = await redis.zscore(RETAINED_KEY, MODEL)
    assert retained_until is not None

    fractional_retention_until = retained_until + 0.0005
    await redis.zadd(RETAINED_KEY, {MODEL: fractional_retention_until})

    with pytest.raises(MonitorStateCorrupt):
        await store.list_states()


async def test_listing_fails_closed_for_active_only_orphan(
    store: RateLimitMonitorStore,
    redis: aioredis.FakeRedis,
) -> None:
    active_until = time.time() + 1_800
    await redis.zadd(ACTIVE_KEY, {MODEL: active_until})

    with pytest.raises(MonitorStateCorrupt):
        await store.list_states()

    assert await redis.zscore(ACTIVE_KEY, MODEL) == active_until
    assert await redis.zscore(RETAINED_KEY, MODEL) is None
    assert await redis.get(snapshot_key(MODEL)) is None


async def test_listing_fails_closed_when_live_snapshot_lacks_active_index(
    store: RateLimitMonitorStore,
    redis: aioredis.FakeRedis,
) -> None:
    await store.activate(MODEL, ("default",))
    raw_before = await redis.get(snapshot_key(MODEL))
    retained_before = await redis.zscore(RETAINED_KEY, MODEL)
    assert raw_before is not None
    assert retained_before is not None
    await redis.zrem(ACTIVE_KEY, MODEL)
    snapshot_ttl_before = await redis.pttl(snapshot_key(MODEL))

    with pytest.raises(MonitorStateCorrupt):
        await store.list_states()

    assert await redis.zscore(ACTIVE_KEY, MODEL) is None
    assert await redis.zscore(RETAINED_KEY, MODEL) == retained_before
    assert await redis.get(snapshot_key(MODEL)) == raw_before
    assert 0 < await redis.pttl(snapshot_key(MODEL)) <= snapshot_ttl_before


async def test_unrelated_orphan_does_not_block_target_hot_paths(store, redis):
    await redis.zadd(ACTIVE_KEY, {MODEL: time.time() + 1_800})
    target = "openai/candidate"

    activated = await store.activate(target, ("default",))
    lease = await store.claim_owner(target, "owner")
    renewal = await store.renew_owner(target, "owner")

    assert lease == ("default",)
    assert renewal is not None
    generation, attempted_at = renewal
    assert await store.publish_source(
        target,
        "owner",
        generation,
        attempted_at,
        _success("default", activated.server_time),
    )
    with pytest.raises(MonitorStateCorrupt):
        await store.list_states()
