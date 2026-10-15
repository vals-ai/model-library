import math
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack
from typing import cast
from unittest.mock import Mock

import pytest
from fakeredis import FakeServer, aioredis

import model_gateway.app as gateway_app
from model_gateway.rate_limit_monitor.manager import RateLimitMonitor
from model_gateway.rate_limit_monitor.state import RateLimitMonitorStore
from model_gateway.rate_limit_monitor.types import MonitorFacts, MonitorSourceFacts
from tests.unit.model_gateway._support import _make_client
from tests.unit.model_gateway.rate_limit_monitor.state._support import (
    MODEL,
    _claim_generation,
    _success,
)

StoreFactory = Callable[[str | None], Awaitable[RateLimitMonitorStore]]


@pytest.fixture
async def stage_store(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[StoreFactory]:
    server = FakeServer()
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setattr(
        gateway_app.async_redis,
        "from_url",
        lambda *args, **kwargs: aioredis.FakeRedis(
            server=server, decode_responses=True
        ),
    )
    monkeypatch.setattr(gateway_app, "set_redis_client", Mock())
    monkeypatch.setattr(
        gateway_app, "runtime_config_manager_from_env", Mock(return_value=None)
    )
    monkeypatch.setattr(gateway_app.telemetry, "configure_telemetry", Mock())
    monkeypatch.setattr(gateway_app.telemetry, "shutdown_telemetry", Mock())
    monkeypatch.setattr(RateLimitMonitor, "start", Mock())

    async with AsyncExitStack() as applications:

        async def create_store(stage: str | None) -> RateLimitMonitorStore:
            if stage is None:
                monkeypatch.delenv("GATEWAY_STAGE", raising=False)
            else:
                monkeypatch.setenv("GATEWAY_STAGE", stage)

            application = _make_client(runtime_role="control").app
            await applications.enter_async_context(
                application.router.lifespan_context(application)
            )
            monitor = cast(RateLimitMonitor, application.state.rate_limit_monitor)
            return cast(RateLimitMonitorStore, monitor._store)

        yield create_store


@pytest.mark.parametrize("stage", ["", " prod", "dev:owner", "dev/prod"])
async def test_invalid_stage_cannot_start_a_monitor(
    stage_store: StoreFactory,
    stage: str,
) -> None:
    with pytest.raises(ValueError, match="Invalid gateway stage"):
        await stage_store(stage)


@pytest.mark.parametrize(
    ("first_stage", "second_stage"),
    [
        ("dev", "prod"),
        ("dev", "preview-one-a1b2c3"),
        ("prod", "preview-one-a1b2c3"),
        ("preview-one-a1b2c3", "preview-two-d4e5f6"),
        ("prod", None),
    ],
)
async def test_stage_cannot_read_or_claim_another_stages_activation(
    stage_store: StoreFactory,
    first_stage: str,
    second_stage: str | None,
) -> None:
    first = await stage_store(first_stage)
    second = await stage_store(second_stage)
    await first.activate(MODEL, ("default",))

    assert (await second.list_states()).states == []
    assert await second.discover_active() == set()
    assert await second.claim_owner(MODEL, "second-owner") is None
    assert await first.claim_owner(MODEL, "first-owner") == ("default",)


@pytest.mark.parametrize(
    "operation",
    ["list", "discover", "claim", "renew", "release", "publish", "activate"],
)
async def test_stage_ignores_legacy_unscoped_state(
    stage_store: StoreFactory,
    operation: str,
) -> None:
    store = await stage_store("prod")
    redis = cast(aioredis.FakeRedis, store.redis)
    active_until = math.ceil(time.time() * 1_000) / 1_000 + 600
    retained_until = active_until + 86_400
    generation = "a" * 32
    update = _success("default", active_until - 600)
    facts = MonitorFacts(
        generation=generation,
        model=MODEL,
        sources=[
            MonitorSourceFacts(
                source="default",
                last_attempt_at=active_until - 600,
                last_attempt_generation=generation,
                last_success_at=active_until - 600,
                rate_limit=update.rate_limit,
            )
        ],
    )
    raw = facts.model_dump_json(exclude_computed_fields=True)
    await redis.zadd("model_gateway:rate_limit_monitor:active", {MODEL: active_until})
    await redis.zadd(
        "model_gateway:rate_limit_monitor:retained", {MODEL: retained_until}
    )
    await redis.set(f"model_gateway:rate_limit_monitor:snapshot:{MODEL}", raw)
    await redis.set(f"model_gateway:rate_limit_monitor:owner:{MODEL}", "legacy-owner")

    if operation == "list":
        assert (await store.list_states()).states == []
    elif operation == "discover":
        assert await store.discover_active() == set()
    elif operation == "claim":
        assert await store.claim_owner(MODEL, "legacy-owner") is None
    elif operation == "renew":
        assert await store.renew_owner(MODEL, "legacy-owner") is None
    elif operation == "release":
        await store.release_owner(MODEL, "legacy-owner")
    elif operation == "publish":
        assert not await store.publish_source(
            MODEL, "legacy-owner", generation, active_until - 600, update
        )
    else:
        activated = await store.activate(MODEL, ("default",))
        assert activated.state.status == "starting"
        assert activated.state.sources[0].rate_limit is None

    assert await redis.get(f"model_gateway:rate_limit_monitor:snapshot:{MODEL}") == raw
    assert (
        await redis.get(f"model_gateway:rate_limit_monitor:owner:{MODEL}")
        == "legacy-owner"
    )
    assert (
        await redis.zscore("model_gateway:rate_limit_monitor:active", MODEL)
        == active_until
    )
    assert (
        await redis.zscore("model_gateway:rate_limit_monitor:retained", MODEL)
        == retained_until
    )


async def test_stages_keep_separate_leases_and_source_snapshots(
    stage_store: StoreFactory,
) -> None:
    development = await stage_store("dev")
    production = await stage_store("prod")
    activated = await development.activate(MODEL, ("default",))
    development_generation = await _claim_generation(development, MODEL, "owner")
    await production.activate(MODEL, ("default",))

    assert await production.claim_owner(MODEL, "production-owner") == ("default",)
    assert await production.renew_owner(MODEL, "owner") is None
    assert not await production.publish_source(
        MODEL,
        "owner",
        development_generation,
        activated.server_time,
        _success("default", activated.server_time),
    )
    await production.release_owner(MODEL, "owner")
    assert await development.publish_source(
        MODEL,
        "owner",
        development_generation,
        activated.server_time,
        _success("default", activated.server_time),
    )

    assert (await development.list_states()).states[0].status == "ok"
    assert (await production.list_states()).states[0].status == "starting"
    assert await development.renew_owner(MODEL, "owner") is not None
    assert await production.renew_owner(MODEL, "production-owner") is not None


@pytest.mark.parametrize("stage", ["dev", "prod", "preview-one-a1b2c3", None])
async def test_workers_in_one_stage_share_snapshots_and_ownership(
    stage_store: StoreFactory,
    stage: str | None,
) -> None:
    first = await stage_store(stage)
    second = await stage_store(stage)
    activated = await first.activate(MODEL, ("default",))

    assert await second.discover_active() == {MODEL}
    generation = await _claim_generation(second, MODEL, "second-owner")
    assert await first.claim_owner(MODEL, "first-owner") is None
    assert await second.publish_source(
        MODEL,
        "second-owner",
        generation,
        activated.server_time,
        _success("default", activated.server_time),
    )

    source = (await first.list_states()).states[0].sources[0]
    assert source.rate_limit is not None
    assert source.rate_limit.requests[0].remaining == 9_000
    await first.release_owner(MODEL, "second-owner")
    assert await first.claim_owner(MODEL, "first-owner") == ("default",)
