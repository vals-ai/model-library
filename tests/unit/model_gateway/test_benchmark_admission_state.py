import asyncio
import logging
from unittest.mock import AsyncMock, patch

import pytest
from fakeredis import aioredis

from model_gateway.benchmark_admission_state import (
    BenchmarkAdmissionConflict,
    BenchmarkAdmissionStore,
    get_benchmark_run_pointer_key,
)
from model_library.retriers.token.benchmark_queue import (
    HEARTBEAT_TTL,
    BenchmarkQueueKeys,
)
from model_library.retriers.token.utils import get_status, set_redis_client

MODEL = "openai/gpt-4"
MODEL_KEY = ("openai.gpt-4", "abc123")
OTHER_MODEL_KEY = ("openai.gpt-4", "other-key")

logger = logging.getLogger("test_benchmark_admission_state")


@pytest.fixture
def redis():
    client = aioredis.FakeRedis(decode_responses=True)
    set_redis_client(client)
    return client


@pytest.fixture
def other_redis():
    return aioredis.FakeRedis(decode_responses=True)


@pytest.fixture
def store(redis):
    return BenchmarkAdmissionStore(redis, logger)


async def _acquire(
    store: BenchmarkAdmissionStore,
    run_id: str,
    *,
    model_registry_key: tuple[str, str] = MODEL_KEY,
    effective_token_limit: int = 10_000,
    total_requests: int | None = None,
    early_release: bool = True,
    immediate_queue_release: bool = False,
):
    return await store.acquire(
        model=MODEL,
        model_registry_key=model_registry_key,
        run_id=run_id,
        effective_token_limit=effective_token_limit,
        total_requests=total_requests,
        early_release=early_release,
        immediate_queue_release=immediate_queue_release,
    )


async def test_acquire_is_idempotent_while_run_is_live(store, redis):
    first = await _acquire(store, "run-1", total_requests=3)
    second = await _acquire(store, "run-1", total_requests=3)

    assert first.state == "acquired"
    assert first.effective_token_limit == 10_000
    assert second.state == "acquired"
    assert second.effective_token_limit == 10_000
    keys = BenchmarkQueueKeys.for_run(MODEL_KEY, "run-1")
    assert await redis.lrange(keys.queue, 0, -1) == ["run-1"]
    assert await redis.hgetall(get_benchmark_run_pointer_key("run-1")) == {
        "model": MODEL,
        "base": keys.base,
    }


async def test_store_uses_injected_redis_instead_of_global_client(
    redis,
    other_redis,
):
    set_redis_client(other_redis)
    injected_store = BenchmarkAdmissionStore(redis, logger)

    await _acquire(injected_store, "run-1")

    keys = BenchmarkQueueKeys.for_run(MODEL_KEY, "run-1")
    assert await redis.lrange(keys.queue, 0, -1) == ["run-1"]
    assert await other_redis.lrange(keys.queue, 0, -1) == []


async def test_acquire_retry_repairs_initialization_interrupted_after_pointer(
    store, redis
):
    async def stop_after_heartbeat(redis_client, keys, *_args):
        await redis_client.set(keys.alive, "1", ex=HEARTBEAT_TTL)
        raise RuntimeError("worker stopped")

    with patch(
        "model_gateway.benchmark_admission_state.initialize_benchmark_run",
        new=AsyncMock(side_effect=stop_after_heartbeat),
    ):
        with pytest.raises(RuntimeError, match="worker stopped"):
            await _acquire(store, "run-1")

    pointer = await redis.hgetall(get_benchmark_run_pointer_key("run-1"))
    repaired = await _acquire(store, "run-1")

    assert pointer["model"] == MODEL
    assert repaired.state == "acquired"


async def test_live_acquire_rejects_incompatible_queue_identity(store):
    await _acquire(store, "run-1")

    with pytest.raises(BenchmarkAdmissionConflict):
        await _acquire(store, "run-1", model_registry_key=OTHER_MODEL_KEY)


async def test_live_acquire_rejects_incompatible_effective_limit(store):
    await _acquire(store, "run-1")

    with pytest.raises(BenchmarkAdmissionConflict):
        await _acquire(store, "run-1", effective_token_limit=20_000)


async def test_terminal_run_id_restarts_at_queue_tail(store, redis):
    await _acquire(store, "run-1")
    waiting = await _acquire(store, "run-2")
    assert waiting.state == "waiting"

    released = await store.release(
        model=MODEL,
        run_id="run-1",
        outcome="finished",
    )
    restarted = await _acquire(store, "run-1")

    assert released.state == "released"
    assert released.outcome == "finished"
    assert restarted.state == "waiting"
    keys = BenchmarkQueueKeys.for_run(MODEL_KEY, "run-1")
    assert await redis.lrange(keys.queue, 0, -1) == ["run-2", "run-1"]


async def test_first_terminal_outcome_wins(store):
    await _acquire(store, "run-1")

    first = await store.release(model=MODEL, run_id="run-1", outcome="failed")
    duplicate = await store.release(
        model=MODEL,
        run_id="run-1",
        outcome="cancelled",
    )

    assert first.outcome == "failed"
    assert duplicate.outcome == "failed"


async def test_duplicate_release_finishes_interrupted_terminal_cleanup(store, redis):
    await _acquire(store, "run-1")
    keys = BenchmarkQueueKeys.for_run(MODEL_KEY, "run-1")
    await redis.hset(keys.run_meta, mapping={"outcome": "failed"})

    released = await store.release(
        model=MODEL,
        run_id="run-1",
        outcome="cancelled",
    )

    assert released.outcome == "failed"
    assert await redis.lpos(keys.queue, "run-1") is None
    assert not await redis.exists(keys.alive)
    assert "completed_at" in await redis.hgetall(keys.run_meta)


async def test_restart_finishes_interrupted_release_before_joining_tail(store, redis):
    await _acquire(store, "run-1")
    await _acquire(store, "run-2")
    keys = BenchmarkQueueKeys.for_run(MODEL_KEY, "run-1")
    await redis.hset(keys.run_meta, mapping={"outcome": "failed"})

    restarted = await _acquire(store, "run-1")

    assert restarted.state == "waiting"
    assert await redis.lrange(keys.queue, 0, -1) == ["run-2", "run-1"]


async def test_wait_returns_current_state_without_owning_a_worker_context(store):
    await _acquire(store, "run-1")
    waiting = await _acquire(store, "run-2")
    assert waiting.state == "waiting"

    timed_out = await store.wait(model=MODEL, run_id="run-2", timeout_seconds=0)
    await store.release(model=MODEL, run_id="run-1", outcome="finished")
    acquired = await store.wait(model=MODEL, run_id="run-2", timeout_seconds=0)

    assert timed_out.state == "waiting"
    assert acquired.state == "acquired"


async def test_renew_terminalizes_expired_run_and_advances_successor(store, redis):
    await _acquire(store, "run-1")
    await _acquire(store, "run-2")
    run_1_keys = BenchmarkQueueKeys.for_run(MODEL_KEY, "run-1")
    await redis.delete(run_1_keys.alive)

    expired = await store.renew(model=MODEL, run_id="run-1")
    successor = await store.renew(model=MODEL, run_id="run-2")

    assert expired.state == "released"
    assert expired.outcome == "failed"
    assert successor.state == "acquired"


async def test_immediate_early_release_advances_queue_but_remains_live(store, redis):
    await _acquire(
        store,
        "run-1",
        total_requests=1,
        immediate_queue_release=True,
    )
    await _acquire(store, "run-2")
    run_1_keys = BenchmarkQueueKeys.for_run(MODEL_KEY, "run-1")
    await redis.sadd(run_1_keys.dispatched, "question-1")

    renewed = await store.renew(model=MODEL, run_id="run-1")
    successor = await store.renew(model=MODEL, run_id="run-2")

    assert renewed.state == "acquired"
    assert successor.state == "acquired"
    assert await redis.lpos(run_1_keys.queue, "run-1") is None
    assert await redis.zscore(run_1_keys.active_heads, "run-1") is None
    assert await redis.exists(run_1_keys.alive)


async def test_grace_early_release_keeps_first_deadline(store, redis, monkeypatch):
    await _acquire(store, "run-1", total_requests=1)
    await _acquire(store, "run-2")
    run_1_keys = BenchmarkQueueKeys.for_run(MODEL_KEY, "run-1")
    await redis.sadd(run_1_keys.dispatched, "question-1")

    monkeypatch.setattr(
        "model_gateway.benchmark_admission_state.time.time",
        lambda: 100.0,
    )
    await asyncio.gather(
        store.renew(model=MODEL, run_id="run-1"),
        store.renew(model=MODEL, run_id="run-1"),
    )
    first_deadline = (await redis.hgetall(run_1_keys.run_meta))[
        "early_release_deadline"
    ]

    assert (await redis.hgetall(run_1_keys.run_meta))["early_release_deadline"] == (
        first_deadline
    )
    assert await redis.lpos(run_1_keys.queue, "run-1") == 0

    monkeypatch.setattr(
        "model_gateway.benchmark_admission_state.time.time",
        lambda: 106.0,
    )
    renewed = await store.renew(model=MODEL, run_id="run-1")

    assert renewed.state == "acquired"
    assert await redis.lpos(run_1_keys.queue, "run-1") is None


async def test_renew_racing_final_release_keeps_terminal_outcome(store, redis):
    await _acquire(
        store,
        "run-1",
        total_requests=1,
        immediate_queue_release=True,
    )
    keys = BenchmarkQueueKeys.for_run(MODEL_KEY, "run-1")
    await redis.sadd(keys.dispatched, "question-1")

    _, released = await asyncio.gather(
        store.renew(model=MODEL, run_id="run-1"),
        store.release(model=MODEL, run_id="run-1", outcome="failed"),
    )
    duplicate = await store.release(
        model=MODEL,
        run_id="run-1",
        outcome="cancelled",
    )

    assert released.outcome == "failed"
    assert duplicate.outcome == "failed"
    assert await redis.lpos(keys.queue, "run-1") is None


async def test_request_driven_state_remains_visible_in_existing_queue_status(store):
    await _acquire(store, "run-1")

    models = (await get_status()).models

    assert len(models) == 1
    assert models[0].queue is not None
    assert models[0].queue.entries[0].run_id == "run-1"
    assert models[0].queue.entries[0].is_active_head
