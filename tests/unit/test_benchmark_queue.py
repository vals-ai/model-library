"""
Unit tests for benchmark queue FIFO serialization.
"""

import asyncio
import importlib
import logging
from unittest.mock import AsyncMock, patch

import fakeredis.aioredis
import pytest

bq_module = importlib.import_module("model_library.retriers.token.benchmark_queue")
from model_library.retriers.token.benchmark_queue import (
    benchmark_queue,
)
from model_library.retriers.token.utils import get_status, set_redis_client

MODEL_KEY = ("openai.gpt-4", "abc123")

logger = logging.getLogger("test_benchmark_queue")


@pytest.fixture
def redis():
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    set_redis_client(client)
    return client


def _queue_key():
    return f"{MODEL_KEY[0]}:{MODEL_KEY[1]}:benchmark:queue"


def _alive_key(run_id: str):
    return f"{MODEL_KEY[0]}:{MODEL_KEY[1]}:benchmark:alive:{run_id}"


def _dispatched_key():
    return f"{MODEL_KEY[0]}:{MODEL_KEY[1]}:tokens:inflight:dispatched"


def _benchmark_run_key():
    return f"{MODEL_KEY[0]}:{MODEL_KEY[1]}:tokens:inflight:benchmark_run"


async def _simulate_process_death(redis, run_id: str):
    """Delete alive key and block heartbeat from refreshing it (simulates kill -9)."""
    key = _alive_key(run_id)
    await redis.delete(key)
    original_set = redis.set

    async def blocked_set(name, *args, **kwargs):
        if name == key:
            return True
        return await original_set(name, *args, **kwargs)

    redis.set = blocked_set


# ── Core queue behavior ──────────────────────────────────────────────


async def test_single_run_proceeds(redis):
    """First run in an empty queue self-notifies and executes immediately."""
    executed = False

    async with benchmark_queue(MODEL_KEY, "run-1", logger):
        executed = True

    assert executed
    assert await redis.llen(_queue_key()) == 0


async def test_fifo_ordering(redis):
    """Three concurrent runs execute strictly in FIFO order."""
    order = []

    async def enqueue(run_id: str):
        async with benchmark_queue(MODEL_KEY, run_id, logger):
            order.append(run_id)
            await asyncio.sleep(0.01)

    await asyncio.gather(enqueue("run-1"), enqueue("run-2"), enqueue("run-3"))

    assert order == ["run-1", "run-2", "run-3"]


async def test_cleanup_on_exception_notifies_next(redis):
    """If a run raises, its slot is released and the next run proceeds."""
    order = []

    async def failing_run():
        with pytest.raises(ValueError):
            async with benchmark_queue(MODEL_KEY, "run-1", logger):
                order.append("run-1-start")
                raise ValueError("boom")

    async def waiting_run():
        async with benchmark_queue(MODEL_KEY, "run-2", logger):
            order.append("run-2")

    await asyncio.gather(failing_run(), waiting_run())

    assert "run-1-start" in order
    assert "run-2" in order
    assert order.index("run-1-start") < order.index("run-2")


async def test_idempotent_enqueue(redis):
    """Re-enqueuing the same run_id doesn't create duplicates (server restart edge case)."""
    # Pre-populate as if the server restarted with this run already queued
    await redis.rpush(_queue_key(), "run-1")

    executed = False
    async with benchmark_queue(MODEL_KEY, "run-1", logger):
        executed = True

    assert executed
    assert await redis.llen(_queue_key()) == 0


async def test_timeout_raises(redis):
    """blpop timeout raises RuntimeError."""
    redis.blpop = AsyncMock(return_value=None)

    with pytest.raises(RuntimeError, match="timed out"):
        async with benchmark_queue(MODEL_KEY, "run-1", logger):
            pass


async def test_independent_model_queues(redis):
    """Runs for different models execute concurrently, not serialized."""
    model_a = ("openai.gpt-4", "key-a")
    model_b = ("anthropic.claude", "key-b")
    active = []

    async def enqueue(model_key: tuple[str, str], run_id: str):
        async with benchmark_queue(model_key, run_id, logger):
            active.append(run_id)
            await asyncio.sleep(0.05)

    await asyncio.gather(enqueue(model_a, "a-1"), enqueue(model_b, "b-1"))

    assert set(active) == {"a-1", "b-1"}


async def test_queue_reuse_after_completion(redis):
    """Queue works correctly for a second batch after the first fully drains."""
    order = []

    async def enqueue(run_id: str):
        async with benchmark_queue(MODEL_KEY, run_id, logger):
            order.append(run_id)
            await asyncio.sleep(0.01)

    await asyncio.gather(enqueue("batch1-a"), enqueue("batch1-b"))
    await asyncio.gather(enqueue("batch2-a"), enqueue("batch2-b"))

    assert order == ["batch1-a", "batch1-b", "batch2-a", "batch2-b"]


# ── Cancellation / error resilience ──────────────────────────────────


async def test_cancel_during_wait_propagates(redis):
    """Task cancelled while blocked on blpop raises CancelledError.

    Note: cleanup of the cancelled waiter's queue entry relies on the heartbeat
    eviction mechanism (the waiter's alive key expires, and the next waiter evicts it).
    """

    async def holder():
        async with benchmark_queue(MODEL_KEY, "run-1", logger):
            await asyncio.sleep(0.1)

    async def waiter():
        async with benchmark_queue(MODEL_KEY, "run-2", logger):
            pass

    holder_task = asyncio.create_task(holder())
    waiter_task = asyncio.create_task(waiter())

    await asyncio.sleep(0.02)
    waiter_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await waiter_task

    await holder_task


async def test_cancel_during_execution_notifies_next(redis):
    """Task cancelled mid-execution still triggers cleanup and unblocks the next run."""
    order = []

    async def run1():
        async with benchmark_queue(MODEL_KEY, "run-1", logger):
            order.append("run-1")
            await asyncio.sleep(10)

    async def run2():
        async with benchmark_queue(MODEL_KEY, "run-2", logger):
            order.append("run-2")

    task1 = asyncio.create_task(run1())
    task2 = asyncio.create_task(run2())

    await asyncio.sleep(0.02)
    task1.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task1

    await asyncio.wait_for(task2, timeout=2.0)

    assert order == ["run-1", "run-2"]


async def test_redis_error_during_cleanup_propagates(redis):
    """If Redis dies during finally block, the error surfaces (queue may be stuck)."""

    async def failing_lrem(*args, **kwargs):
        raise ConnectionError("Redis connection lost")

    redis.lrem = failing_lrem

    with pytest.raises(ConnectionError, match="Redis connection lost"):
        async with benchmark_queue(MODEL_KEY, "run-1", logger):
            pass

    # run-1 is still in the queue — cleanup failed
    assert await redis.lpos(_queue_key(), "run-1") is not None


# ── Heartbeat / dead entry eviction ─────────────────────────────────


async def test_heartbeat_sets_alive_key(redis):
    """Alive key is set on entry and deleted on exit."""
    key = _alive_key("run-1")

    async with benchmark_queue(MODEL_KEY, "run-1", logger):
        assert await redis.exists(key) == 1
        assert await redis.ttl(key) > 0

    assert await redis.exists(key) == 0


@patch.object(bq_module, "HEARTBEAT_INTERVAL", 0.05)
async def test_heartbeat_evicts_dead_head(redis):
    """Waiter detects dead head via missing alive key and self-promotes."""
    # Simulate a dead run: in the queue but no alive key
    await redis.rpush(_queue_key(), "dead-run")

    executed = False
    async with benchmark_queue(MODEL_KEY, "run-2", logger):
        executed = True

    assert executed
    assert await redis.lpos(_queue_key(), "dead-run") is None


@patch.object(bq_module, "HEARTBEAT_INTERVAL", 0.05)
async def test_heartbeat_evicts_dead_head_with_live_waiter(redis):
    """run-1 dies (kill -9), run-2 is waiting — heartbeat evicts run-1, run-2 proceeds."""
    order = []

    async def run1():
        async with benchmark_queue(MODEL_KEY, "run-1", logger):
            order.append("run-1")
            # Simulate kill -9: alive key expires and can't be refreshed
            await _simulate_process_death(redis, "run-1")
            await asyncio.sleep(100)

    async def run2():
        async with benchmark_queue(MODEL_KEY, "run-2", logger):
            order.append("run-2")

    task1 = asyncio.create_task(run1())
    task2 = asyncio.create_task(run2())

    await asyncio.wait_for(task2, timeout=5.0)

    task1.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task1

    assert "run-2" in order


@patch.object(bq_module, "HEARTBEAT_INTERVAL", 0.05)
async def test_cancel_during_wait_does_not_break_holder(redis):
    """Cancelling a waiter doesn't affect the currently executing holder."""
    holder_completed = False

    async def holder():
        nonlocal holder_completed
        async with benchmark_queue(MODEL_KEY, "run-1", logger):
            await asyncio.sleep(0.1)
            holder_completed = True

    async def waiter():
        async with benchmark_queue(MODEL_KEY, "run-2", logger):
            pass

    holder_task = asyncio.create_task(holder())
    waiter_task = asyncio.create_task(waiter())

    await asyncio.sleep(0.02)
    waiter_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await waiter_task

    await holder_task
    assert holder_completed


@patch.object(bq_module, "HEARTBEAT_INTERVAL", 0.05)
async def test_concurrent_eviction_only_promotes_one(redis):
    """When head dies, only one of multiple waiters should acquire the slot.

    Without the eviction lock, both waiters would see the dead head, both evict it,
    and both self-notify — breaking mutual exclusion.
    """
    order = []
    active_count = 0

    async def head():
        """Head of queue — dies immediately."""
        async with benchmark_queue(MODEL_KEY, "run-1", logger):
            order.append("run-1")
            await _simulate_process_death(redis, "run-1")
            await asyncio.sleep(100)

    async def waiter(run_id: str):
        nonlocal active_count
        async with benchmark_queue(MODEL_KEY, run_id, logger):
            active_count += 1
            assert active_count == 1, (
                f"mutual exclusion violated: {active_count} runs active"
            )
            order.append(run_id)
            await asyncio.sleep(0.05)
            active_count -= 1

    head_task = asyncio.create_task(head())
    waiter2 = asyncio.create_task(waiter("run-2"))
    waiter3 = asyncio.create_task(waiter("run-3"))

    await asyncio.wait_for(asyncio.gather(waiter2, waiter3), timeout=5.0)

    head_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await head_task

    # both waiters executed, but strictly one at a time
    assert "run-2" in order
    assert "run-3" in order
    assert order.index("run-2") < order.index("run-3")


# ── Queue status ─────────────────────────────────────────────────────


async def test_get_queue_status_during_run(redis):
    """Status returns ordered entries with alive heartbeats while runs are active."""

    async def holder():
        async with benchmark_queue(MODEL_KEY, "run-1", logger):
            await asyncio.sleep(10)

    async def waiter():
        async with benchmark_queue(MODEL_KEY, "run-2", logger):
            pass

    holder_task = asyncio.create_task(holder())
    waiter_task = asyncio.create_task(waiter())

    await asyncio.sleep(0.02)

    statuses = (await get_status()).benchmark_queue

    assert len(statuses) == 1
    status = statuses[0]
    assert status.length == 2
    assert status.entries[0].run_id == "run-1"
    assert status.entries[1].run_id == "run-2"
    assert status.entries[0].alive is True
    assert status.entries[1].alive is True
    assert status.entries[0].heartbeat_ttl > 0

    holder_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await holder_task
    await asyncio.wait_for(waiter_task, timeout=2.0)


async def test_get_queue_status_empty(redis):
    """Status on empty queue returns empty list."""
    statuses = (await get_status()).benchmark_queue

    assert statuses == []


async def test_get_queue_status_dead_entry(redis):
    """Status correctly reports dead entries (no alive key)."""
    await redis.rpush(_queue_key(), "dead-run")

    statuses = (await get_status()).benchmark_queue

    assert len(statuses) == 1
    status = statuses[0]
    assert status.length == 1
    assert status.entries[0].run_id == "dead-run"
    assert status.entries[0].alive is False
    assert status.entries[0].heartbeat_ttl == -1


async def test_get_queue_status_all_queues(redis):
    """Status with no args returns all active queues."""
    model_a = ("openai.gpt-4", "key-a")
    model_b = ("anthropic.claude", "key-b")

    async def holder(model_key, run_id):
        async with benchmark_queue(model_key, run_id, logger):
            await asyncio.sleep(10)

    task_a = asyncio.create_task(holder(model_a, "a-1"))
    task_b = asyncio.create_task(holder(model_b, "b-1"))

    await asyncio.sleep(0.02)

    statuses = (await get_status()).benchmark_queue

    assert len(statuses) == 2
    queue_keys = {s.queue_key for s in statuses}
    assert f"{model_a[0]}:{model_a[1]}:benchmark:queue" in queue_keys
    assert f"{model_b[0]}:{model_b[1]}:benchmark:queue" in queue_keys

    task_a.cancel()
    task_b.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task_a
    with pytest.raises(asyncio.CancelledError):
        await task_b


# ── Early release / dispatch tracking ────────────────────────────────


@patch.object(bq_module, "HEARTBEAT_INTERVAL", 0.05)
async def test_early_release_when_all_dispatched(redis):
    """Heartbeat releases slot early when dispatched count reaches total_requests."""
    total = 3
    order = []

    async def run1():
        async with benchmark_queue(MODEL_KEY, "run-1", logger, total_requests=total):
            order.append("run-1-start")
            # simulate all requests being dispatched
            for i in range(total):
                await redis.sadd(_dispatched_key(), f"req-{i}")
            # hold the slot — heartbeat should early-release before this finishes
            await asyncio.sleep(10)
            order.append("run-1-end")

    async def run2():
        async with benchmark_queue(MODEL_KEY, "run-2", logger):
            order.append("run-2")

    task1 = asyncio.create_task(run1())
    task2 = asyncio.create_task(run2())

    # run-2 should proceed before run-1 ends
    await asyncio.wait_for(task2, timeout=5.0)

    assert "run-2" in order
    assert "run-1-end" not in order  # run-1 still running

    task1.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task1


async def test_benchmark_run_key_set_in_redis(redis):
    """benchmark_run key is set after acquiring slot and cleaned on exit."""
    async with benchmark_queue(MODEL_KEY, "run-1", logger):
        val = await redis.get(_benchmark_run_key())
        assert val == "run-1"

    # key is cleaned up by atomic cleanup on normal exit
    val = await redis.get(_benchmark_run_key())
    assert val is None


async def test_dispatched_key_cleaned_on_exit(redis):
    """Dispatched set is deleted in finally block."""
    async with benchmark_queue(MODEL_KEY, "run-1", logger, total_requests=5):
        await redis.sadd(_dispatched_key(), "req-1", "req-2")
        assert await redis.scard(_dispatched_key()) == 2

    assert await redis.exists(_dispatched_key()) == 0


async def test_stale_dispatched_cleared_on_start(redis):
    """Stale dispatched set from a previous run is cleared when slot is acquired."""
    # simulate stale data from a crashed run
    await redis.sadd(_dispatched_key(), "old-req-1", "old-req-2")

    async with benchmark_queue(MODEL_KEY, "run-1", logger, total_requests=5):
        count = await redis.scard(_dispatched_key())
        assert count == 0  # stale data cleared


async def test_no_early_release_without_total_requests(redis):
    """Without total_requests, dispatched set is ignored for release."""
    order = []

    async def run1():
        async with benchmark_queue(MODEL_KEY, "run-1", logger):
            order.append("run-1-start")
            # add dispatched items — should not trigger early release
            await redis.sadd(_dispatched_key(), "req-1", "req-2", "req-3")
            await asyncio.sleep(0.1)
            order.append("run-1-end")

    async def run2():
        async with benchmark_queue(MODEL_KEY, "run-2", logger):
            order.append("run-2")

    await asyncio.gather(run1(), run2())

    # run-2 only starts after run-1 finishes (no early release)
    assert order == ["run-1-start", "run-1-end", "run-2"]


@patch.object(bq_module, "HEARTBEAT_INTERVAL", 0.05)
async def test_early_release_notifies_next_run(redis):
    """After early release, the next queued run starts while the first is still executing."""
    total = 2
    run2_started = asyncio.Event()

    async def run1():
        async with benchmark_queue(MODEL_KEY, "run-1", logger, total_requests=total):
            for i in range(total):
                await redis.sadd(_dispatched_key(), f"req-{i}")
            # wait for run-2 to start (proves early release worked)
            await asyncio.wait_for(run2_started.wait(), timeout=5.0)

    async def run2():
        async with benchmark_queue(MODEL_KEY, "run-2", logger):
            run2_started.set()

    await asyncio.gather(run1(), run2())


@patch.object(bq_module, "HEARTBEAT_INTERVAL", 0.05)
async def test_waiting_run_not_removed_by_stale_dispatched(redis):
    """A waiting run must not early-release itself when it sees another run's dispatched count.

    Regression: dispatched set is shared per-model. Without the slot_acquired guard,
    a waiting run's heartbeat would see the dispatched count, trigger early release,
    remove itself from the queue, and get stuck on blpop forever.

    Setup: pre-populate the dispatched set and have run-1 NOT use total_requests,
    so only the waiting runs' heartbeats check the dispatched count.
    """
    total = 3

    # pre-populate dispatched set (simulates stale data from executing run)
    for i in range(total):
        await redis.sadd(_dispatched_key(), f"req-{i}")

    async def run1():
        async with benchmark_queue(MODEL_KEY, "run-1", logger):
            # wait for multiple heartbeat cycles
            await asyncio.sleep(0.3)
            # verify waiters are still in the queue
            queue = await redis.lrange(_queue_key(), 0, -1)
            assert "run-2" in queue, "run-2 removed from queue by stale dispatched data"
            assert "run-3" in queue, "run-3 removed from queue by stale dispatched data"

    async def run2():
        async with benchmark_queue(MODEL_KEY, "run-2", logger, total_requests=total):
            pass

    async def run3():
        async with benchmark_queue(MODEL_KEY, "run-3", logger, total_requests=total):
            pass

    await asyncio.wait_for(asyncio.gather(run1(), run2(), run3()), timeout=10.0)


# ── Key formatting ───────────────────────────────────────────────────


async def test_redis_keys_use_colon_separated_format(redis):
    """Keys use 'provider:hash:benchmark:...' not tuple repr like "('provider', 'hash'):..."."""
    async with benchmark_queue(MODEL_KEY, "run-1", logger):
        keys = await redis.keys("*benchmark*")
        for key in keys:
            assert "(" not in key and ")" not in key, (
                f"key uses tuple repr: {key}"
            )


# ── Cleanup safety ───────────────────────────────────────────────────


async def test_finally_safe_when_blpop_times_out(redis):
    """Finally block doesn't crash with UnboundLocalError when blpop times out."""
    redis.blpop = AsyncMock(return_value=None)

    with pytest.raises(RuntimeError, match="timed out"):
        async with benchmark_queue(MODEL_KEY, "run-1", logger):
            pass

    # benchmark_run_key should NOT have been set (slot was never acquired)
    assert await redis.get(_benchmark_run_key()) is None


async def test_cleanup_does_not_delete_next_runs_keys(redis):
    """Atomic cleanup skips deletion when another run has taken over benchmark_run_key."""
    # run-1 acquires slot, then another run overwrites benchmark_run_key
    async with benchmark_queue(MODEL_KEY, "run-1", logger, total_requests=5):
        await redis.sadd(_dispatched_key(), "req-1")
        # simulate run-2 taking over (as if early release happened and run-2 acquired)
        await redis.set(_benchmark_run_key(), "run-2")
        await redis.sadd(_dispatched_key(), "run-2-req")

    # run-1's cleanup should NOT have deleted run-2's keys
    assert await redis.get(_benchmark_run_key()) == "run-2"
    assert await redis.scard(_dispatched_key()) == 2


@patch.object(bq_module, "HEARTBEAT_INTERVAL", 0.05)
async def test_self_promote_when_head_crashes_between_lrem_and_notify(redis):
    """If previous head is removed from queue but _notify_next never fires
    (process killed between lrem and _notify_next), the next run's heartbeat
    self-promotes by notifying itself when it sees it's at head."""

    async def run1():
        async with benchmark_queue(MODEL_KEY, "run-1", logger):
            # Simulate: run-1 removes itself from queue but crashes before
            # calling _notify_next. We do this by directly manipulating Redis.
            await redis.lrem(_queue_key(), 1, "run-1")
            # Don't call _notify_next — simulates crash between lrem and notify.
            # Now sleep long enough for run-2's heartbeat to self-promote.
            await asyncio.sleep(0.3)

    async def run2():
        async with benchmark_queue(MODEL_KEY, "run-2", logger):
            pass

    task1 = asyncio.create_task(run1())
    task2 = asyncio.create_task(run2())

    # run-2 should self-promote via heartbeat within a few intervals
    await asyncio.wait_for(task2, timeout=5.0)

    task1.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task1
