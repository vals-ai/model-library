"""
Unit tests for benchmark queue FIFO serialization.
"""

import asyncio
import importlib
import logging
from unittest.mock import AsyncMock, patch

import pytest
from fakeredis import aioredis
from model_library.retriers.token.benchmark_queue import (
    BENCHMARK_NOTIFY_CANCELLED,
    BENCHMARK_NOTIFY_PROCEED,
    BenchmarkQueueCancelled,
    benchmark_queue,
)
from model_library.retriers.token.utils import KEY_PREFIX, get_status, set_redis_client

bq_module = importlib.import_module("model_library.retriers.token.benchmark_queue")

MODEL_KEY = ("openai.gpt-4", "abc123")

logger = logging.getLogger("test_benchmark_queue")


@pytest.fixture
def redis():
    client = aioredis.FakeRedis(decode_responses=True)
    set_redis_client(client)
    return client


def _queue_key():
    return f"{KEY_PREFIX}:{MODEL_KEY[0]}:{MODEL_KEY[1]}:benchmark:queue"


def _alive_key(run_id: str):
    return f"{KEY_PREFIX}:{MODEL_KEY[0]}:{MODEL_KEY[1]}:benchmark:alive:{run_id}"


def _notify_key(run_id: str):
    return f"{KEY_PREFIX}:{MODEL_KEY[0]}:{MODEL_KEY[1]}:benchmark:notify:{run_id}"


def _run_dispatched_key(run_id: str):
    return (
        f"{KEY_PREFIX}:{MODEL_KEY[0]}:{MODEL_KEY[1]}:benchmark:run:{run_id}:dispatched"
    )


def _active_heads_key():
    return f"{KEY_PREFIX}:{MODEL_KEY[0]}:{MODEL_KEY[1]}:benchmark:active_heads"


def _window_key():
    return f"{KEY_PREFIX}:{MODEL_KEY[0]}:{MODEL_KEY[1]}:benchmark:active_head_window"


def _unhealthy_since_key():
    return f"{KEY_PREFIX}:{MODEL_KEY[0]}:{MODEL_KEY[1]}:benchmark:unhealthy_since"


def _token_key():
    return f"{KEY_PREFIX}:{MODEL_KEY[0]}:{MODEL_KEY[1]}:tokens"


def _run_meta_key(run_id: str):
    return f"{KEY_PREFIX}:{MODEL_KEY[0]}:{MODEL_KEY[1]}:benchmark:run:{run_id}"


async def _set_token_health(
    redis,
    *,
    current: int,
    limit: int,
    ewma_ratio: float,
    short_ewma_ratio: float | None = None,
):
    token_key = _token_key()
    await redis.set(token_key, current)
    await redis.set(f"{token_key}:limit", limit)
    await redis.set(f"{token_key}:remaining_ratio_ewma_2m", ewma_ratio)
    await redis.set(
        f"{token_key}:remaining_ratio_ewma_15s",
        ewma_ratio if short_ewma_ratio is None else short_ewma_ratio,
    )


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


@patch.object(bq_module, "HOURS_24", 0)
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


@patch.object(bq_module, "ACTIVE_HEAD_SCALE_UP_INTERVAL", 0)
async def test_healthy_token_state_admits_additional_active_head(redis):
    """Healthy token state increases the active-head window and admits a second run."""
    await _set_token_health(redis, current=30, limit=100, ewma_ratio=0.30)
    run1_entered = asyncio.Event()
    run2_entered = asyncio.Event()
    run1_done = False

    async def run1():
        nonlocal run1_done
        async with benchmark_queue(MODEL_KEY, "run-1", logger):
            run1_entered.set()
            await asyncio.wait_for(run2_entered.wait(), timeout=2.0)
        run1_done = True

    async def run2():
        await run1_entered.wait()
        async with benchmark_queue(MODEL_KEY, "run-2", logger):
            assert not run1_done
            run2_entered.set()

    await asyncio.gather(run1(), run2())


async def test_missing_token_health_keeps_single_active_head(redis):
    """Extra heads require token health; the first run still proceeds without it."""
    run1_entered = asyncio.Event()
    run2_entered = False

    async def run1():
        async with benchmark_queue(MODEL_KEY, "run-1", logger):
            run1_entered.set()
            await asyncio.sleep(0.1)

    async def run2():
        nonlocal run2_entered
        await run1_entered.wait()
        async with benchmark_queue(MODEL_KEY, "run-2", logger):
            run2_entered = True

    task1 = asyncio.create_task(run1())
    task2 = asyncio.create_task(run2())
    await asyncio.sleep(0.05)

    assert not run2_entered

    await asyncio.gather(task1, task2)


@patch.object(bq_module, "ACTIVE_HEAD_SCALE_UP_INTERVAL", 0)
async def test_mixed_token_health_uses_lower_current_or_ewma_ratio(redis):
    """A low EWMA blocks scale-up even when current remaining tokens are high."""
    await _set_token_health(redis, current=90, limit=100, ewma_ratio=0.20)
    run1_entered = asyncio.Event()
    run2_entered = False

    async def run1():
        async with benchmark_queue(MODEL_KEY, "run-1", logger):
            run1_entered.set()
            await asyncio.sleep(0.1)

    async def run2():
        nonlocal run2_entered
        await run1_entered.wait()
        async with benchmark_queue(MODEL_KEY, "run-2", logger):
            run2_entered = True

    task1 = asyncio.create_task(run1())
    task2 = asyncio.create_task(run2())
    await asyncio.sleep(0.05)

    assert not run2_entered

    await asyncio.gather(task1, task2)


@patch.object(bq_module, "ACTIVE_HEAD_QUEUE_SCAN_LIMIT", 2)
@patch.object(bq_module, "ACTIVE_HEAD_SCALE_UP_INTERVAL", 0)
async def test_active_head_admission_scans_bounded_queue_prefix(redis):
    """Active-head control does not scan/admit beyond the bounded FIFO prefix."""
    now = 1000.0
    await _set_token_health(redis, current=90, limit=100, ewma_ratio=0.90)
    await redis.rpush(_queue_key(), "run-1", "run-2", "run-3")
    for run_id in ("run-1", "run-2", "run-3"):
        await redis.set(_alive_key(run_id), "1")
    await redis.set(_window_key(), "3")

    with patch.object(bq_module.time, "time", return_value=now):
        await bq_module._control_and_admit_heads(  # pyright: ignore[reportPrivateUsage]
            redis,
            _queue_key(),
            f"{KEY_PREFIX}:{MODEL_KEY[0]}:{MODEL_KEY[1]}:benchmark",
            _token_key(),
            logger,
        )

    assert await redis.zrange(_active_heads_key(), 0, -1) == ["run-1", "run-2"]
    assert await redis.llen(_notify_key("run-3")) == 0


@patch.object(bq_module, "ACTIVE_HEAD_SCALE_UP_INTERVAL", 0)
async def test_sustained_unhealthy_token_state_scales_window_down_by_attrition(redis):
    """Below 15% short health for 15 seconds lowers the window by attrition."""
    now = 1000.0
    await _set_token_health(redis, current=10, limit=100, ewma_ratio=0.10)
    await redis.rpush(_queue_key(), "run-1", "run-2")
    await redis.set(_alive_key("run-1"), "1")
    await redis.set(_alive_key("run-2"), "1")
    await redis.set(_window_key(), "2")
    await redis.set(_unhealthy_since_key(), str(now - 15))
    await redis.zadd(_active_heads_key(), {"run-1": now - 10, "run-2": now - 9})

    with patch.object(bq_module.time, "time", return_value=now):
        await bq_module._control_and_admit_heads(  # pyright: ignore[reportPrivateUsage]
            redis,
            _queue_key(),
            f"{KEY_PREFIX}:{MODEL_KEY[0]}:{MODEL_KEY[1]}:benchmark",
            _token_key(),
            logger,
        )

    assert await redis.get(_window_key()) == "1"
    assert set(await redis.zrange(_active_heads_key(), 0, -1)) == {"run-1", "run-2"}


@patch.object(bq_module, "ACTIVE_HEAD_SCALE_UP_INTERVAL", 0)
async def test_scale_down_waits_for_short_ewma_to_drop(redis):
    """A transient current-token dip does not downscale while 15s health is okay."""
    now = 1000.0
    await _set_token_health(
        redis,
        current=10,
        limit=100,
        ewma_ratio=0.10,
        short_ewma_ratio=0.20,
    )
    await redis.rpush(_queue_key(), "run-1", "run-2")
    await redis.set(_alive_key("run-1"), "1")
    await redis.set(_alive_key("run-2"), "1")
    await redis.set(_window_key(), "2")
    await redis.set(_unhealthy_since_key(), str(now - 15))
    await redis.zadd(_active_heads_key(), {"run-1": now - 10, "run-2": now - 9})

    with patch.object(bq_module.time, "time", return_value=now):
        await bq_module._control_and_admit_heads(  # pyright: ignore[reportPrivateUsage]
            redis,
            _queue_key(),
            f"{KEY_PREFIX}:{MODEL_KEY[0]}:{MODEL_KEY[1]}:benchmark",
            _token_key(),
            logger,
        )

    assert await redis.get(_window_key()) == "2"
    assert await redis.exists(_unhealthy_since_key()) == 0


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


async def test_cancel_check_exits_after_clearing_stale_notification(redis):
    """A pre-entry cancel check prevents deleting a valid cancellation and then blocking."""
    executed = False

    async def is_cancelled():
        return True

    await redis.rpush(_notify_key("run-1"), BENCHMARK_NOTIFY_CANCELLED)

    with pytest.raises(BenchmarkQueueCancelled):
        async with benchmark_queue(
            MODEL_KEY, "run-1", logger, is_cancelled=is_cancelled
        ):
            executed = True

    assert not executed
    assert await redis.llen(_queue_key()) == 0
    assert await redis.llen(_notify_key("run-1")) == 0


async def test_stale_cancelled_notification_is_cleared_when_not_cancelled(redis):
    """A stale cancellation sentinel from a prior attempt does not block a resumed run."""
    executed = False

    async def is_cancelled():
        return False

    await redis.rpush(_notify_key("run-1"), BENCHMARK_NOTIFY_CANCELLED)

    async with benchmark_queue(MODEL_KEY, "run-1", logger, is_cancelled=is_cancelled):
        executed = True

    assert executed
    assert await redis.llen(_queue_key()) == 0


async def test_cancel_check_exits_after_valid_proceed_before_yield(redis):
    """Cancellation after proceed validation but before yield does not let the body run."""
    cancel_now = False
    executed = False
    original_zscore = redis.zscore

    async def is_cancelled():
        return cancel_now

    async def zscore_and_cancel(name, value):
        nonlocal cancel_now
        score = await original_zscore(name, value)
        if name == _active_heads_key() and value == "run-1" and score is not None:
            cancel_now = True
        return score

    redis.zscore = zscore_and_cancel

    with pytest.raises(BenchmarkQueueCancelled):
        async with benchmark_queue(
            MODEL_KEY, "run-1", logger, is_cancelled=is_cancelled
        ):
            executed = True

    assert not executed
    assert await redis.llen(_queue_key()) == 0


async def test_waiting_run_uses_cancel_notification_not_cancel_polling(redis):
    """A queued run waits for the backend cancellation sentinel instead of polling."""
    holder_entered = asyncio.Event()
    cancel_now = False
    executed = False

    async def is_cancelled():
        return cancel_now

    async def holder():
        async with benchmark_queue(MODEL_KEY, "run-1", logger):
            holder_entered.set()
            await asyncio.sleep(0.2)

    async def waiter():
        nonlocal executed
        with pytest.raises(BenchmarkQueueCancelled):
            async with benchmark_queue(
                MODEL_KEY, "run-2", logger, is_cancelled=is_cancelled
            ):
                executed = True

    holder_task = asyncio.create_task(holder())
    await holder_entered.wait()

    waiter_task = asyncio.create_task(waiter())
    while await redis.lpos(_queue_key(), "run-2") is None:
        await asyncio.sleep(0.01)

    cancel_now = True
    await asyncio.sleep(0.05)
    assert not waiter_task.done()

    await redis.delete(_notify_key("run-2"))
    await redis.rpush(_notify_key("run-2"), BENCHMARK_NOTIFY_CANCELLED)
    await redis.lrem(_queue_key(), 1, "run-2")

    await asyncio.wait_for(waiter_task, timeout=2.0)
    await holder_task

    assert not executed
    assert await redis.lpos(_queue_key(), "run-2") is None


async def test_cancelled_notification_exits_waiter(redis):
    """A queued run can be woken with a cancellation sentinel without running the body."""
    holder_entered = asyncio.Event()
    executed = False

    async def holder():
        async with benchmark_queue(MODEL_KEY, "run-1", logger):
            holder_entered.set()
            await asyncio.sleep(0.1)

    async def waiter():
        nonlocal executed
        with pytest.raises(BenchmarkQueueCancelled):
            async with benchmark_queue(MODEL_KEY, "run-2", logger):
                executed = True

    holder_task = asyncio.create_task(holder())
    await holder_entered.wait()

    waiter_task = asyncio.create_task(waiter())
    while await redis.lpos(_queue_key(), "run-2") is None:
        await asyncio.sleep(0.01)

    await redis.rpush(_notify_key("run-2"), BENCHMARK_NOTIFY_CANCELLED)

    await asyncio.wait_for(waiter_task, timeout=2.0)
    await holder_task

    assert not executed
    assert await redis.lpos(_queue_key(), "run-2") is None


async def test_stale_proceed_notification_for_non_head_is_ignored(redis):
    """A stale proceed token must not let a non-head run acquire the queue slot."""
    holder_entered = asyncio.Event()
    executed = False

    async def holder():
        async with benchmark_queue(MODEL_KEY, "run-1", logger):
            holder_entered.set()
            await asyncio.sleep(0.1)

    async def waiter():
        nonlocal executed
        with pytest.raises(BenchmarkQueueCancelled):
            async with benchmark_queue(MODEL_KEY, "run-2", logger):
                executed = True

    holder_task = asyncio.create_task(holder())
    await holder_entered.wait()

    waiter_task = asyncio.create_task(waiter())
    while await redis.lpos(_queue_key(), "run-2") is None:
        await asyncio.sleep(0.01)

    await redis.rpush(_notify_key("run-2"), BENCHMARK_NOTIFY_PROCEED)
    await asyncio.sleep(0.05)
    assert not executed

    # Match backend cleanup: stale notifications are deleted before pushing cancel.
    await redis.delete(_notify_key("run-2"))
    await redis.rpush(_notify_key("run-2"), BENCHMARK_NOTIFY_CANCELLED)
    await redis.lrem(_queue_key(), 1, "run-2")
    await asyncio.wait_for(waiter_task, timeout=2.0)
    await holder_task

    assert not executed


async def test_cancelled_notification_survives_stale_proceed_race(redis):
    """A cancel wake pushed after a stale proceed pop must not be deleted before it is read."""
    holder_entered = asyncio.Event()
    executed = False
    simulate_cleanup_race = False
    original_zscore = redis.zscore

    async def racing_zscore(name, value):
        nonlocal simulate_cleanup_race
        if simulate_cleanup_race and name == _active_heads_key() and value == "run-2":
            simulate_cleanup_race = False
            await redis.zrem(_active_heads_key(), "run-2")
            await redis.lrem(_queue_key(), 1, "run-2")
            await redis.rpush(_notify_key("run-2"), BENCHMARK_NOTIFY_CANCELLED)
        return await original_zscore(name, value)

    redis.zscore = racing_zscore

    async def holder():
        async with benchmark_queue(MODEL_KEY, "run-1", logger):
            holder_entered.set()
            await asyncio.sleep(0.1)

    async def waiter():
        nonlocal executed
        with pytest.raises(BenchmarkQueueCancelled):
            async with benchmark_queue(MODEL_KEY, "run-2", logger):
                executed = True

    holder_task = asyncio.create_task(holder())
    await holder_entered.wait()

    waiter_task = asyncio.create_task(waiter())
    while await redis.lpos(_queue_key(), "run-2") is None:
        await asyncio.sleep(0.01)

    simulate_cleanup_race = True
    await redis.rpush(_notify_key("run-2"), BENCHMARK_NOTIFY_PROCEED)

    await asyncio.wait_for(waiter_task, timeout=2.0)
    await holder_task

    assert not executed


async def test_cancelled_cleanup_does_not_immediately_notify_next(redis):
    """A cancelled head run leaves the next waiter blocked until heartbeat self-promotion."""
    cancel_now = False

    async def is_cancelled():
        return cancel_now

    async with benchmark_queue(MODEL_KEY, "run-1", logger, is_cancelled=is_cancelled):
        await redis.rpush(_queue_key(), "run-2")
        cancel_now = True

    assert await redis.llen(_notify_key("run-2")) == 0


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

    models = (await get_status()).models

    assert len(models) == 1
    queue = models[0].queue
    assert queue is not None
    assert queue.length == 2
    assert queue.entries[0].run_id == "run-1"
    assert queue.entries[1].run_id == "run-2"
    assert queue.entries[0].alive
    assert queue.entries[1].alive
    assert queue.entries[0].heartbeat_ttl > 0
    assert queue.active_head_window == 1
    assert queue.active_heads == ["run-1"]
    assert queue.entries[0].is_active_head
    assert queue.entries[0].display_state == "ACTIVE_HEAD"
    assert not queue.entries[1].is_active_head

    holder_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await holder_task
    await asyncio.wait_for(waiter_task, timeout=2.0)


async def test_get_queue_status_empty(redis):
    """Status on empty queue returns empty models list."""
    models = (await get_status()).models

    assert models == []


async def test_get_queue_status_respects_queue_entry_limit(redis):
    """Bounded status keeps full queue length but fetches only the visible prefix."""
    await redis.rpush(_queue_key(), "run-1", "run-2", "run-3")
    for run_id in ("run-1", "run-2", "run-3"):
        await redis.set(_alive_key(run_id), "1")

    models = (await get_status(queue_entry_limit=2, include_historical=False)).models

    assert len(models) == 1
    queue = models[0].queue
    assert queue is not None
    assert queue.length == 3
    assert [entry.run_id for entry in queue.entries] == ["run-1", "run-2"]


async def test_fast_status_classifies_popped_token_active_run(redis):
    """Fast polling should not mislabel early-released queued runs as direct."""
    await redis.set(_token_key(), "100")
    run_id = "popped-active-run"
    run_meta = f"{KEY_PREFIX}:{MODEL_KEY[0]}:{MODEL_KEY[1]}:benchmark:run:{run_id}"
    question_id = f"{run_id}:question-1"
    question_meta = f"{_token_key()}:inflight:{question_id}"
    await redis.hset(
        run_meta,
        mapping={
            "total_requests": 1,
            "slot_acquired": 1,
            "enqueued_at": 1.0,
            "slot_acquired_at": 2.0,
            "popped_at": 3.0,
        },
    )
    await redis.hset(question_meta, mapping={"run_id": run_id, "priority": "0"})
    await redis.zadd(
        f"{KEY_PREFIX}:{MODEL_KEY[0]}:{MODEL_KEY[1]}:priority:0",
        {question_id: 1_000_000_000_000.0},
    )

    models = (await get_status(queue_entry_limit=100, include_historical=False)).models

    assert len(models) == 1
    queue = models[0].queue
    assert queue is not None
    assert queue.entries[0].run_id == run_id
    assert queue.entries[0].display_state == "POPPED"
    assert queue.entries[0].is_queued
    assert queue.entries[0].popped


async def test_get_queue_status_can_skip_historical_scan(redis):
    """Fast polling can omit popped/completed metadata that has no live token state."""
    await redis.set(_token_key(), "100")
    run_meta = f"{KEY_PREFIX}:{MODEL_KEY[0]}:{MODEL_KEY[1]}:benchmark:run:done-run"
    await redis.hset(
        run_meta,
        mapping={"total_requests": 1, "enqueued_at": 1.0, "completed_at": 2.0},
    )

    models = (await get_status()).models
    assert models[0].queue is not None
    assert models[0].queue.entries[0].run_id == "done-run"

    fast_models = (await get_status(include_historical=False)).models

    assert len(fast_models) == 1
    assert fast_models[0].queue is None


async def test_get_queue_status_includes_history_with_queue_entry_limit(redis):
    """Bounded queue reads can still include completed benchmark history."""
    await redis.set(_token_key(), "100")
    run_meta = f"{KEY_PREFIX}:{MODEL_KEY[0]}:{MODEL_KEY[1]}:benchmark:run:done-run"
    await redis.hset(
        run_meta,
        mapping={
            "total_requests": 1,
            "slot_acquired": 1,
            "enqueued_at": 1.0,
            "popped_at": 2.0,
            "completed_at": 3.0,
        },
    )

    models = (await get_status(queue_entry_limit=100, include_historical=True)).models

    assert len(models) == 1
    queue = models[0].queue
    assert queue is not None
    assert queue.entries[0].run_id == "done-run"
    assert queue.entries[0].display_state == "HISTORY_DONE"


async def test_get_queue_status_dead_entry(redis):
    """Status correctly reports dead entries (no alive key)."""
    await redis.rpush(_queue_key(), "dead-run")

    models = (await get_status()).models

    assert len(models) == 1
    queue = models[0].queue
    assert queue is not None
    assert queue.length == 1
    assert queue.entries[0].run_id == "dead-run"
    assert not queue.entries[0].alive
    assert queue.entries[0].heartbeat_ttl == -1


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

    models = (await get_status()).models

    queue_keys = {m.queue.queue_key for m in models if m.queue}
    assert len(queue_keys) == 2
    assert f"{KEY_PREFIX}:{model_a[0]}:{model_a[1]}:benchmark:queue" in queue_keys
    assert f"{KEY_PREFIX}:{model_b[0]}:{model_b[1]}:benchmark:queue" in queue_keys

    task_a.cancel()
    task_b.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task_a
    with pytest.raises(asyncio.CancelledError):
        await task_b


# ── Early release / dispatch tracking ────────────────────────────────


@patch.object(bq_module, "EARLY_RELEASE_GRACE_PERIOD", 0)
@patch.object(bq_module, "HEARTBEAT_INTERVAL", 0.05)
async def test_early_release_when_all_dispatched(redis):
    """Heartbeat releases slot early when dispatched count reaches total_requests."""
    total = 3
    order = []

    async def run1():
        async with benchmark_queue(MODEL_KEY, "run-1", logger, total_requests=total):
            order.append("run-1-start")
            # simulate all requests being dispatched (per-run dispatched SET)
            for i in range(total):
                await redis.sadd(_run_dispatched_key("run-1"), f"req-{i}")
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


@patch.object(bq_module, "EARLY_RELEASE_GRACE_PERIOD", 0.3)
@patch.object(bq_module, "HEARTBEAT_INTERVAL", 0.05)
async def test_early_release_grace_period(redis):
    """Early release waits for grace period after all requests are dispatched."""
    total = 3
    order = []

    async def run1():
        async with benchmark_queue(MODEL_KEY, "run-1", logger, total_requests=total):
            order.append("run-1-start")
            for i in range(total):
                await redis.sadd(_run_dispatched_key("run-1"), f"req-{i}")
            # grace period is 0.3s — run-2 should NOT start immediately
            await asyncio.sleep(0.1)
            order.append("run-1-after-dispatch")
            await asyncio.sleep(10)
            order.append("run-1-end")

    async def run2():
        async with benchmark_queue(MODEL_KEY, "run-2", logger):
            order.append("run-2")

    task1 = asyncio.create_task(run1())
    task2 = asyncio.create_task(run2())

    # wait a bit — less than grace period
    await asyncio.sleep(0.15)
    assert "run-2" not in order, "run-2 started before grace period expired"

    # now wait for run-2 to proceed (after grace period expires)
    await asyncio.wait_for(task2, timeout=5.0)
    assert "run-2" in order
    assert "run-1-after-dispatch" in order
    assert "run-1-end" not in order

    task1.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task1


async def test_no_early_release_without_total_requests(redis):
    """Without total_requests, dispatched set is ignored for release."""
    order = []

    async def run1():
        async with benchmark_queue(MODEL_KEY, "run-1", logger):
            order.append("run-1-start")
            # add dispatched items — should not trigger early release (no total_requests)
            await redis.sadd(_run_dispatched_key("run-1"), "req-1", "req-2", "req-3")
            await asyncio.sleep(0.1)
            order.append("run-1-end")

    async def run2():
        async with benchmark_queue(MODEL_KEY, "run-2", logger):
            order.append("run-2")

    await asyncio.gather(run1(), run2())

    # run-2 only starts after run-1 finishes (no early release)
    assert order == ["run-1-start", "run-1-end", "run-2"]


@patch.object(bq_module, "HEARTBEAT_INTERVAL", 0.05)
async def test_no_early_release_when_disabled(redis):
    """early_release=False prevents dispatched-based release even with total_requests."""
    total = 3
    order = []

    async def run1():
        async with benchmark_queue(
            MODEL_KEY, "run-1", logger, total_requests=total, early_release=False
        ):
            order.append("run-1-start")
            for i in range(total):
                await redis.sadd(_run_dispatched_key("run-1"), f"req-{i}")
            await asyncio.sleep(0.2)
            order.append("run-1-end")

    async def run2():
        async with benchmark_queue(MODEL_KEY, "run-2", logger):
            order.append("run-2")

    await asyncio.gather(run1(), run2())

    # run-2 only starts after run-1 finishes (early release disabled)
    assert order == ["run-1-start", "run-1-end", "run-2"]


@patch.object(bq_module, "EARLY_RELEASE_GRACE_PERIOD", 0)
@patch.object(bq_module, "HEARTBEAT_INTERVAL", 0.05)
async def test_early_release_notifies_next_run(redis):
    """After early release, the next queued run starts while the first is still executing."""
    total = 2
    run2_started = asyncio.Event()

    async def run1():
        async with benchmark_queue(MODEL_KEY, "run-1", logger, total_requests=total):
            for i in range(total):
                await redis.sadd(_run_dispatched_key("run-1"), f"req-{i}")
            # wait for run-2 to start (proves early release worked)
            await asyncio.wait_for(run2_started.wait(), timeout=5.0)

    async def run2():
        async with benchmark_queue(MODEL_KEY, "run-2", logger):
            run2_started.set()

    await asyncio.gather(run1(), run2())


@patch.object(bq_module, "EARLY_RELEASE_GRACE_PERIOD", 0)
@patch.object(bq_module, "HEARTBEAT_INTERVAL", 0.05)
async def test_cancelled_run_does_not_early_release_to_next_waiter(redis):
    """Early release does not notify the next waiter once the active run is cancelled."""
    total = 2
    cancel_now = False
    run2_started = False

    async def is_cancelled():
        return cancel_now

    async def run1():
        async with benchmark_queue(
            MODEL_KEY, "run-1", logger, total_requests=total, is_cancelled=is_cancelled
        ):
            for i in range(total):
                await redis.sadd(_run_dispatched_key("run-1"), f"req-{i}")
            await asyncio.sleep(0.2)

    async def run2():
        nonlocal run2_started
        async with benchmark_queue(MODEL_KEY, "run-2", logger):
            run2_started = True

    task1 = asyncio.create_task(run1())
    task2 = asyncio.create_task(run2())

    while await redis.lpos(_queue_key(), "run-2") is None:
        await asyncio.sleep(0.01)
    cancel_now = True
    await asyncio.sleep(0.15)

    assert not run2_started
    assert await redis.llen(_notify_key("run-2")) == 0

    task1.cancel()
    task2.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task1
    with pytest.raises(asyncio.CancelledError):
        await task2


@patch.object(bq_module, "EARLY_RELEASE_GRACE_PERIOD", 0)
@patch.object(bq_module, "HEARTBEAT_INTERVAL", 0.05)
async def test_reused_run_id_clears_stale_dispatched_and_lifecycle_fields(redis):
    total = 3
    dispatched_key = _run_dispatched_key("run-1")
    meta_key = _run_meta_key("run-1")
    for i in range(total):
        await redis.sadd(dispatched_key, f"old-req-{i}")
    await redis.hset(
        meta_key,
        mapping={
            "popped_at": 1.0,
            "completed_at": 2.0,
            "slot_acquired_at": 3.0,
        },
    )

    async with benchmark_queue(MODEL_KEY, "run-1", logger, total_requests=total):
        assert await redis.scard(dispatched_key) == 0
        meta = await redis.hgetall(meta_key)
        assert "popped_at" not in meta
        assert "completed_at" not in meta
        assert meta["slot_acquired"] == "1"
        try:
            slot_acquired_at = float(meta["slot_acquired_at"])
        except (TypeError, ValueError):
            pytest.fail("slot_acquired_at must be a numeric Redis value")
        else:
            assert slot_acquired_at > 3.0


@patch.object(bq_module, "EARLY_RELEASE_GRACE_PERIOD", 0)
@patch.object(bq_module, "HEARTBEAT_INTERVAL", 0.05)
async def test_waiting_run_not_released_before_slot_acquired(redis):
    """A waiting run must not early-release before acquiring the slot.

    The slot_acquired guard prevents a run's heartbeat from checking dispatched
    count while still waiting in queue. Even if the per-run dispatched set is
    pre-populated, the run should wait for its turn.
    """
    total = 3

    # pre-populate run-2's dispatched set (simulates unexpected state)
    for i in range(total):
        await redis.sadd(_run_dispatched_key("run-2"), f"req-{i}")

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
            assert "(" not in key and ")" not in key, f"key uses tuple repr: {key}"


# ── Cleanup safety ───────────────────────────────────────────────────


@patch.object(bq_module, "HOURS_24", 0)
async def test_finally_safe_when_blpop_times_out(redis):
    """Finally block doesn't crash with UnboundLocalError when blpop times out."""
    redis.blpop = AsyncMock(return_value=None)

    with pytest.raises(RuntimeError, match="timed out"):
        async with benchmark_queue(MODEL_KEY, "run-1", logger):
            pass


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
