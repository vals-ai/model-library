import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from redis.exceptions import RedisError

import model_gateway.rate_limit_monitor.manager as monitor_module
from model_gateway.rate_limit_monitor.manager import (
    OWNERSHIP_RETRY_SECONDS,
    _SourceProbe,
)
from model_gateway.rate_limit_monitor.state import (
    MonitorStateCorrupt,
)

from tests.unit.model_gateway.rate_limit_monitor.manager._support import (
    MODEL,
    FakeStore,
    ControlledSleep,
    FakeProbe,
    _rate_limit,
    _monitor,
    _stub_sources,
    _wait_until,
    _isolate_monitor_metrics as _isolate_monitor_metrics,
)


async def test_discovery_runs_active_model_without_registry_filter(monkeypatch):
    removed_model = "openai/removed-model"
    store = FakeStore()
    store.active = {removed_model}
    monitor = _monitor(monkeypatch, store)
    sleep = ControlledSleep()
    monitor._sleep = sleep
    model_worker = AsyncMock()
    monkeypatch.setattr(monitor, "_model_worker", model_worker)
    discovery_task = asyncio.create_task(monitor._discovery_loop())
    try:
        await sleep.next()

        model_worker.assert_awaited_once_with(removed_model)
        assert removed_model in monitor._model_tasks
    finally:
        discovery_task.cancel()
        await asyncio.gather(discovery_task, return_exceptions=True)


async def test_discovery_keeps_contended_standby_task_until_model_becomes_inactive(
    monkeypatch,
):
    store = FakeStore()
    store.leases = {}
    monitor = _monitor(monkeypatch, store)
    sleep = ControlledSleep()
    monitor._sleep = sleep
    ownership = Mock()
    monkeypatch.setattr(
        monitor_module,
        "record_rate_limit_monitor_ownership",
        ownership,
    )
    discovery_task = asyncio.create_task(monitor._discovery_loop())
    try:
        await _wait_until(lambda: len(store.claim_calls) == 1)
        task = monitor._model_tasks[MODEL]
        first_seconds, first_future = await sleep.next()
        second_seconds, second_future = await sleep.next()
        sleeps = {
            first_seconds: first_future,
            second_seconds: second_future,
        }

        assert set(sleeps) == {
            monitor_module.DISCOVERY_SECONDS,
            OWNERSHIP_RETRY_SECONDS,
        }
        sleeps[monitor_module.DISCOVERY_SECONDS].set_result(None)
        await _wait_until(lambda: store.discover_calls == 2)
        seconds, inactive_refresh = await sleep.next()

        assert seconds == monitor_module.DISCOVERY_SECONDS == 1.0
        assert monitor._model_tasks[MODEL] is task
        assert len(store.claim_calls) == 1
        assert not task.done()

        store.active = set()
        inactive_refresh.set_result(None)
        await _wait_until(lambda: MODEL not in monitor._model_tasks)

        assert task.cancelled()
        assert len(store.claim_calls) == 1
        ownership.assert_called_once_with("contended")
    finally:
        discovery_task.cancel()
        await asyncio.gather(discovery_task, return_exceptions=True)
        await monitor._cancel_model_tasks()


async def test_discovery_runs_every_second_and_cancels_inactive_model(monkeypatch):
    store = FakeStore()
    monitor = _monitor(monkeypatch, store)
    sleep = ControlledSleep()
    monitor._sleep = sleep
    worker_started = asyncio.Event()
    worker_cancelled = asyncio.Event()

    async def model_worker(model: str) -> None:
        assert model == MODEL
        worker_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            worker_cancelled.set()

    monkeypatch.setattr(monitor, "_model_worker", model_worker)
    discovery_task = asyncio.create_task(monitor._discovery_loop())
    try:
        await worker_started.wait()
        seconds, future = await sleep.next()
        assert seconds == monitor_module.DISCOVERY_SECONDS == 1.0
        assert store.discover_calls == 1

        store.active = set()
        future.set_result(None)
        await worker_cancelled.wait()
        seconds, _ = await sleep.next()
        assert seconds == 1.0
        assert store.discover_calls == 2
    finally:
        discovery_task.cancel()
        await asyncio.gather(discovery_task, return_exceptions=True)
        await monitor._cancel_model_tasks()


async def test_start_deduplicates_model_task_and_close_cancels_it(monkeypatch):
    store = FakeStore()
    monitor = _monitor(monkeypatch, store)
    probe = FakeProbe([_rate_limit()])
    probe.block = True
    _stub_sources(monkeypatch, monitor, _SourceProbe("default", "openai", probe))

    monitor.start()
    discovery_task = monitor._discovery_task
    monitor.start()
    assert monitor._discovery_task is discovery_task
    await probe.called.wait()
    first_task = monitor._model_tasks[MODEL]

    assert monitor._model_tasks[MODEL] is first_task
    await monitor.close()
    assert monitor._model_tasks == {}
    assert len(store.release_calls) == 1


async def test_check_health_tracks_monitor_lifecycle(monkeypatch):
    store = FakeStore()
    store.active = set()
    monitor = _monitor(monkeypatch, store)

    with pytest.raises(RuntimeError, match="not running"):
        monitor.check_health()

    monitor.start()
    await _wait_until(lambda: store.discover_calls == 1)
    monitor.check_health()

    await monitor.close()
    with pytest.raises(RuntimeError, match="not running"):
        monitor.check_health()


async def test_discovery_retries_redis_failure_and_readiness_recovers(monkeypatch):
    store = FakeStore()
    store.active = set()
    store.discover_error = RedisError("Timeout connecting to server")
    monitor = _monitor(monkeypatch, store)
    sleep = ControlledSleep()
    monitor._sleep = sleep
    now = [100.0]
    monitor._monotonic = lambda: now[0]
    log_event = Mock()
    monkeypatch.setattr(monitor_module, "log_gateway_event", log_event)

    monitor.start()
    try:
        seconds, future = await sleep.next()
        assert seconds == monitor_module.DISCOVERY_SECONDS
        assert store.discover_calls == 1
        monitor.check_health()
        assert log_event.call_count == 1
        assert log_event.call_args.args == (
            "gateway.rate_limit_monitor.discovery_failure",
        )

        future.set_result(None)
        seconds, future = await sleep.next()
        assert seconds == monitor_module.DISCOVERY_SECONDS
        assert store.discover_calls == 2
        assert log_event.call_count == 1

        now[0] += monitor_module.DISCOVERY_FAILURE_GRACE_SECONDS + 1
        with pytest.raises(RuntimeError, match="discovery is failing"):
            monitor.check_health()

        store.discover_error = None
        future.set_result(None)
        _, future = await sleep.next()
        assert store.discover_calls == 3
        monitor.check_health()
        assert [call.args[0] for call in log_event.call_args_list] == [
            "gateway.rate_limit_monitor.discovery_failure",
            "gateway.rate_limit_monitor.discovery_recovered",
        ]

        future.set_result(None)
        _, future = await sleep.next()
        assert store.discover_calls == 4
        assert log_event.call_count == 2

        store.discover_error = RedisError("Timeout connecting to server")
        future.set_result(None)
        _, future = await sleep.next()
        assert store.discover_calls == 5
        assert [call.args[0] for call in log_event.call_args_list] == [
            "gateway.rate_limit_monitor.discovery_failure",
            "gateway.rate_limit_monitor.discovery_recovered",
            "gateway.rate_limit_monitor.discovery_failure",
        ]

        store.discover_error = None
        future.set_result(None)
        await sleep.next()
        assert store.discover_calls == 6
        assert log_event.call_args.args == (
            "gateway.rate_limit_monitor.discovery_recovered",
        )
        assert log_event.call_count == 4
    finally:
        await monitor.close()


async def test_discovery_reaps_exceptional_worker_without_restart(monkeypatch):
    store = FakeStore()
    monitor = _monitor(monkeypatch, store)

    async def fail() -> None:
        raise MonitorStateCorrupt("worker failed")

    worker = asyncio.create_task(fail())
    await asyncio.gather(worker, return_exceptions=True)
    monitor._model_tasks[MODEL] = worker

    with pytest.raises(MonitorStateCorrupt, match="worker failed"):
        await monitor._discovery_loop()

    assert MODEL not in monitor._model_tasks


async def test_discovery_restarts_normally_completed_active_worker(monkeypatch):
    store = FakeStore()
    monitor = _monitor(monkeypatch, store)
    sleep = ControlledSleep()
    monitor._sleep = sleep
    replacement_started = asyncio.Event()

    completed = asyncio.create_task(asyncio.sleep(0))
    await completed
    monitor._model_tasks[MODEL] = completed

    async def replacement(model: str) -> None:
        assert model == MODEL
        replacement_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(monitor, "_model_worker", replacement)
    discovery = asyncio.create_task(monitor._discovery_loop())
    try:
        await replacement_started.wait()
        await sleep.next()
        replacement_task = monitor._model_tasks[MODEL]
        assert replacement_task is not completed
        assert not replacement_task.done()
    finally:
        discovery.cancel()
        await asyncio.gather(discovery, return_exceptions=True)
        await monitor._cancel_model_tasks()


async def test_discovery_cancels_inactive_claimed_worker_and_releases(monkeypatch):
    store = FakeStore()
    monitor = _monitor(monkeypatch, store)
    sleep = ControlledSleep()
    monitor._sleep = sleep
    probe = FakeProbe([_rate_limit()])
    probe.block = True
    _stub_sources(monkeypatch, monitor, _SourceProbe("default", "openai", probe))
    discovery = asyncio.create_task(monitor._discovery_loop())
    try:
        await probe.called.wait()
        pending_sleeps = [await sleep.next(), await sleep.next()]
        discovery_sleep = next(item for item in pending_sleeps if item[0] == 1.0)
        store.active = set()
        discovery_sleep[1].set_result(None)
        await _wait_until(lambda: MODEL not in monitor._model_tasks)

        assert probe.finished.is_set()
        assert len(store.release_calls) == 1
    finally:
        discovery.cancel()
        await asyncio.gather(discovery, return_exceptions=True)
        await monitor._cancel_model_tasks()
