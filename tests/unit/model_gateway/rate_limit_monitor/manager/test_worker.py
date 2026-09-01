import asyncio
from unittest.mock import Mock, call

import pytest

import model_gateway.rate_limit_monitor.manager as monitor_module
from model_gateway.rate_limit_monitor.manager import (
    OWNERSHIP_RETRY_SECONDS,
    POLL_INTERVAL_SECONDS,
    _SourceProbe,
)
from tests.unit.model_gateway.rate_limit_monitor.manager._support import (
    MODEL,
    ANTHROPIC_MODEL,
    GENERATION_A,
    GENERATION_B,
    FakeStore,
    _control_clock,
    ControlledSleep,
    FakeProbe,
    _rate_limit,
    _monitor,
    _stub_sources,
    _wait_until,
    _isolate_monitor_metrics as _isolate_monitor_metrics,
)


async def test_contended_worker_retries_after_five_seconds_and_acquires(
    monkeypatch,
):
    store = FakeStore()
    store.leases = {}
    monitor = _monitor(monkeypatch, store)
    ownership = Mock()
    monkeypatch.setattr(
        monitor_module,
        "record_rate_limit_monitor_ownership",
        ownership,
    )
    sleep = ControlledSleep()
    monitor._sleep = sleep
    probe = FakeProbe([_rate_limit()])
    probe.block = True
    _stub_sources(monkeypatch, monitor, _SourceProbe("default", "openai", probe))

    task = asyncio.create_task(monitor._model_worker(MODEL))
    monitor._model_tasks[MODEL] = task
    await _wait_until(lambda: len(store.claim_calls) == 1)
    seconds, retry = await sleep.next()

    assert seconds == OWNERSHIP_RETRY_SECONDS == 5.0
    assert monitor._model_tasks[MODEL] is task
    assert not task.done()

    store.leases[MODEL] = ("default",)
    retry.set_result(None)
    await probe.called.wait()

    assert len(store.claim_calls) == 2
    assert ownership.call_args_list == [call("contended"), call("acquired")]

    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert len(store.release_calls) == 1


async def test_ownership_loss_is_recorded_when_renewal_is_lost(monkeypatch):
    store = FakeStore()
    store.renew_results = [None]
    monitor = _monitor(monkeypatch, store)
    ownership = Mock()
    monkeypatch.setattr(
        monitor_module,
        "record_rate_limit_monitor_ownership",
        ownership,
    )

    assert await monitor._renew_owner(MODEL, "owner") is None

    ownership.assert_called_once_with("lost")


async def test_probe_is_immediate_non_overlapping_and_uses_fixed_targets(
    monkeypatch,
):
    store = FakeStore()
    monitor = _monitor(monkeypatch, store)
    clock = _control_clock(monitor)
    probe = FakeProbe([_rate_limit(), _rate_limit()])
    probe.block = True
    poll_metric = Mock()
    monkeypatch.setattr(
        monitor_module,
        "record_rate_limit_monitor_poll",
        poll_metric,
    )
    perf_counter_values = iter([2.0, 2.012, 3.0, 3.004])
    monkeypatch.setattr(
        monitor_module.time,
        "perf_counter",
        lambda: next(perf_counter_values),
    )

    task = asyncio.create_task(
        monitor._source_loop(MODEL, "owner", _SourceProbe("default", "openai", probe))
    )
    await probe.called.wait()
    assert probe.calls == 1
    assert clock.sleeps.empty()

    clock.advance(3.0)
    probe.release.set()
    await _wait_until(lambda: len(store.publications) == 1)
    seconds, future = await clock.next_sleep()

    assert seconds == 2.0
    assert probe.calls == 1
    assert probe.max_active_calls == 1
    assert store.publications[0][3] == 1_000.0
    assert store.publications[0][4].status == "ok"
    assert poll_metric.call_args_list[0].kwargs == {
        "provider": "openai",
        "source": "default",
        "outcome": "ok",
        "latency_ms": pytest.approx(12.0),
    }

    future.set_result(None)
    await _wait_until(lambda: len(store.publications) == 2)
    assert clock.monotonic() == 5.0
    assert probe.calls == 2
    assert probe.max_active_calls == 1
    assert [publication[3] for publication in store.publications] == [
        1_000.0,
        1_001.0,
    ]
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def test_rejected_generation_refreshes_at_next_scheduled_poll(monkeypatch):
    store = FakeStore()
    store.publish_results = [False, True]
    monitor = _monitor(monkeypatch, store)
    clock = _control_clock(monitor)
    probe = FakeProbe([_rate_limit(), _rate_limit()])
    publish_metric = Mock()
    monkeypatch.setattr(
        monitor_module,
        "record_rate_limit_monitor_publish",
        publish_metric,
    )

    async def publish_source(model, token, generation, attempted_at, source):
        store.publications.append((model, token, generation, attempted_at, source))
        if len(store.publications) == 1:
            store.generations[MODEL] = GENERATION_B
            return False
        return True

    store.publish_source = publish_source  # type: ignore[method-assign]
    task = asyncio.create_task(
        monitor._source_loop(MODEL, "owner", _SourceProbe("default", "openai", probe))
    )

    await _wait_until(lambda: len(store.publications) == 1)
    seconds, future = await clock.next_sleep()

    assert seconds == POLL_INTERVAL_SECONDS
    assert probe.calls == 1
    future.set_result(None)
    await _wait_until(lambda: len(store.publications) == 2)
    assert [publication[2] for publication in store.publications] == [
        GENERATION_A,
        GENERATION_B,
    ]
    assert probe.calls == 2
    assert publish_metric.call_args_list == [call("rejected"), call("accepted")]
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def test_cancel_during_provider_probe_stops_probe(monkeypatch):
    store = FakeStore()
    monitor = _monitor(monkeypatch, store)
    probe = FakeProbe([_rate_limit()])
    probe.block = True

    task = asyncio.create_task(
        monitor._source_loop(MODEL, "owner", _SourceProbe("default", "openai", probe))
    )
    await probe.called.wait()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert probe.finished.is_set()


async def test_dual_sources_run_independently_and_release_after_children_stop(
    monkeypatch,
):
    store = FakeStore()
    store.active = {ANTHROPIC_MODEL}
    monitor = _monitor(monkeypatch, store, {ANTHROPIC_MODEL: "anthropic"})

    async def block_sleep(seconds: float) -> None:
        await asyncio.Event().wait()

    monitor._sleep = block_sleep
    pool_1 = FakeProbe([_rate_limit()])
    pool_2 = FakeProbe([_rate_limit()])
    pool_1.block = True
    pool_2.block = True
    _stub_sources(
        monkeypatch,
        monitor,
        _SourceProbe("pool_1", "openai", pool_1),
        _SourceProbe("pool_2", "anthropic", pool_2),
    )

    def assert_children_stopped() -> None:
        assert pool_1.finished.is_set()
        assert pool_2.finished.is_set()

    store.release_check = assert_children_stopped

    task = asyncio.create_task(monitor._model_worker(ANTHROPIC_MODEL))
    monitor._model_tasks[ANTHROPIC_MODEL] = task
    await asyncio.gather(pool_1.called.wait(), pool_2.called.wait())

    assert len(store.claim_calls) == 1
    assert pool_1.active_calls == 1
    assert pool_2.active_calls == 1
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert len(store.release_calls) == 1


async def test_heartbeat_lease_loss_stops_probe_and_allows_reactivation(monkeypatch):
    store = FakeStore()
    store.renew_results = [(GENERATION_A, 1_000.0), None]
    monitor = _monitor(monkeypatch, store)
    ownership = Mock()
    monkeypatch.setattr(
        monitor_module,
        "record_rate_limit_monitor_ownership",
        ownership,
    )
    sleep = ControlledSleep()
    monitor._sleep = sleep
    old_probe = FakeProbe([_rate_limit()])
    old_probe.block = True
    _stub_sources(monkeypatch, monitor, _SourceProbe("default", "openai", old_probe))

    old_task = asyncio.create_task(monitor._model_worker(MODEL))
    monitor._model_tasks[MODEL] = old_task
    await old_probe.called.wait()
    seconds, heartbeat_sleep = await sleep.next()
    assert seconds == 10.0
    heartbeat_sleep.set_result(None)
    await old_task

    assert old_probe.finished.is_set()
    assert len(store.release_calls) == 1
    assert ownership.call_args_list == [call("acquired"), call("lost")]

    store.active.add(MODEL)
    store.leases[MODEL] = ("default",)
    store.generations[MODEL] = GENERATION_B
    store.renew_results = [(GENERATION_B, 1_001.0)]
    new_probe = FakeProbe([_rate_limit()])
    new_probe.block = True
    _stub_sources(monkeypatch, monitor, _SourceProbe("default", "openai", new_probe))
    new_task = asyncio.create_task(monitor._model_worker(MODEL))
    monitor._model_tasks[MODEL] = new_task
    await new_probe.called.wait()

    assert len(store.claim_calls) == 2
    assert monitor._model_tasks[MODEL] is new_task
    new_task.cancel()
    await asyncio.gather(new_task, return_exceptions=True)


async def test_cancel_model_tasks_removes_task_cancelled_before_start(monkeypatch):
    store = FakeStore()
    monitor = _monitor(monkeypatch, store)
    task = asyncio.create_task(monitor._model_worker(MODEL))
    monitor._model_tasks[MODEL] = task

    await monitor._cancel_model_tasks()

    assert monitor._model_tasks == {}
    assert store.claim_calls == []
