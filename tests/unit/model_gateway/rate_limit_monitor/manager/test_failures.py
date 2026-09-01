import asyncio
import logging
from unittest.mock import Mock

import httpx
import pytest
from redis.exceptions import RedisError

import model_gateway.rate_limit_monitor.manager as monitor_module
from model_gateway.rate_limit_monitor.manager import (
    POLL_INTERVAL_SECONDS,
    RateLimitMonitor,
    _SourceProbe,
)
from model_gateway.rate_limit_monitor.state import (
    MonitorSourceUpdate,
    MonitorStateCorrupt,
)

from tests.unit.model_gateway.rate_limit_monitor.manager._support import (
    MODEL,
    FakeStore,
    _control_clock,
    FakeProbe,
    _rate_limit,
    _monitor,
    _stub_sources,
    _wait_until,
    _isolate_monitor_metrics as _isolate_monitor_metrics,
)


@pytest.mark.parametrize(
    "error",
    [RedisError("redis unavailable"), MonitorStateCorrupt("corrupt state")],
    ids=["redis-error", "corrupt-state"],
)
async def test_fatal_state_errors_propagate_without_release(monkeypatch, error):
    store = FakeStore()
    store.renew_error = error
    monitor = _monitor(monkeypatch, store)
    probe = FakeProbe([_rate_limit()])
    _stub_sources(monkeypatch, monitor, _SourceProbe("default", "openai", probe))

    with pytest.raises(type(error), match=str(error)):
        await monitor._model_worker(MODEL)

    assert store.release_calls == []


async def test_construction_error_propagates_without_release(monkeypatch):
    store = FakeStore()
    monitor = _monitor(monkeypatch, store)
    constructor_error = RuntimeError("invalid provider configuration")
    monkeypatch.setattr(monitor, "_build_sources", Mock(side_effect=constructor_error))

    with pytest.raises(RuntimeError, match="invalid provider configuration"):
        await monitor._model_worker(MODEL)

    assert store.release_calls == []
    assert store.publications == []


async def test_provider_constructor_error_attempts_once_without_publication(
    monkeypatch,
):
    store = FakeStore()
    monitor = _monitor(monkeypatch, store)
    factory = Mock(side_effect=RuntimeError("constructor failed"))
    monkeypatch.setattr(monitor_module, "get_registry_model", factory)

    with pytest.raises(RuntimeError, match="constructor failed"):
        await monitor._model_worker(MODEL)

    factory.assert_called_once_with(MODEL)
    assert store.publications == []
    assert store.release_calls == []


async def test_provider_failure_publishes_generic_error_then_poll_succeeds(
    monkeypatch,
):
    store = FakeStore()
    monitor = _monitor(monkeypatch, store)
    clock = _control_clock(monitor)
    probe = FakeProbe([httpx.HTTPError("secret-marker"), _rate_limit()])
    task = asyncio.create_task(
        monitor._source_loop(MODEL, "owner", _SourceProbe("default", "openai", probe))
    )

    await _wait_until(lambda: len(store.publications) == 1)
    seconds, future = await clock.next_sleep()
    first = store.publications[0][4]
    assert first == MonitorSourceUpdate(source="default", status="error")
    assert seconds == POLL_INTERVAL_SECONDS

    future.set_result(None)
    await _wait_until(lambda: len(store.publications) == 2)
    assert store.publications[1][4].status == "ok"
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def test_programmer_error_terminates_source_loop(monkeypatch):
    store = FakeStore()
    monitor = _monitor(monkeypatch, store)
    probe = FakeProbe([AssertionError("programmer defect")])

    with pytest.raises(AssertionError, match="programmer defect"):
        await monitor._source_loop(
            MODEL,
            "owner",
            _SourceProbe("default", "openai", probe),
        )

    assert store.publications == []


async def test_unsupported_probe_result_is_published(monkeypatch):
    store = FakeStore()
    monitor = _monitor(monkeypatch, store)
    clock = _control_clock(monitor)
    probe = FakeProbe([None])
    task = asyncio.create_task(
        monitor._source_loop(MODEL, "owner", _SourceProbe("default", "openai", probe))
    )

    await _wait_until(lambda: len(store.publications) == 1)
    source = store.publications[0][4]
    assert source == MonitorSourceUpdate(source="default", status="unsupported")
    await clock.next_sleep()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


def test_provider_failure_logging_redacts_exception_message(caplog):
    secret = "secret-marker"

    with caplog.at_level(logging.WARNING):
        RateLimitMonitor._log_provider_failure(
            MODEL,
            "default",
            RuntimeError(secret),
        )

    assert secret not in caplog.text
    assert "RuntimeError" in caplog.text
    assert "provider_error" in caplog.text
