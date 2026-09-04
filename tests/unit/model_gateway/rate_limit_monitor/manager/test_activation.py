import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

import model_gateway.rate_limit_monitor.manager as monitor_module
from model_gateway.rate_limit_monitor.manager import (
    MonitorInvalidModel,
    _SourceProbe,
)
from tests.unit.model_gateway.rate_limit_monitor.manager._support import (
    MODEL,
    ANTHROPIC_MODEL,
    FakeStore,
    FakeProbe,
    _monitor,
    _isolate_monitor_metrics as _isolate_monitor_metrics,
)


async def test_unknown_model_is_rejected_without_store_or_provider_work(monkeypatch):
    store = FakeStore()
    monitor = _monitor(monkeypatch, store)
    provider_factory = AsyncMock()
    monkeypatch.setattr(monitor_module, "get_registry_model", provider_factory)

    with pytest.raises(MonitorInvalidModel):
        await monitor.activate("openai/not-registered")

    assert store.activate_calls == []
    provider_factory.assert_not_called()


async def test_unsupported_model_is_rejected_without_store_or_provider_work(
    monkeypatch,
):
    store = FakeStore()
    monitor = _monitor(monkeypatch, store, supported_models=set())
    provider_factory = AsyncMock()
    monkeypatch.setattr(monitor_module, "get_registry_model", provider_factory)

    with pytest.raises(MonitorInvalidModel):
        await monitor.activate(MODEL)

    assert store.activate_calls == []
    provider_factory.assert_not_called()


async def test_activation_accepts_same_and_cross_provider_alternative_keys(
    monkeypatch,
):
    same_provider_alias = "openai/gpt-4o-alias"
    cross_provider_key = "azure/gpt-4o"
    store = FakeStore()
    monitor = _monitor(
        monkeypatch,
        store,
        {
            MODEL: "openai",
            same_provider_alias: "openai",
            cross_provider_key: "azure",
        },
        alternative_keys={MODEL: [same_provider_alias, cross_provider_key]},
    )

    try:
        same_provider = await monitor.activate(same_provider_alias)
        cross_provider = await monitor.activate(cross_provider_key)

        assert same_provider.state.model == same_provider_alias
        assert cross_provider.state.model == cross_provider_key
        assert store.activate_calls == [
            (same_provider_alias, ("default",)),
            (cross_provider_key, ("default",)),
        ]
    finally:
        await monitor.close()


async def test_activate_and_list_do_not_probe_inline(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "pool-one-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY_2", "pool-two-secret")
    store = FakeStore()
    monitor = _monitor(
        monkeypatch,
        store,
        {MODEL: "openai", ANTHROPIC_MODEL: "anthropic"},
    )
    provider_factory = AsyncMock()
    monkeypatch.setattr(monitor_module, "get_registry_model", provider_factory)

    try:
        activation = await monitor.activate(ANTHROPIC_MODEL)
        listing = await monitor.list_states()

        expected_sources = ("default",)
        assert store.activate_calls == [(ANTHROPIC_MODEL, expected_sources)]
        assert activation.state.status == "starting"
        assert listing.states == []
        provider_factory.assert_not_called()
    finally:
        await monitor.close()


async def test_close_during_activation_does_not_start_a_worker(monkeypatch):
    store = FakeStore()
    monitor = _monitor(monkeypatch, store)
    activation_started = asyncio.Event()
    activation_finished = asyncio.Event()
    original_activate = store.activate

    async def activate(model, source_names):
        activation_started.set()
        await activation_finished.wait()
        return await original_activate(model, source_names)

    store.activate = activate  # type: ignore[method-assign]
    model_worker = AsyncMock()
    monkeypatch.setattr(monitor, "_model_worker", model_worker)

    activation_task = asyncio.create_task(monitor.activate(MODEL))
    await activation_started.wait()
    await monitor.close()
    activation_finished.set()
    await activation_task

    assert monitor._model_tasks == {}
    model_worker.assert_not_awaited()


def test_standard_factory_constructs_one_default_direct_probe(monkeypatch):
    store = FakeStore()
    monitor = _monitor(monkeypatch, store)
    probe = FakeProbe([None])
    factory = Mock(return_value=probe)
    monkeypatch.setattr(monitor_module, "get_registry_model", factory)

    sources = monitor._build_sources(MODEL, ("default",))

    assert sources == [_SourceProbe("default", "openai", probe)]
    factory.assert_called_once_with(MODEL)
