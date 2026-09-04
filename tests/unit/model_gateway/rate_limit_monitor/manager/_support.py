import asyncio
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import model_gateway.rate_limit_monitor.manager as monitor_module
from model_gateway.rate_limit_monitor.manager import RateLimitMonitor, _SourceProbe
from model_gateway.rate_limit_monitor.state import MonitorSourceUpdate
from model_gateway.rate_limit_monitor.types import (
    MonitorActivationResponse,
    MonitorListResponse,
    MonitorSourceName,
    MonitorSourceState,
    MonitorState,
)
from model_library.rate_limits import (
    RateLimit,
    RateLimitCapacity,
    RequestRateLimit,
    TokenRateLimit,
)

MODEL = "openai/gpt-4o"

ANTHROPIC_MODEL = "anthropic/claude-sonnet-4-5"

GENERATION_A = "a" * 32

GENERATION_B = "b" * 32


@pytest.fixture(autouse=True)
def _isolate_monitor_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "record_rate_limit_monitor_ownership",
        "record_rate_limit_monitor_poll",
        "record_rate_limit_monitor_publish",
    ):
        monkeypatch.setattr(monitor_module, name, Mock())


class FakeStore:
    def __init__(self):
        self.active: set[str] = {MODEL}
        self.leases: dict[str, tuple[MonitorSourceName, ...]] = {
            MODEL: ("default",),
            ANTHROPIC_MODEL: ("pool_1", "pool_2"),
        }
        self.generations = {
            MODEL: GENERATION_A,
            ANTHROPIC_MODEL: GENERATION_A,
        }
        self.activate_calls: list[tuple[str, tuple[MonitorSourceName, ...]]] = []
        self.claim_calls: list[tuple[str, str]] = []
        self.renew_calls: list[tuple[str, str]] = []
        self.release_calls: list[tuple[str, str]] = []
        self.publications: list[tuple[str, str, str, float, MonitorSourceUpdate]] = []
        self.publish_results: list[bool] = []
        self.renew_error: Exception | None = None
        self.renew_results: list[tuple[str, float] | None] = []
        self.renewed_at = 1_000.0
        self.discover_error: Exception | None = None
        self.discover_calls = 0
        self.renew_check: Callable[[], None] | None = None
        self.release_check: Callable[[], None] | None = None

    async def activate(
        self,
        model: str,
        expected_sources: tuple[MonitorSourceName, ...],
    ) -> MonitorActivationResponse:
        self.activate_calls.append((model, expected_sources))
        sources = [
            MonitorSourceState(source=source, status="starting")
            for source in expected_sources
        ]
        return MonitorActivationResponse(
            server_time=1.0,
            state=MonitorState(
                model=model,
                active=True,
                active_until=1_801.0,
                retention_until=88_201.0,
                status="starting",
                sources=sources,
            ),
        )

    async def list_states(self) -> MonitorListResponse:
        return MonitorListResponse(
            server_time=1.0,
            states=[],
        )

    async def discover_active(self) -> set[str]:
        self.discover_calls += 1
        if self.discover_error is not None:
            raise self.discover_error
        return set(self.active)

    async def claim_owner(
        self,
        model: str,
        token: str,
    ) -> tuple[MonitorSourceName, ...] | None:
        self.claim_calls.append((model, token))
        return self.leases.get(model)

    async def renew_owner(
        self,
        model: str,
        token: str,
    ) -> tuple[str, float] | None:
        if self.renew_check is not None:
            self.renew_check()
        self.renew_calls.append((model, token))
        if self.renew_error is not None:
            raise self.renew_error
        if self.renew_results:
            return self.renew_results.pop(0)
        if model not in self.leases:
            return None
        attempted_at = self.renewed_at
        self.renewed_at += 1.0
        return self.generations[model], attempted_at

    async def release_owner(self, model: str, token: str) -> None:
        if self.release_check is not None:
            self.release_check()
        self.release_calls.append((model, token))

    async def publish_source(
        self,
        model: str,
        token: str,
        captured_generation: str,
        attempted_at: float,
        source: MonitorSourceUpdate,
    ) -> bool:
        self.publications.append(
            (model, token, captured_generation, attempted_at, source)
        )
        if self.publish_results:
            return self.publish_results.pop(0)
        return True


class ControlledClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps: asyncio.Queue[tuple[float, asyncio.Future[None]]] = asyncio.Queue()

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

    async def sleep(self, seconds: float) -> None:
        future = asyncio.get_running_loop().create_future()
        await self.sleeps.put((seconds, future))
        await future
        self.now += seconds

    async def next_sleep(self) -> tuple[float, asyncio.Future[None]]:
        return await self.sleeps.get()


def _control_clock(monitor: RateLimitMonitor) -> ControlledClock:
    clock = ControlledClock()
    monitor._sleep = clock.sleep
    monitor._monotonic = clock.monotonic
    return clock


class ControlledSleep:
    def __init__(self):
        self.calls: asyncio.Queue[tuple[float, asyncio.Future[None]]] = asyncio.Queue()

    async def __call__(self, seconds: float) -> None:
        future = asyncio.get_running_loop().create_future()
        await self.calls.put((seconds, future))
        await future

    async def next(self) -> tuple[float, asyncio.Future[None]]:
        return await self.calls.get()


class FakeProbe:
    def __init__(
        self,
        responses: list[RateLimit | None | BaseException],
        *,
        provider: str = "openai",
    ):
        self.responses = responses
        self.provider = provider
        self.calls = 0
        self.active_calls = 0
        self.max_active_calls = 0
        self.called = asyncio.Event()
        self.release = asyncio.Event()
        self.block = False
        self.finished = asyncio.Event()

    async def get_rate_limit(self) -> RateLimit | None:
        self.calls += 1
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        self.called.set()
        try:
            if self.block:
                await self.release.wait()
            value = self.responses.pop(0)
            if isinstance(value, BaseException):
                raise value
            return value
        finally:
            self.active_calls -= 1
            self.finished.set()


def _rate_limit() -> RateLimit:
    return RateLimit(
        requests=(RequestRateLimit(limit=10_000, remaining=9_000),),
        tokens=TokenRateLimit(
            input=RateLimitCapacity(limit=600_000, remaining=550_000),
            output=RateLimitCapacity(limit=400_000, remaining=350_000),
        ),
        unix_timestamp=50.0,
    )


def _monitor(
    monkeypatch,
    store: FakeStore,
    providers: dict[str, str] | None = None,
    supported_models: set[str] | None = None,
    alternative_keys: dict[str, list[str | dict[str, object]]] | None = None,
) -> RateLimitMonitor:
    providers = providers or {MODEL: "openai"}
    if supported_models is None:
        supported_models = set(providers)
    alternative_keys = alternative_keys or {}
    registry = {
        model: SimpleNamespace(
            provider_name=provider,
            rate_limit=SimpleNamespace(
                supports_live_monitoring=model in supported_models,
            ),
            alternative_keys=alternative_keys.get(model, []),
        )
        for model, provider in providers.items()
    }
    monkeypatch.setattr(monitor_module, "get_registry_config", registry.get)
    return RateLimitMonitor(store)


def _stub_sources(
    monkeypatch: pytest.MonkeyPatch,
    monitor: RateLimitMonitor,
    *sources: _SourceProbe,
) -> None:
    specs = tuple(
        SimpleNamespace(source=source.source, key_setting=None) for source in sources
    )
    monkeypatch.setattr(monitor_module, "_source_specs", lambda model: specs)
    monkeypatch.setattr(
        monitor, "_build_sources", lambda model, expected: list(sources)
    )


async def _wait_until(predicate: Callable[[], bool]) -> None:
    for _ in range(100):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition did not become true")
