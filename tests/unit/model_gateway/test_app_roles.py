import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import ASGITransport, AsyncClient
import pytest

import model_gateway.app as gateway_app
from model_library.retriers.token import utils as token_utils


class ServerSettings:
    MODEL_GATEWAY_API_KEYS = '{"test":"sk-test"}'
    MODEL_GATEWAY_HMAC_SECRET = "test-secret"

    def get(self, name: str, default: str = "") -> str:
        return getattr(self, name, default)

    def unset(self, _key: str) -> None:
        pass


def _create_app(monkeypatch: pytest.MonkeyPatch, role: str):
    monkeypatch.setenv("GATEWAY_RUNTIME_ROLE", role)
    monkeypatch.delenv("GATEWAY_STARTUP_CANARY_ENABLED", raising=False)
    with patch.object(gateway_app, "model_library_settings", ServerSettings()):
        return gateway_app.create_app()


def _route_paths(app) -> set[str]:
    return {route.path for route in app.routes}


async def _get_readiness(app):
    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        return await client.get("/health/ready")


async def _wait_for_metrics_stop(
    stop: asyncio.Event,
    *,
    publishers: object,
) -> None:
    del publishers
    await stop.wait()


@contextmanager
def _quiet_lifecycle() -> Iterator[None]:
    with (
        patch.object(
            gateway_app,
            "publish_metrics_periodically",
            _wait_for_metrics_stop,
        ),
        patch.object(gateway_app, "install_loop_exception_handler", return_value=None),
        patch.object(gateway_app, "log_process_lifecycle"),
        patch.object(gateway_app.telemetry, "configure_telemetry"),
        patch.object(gateway_app.telemetry, "shutdown_telemetry"),
        patch.object(
            gateway_app,
            "install_telemetry_delivery_metric_handler",
            return_value=MagicMock(),
        ),
        patch.object(gateway_app, "remove_telemetry_delivery_metric_handler"),
    ):
        yield


def test_combined_role_remains_the_direct_app_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GATEWAY_RUNTIME_ROLE", raising=False)
    with patch.object(gateway_app, "model_library_settings", ServerSettings()):
        app = gateway_app.create_app()

    paths = _route_paths(app)
    assert app.state.runtime_role == "combined"
    assert "/query" in paths
    assert "/benchmark-runs/acquire" in paths
    assert "/rate-limit-monitor" in paths
    assert "/rate-limit-monitor/activate" in paths
    assert "/docs" in paths
    assert app.state.rate_limit_monitor is None


def test_query_role_excludes_benchmark_admission(monkeypatch: pytest.MonkeyPatch):
    app = _create_app(monkeypatch, "query")

    paths = _route_paths(app)
    assert "/query" in paths
    assert "/token-retry/status" in paths
    assert not any(path.startswith("/benchmark-runs/") for path in paths)
    assert not any(path.startswith("/rate-limit-monitor") for path in paths)
    assert app.state.rate_limit_monitor is None


def test_control_role_exposes_only_health_and_control_operations(
    monkeypatch: pytest.MonkeyPatch,
):
    app = _create_app(monkeypatch, "control")

    assert _route_paths(app) == {
        "/health/live",
        "/health/ready",
        "/benchmark-runs/acquire",
        "/benchmark-runs/wait",
        "/benchmark-runs/renew",
        "/benchmark-runs/release",
        "/rate-limit-monitor",
        "/rate-limit-monitor/activate",
    }
    assert app.state.startup_canary == {
        "enabled": False,
        "status": "disabled",
        "error": "",
    }
    assert not app.state.usage_ledger.enabled
    assert app.state.rate_limit_monitor is None


async def test_control_monitor_validation_uses_canonical_invalid_request(
    monkeypatch: pytest.MonkeyPatch,
):
    app = _create_app(monkeypatch, "control")
    monitor = MagicMock()
    monitor.activate = AsyncMock()
    app.state.rate_limit_monitor = monitor
    headers = {"Authorization": "Bearer sk-test"}

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        malformed = await client.post(
            "/rate-limit-monitor/activate",
            headers={**headers, "Content-Type": "application/json"},
            content=b"{",
        )
        wrong_shape = await client.post(
            "/rate-limit-monitor/activate",
            headers=headers,
            json={},
        )

    for response in (malformed, wrong_shape):
        assert response.status_code == 400
        assert response.json()["code"] == "invalid_request"
    monitor.activate.assert_not_awaited()


@pytest.mark.parametrize("role", ["control", "combined"])
async def test_control_enabled_lifespan_starts_one_monitor_and_closes_before_redis(
    monkeypatch: pytest.MonkeyPatch,
    role: str,
):
    monkeypatch.setenv("REDIS_URL", "redis://monitor.test")
    with patch.object(
        gateway_app,
        "create_usage_ledger_from_env",
        return_value=gateway_app.NoopUsageLedger(),
    ):
        app = _create_app(monkeypatch, role)
    events: list[str] = []
    redis_client = MagicMock()

    async def close_redis() -> None:
        events.append("redis")

    redis_client.aclose = AsyncMock(side_effect=close_redis)
    store = object()
    monitor = MagicMock(spec=["start", "close"])

    async def close_monitor() -> None:
        events.append("monitor")

    monitor.close = AsyncMock(side_effect=close_monitor)

    with (
        _quiet_lifecycle(),
        patch.object(gateway_app.async_redis, "from_url", return_value=redis_client),
        patch.object(gateway_app, "set_redis_client") as set_redis_client,
        patch.object(
            gateway_app,
            "RateLimitMonitorStore",
            return_value=store,
        ) as store_factory,
        patch.object(
            gateway_app,
            "RateLimitMonitor",
            return_value=monitor,
        ) as monitor_factory,
    ):
        async with app.router.lifespan_context(app):
            assert app.state.rate_limit_monitor is monitor
            monitor.start.assert_called_once_with()

        assert app.state.rate_limit_monitor is None

    set_redis_client.assert_called_once_with(redis_client)
    store_factory.assert_called_once_with(redis_client)
    monitor_factory.assert_called_once_with(store)
    monitor.close.assert_awaited_once_with()
    redis_client.aclose.assert_awaited_once_with()
    assert events == ["monitor", "redis"]


@pytest.mark.parametrize("role", ["control", "combined"])
async def test_control_enabled_lifespan_requires_redis(
    monkeypatch: pytest.MonkeyPatch,
    role: str,
):
    monkeypatch.delenv("REDIS_URL", raising=False)
    app = _create_app(monkeypatch, role)

    with (
        _quiet_lifecycle(),
        patch.object(gateway_app, "RateLimitMonitor") as monitor_factory,
        pytest.raises(RuntimeError, match="REDIS_URL"),
    ):
        async with app.router.lifespan_context(app):
            pass

    monitor_factory.assert_not_called()
    assert app.state.rate_limit_monitor is None


async def test_query_lifespan_with_redis_does_not_create_monitor(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("REDIS_URL", "redis://query.test")
    with patch.object(
        gateway_app,
        "create_usage_ledger_from_env",
        return_value=gateway_app.NoopUsageLedger(),
    ):
        app = _create_app(monkeypatch, "query")
    redis_client = MagicMock()
    redis_client.aclose = AsyncMock()

    with (
        _quiet_lifecycle(),
        patch.object(gateway_app.async_redis, "from_url", return_value=redis_client),
        patch.object(gateway_app, "set_redis_client"),
        patch.object(gateway_app, "RateLimitMonitor") as monitor_factory,
    ):
        async with app.router.lifespan_context(app):
            assert app.state.rate_limit_monitor is None

    monitor_factory.assert_not_called()
    redis_client.aclose.assert_awaited_once_with()


async def test_query_lifespan_closes_rate_limit_service_before_telemetry(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("REDIS_URL", raising=False)
    events: list[str] = []
    rate_limit_probe_service = MagicMock(spec=["close"])

    async def close_rate_limit_probe_service() -> None:
        events.append("rate_limit_probe_service")

    rate_limit_probe_service.close = AsyncMock(
        side_effect=close_rate_limit_probe_service
    )
    with patch.object(
        gateway_app,
        "register_rate_limit_route",
        return_value=rate_limit_probe_service,
    ):
        app = _create_app(monkeypatch, "query")

    telemetry_shutdown = MagicMock(side_effect=lambda: events.append("telemetry"))
    with (
        _quiet_lifecycle(),
        patch.object(
            gateway_app.telemetry,
            "shutdown_telemetry",
            telemetry_shutdown,
        ),
    ):
        async with app.router.lifespan_context(app):
            pass

    rate_limit_probe_service.close.assert_awaited_once_with()
    telemetry_shutdown.assert_called_once_with()
    assert events == ["rate_limit_probe_service", "telemetry"]


@pytest.mark.parametrize("role", ["control", "combined"])
async def test_control_enabled_readiness_requires_redis(
    monkeypatch: pytest.MonkeyPatch,
    role: str,
):
    app = _create_app(monkeypatch, role)
    monitor = MagicMock(spec=["check_health"])
    app.state.rate_limit_monitor = monitor
    monkeypatch.setattr(token_utils, "redis_client", None)

    response = await _get_readiness(app)

    assert response.status_code == 503
    assert response.json() == {"status": "redis unavailable"}
    monitor.check_health.assert_not_called()


@pytest.mark.parametrize("role", ["control", "combined"])
async def test_control_enabled_readiness_checks_redis_and_monitor(
    monkeypatch: pytest.MonkeyPatch,
    role: str,
):
    app = _create_app(monkeypatch, role)
    monitor = MagicMock(spec=["check_health"])
    app.state.rate_limit_monitor = monitor
    redis_client = MagicMock()
    redis_client.ping = AsyncMock(return_value=True)
    monkeypatch.setattr(token_utils, "redis_client", redis_client)

    response = await _get_readiness(app)

    assert response.status_code == 200
    redis_client.ping.assert_awaited_once_with()
    monitor.check_health.assert_called_once_with()


@pytest.mark.parametrize("role", ["control", "combined"])
async def test_terminal_monitor_makes_control_enabled_readiness_unhealthy(
    monkeypatch: pytest.MonkeyPatch,
    role: str,
):
    app = _create_app(monkeypatch, role)
    monitor = MagicMock(spec=["check_health"])
    monitor.check_health.side_effect = RuntimeError("monitor failed")
    app.state.rate_limit_monitor = monitor
    redis_client = MagicMock()
    redis_client.ping = AsyncMock(return_value=True)
    monkeypatch.setattr(token_utils, "redis_client", redis_client)

    response = await _get_readiness(app)

    assert response.status_code == 500
    redis_client.ping.assert_awaited_once_with()
    monitor.check_health.assert_called_once_with()


async def test_query_readiness_does_not_depend_on_redis(
    monkeypatch: pytest.MonkeyPatch,
):
    app = _create_app(monkeypatch, "query")
    monkeypatch.setattr(token_utils, "redis_client", None)

    response = await _get_readiness(app)

    assert response.status_code == 200


def test_invalid_runtime_role_is_rejected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GATEWAY_RUNTIME_ROLE", "worker")
    with (
        patch.object(gateway_app, "model_library_settings", ServerSettings()),
        pytest.raises(ValueError, match="GATEWAY_RUNTIME_ROLE"),
    ):
        gateway_app.create_app()
