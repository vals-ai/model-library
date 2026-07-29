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
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        return await client.get("/health/ready")


def test_combined_role_remains_the_direct_app_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GATEWAY_RUNTIME_ROLE", raising=False)
    with patch.object(gateway_app, "model_library_settings", ServerSettings()):
        app = gateway_app.create_app()

    paths = _route_paths(app)
    assert app.state.runtime_role == "combined"
    assert "/query" in paths
    assert "/benchmark-runs/acquire" in paths
    assert "/docs" in paths


def test_query_role_excludes_benchmark_admission(monkeypatch: pytest.MonkeyPatch):
    app = _create_app(monkeypatch, "query")

    paths = _route_paths(app)
    assert "/query" in paths
    assert "/token-retry/status" in paths
    assert not any(path.startswith("/benchmark-runs/") for path in paths)


def test_control_role_exposes_only_health_and_benchmark_admission(
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
    }
    assert app.state.startup_canary == {
        "enabled": False,
        "status": "disabled",
        "error": "",
    }
    assert not app.state.usage_ledger.enabled


async def test_control_readiness_requires_redis(monkeypatch: pytest.MonkeyPatch):
    app = _create_app(monkeypatch, "control")
    monkeypatch.setattr(token_utils, "redis_client", None)

    response = await _get_readiness(app)

    assert response.status_code == 503
    assert response.json() == {"status": "redis unavailable"}


async def test_control_readiness_pings_redis(monkeypatch: pytest.MonkeyPatch):
    app = _create_app(monkeypatch, "control")
    redis_client = MagicMock()
    redis_client.ping = AsyncMock(return_value=True)
    monkeypatch.setattr(token_utils, "redis_client", redis_client)

    response = await _get_readiness(app)

    assert response.status_code == 200
    redis_client.ping.assert_awaited_once_with()


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
