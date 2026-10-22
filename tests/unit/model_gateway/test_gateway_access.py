"""The `access` AppConfig profile decides which Gateway keys may call /admin."""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

import model_gateway.app as gateway_app
import model_gateway.runtime_config as runtime_config_module
from tests.unit.model_gateway.test_app_roles import ServerSettings, _quiet_lifecycle
from tests.unit.model_gateway.test_key_pools_operator_path import (
    API_KEYS,
    HARNESS,
    OPERATOR,
    access_document,
)

REFRESH_SECONDS = 0.05
MODELS = json.dumps(
    {
        "schema_version": 1,
        "revision": 1,
        "originated_at": "2026-10-19T00:00:00+00:00",
        "documents": {
            "dummy_model.yaml": Path("tests/unit/model_gateway/dummy_model.yaml").read_text()
        },
    }
).encode()
POOLS = json.dumps({"pools": {}, "model_pools": [], "weights": {}, "revision": 1})


class Agent:
    """What the task's AppConfig Agent serves; `access=None` answers `missing`."""

    def __init__(self, access: bytes | None = None, missing: int = 404) -> None:
        self.access = access
        self.missing = missing

    def handle(self, request: httpx.Request) -> httpx.Response:
        content = {
            "models-profile": MODELS,
            "credentials-profile": b"{}",
            "pools-profile": POOLS.encode(),
            "access-profile": self.access,
        }[request.url.path.rsplit("/", 1)[-1]]
        if content is None:
            return httpx.Response(self.missing)
        return httpx.Response(
            200, content=content, headers={"Configuration-Version": "1"}
        )


@asynccontextmanager
async def _control_task(
    monkeypatch: pytest.MonkeyPatch, agent: Agent, api_keys: dict[str, str] = API_KEYS
) -> AsyncIterator[AsyncClient]:
    for name, value in {
        "GATEWAY_RUNTIME_ROLE": "control",
        "REDIS_URL": "redis://control.test",
        "GATEWAY_APPCONFIG_APPLICATION": "app-1",
        "GATEWAY_APPCONFIG_ENVIRONMENT": "env-1",
        "GATEWAY_APPCONFIG_MODEL_PROFILE": "models-profile",
        "GATEWAY_APPCONFIG_CREDENTIAL_PROFILE": "credentials-profile",
        "GATEWAY_APPCONFIG_POOL_PROFILE": "pools-profile",
        "GATEWAY_APPCONFIG_ACCESS_PROFILE": "access-profile",
        "GATEWAY_APPCONFIG_DEPLOYMENT_STRATEGY": "strategy-1",
        "GATEWAY_APPCONFIG_ACCESS_TOKEN": "local-token",
        "GATEWAY_RUNTIME_REFRESH_SECONDS": str(REFRESH_SECONDS),
    }.items():
        monkeypatch.setenv(name, value)
    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(
        runtime_config_module.httpx,
        "AsyncClient",
        lambda: real_async_client(transport=httpx.MockTransport(agent.handle)),
    )
    settings = ServerSettings()
    settings.MODEL_GATEWAY_API_KEYS = json.dumps(api_keys)
    with patch.object(gateway_app, "model_library_settings", settings):
        app = gateway_app.create_app()
    redis_client = MagicMock(aclose=AsyncMock())
    monitor = MagicMock(close=AsyncMock())
    with (
        _quiet_lifecycle(),
        patch.object(gateway_app.async_redis, "from_url", return_value=redis_client),
        patch.object(gateway_app, "set_redis_client"),
        patch.object(gateway_app, "RateLimitMonitor", return_value=monitor),
    ):
        async with app.router.lifespan_context(app):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                yield client


async def _status(client: AsyncClient, headers: dict[str, str]) -> int:
    return (await client.get("/admin/key-pools", headers=headers)).status_code


async def _wait_for(client: AsyncClient, headers: dict[str, str], status: int) -> None:
    """Wait until /admin/key-pools answers `status`, by the background refresh alone."""
    loop = asyncio.get_running_loop()
    started = loop.time()
    while loop.time() - started < 40 * REFRESH_SECONDS:
        if await _status(client, headers) == status:
            return
        await asyncio.sleep(REFRESH_SECONDS / 5)
    raise AssertionError(f"/admin/key-pools never answered {status}")


@pytest.mark.parametrize("missing", [404, 503])
async def test_a_control_task_without_an_access_document_starts_and_refuses_admin(
    monkeypatch: pytest.MonkeyPatch, missing: int
) -> None:
    async with _control_task(monkeypatch, Agent(missing=missing)) as client:
        response = await client.get("/admin/key-pools", headers=OPERATOR)

    assert response.status_code == 403
    assert response.json()["code"] == "forbidden"


async def test_the_background_refresh_grants_and_revokes_admin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = Agent()

    async with _control_task(monkeypatch, agent) as client:
        assert await _status(client, OPERATOR) == 403

        agent.access = access_document(["dashboards"])
        await _wait_for(client, OPERATOR, 200)
        assert await _status(client, HARNESS) == 403

        agent.access = access_document([])
        await _wait_for(client, OPERATOR, 403)


@pytest.mark.parametrize("missing", [404, 503])
async def test_admin_closes_when_the_access_document_goes_missing(
    monkeypatch: pytest.MonkeyPatch, missing: int
) -> None:
    agent = Agent(access_document(["dashboards"]), missing=missing)

    async with _control_task(monkeypatch, agent) as client:
        assert await _status(client, OPERATOR) == 200

        agent.access = None
        await _wait_for(client, OPERATOR, 403)


async def test_only_names_that_are_gateway_keys_with_their_own_value_grant_admin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_keys = {"dashboards": "sk-shared", "ops": "sk-shared", "harness": "sk-harness"}
    agent = Agent(access_document(["ops", "sk-harness", "nobody", "harness"]))

    async with _control_task(monkeypatch, agent, api_keys) as client:
        shared = await _status(client, {"Authorization": "Bearer sk-shared"})
        harness = await _status(client, HARNESS)

    assert shared == 403
    assert harness == 200
