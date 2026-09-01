import asyncio
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from redis.exceptions import ConnectionError as RedisConnectionError
from starlette.testclient import TestClient

import model_gateway.rate_limit_monitor.routes as monitor_routes
from model_gateway.auth import create_auth_middleware
from model_gateway.rate_limit_monitor.manager import MonitorInvalidModel
from model_gateway.rate_limit_monitor.state import MonitorStateCorrupt
from model_gateway.rate_limit_monitor.types import (
    MonitorActivationResponse,
    MonitorListResponse,
    MonitorSourceState,
    MonitorState,
)
from model_library.rate_limits import RateLimit, RequestRateLimit

HEADERS = {"Authorization": "Bearer sk-test"}
MODEL = "openai/gpt-4o"
SERVER_TIME = 1_700_000_000.0


def _state(model: str = MODEL) -> MonitorState:
    return MonitorState(
        model=model,
        active=True,
        active_until=SERVER_TIME + 1_800,
        retention_until=SERVER_TIME + 88_200,
        status="starting",
        sources=[MonitorSourceState(source="default", status="starting")],
    )


class FakeMonitor:
    def __init__(self) -> None:
        self.activate_calls: list[str] = []
        self.list_calls = 0
        self.activate_error: BaseException | None = None
        self.list_error: BaseException | None = None
        self.block_activate = False
        self.block_list = False

    async def activate(self, model: str) -> MonitorActivationResponse:
        self.activate_calls.append(model)
        if self.activate_error is not None:
            raise self.activate_error
        if self.block_activate:
            await asyncio.Event().wait()
        return MonitorActivationResponse(
            server_time=SERVER_TIME,
            state=_state(model),
        )

    async def list_states(self) -> MonitorListResponse:
        self.list_calls += 1
        if self.list_error is not None:
            raise self.list_error
        if self.block_list:
            await asyncio.Event().wait()
        return MonitorListResponse(
            server_time=SERVER_TIME,
            states=[_state()],
        )


def _client(
    monitor: FakeMonitor,
    *,
    raise_server_exceptions: bool = True,
) -> TestClient:
    app = FastAPI()
    app.state.rate_limit_monitor = monitor
    monitor_routes.register_rate_limit_monitor_routes(app)
    app.middleware("http")(create_auth_middleware({"test": "sk-test"}))
    return TestClient(app, raise_server_exceptions=raise_server_exceptions)


def test_get_and_post_return_shared_state() -> None:
    monitor = FakeMonitor()
    client = _client(monitor)

    listed = client.get("/rate-limit-monitor", headers=HEADERS)
    activated = client.post(
        "/rate-limit-monitor/activate",
        headers=HEADERS,
        json={"model": f"  {MODEL}  "},
    )

    assert listed.status_code == 200
    assert listed.json() == MonitorListResponse(
        server_time=SERVER_TIME,
        states=[_state()],
    ).model_dump(mode="json")
    assert activated.status_code == 200
    assert activated.json() == MonitorActivationResponse(
        server_time=SERVER_TIME,
        state=_state(),
    ).model_dump(mode="json")
    assert monitor.list_calls == 1
    assert monitor.activate_calls == [MODEL]


def test_get_round_trips_fixed_rate_limit() -> None:
    monitor = FakeMonitor()
    rate_limit = RateLimit(
        requests=(RequestRateLimit(limit=25, remaining=20, mode="concurrency"),),
        scope="api_key",
        unix_timestamp=SERVER_TIME - 1,
    )
    monitor.list_states = AsyncMock(
        return_value=MonitorListResponse(
            server_time=SERVER_TIME,
            states=[
                MonitorState(
                    model=MODEL,
                    active=True,
                    active_until=SERVER_TIME + 1_800,
                    retention_until=SERVER_TIME + 88_200,
                    status="ok",
                    sources=[
                        MonitorSourceState(
                            source="default",
                            status="ok",
                            last_attempt_at=SERVER_TIME,
                            last_success_at=SERVER_TIME,
                            rate_limit=rate_limit,
                        )
                    ],
                )
            ],
        )
    )
    client = _client(monitor)

    response = client.get("/rate-limit-monitor", headers=HEADERS)

    assert response.status_code == 200
    payload = response.json()["states"][0]["sources"][0]["rate_limit"]
    assert payload == {
        "requests": [
            {
                "limit": 25,
                "remaining": 20,
                "mode": "concurrency",
            }
        ],
        "tokens": None,
        "scope": "api_key",
        "unix_timestamp": SERVER_TIME - 1,
    }


def test_existing_auth_rejects_requests_before_monitor_work() -> None:
    monitor = FakeMonitor()
    client = _client(monitor)

    response = client.get("/rate-limit-monitor")

    assert response.status_code == 401
    assert response.json()["code"] == "unauthorized"
    assert monitor.list_calls == 0
    assert monitor.activate_calls == []


def test_unknown_model_uses_fixed_error_without_exception_text() -> None:
    monitor = FakeMonitor()
    monitor.activate_error = MonitorInvalidModel("secret dynamic model detail")
    client = _client(monitor)

    response = client.post(
        "/rate-limit-monitor/activate",
        headers=HEADERS,
        json={"model": "unknown/model"},
    )

    assert response.status_code == 400
    assert response.json() == {
        "code": "invalid_model",
        "message": "Invalid model activation request",
    }
    assert "secret" not in response.text


@pytest.mark.parametrize(
    "error",
    [
        RedisConnectionError("secret redis endpoint"),
        MonitorStateCorrupt("secret persisted value"),
    ],
)
@pytest.mark.parametrize("operation", ["list", "activate"])
def test_state_failures_reach_application_error_boundary(
    error: BaseException,
    operation: str,
) -> None:
    monitor = FakeMonitor()
    client = _client(monitor, raise_server_exceptions=False)

    if operation == "list":
        monitor.list_error = error
        response = client.get("/rate-limit-monitor", headers=HEADERS)
    else:
        monitor.activate_error = error
        response = client.post(
            "/rate-limit-monitor/activate",
            headers=HEADERS,
            json={"model": MODEL},
        )

    assert response.status_code == 500
    assert response.text == "Internal Server Error"
    assert "secret" not in response.text


@pytest.mark.parametrize("operation", ["list", "activate"])
def test_operation_timeout_reaches_application_error_boundary(
    monkeypatch,
    operation: str,
) -> None:
    monkeypatch.setattr(monitor_routes, "OPERATION_TIMEOUT_SECONDS", 0)
    monitor = FakeMonitor()
    client = _client(monitor, raise_server_exceptions=False)

    if operation == "list":
        monitor.block_list = True
        response = client.get("/rate-limit-monitor", headers=HEADERS)
    else:
        monitor.block_activate = True
        response = client.post(
            "/rate-limit-monitor/activate",
            headers=HEADERS,
            json={"model": MODEL},
        )

    assert response.status_code == 500
    assert response.text == "Internal Server Error"
    if operation == "list":
        assert monitor.list_calls == 1
        assert monitor.activate_calls == []
    else:
        assert monitor.list_calls == 0
        assert monitor.activate_calls == [MODEL]
