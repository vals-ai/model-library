import asyncio
import json
import logging
from typing import Any, cast
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient
from starlette.types import Message, Receive, Scope, Send
from uvicorn import Config

import model_gateway.app as gateway_app
from model_gateway import asgi_observability
from model_gateway import metrics
from model_gateway.asgi_observability import GatewayObservabilityMiddleware
from model_gateway.observability import (
    ALB_TRACE_HEADER,
    install_loop_exception_handler,
    log_gateway_event,
    request_log_fields_from_scope,
    runtime_snapshot,
)


@pytest.fixture(autouse=True)
def reset_metrics_state():
    metrics.flush_metrics()
    yield
    metrics.flush_metrics()


async def _noop_receive() -> Message:
    return {"type": "http.request", "body": b"", "more_body": False}


def _http_scope(
    *,
    path: str = "/query",
    method: str = "POST",
    headers: list[tuple[bytes, bytes]] | None = None,
) -> Scope:
    return {
        "type": "http",
        "http_version": "1.1",
        "method": method,
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": headers or [],
        "client": ("203.0.113.10", 12345),
        "server": ("gateway.test", 443),
        "scheme": "https",
        "state": {},
    }


def _json_log_records(caplog: pytest.LogCaptureFixture) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for record in caplog.records:
        try:
            parsed = json.loads(record.getMessage())
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            records.append(cast(dict[str, object], parsed))
    return records


def _enable_gateway_observability_logs(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="uvicorn.error.gateway_observability")


def _emf_payloads(capsys: pytest.CaptureFixture[str]) -> list[dict[str, object]]:
    return [
        cast(dict[str, object], json.loads(line))
        for line in capsys.readouterr().out.splitlines()
    ]


def test_observability_logger_emits_info_under_uvicorn_default_config(
    capsys: pytest.CaptureFixture[str],
):
    logger_names = (
        "uvicorn",
        "uvicorn.error",
        "uvicorn.error.gateway_observability",
    )
    saved = {
        name: (
            logging.getLogger(name).level,
            list(logging.getLogger(name).handlers),
            logging.getLogger(name).propagate,
        )
        for name in logger_names
    }
    try:
        Config("model_gateway.main:create_app", factory=True).configure_logging()
        log_gateway_event("gateway.test.visible", safe_count=1)
        captured = capsys.readouterr()
    finally:
        for name, (level, handlers, propagate) in saved.items():
            logger = logging.getLogger(name)
            logger.setLevel(level)
            logger.handlers = handlers
            logger.propagate = propagate

    assert '"event":"gateway.test.visible"' in captured.err
    assert '"safe_count":1' in captured.err


def test_request_log_fields_from_scope_uses_alb_trace_and_bounds_values():
    scope = _http_scope(
        headers=[
            (ALB_TRACE_HEADER.encode(), b"Root=1-abc"),
            (b"x-run-id", b"r" * 300),
            (b"x-question-id", b"q1"),
            (b"x-query-id", b"query-a"),
        ]
    )

    fields = request_log_fields_from_scope(scope)

    assert fields["alb_trace_id"] == "Root=1-abc"
    assert fields["run_id"] is not None
    assert len(fields["run_id"]) <= 128
    assert fields["question_id"] == "q1"
    assert fields["query_id"] == "query-a"


def test_log_gateway_event_is_json_and_excludes_none(caplog: pytest.LogCaptureFixture):
    _enable_gateway_observability_logs(caplog)

    log_gateway_event(
        "gateway.request.test",
        alb_trace_id="Root=1-abc",
        run_id="run-a",
        question_id=None,
        query_id="query-a",
        phase="provider_call",
        optional_field=None,
    )

    [record] = _json_log_records(caplog)
    assert record["event"] == "gateway.request.test"
    assert record["alb_trace_id"] == "Root=1-abc"
    assert record["run_id"] == "run-a"
    assert record["query_id"] == "query-a"
    assert "question_id" not in record
    assert "optional_field" not in record


@pytest.mark.asyncio
async def test_loop_exception_handler_delegates_sanitized_context(
    caplog: pytest.LogCaptureFixture,
):
    class SecretTransport:
        def __repr__(self) -> str:
            return "SECRET=top-secret"

    delegated_contexts: list[dict[str, Any]] = []
    _enable_gateway_observability_logs(caplog)
    loop = asyncio.get_running_loop()
    original_handler = loop.get_exception_handler()

    def previous_handler(
        _loop: asyncio.AbstractEventLoop, context: dict[str, Any]
    ) -> None:
        delegated_contexts.append(context)

    context = {
        "message": "Fatal read error on socket transport",
        "transport": SecretTransport(),
        "payload": "raw request body",
        "exception": RuntimeError("secret in exception text"),
    }
    loop.set_exception_handler(previous_handler)
    try:
        install_loop_exception_handler(loop)
        loop.call_exception_handler(context)
    finally:
        loop.set_exception_handler(original_handler)

    [record] = _json_log_records(caplog)
    assert record["event"] == "gateway.event_loop.exception"
    assert record["context_message"] == "Fatal read error on socket transport"
    assert record["exception_type"] == "RuntimeError"
    assert "exception_message" not in record
    assert "transport" not in record
    assert "payload" not in record
    assert "top-secret" not in json.dumps(record)
    assert delegated_contexts == [context]


@pytest.mark.asyncio
async def test_loop_exception_handler_logs_only_exception_type_for_os_error(
    caplog: pytest.LogCaptureFixture,
):
    delegated_contexts: list[dict[str, Any]] = []
    _enable_gateway_observability_logs(caplog)
    loop = asyncio.get_running_loop()
    original_handler = loop.get_exception_handler()

    def previous_handler(
        _loop: asyncio.AbstractEventLoop, context: dict[str, Any]
    ) -> None:
        delegated_contexts.append(context)

    context = {
        "message": "Exception in callback UVTransport._call_connection_made",
        "exception": OSError(24, "Too many open files"),
        "handle": object(),
    }
    loop.set_exception_handler(previous_handler)
    try:
        install_loop_exception_handler(loop)
        loop.call_exception_handler(context)
    finally:
        loop.set_exception_handler(original_handler)

    [record] = _json_log_records(caplog)
    assert record["event"] == "gateway.event_loop.exception"
    assert record["exception_type"] == "OSError"
    assert (
        record["context_message"]
        == "Exception in callback UVTransport._call_connection_made"
    )
    assert "exception_errno" not in record
    assert "exception_strerror" not in record
    assert "exception_message" not in record
    assert "context_handle_type" not in record
    assert delegated_contexts == [context]


@pytest.mark.asyncio
async def test_loop_exception_handler_logs_bounded_context_message(
    caplog: pytest.LogCaptureFixture,
):
    _enable_gateway_observability_logs(caplog)
    loop = asyncio.get_running_loop()
    original_handler = loop.get_exception_handler()

    def previous_handler(
        _loop: asyncio.AbstractEventLoop, _context: dict[str, Any]
    ) -> None:
        return None

    loop.set_exception_handler(previous_handler)
    try:
        install_loop_exception_handler(loop)
        loop.call_exception_handler(
            {"message": "Fatal read error on socket transport", "transport": object()}
        )
    finally:
        loop.set_exception_handler(original_handler)

    [record] = _json_log_records(caplog)
    assert record["event"] == "gateway.event_loop.exception"
    assert record["context_message"] == "Fatal read error on socket transport"
    assert "message" not in record
    assert "transport" not in record
    assert "runtime" not in record


@pytest.mark.asyncio
async def test_asgi_middleware_logs_pre_response_start_errors(
    caplog: pytest.LogCaptureFixture,
):
    _enable_gateway_observability_logs(caplog)

    async def app(
        _scope: Scope,
        _receive: Receive,
        _send: Send,
    ) -> None:
        raise RuntimeError("boom")

    async def send(_message: Message) -> None:
        return None

    middleware = GatewayObservabilityMiddleware(app)

    with pytest.raises(RuntimeError, match="boom"):
        await middleware(_http_scope(), _noop_receive, send)

    records = _json_log_records(caplog)
    assert {record["event"] for record in records} >= {
        "gateway.request.error",
        "gateway.response.missing_start",
        "gateway.request.done",
    }
    request_error = next(
        record for record in records if record.get("event") == "gateway.request.error"
    )
    missing_start = next(
        record
        for record in records
        if record.get("event") == "gateway.response.missing_start"
    )
    assert request_error["response_started"] is False
    assert "runtime" in request_error
    assert "runtime" in missing_start


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "method"),
    [
        ("/health/live", "GET"),
        ("/query", "POST"),
    ],
)
async def test_asgi_middleware_does_not_log_success_lifecycle_by_default(
    caplog: pytest.LogCaptureFixture,
    path: str,
    method: str,
):
    _enable_gateway_observability_logs(caplog)

    async def app(
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok", "more_body": False})

    async def send(_message: Message) -> None:
        return None

    middleware = GatewayObservabilityMiddleware(app)
    await middleware(_http_scope(path=path, method=method), _noop_receive, send)

    assert _json_log_records(caplog) == []


@pytest.mark.asyncio
async def test_asgi_middleware_logs_non_2xx_query(
    caplog: pytest.LogCaptureFixture,
):
    _enable_gateway_observability_logs(caplog)

    async def app(
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        await send({"type": "http.response.start", "status": 429, "headers": []})
        await send({"type": "http.response.body", "body": b"busy", "more_body": False})

    async def send(_message: Message) -> None:
        return None

    middleware = GatewayObservabilityMiddleware(app)
    await middleware(_http_scope(), _noop_receive, send)

    records = _json_log_records(caplog)
    assert [record["event"] for record in records] == ["gateway.request.done"]
    assert records[0]["status_code"] == 429
    assert "runtime" in records[0]


@pytest.mark.asyncio
async def test_asgi_middleware_send_error_on_response_start_is_not_success(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
):
    _enable_gateway_observability_logs(caplog)

    async def app(
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})

    async def send(_message: Message) -> None:
        raise BrokenPipeError("client went away")

    middleware = GatewayObservabilityMiddleware(app)

    with pytest.raises(BrokenPipeError):
        await middleware(_http_scope(), _noop_receive, send)

    records = _json_log_records(caplog)
    error_record = next(
        record
        for record in records
        if record.get("event") == "gateway.response.send_error"
    )
    assert any(
        record.get("event") == "gateway.response.missing_start" for record in records
    )
    assert error_record["exception_type"] == "BrokenPipeError"
    assert error_record["response_started"] is False
    assert metrics.flush_metrics() == 0
    assert _emf_payloads(capsys) == []


@pytest.mark.asyncio
async def test_asgi_middleware_send_error_logs_runtime_data(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(asgi_observability, "runtime_snapshot", lambda: {"pid": 1000})
    _enable_gateway_observability_logs(caplog)

    async def app(
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})

    async def send(_message: Message) -> None:
        raise BrokenPipeError("client went away")

    middleware = GatewayObservabilityMiddleware(app)

    with pytest.raises(BrokenPipeError):
        await middleware(_http_scope(), _noop_receive, send)

    records = _json_log_records(caplog)
    assert {record.get("event") for record in records} >= {
        "gateway.response.send_error",
        "gateway.response.missing_start",
        "gateway.request.done",
    }
    assert all(record.get("runtime") == {"pid": 1000} for record in records)


@pytest.mark.asyncio
async def test_asgi_middleware_send_error_on_final_body_is_not_done_success(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
):
    _enable_gateway_observability_logs(caplog)

    async def app(
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send(
            {"type": "http.response.body", "body": b"partial", "more_body": True}
        )
        await send({"type": "http.response.body", "body": b"done", "more_body": False})

    async def send(message: Message) -> None:
        if message.get("type") == "http.response.body" and not message.get(
            "more_body", False
        ):
            raise BrokenPipeError("client went away")

    middleware = GatewayObservabilityMiddleware(app)

    with pytest.raises(BrokenPipeError):
        await middleware(_http_scope(), _noop_receive, send)

    records = _json_log_records(caplog)
    done = next(
        record for record in records if record.get("event") == "gateway.request.done"
    )
    assert done["response_started"] is True
    assert done["first_body_sent"] is True
    assert done["response_done"] is False
    assert done["send_error"] is True
    assert metrics.flush_metrics() == 0
    assert _emf_payloads(capsys) == []


def test_create_app_observability_wraps_auth_short_circuit(
    caplog: pytest.LogCaptureFixture,
):
    class ServerSettings:
        MODEL_GATEWAY_API_KEYS = "sk-test"
        MODEL_GATEWAY_HMAC_SECRET = "test-secret"

        def get(self, name: str, default: str = "") -> str:
            return getattr(self, name, default)

        def unset(self, _key: str) -> None:
            return None

    _enable_gateway_observability_logs(caplog)
    with patch.object(gateway_app, "model_library_settings", ServerSettings()):
        app = gateway_app.create_app()
        response = TestClient(app).post(
            "/query",
            headers={ALB_TRACE_HEADER: "Root=1-auth-short-circuit"},
            json={"model": "openai/gpt-4o", "input": "hello"},
        )

    assert response.status_code == 401
    records = _json_log_records(caplog)
    assert {record["event"] for record in records} >= {"gateway.request.done"}
    boundary_records = [
        record
        for record in records
        if str(record.get("event", "")).startswith("gateway.request")
        or str(record.get("event", "")).startswith("gateway.response")
    ]
    assert boundary_records
    assert all(
        record.get("alb_trace_id") == "Root=1-auth-short-circuit"
        for record in boundary_records
    )


@pytest.mark.asyncio
async def test_asgi_middleware_logs_client_disconnect(caplog: pytest.LogCaptureFixture):
    _enable_gateway_observability_logs(caplog)

    async def app(
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        await receive()
        await send({"type": "http.response.start", "status": 499, "headers": []})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    async def receive() -> Message:
        return {"type": "http.disconnect"}

    async def send(_message: Message) -> None:
        return None

    middleware = GatewayObservabilityMiddleware(app)
    await middleware(_http_scope(), receive, send)

    records = _json_log_records(caplog)
    assert any(
        record.get("event") == "gateway.request.client_disconnect" for record in records
    )
    done = [
        record for record in records if record.get("event") == "gateway.request.done"
    ]
    assert done and done[0]["client_disconnected"] is True


def test_create_app_lifespan_logs_process_lifecycle(
    caplog: pytest.LogCaptureFixture,
):
    class ServerSettings:
        MODEL_GATEWAY_API_KEYS = "sk-test"
        MODEL_GATEWAY_HMAC_SECRET = "test-secret"

        def get(self, name: str, default: str = "") -> str:
            return getattr(self, name, default)

        def unset(self, _key: str) -> None:
            return None

    class FakeUsageLedger:
        async def start(self) -> None:
            return None

        async def close(self) -> None:
            return None

    _enable_gateway_observability_logs(caplog)
    with (
        patch.dict(
            "os.environ",
            {
                "GATEWAY_STARTUP_CANARY_ENABLED": "false",
                "REDIS_URL": "",
                "GATEWAY_STAGE": "preview-test",
                "GATEWAY_SERVICE": "gateway-test",
            },
        ),
        patch.object(gateway_app, "model_library_settings", ServerSettings()),
        patch.object(gateway_app, "get_model_names", return_value=["openai/gpt-4o"]),
        patch.object(
            gateway_app, "create_usage_ledger_from_env", return_value=FakeUsageLedger()
        ),
    ):
        app = gateway_app.create_app()
        with TestClient(app):
            pass

    lifecycle_records = [
        record
        for record in _json_log_records(caplog)
        if str(record.get("event", "")).startswith("gateway.process.")
    ]
    events = [record.get("event") for record in lifecycle_records]
    assert events.count("gateway.process.startup") == 1
    assert events.count("gateway.process.shutdown_start") == 1
    assert events.count("gateway.process.shutdown_done") == 1
    assert all(record["stage"] == "preview-test" for record in lifecycle_records)
    assert all(record["service"] == "gateway-test" for record in lifecycle_records)
    assert all(isinstance(record["worker_id"], int) for record in lifecycle_records)
    assert all("runtime" not in record for record in lifecycle_records)


def test_runtime_snapshot_contains_safe_runtime_keys():
    snapshot = runtime_snapshot()

    assert set(snapshot) >= {
        "pid",
        "thread_count",
        "asyncio_task_count",
        "open_fd_count",
        "inbound_socket_count",
        "outbound_socket_count",
        "rss_bytes",
    }
    assert all(isinstance(value, int) for value in snapshot.values())


@pytest.mark.asyncio
async def test_record_runtime_current_samples_loop_lag_before_snapshot(
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeLoop:
        def __init__(self) -> None:
            self.calls = 0
            self.now = 100.25

        def time(self) -> float:
            self.calls += 1
            if self.calls == 1:
                return 100.0
            return self.now

    fake_loop = FakeLoop()
    captured: list[dict[str, int]] = []

    def slow_runtime_snapshot() -> dict[str, int]:
        fake_loop.now = 105.25
        return {"pid": 123}

    def capture_runtime(snapshot: dict[str, int]) -> None:
        captured.append(snapshot.copy())

    monkeypatch.setattr(gateway_app, "runtime_snapshot", slow_runtime_snapshot)
    monkeypatch.setattr(gateway_app, "record_runtime", capture_runtime)

    await gateway_app._record_runtime_current(
        cast(asyncio.AbstractEventLoop, fake_loop)
    )

    assert captured == [{"pid": 123, "event_loop_lag_ms": 250}]


def test_record_runtime_metrics_emits_low_cardinality_metrics(
    capsys: pytest.CaptureFixture[str],
):
    metrics.record_runtime(
        {
            "pid": 123,
            "thread_count": 2,
            "asyncio_task_count": 3,
            "open_fd_count": 4,
            "inbound_socket_count": 5,
            "outbound_socket_count": 6,
            "rss_bytes": 7,
            "event_loop_lag_ms": 8,
        }
    )

    assert metrics.flush_metrics() == 1
    [payload] = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert payload["WorkerId"] == "123"
    assert payload["ThreadCount"] == 2
    assert payload["AsyncioTaskCount"] == 3
    assert payload["OpenFileDescriptors"] == 4
    assert payload["InboundSocketCount"] == 5
    assert payload["OutboundSocketCount"] == 6
    assert payload["RssBytes"] == 7
    assert payload["EventLoopLagMs"] == 8


def test_record_runtime_metrics_requires_complete_snapshot():
    with pytest.raises(KeyError, match="event_loop_lag_ms"):
        metrics.record_runtime(
            {
                "pid": 123,
                "thread_count": 2,
                "asyncio_task_count": 3,
                "open_fd_count": 4,
                "inbound_socket_count": 5,
                "outbound_socket_count": 6,
                "rss_bytes": 7,
            }
        )
