import base64
import io
import json
import os
import threading
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Literal, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx  # pyright: ignore[reportMissingImports]
import pytest
from pydantic import SecretStr, ValidationError
from redis.exceptions import TimeoutError as RedisTimeoutError
from starlette.testclient import TestClient

import model_gateway.app as gateway_app
import model_gateway.model_helpers as model_helpers
from model_gateway import startup_canary
from model_gateway import telemetry_helpers
from model_library.base import LLMConfig, TextInput, dump_gateway_config


def _make_client(*, client: tuple[str, int] = ("testclient", 50000)):
    from model_gateway import main

    class ServerSettings:
        MODEL_GATEWAY_API_KEYS = "sk-test"
        MODEL_GATEWAY_HMAC_SECRET = "test-secret"

        def get(self, name: str, default: str = "") -> str:
            return getattr(self, name, default)

        def unset(self, key: str):
            pass

    with patch.object(gateway_app, "model_library_settings", ServerSettings()):
        app = main.create_app()
        return TestClient(app, client=client)


HEADERS = {"Authorization": "Bearer sk-test"}


def test_query_result_response_body_renames_history_to_signed_history():
    from model_gateway.types import query_result_response_body
    from model_library.base.input import TextInput
    from model_library.base.output import QueryResult

    result = QueryResult(output_text="ok", history=[TextInput(text="hi")])

    body = query_result_response_body(result, signed_history="signed")

    assert body == {
        **QueryResult.model_dump(
            result, mode="json", exclude={"history"}, exclude_none=True
        ),
        "signed_history": "signed",
    }
    assert "history" not in body


@pytest.mark.parametrize(
    ("response_cls", "payload_field", "payload_value"),
    [
        (
            "UploadFileResponse",
            "file",
            {
                "type": "file",
                "name": "doc.pdf",
                "mime": "application/pdf",
                "file_id": "file-test",
            },
        ),
        ("EmbeddingResponse", "embedding", [0.1, 0.2, 0.3]),
        ("ModerationResponse", "response", {"id": "modr-test", "results": []}),
    ],
)
def test_provider_operation_responses_require_success_xor_error(
    response_cls: str, payload_field: str, payload_value: object
):
    from model_gateway import types
    from model_gateway.types import ProviderError

    cls = getattr(types, response_cls)
    error = ProviderError(code="internal_error", message="provider failed")

    success = cls(**{payload_field: payload_value})
    assert getattr(success, payload_field) is not None
    assert payload_field in success.model_dump(exclude_none=True)
    assert cls(error=error).model_dump(mode="json", exclude_none=True) == {
        "error": {
            "type": "ProviderError",
            "code": "internal_error",
            "message": "provider failed",
        }
    }

    with pytest.raises(ValidationError):
        cls()
    with pytest.raises(ValidationError):
        cls(error=error, **{payload_field: payload_value})


def test_llm_config_telemetry_attributes_keeps_arbitrary_safe_provider_config_keys():
    attrs = telemetry_helpers.llm_config_telemetry_attributes(  # pyright: ignore[reportPrivateUsage]
        {
            "provider_config": {
                "new_boolean": True,
                "new_limit": 3,
                "new_mode": "fast",
                "prompt": "raw prompt",
                "api_key": "secret",
                "nested": {"value": "not scalar"},
            }
        }
    )

    assert attrs == {
        "llm.config.provider_config": '{"api_key":"<redacted>","nested":{"value":"not scalar"},"new_boolean":true,"new_limit":3,"new_mode":"fast","prompt":"raw prompt"}'
    }


def test_query_telemetry_buckets_custom_endpoint_without_raw_url():
    from model_gateway.types import QueryRequest

    raw_endpoint = "https://private-provider.example.internal/v1"
    config = dump_gateway_config(
        LLMConfig(
            custom_endpoint=raw_endpoint,
            custom_api_key=SecretStr("sk-provider"),
        )
    )
    attrs = telemetry_helpers.query_telemetry_attributes(
        QueryRequest(
            model="openai/gpt-4o",
            inputs=[TextInput(text="hi")],
            config=LLMConfig(
                custom_endpoint=raw_endpoint,
                custom_api_key=SecretStr("sk-provider"),
            ),
        ),
        config,
        config_query_params=telemetry_helpers.query_config_params(config),
        token_retry_params=None,
    )

    assert attrs["model.provider_endpoint"] == "custom"
    assert attrs["llm.config.custom_endpoint"] == f'"{raw_endpoint}"'
    assert attrs["llm.config.custom_api_key"] == '"**********"'
    assert "sk-provider" not in attrs.values()


def test_health_live_no_auth():
    client = _make_client()
    assert client.get("/health/live").status_code == 200


def test_health_live_does_not_emit_sentry_http_span():
    from model_gateway import metrics

    client = _make_client()
    with patch.object(
        metrics.telemetry, "start_span", return_value=nullcontext()
    ) as span:
        resp = client.get("/health/live")

    assert resp.status_code == 200
    span.assert_not_called()


def test_metrics_http_span_uses_method_and_route_name():
    from model_gateway import metrics

    client = _make_client()
    with patch.object(
        metrics.telemetry, "start_span", return_value=nullcontext()
    ) as span:
        resp = client.get("/models", headers=HEADERS)

    assert resp.status_code == 200
    assert span.call_args.args[0] == "GET /models"


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (
            {"MODEL_GATEWAY_API_KEYS": "", "MODEL_GATEWAY_HMAC_SECRET": "test-secret"},
            "MODEL_GATEWAY_API_KEYS must be set",
        ),
        (
            {"MODEL_GATEWAY_HMAC_SECRET": "test-secret"},
            "MODEL_GATEWAY_API_KEYS must be set",
        ),
        (
            {"MODEL_GATEWAY_API_KEYS": "sk-test", "MODEL_GATEWAY_HMAC_SECRET": ""},
            "MODEL_GATEWAY_HMAC_SECRET must be set",
        ),
        (
            {"MODEL_GATEWAY_API_KEYS": "sk-test"},
            "MODEL_GATEWAY_HMAC_SECRET must be set",
        ),
    ],
)
def test_create_app_requires_gateway_keys_and_hmac_secret(
    settings: dict[str, str], message: str
):
    from model_gateway import main

    class GatewaySettings:
        def get(self, name: str, default: str | None = None) -> str | None:
            return settings.get(name, default)

        def unset(self, key: str):
            pass

    with (
        patch.object(gateway_app, "model_library_settings", GatewaySettings()),
        patch.dict(os.environ, {}, clear=True),
        pytest.raises(RuntimeError, match=message),
    ):
        main.create_app()

    client = _make_client()
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_lifespan_survives_malformed_otel_env():
    from model_library import telemetry
    from model_gateway import main

    class ServerSettings:
        MODEL_GATEWAY_API_KEYS = "sk-test"
        MODEL_GATEWAY_HMAC_SECRET = "test-secret"

        def get(self, name: str, default: str = "") -> str:
            return getattr(self, name, default)

        def unset(self, key: str):
            pass

    telemetry.shutdown_telemetry()
    with (
        patch.dict(
            os.environ,
            {
                "GATEWAY_OTEL_ENABLED": "true",
                "OTEL_EXPORTER_OTLP_TIMEOUT": "not-a-float",
            },
            clear=True,
        ),
        patch.object(gateway_app, "model_library_settings", ServerSettings()),
        patch.object(gateway_app, "get_model_names", return_value=["openai/gpt-4o"]),
    ):
        app = main.create_app()
        with TestClient(app) as client:
            resp = client.get("/health/live")

    telemetry.shutdown_telemetry()
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_startup_canary_executes_authenticated_local_query():

    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/health/live":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/query":
            return httpx.Response(
                200, json={"signed_history": "signed", "output_text": "ok"}
            )
        return httpx.Response(404)

    await startup_canary.execute_startup_canary(
        api_key="sk-test",
        base_url="http://127.0.0.1:8000",
        model="openai/gpt-5.4-nano-2026-03-17",
        max_tokens=16,
        timeout_seconds=5,
        wait_timeout_seconds=1,
        transport=httpx.MockTransport(handler),
    )

    query_request = requests[-1]
    assert query_request.url.path == "/query"
    assert query_request.headers["Authorization"] == "Bearer sk-test"
    body = json.loads(query_request.content)
    assert body["model"] == "openai/gpt-5.4-nano-2026-03-17"
    assert body["config"] == {
        "max_tokens": 16,
        "temperature": 0,
        "reasoning_effort": "low",
    }
    assert body["run_id"] == "gateway-startup-canary"


async def test_run_startup_canary_uses_doubled_output_budget():
    app = MagicMock()
    app.state.startup_canary = {
        "enabled": True,
        "status": "pending",
        "error": "",
    }

    with (
        patch.dict(os.environ, {"GATEWAY_PORT": "8123"}),
        patch.object(
            startup_canary, "execute_startup_canary", new_callable=AsyncMock
        ) as execute,
    ):
        await startup_canary.run_startup_canary(app, "sk-test")

    execute.assert_awaited_once_with(
        api_key="sk-test",
        base_url="http://127.0.0.1:8123",
        model="openai/gpt-5.4-nano-2026-03-17",
        max_tokens=32,
        timeout_seconds=30,
        reasoning_effort="low",
        wait_timeout_seconds=30,
    )
    assert app.state.startup_canary["status"] == "passed"


async def test_startup_canary_fails_without_signed_history():

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health/live":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(200, json={"output_text": "ok"})

    try:
        await startup_canary.execute_startup_canary(
            api_key="sk-test",
            base_url="http://127.0.0.1:8000",
            model="openai/gpt-5.4-nano-2026-03-17",
            max_tokens=16,
            timeout_seconds=5,
            wait_timeout_seconds=1,
            transport=httpx.MockTransport(handler),
        )
    except RuntimeError as exc:
        assert "signed_history" in str(exc)
    else:
        raise AssertionError("startup canary should fail without signed_history")


def test_health_ready_waits_for_enabled_startup_canary():
    client = _make_client()
    cast(Any, client.app).state.startup_canary = {
        "enabled": True,
        "status": "pending",
        "error": "",
    }

    pending = client.get("/health/ready")
    assert pending.status_code == 503
    assert pending.json() == {"status": "startup canary pending"}

    cast(Any, client.app).state.startup_canary = {
        "enabled": True,
        "status": "failed",
        "error": "RuntimeError: boom",
    }
    failed = client.get("/health/ready")
    assert failed.status_code == 503
    assert failed.json() == {"status": "startup canary failed"}

    cast(Any, client.app).state.startup_canary = {
        "enabled": True,
        "status": "passed",
        "error": "",
    }
    assert client.get("/health/ready").status_code == 200


def test_lifespan_loads_model_registry_at_startup():
    from model_gateway import main

    class ServerSettings:
        MODEL_GATEWAY_API_KEYS = "sk-test"
        MODEL_GATEWAY_HMAC_SECRET = "test-secret"

        def get(self, name: str, default: str = "") -> str:
            return getattr(self, name, default)

        def unset(self, key: str):
            pass

    with (
        patch.dict(os.environ, {}, clear=True),
        patch.object(gateway_app, "model_library_settings", ServerSettings()),
        patch.object(
            gateway_app, "get_model_names", return_value=["openai/gpt-4o"]
        ) as mock_get_model_names,
    ):
        app = main.create_app()
        with TestClient(app):
            mock_get_model_names.assert_called_once_with()


def test_registry_snapshot_requires_auth_and_returns_full_configs():
    client = _make_client()
    assert client.get("/registry").status_code == 401

    config = MagicMock()
    config.model_dump.return_value = {"full_key": "openai/gpt-4o"}
    with patch(
        "model_gateway.routes.models.get_model_registry",
        return_value={"openai/gpt-4o": config},
    ):
        resp = client.get("/registry", headers=HEADERS)

    assert resp.status_code == 200
    assert resp.json() == {"models": {"openai/gpt-4o": {"full_key": "openai/gpt-4o"}}}
    config.model_dump.assert_called_once_with(mode="json")


def test_model_resolve_requires_auth_and_returns_effective_and_registry_config():
    client = _make_client()
    assert (
        client.post("/models/resolve", json={"model": "openai/gpt-4o"}).status_code
        == 401
    )

    resp = client.post(
        "/models/resolve",
        headers=HEADERS,
        json={
            "model": "openai/gpt-4o",
            "config": {"max_tokens": 123, "custom_api_key": "provider-key"},
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["exists"] is True
    assert data["model"] == "openai/gpt-4o"
    assert "capabilities" not in data
    assert "default_parameters" not in data
    assert data["effective_config"]["max_tokens"] == 123
    assert data["effective_config"]["supports_batch"] is True
    assert data["effective_config"]["custom_api_key"] == "**********"
    assert data["registry_config"]["full_key"] == "openai/gpt-4o"


def test_token_count_requires_auth_and_returns_count_with_restored_raw_input():
    from model_library.base.base import LLM
    from model_library.base.input import RawInput, TextInput

    seen: dict[str, object] = {}
    signed_inputs = LLM.serialize_input(
        [
            RawInput(input={"messages": [{"role": "user", "content": "hi"}]}),
            TextInput(text="count this"),
        ],
        secret=b"test-secret",
    )

    class FakeLLM:
        async def count_tokens(self, inputs, *, tools, **kwargs):
            seen["inputs"] = inputs
            seen["tools"] = tools
            seen["kwargs"] = kwargs
            return 17

    def fake_get_registry_model(model, config):
        seen["model"] = model
        seen["max_tokens"] = config.max_tokens
        return FakeLLM()

    client = _make_client()
    token_body = {
        "model": "openai/gpt-4o",
        "inputs": json.loads(signed_inputs),
        "tools": [
            {
                "name": "lookup",
                "body": {
                    "name": "lookup",
                    "description": "Lookup a value",
                    "properties": {},
                    "required": [],
                },
            }
        ],
        "config": {"max_tokens": 7},
    }
    assert client.post("/tokens/count", json=token_body).status_code == 401

    with patch.object(
        model_helpers, "get_registry_model", side_effect=fake_get_registry_model
    ):
        token_resp = client.post("/tokens/count", headers=HEADERS, json=token_body)

    assert token_resp.status_code == 200
    assert token_resp.json() == {"tokens": 17}
    assert seen["model"] == "openai/gpt-4o"
    assert seen["max_tokens"] == 7
    seen_inputs = cast(list[object], seen["inputs"])
    assert isinstance(seen_inputs[0], RawInput)
    assert seen_inputs[0].input == {"messages": [{"role": "user", "content": "hi"}]}
    assert isinstance(seen_inputs[1], TextInput)
    assert seen_inputs[1].text == "count this"
    assert len(cast(list[object], seen["tools"])) == 1
    assert seen["kwargs"] == {}


def test_token_count_is_not_rejected_by_query_capacity_limit():
    from model_gateway.capacity import GatewayCapacityLimiter

    class FakeLLM:
        async def count_tokens(self, inputs, *, tools, **kwargs):
            return 7

    client = _make_client()
    limiter = GatewayCapacityLimiter(
        max_active=1,
        max_queued=0,
        queue_timeout_seconds=1,
        request_timeout_seconds=1,
    )
    limiter._active = 1  # pyright: ignore[reportPrivateUsage]
    cast(Any, client.app).state.capacity_limiter = limiter

    with patch.object(model_helpers, "get_registry_model", return_value=FakeLLM()):
        resp = client.post(
            "/tokens/count",
            json={
                "model": "openai/gpt-4o",
                "inputs": [{"kind": "text", "text": "hi"}],
                "tools": [],
                "config": {},
            },
            headers=HEADERS,
        )

    assert resp.status_code == 200
    assert resp.json() == {"tokens": 7}


def test_token_count_rejects_malformed_raw_history_with_400():
    class FakeLLM:
        async def count_tokens(self, inputs, *, tools, **kwargs):
            raise AssertionError("count_tokens should not be called")

    client = _make_client()
    with (
        patch.object(model_helpers, "get_registry_model", return_value=FakeLLM()),
        patch.object(gateway_app.telemetry, "record_exception") as record_exception,
    ):
        resp = client.post(
            "/tokens/count",
            json={
                "model": "openai/gpt-4o",
                "inputs": [
                    {
                        "kind": "raw_input",
                        "input": {"messages": [{"role": "user", "content": "hi"}]},
                    }
                ],
                "tools": [],
                "config": {},
            },
            headers=HEADERS,
        )

    assert resp.status_code == 400
    assert resp.json()["code"] == "hmac_verification_failed"
    record_exception.assert_called_once()
    error_attrs = record_exception.call_args.args[1]
    assert error_attrs["gateway.error.code"] == "hmac_verification_failed"
    assert error_attrs["gateway.error.phase"] == "restore_history"
    assert error_attrs["http.response.status_code"] == 400


def test_rate_limit_endpoint_is_token_retry_only():
    client = _make_client()

    rate_body = {"model": "openai/gpt-4o", "config": {}}
    assert client.post("/rate-limit", json=rate_body).status_code == 401
    rate_resp = client.post("/rate-limit", headers=HEADERS, json=rate_body)
    assert rate_resp.status_code == 501
    assert rate_resp.json() == {"detail": "Gateway token retry use only"}


def test_token_retry_status_requires_auth_and_returns_redis_status():
    client = _make_client()
    assert client.get("/token-retry/status").status_code == 401

    status = MagicMock()
    status.model_dump.return_value = {"models": []}
    with (
        patch(
            "model_gateway.routes.token_retry.validate_redis_client",
            new_callable=AsyncMock,
        ),
        patch(
            "model_gateway.routes.token_retry.get_token_retry_status",
            new_callable=AsyncMock,
            return_value=status,
        ),
    ):
        resp = client.get("/token-retry/status", headers=HEADERS)

    assert resp.status_code == 200
    assert resp.json() == {"models": []}
    status.model_dump.assert_called_once_with(mode="json")


def test_token_retry_status_is_cached_briefly_per_process():
    client = _make_client()
    status = MagicMock()
    status.model_dump.return_value = {"models": []}
    with (
        patch(
            "model_gateway.routes.token_retry.validate_redis_client",
            new_callable=AsyncMock,
        ),
        patch(
            "model_gateway.routes.token_retry.get_token_retry_status",
            new_callable=AsyncMock,
            return_value=status,
        ) as get_status,
    ):
        first = client.get("/token-retry/status", headers=HEADERS)
        second = client.get("/token-retry/status", headers=HEADERS)

    assert first.status_code == 200
    assert second.status_code == 200
    assert get_status.await_count == 1


def test_token_retry_status_returns_503_when_redis_is_not_configured():
    client = _make_client()
    with patch(
        "model_gateway.routes.token_retry.validate_redis_client",
        new_callable=AsyncMock,
        side_effect=Exception("redis client not set, run `set_redis_client`"),
    ):
        resp = client.get("/token-retry/status", headers=HEADERS)

    assert resp.status_code == 503
    assert resp.json() == {
        "code": "redis_not_configured",
        "message": "redis client not set, run `set_redis_client`",
    }


def test_model_resolve_returns_exists_false_for_unknown_model():
    client = _make_client()
    resp = client.post(
        "/models/resolve",
        headers=HEADERS,
        json={"model": "unknown-provider/new-model"},
    )

    assert resp.status_code == 200
    assert resp.json() == {
        "exists": False,
        "model": "unknown-provider/new-model",
        "effective_config": None,
        "registry_config": None,
        "input_context_window": None,
    }


def test_model_resolve_maps_invalid_config_to_error_envelope():
    client = _make_client()

    resp = client.post(
        "/models/resolve",
        headers=HEADERS,
        json={
            "model": "openai/gpt-4o",
            "config": {"custom_endpoint": "https://provider.example/v1"},
        },
    )

    assert resp.status_code == 400
    assert resp.json()["code"] == "custom_key_rejected"


def test_model_resolve_rejects_unknown_provider_config_keys():
    client = _make_client()

    resp = client.post(
        "/models/resolve",
        headers=HEADERS,
        json={
            "model": "openai/gpt-4o",
            "config": {"provider_config": {"verbostiy": "low"}},
        },
    )

    assert resp.status_code == 400
    assert resp.json()["code"] == "invalid_request"
    assert "verbostiy" in resp.json()["message"]


def test_lifespan_starts_and_closes_usage_ledger():
    from model_gateway import main

    class ServerSettings:
        MODEL_GATEWAY_API_KEYS = "sk-test"
        MODEL_GATEWAY_HMAC_SECRET = "test-secret"

        def get(self, name: str, default: str = "") -> str:
            return getattr(self, name, default)

        def unset(self, key: str):
            pass

    class FakeUsageLedger:
        enabled = False
        started = 0
        closed = 0

        async def start(self) -> None:
            self.started += 1

        async def close(self) -> None:
            self.closed += 1

        async def write_success(self, event: dict[str, object]) -> None:
            _ = event

    fake_ledger = FakeUsageLedger()
    with (
        patch.dict(os.environ, {"GATEWAY_STARTUP_CANARY_ENABLED": "false"}),
        patch.object(
            gateway_app, "create_usage_ledger_from_env", return_value=fake_ledger
        ),
        patch.object(gateway_app, "get_model_names", return_value=["openai/gpt-4o"]),
        patch.object(gateway_app, "model_library_settings", ServerSettings()),
    ):
        app = main.create_app()
        with TestClient(app):
            assert fake_ledger.started == 1
            assert fake_ledger.closed == 0

    assert fake_ledger.started == 1
    assert fake_ledger.closed == 1


def test_lifespan_close_continues_after_usage_ledger_close_failure():
    from model_gateway import main

    class ServerSettings:
        MODEL_GATEWAY_API_KEYS = "sk-test"
        MODEL_GATEWAY_HMAC_SECRET = "test-secret"

        def get(self, name: str, default: str = "") -> str:
            return getattr(self, name, default)

        def unset(self, key: str):
            pass

    class FailingCloseUsageLedger:
        enabled = False

        async def start(self) -> None:
            return

        async def close(self) -> None:
            raise RuntimeError("ledger close failed")

        async def write_success(self, event: dict[str, object]) -> None:
            _ = event

    class FakeRedis:
        closed = False

        async def aclose(self) -> None:
            self.closed = True

    fake_redis = FakeRedis()
    with (
        patch.dict(
            os.environ,
            {
                "GATEWAY_STARTUP_CANARY_ENABLED": "false",
                "REDIS_URL": "redis://localhost:6379/0",
            },
        ),
        patch.object(gateway_app.async_redis, "from_url", return_value=fake_redis),
        patch.object(
            gateway_app,
            "create_usage_ledger_from_env",
            return_value=FailingCloseUsageLedger(),
        ),
        patch.object(
            gateway_app.telemetry, "shutdown_telemetry"
        ) as mock_shutdown_telemetry,
        patch.object(gateway_app, "get_model_names", return_value=["openai/gpt-4o"]),
        patch.object(gateway_app, "model_library_settings", ServerSettings()),
    ):
        app = main.create_app()
        with TestClient(app):
            pass

    assert fake_redis.closed is True
    mock_shutdown_telemetry.assert_called_once_with()


def test_lifespan_closes_owned_redis_client():
    from model_gateway import main

    class ServerSettings:
        MODEL_GATEWAY_API_KEYS = "sk-test"
        MODEL_GATEWAY_HMAC_SECRET = "test-secret"

        def get(self, name: str, default: str = "") -> str:
            return getattr(self, name, default)

        def unset(self, key: str):
            pass

    class FakeRedis:
        closed = False

        async def aclose(self) -> None:
            self.closed = True

    fake_redis = FakeRedis()
    with (
        patch.dict(os.environ, {"REDIS_URL": "redis://localhost:6379/0"}),
        patch.object(
            gateway_app.async_redis, "from_url", return_value=fake_redis
        ) as mock_from_url,
        patch.object(gateway_app, "set_redis_client") as mock_set_redis_client,
        patch.object(gateway_app, "model_library_settings", ServerSettings()),
    ):
        app = main.create_app()
        with TestClient(app):
            mock_set_redis_client.assert_called_once_with(fake_redis)

    mock_from_url.assert_called_once_with(
        "redis://localhost:6379/0",
        decode_responses=True,
        socket_timeout=30,
        socket_connect_timeout=10,
        socket_keepalive=True,
        health_check_interval=30,
        retry_on_error=[RedisTimeoutError],
        max_connections=200,
    )
    assert fake_redis.closed is True
    mock_set_redis_client.assert_called_once_with(fake_redis)


def test_auth_failure_trace_paths_match_protected_routes():
    from fastapi.routing import APIRoute
    from model_gateway import auth

    client = _make_client()
    app = cast(Any, client.app)
    protected_routes = {
        route.path
        for route in app.routes
        if isinstance(route, APIRoute) and route.path not in auth.EXEMPT_PATHS
    }

    assert auth.TRACE_AUTH_FAILURE_PATHS == protected_routes


def test_http_trace_allowed_routes_match_protected_routes():
    from fastapi.routing import APIRoute
    from model_library import telemetry
    from model_gateway import auth

    client = _make_client()
    app = cast(Any, client.app)
    protected_routes = {
        route.path
        for route in app.routes
        if isinstance(route, APIRoute) and route.path not in auth.EXEMPT_PATHS
    }

    assert telemetry.HTTP_TRACE_ALLOWED_ROUTES == protected_routes


def test_unknown_unauthorized_path_does_not_emit_sentry_span():
    from model_gateway import auth

    client = _make_client()
    with patch.object(auth.telemetry, "start_span", return_value=nullcontext()) as span:
        resp = client.get("/.env")

    assert resp.status_code == 401
    span.assert_not_called()


def test_query_requires_auth():
    from model_gateway import auth

    client = _make_client()
    with patch.object(auth.telemetry, "start_span", return_value=nullcontext()) as span:
        resp = client.post(
            "/query",
            json={
                "model": "openai/gpt-4o",
                "inputs": [{"kind": "text", "text": "hi"}],
            },
        )
    assert resp.status_code == 401
    span.assert_called_once()
    assert span.call_args.args[0] == "POST /query"
    attrs = span.call_args.args[1]
    assert attrs["gateway.route"] == "/query"
    assert attrs["gateway.error.code"] == "access_denied"
    assert attrs["gateway.error.phase"] == "access_control"
    assert attrs["gateway.operation"] == "access_check"
    assert attrs["http.response.status_code"] == 401


def test_query_invalid_model_returns_400():
    client = _make_client()
    resp = client.post(
        "/query",
        json={
            "model": "nonexistent/model",
            "inputs": [{"kind": "text", "text": "hi"}],
        },
        headers=HEADERS,
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "invalid_model"


def test_query_request_config_is_llm_config():
    from model_library.base import LLMConfig
    from model_gateway.types import QueryRequest

    request = QueryRequest.model_validate(
        {
            "model": "openai/gpt-4o",
            "inputs": [{"kind": "text", "text": "hi"}],
            "config": {"max_tokens": 7, "custom_api_key": "provider-key"},
        }
    )

    assert isinstance(request.config, LLMConfig)
    assert request.config.max_tokens == 7
    assert request.config.custom_api_key is not None
    assert request.config.custom_api_key.get_secret_value() == "provider-key"


def test_query_request_config_rejects_unknown_fields():

    client = _make_client()

    with (
        patch.object(gateway_app.telemetry, "set_attributes") as set_attributes,
        patch.object(gateway_app.telemetry, "set_status_error") as set_status_error,
        patch.object(gateway_app.telemetry, "add_event") as add_event,
    ):
        resp = client.post(
            "/query",
            json={
                "model": "openai/gpt-4o",
                "inputs": [{"kind": "text", "text": "hi"}],
                "config": {"max_tokens": 7, "unknown_config": "ignored locally"},
            },
            headers=HEADERS,
        )

    assert resp.status_code == 400
    assert resp.json()["code"] == "invalid_request"
    assert "unknown_config" in resp.json()["message"]
    validation_attrs: dict[str, object | None] | None = None
    for call in set_attributes.call_args_list:
        attrs = call.args[0]
        if (
            isinstance(attrs, dict)
            and attrs.get("gateway.error.phase") == "request_validation"
        ):
            validation_attrs = cast(dict[str, object | None], attrs)
            break
    assert validation_attrs is not None
    assert validation_attrs["gateway.error.code"] == "invalid_request"
    assert validation_attrs["http.response.status_code"] == 400
    set_status_error.assert_any_call("invalid_request")
    add_event.assert_any_call("gateway.request_validation.error", validation_attrs)


def test_query_maps_provider_quota_errors_to_429():
    from model_gateway.errors import map_exception_to_error

    err = map_exception_to_error(
        RuntimeError("OpenAI insufficient_quota: You exceeded your current quota")
    )

    assert err.status_code == 429
    assert err.body.code == "provider_quota_exceeded"
    assert err.body.provider == "openai"


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(
            "direct_no_output",
            id="direct-model-no-output",
        ),
        pytest.param(
            "exhausted_no_output",
            id="exhausted-model-no-output",
        ),
    ],
)
def test_query_returns_provider_exception_envelope_without_gateway_mapping(exc: str):
    from model_library.exceptions import (
        ImmediateRetryExhaustedError,
        ModelNoOutputError,
    )

    original = ModelNoOutputError("model returned empty response")

    class FakeLLM:
        async def query(self, inputs, **kwargs):
            _ = inputs, kwargs
            if exc == "exhausted_no_output":
                raise ImmediateRetryExhaustedError(10, 10, original)
            raise original

    client = _make_client()
    with patch.object(model_helpers, "get_registry_model", return_value=FakeLLM()):
        resp = client.post(
            "/query",
            json={
                "model": "openai/gpt-4o",
                "inputs": [{"kind": "text", "text": "hi"}],
            },
            headers=HEADERS,
        )

    assert resp.status_code == 200
    assert resp.json() == {
        "error": {
            "type": "ProviderError",
            "message": "model returned empty response",
            "provider": "openai",
            "exception_type": "ModelNoOutputError",
        }
    }


def test_query_provider_exception_envelope_preserves_raw_code_and_status_when_present():
    class ProviderStatusError(Exception):
        code = "rate_limit_exceeded"
        status_code = 429

    class FakeLLM:
        async def query(self, inputs, **kwargs):
            _ = inputs, kwargs
            raise ProviderStatusError("provider says slow down")

    client = _make_client()
    with patch.object(model_helpers, "get_registry_model", return_value=FakeLLM()):
        resp = client.post(
            "/query",
            json={
                "model": "openai/gpt-4o",
                "inputs": [{"kind": "text", "text": "hi"}],
            },
            headers=HEADERS,
        )

    assert resp.status_code == 200
    assert resp.json() == {
        "error": {
            "type": "ProviderError",
            "code": "rate_limit_exceeded",
            "message": "provider says slow down",
            "provider": "openai",
            "exception_type": "ProviderStatusError",
            "status_code": 429,
        }
    }


def test_query_provider_exception_envelope_does_not_copy_alias_code_or_status():
    class ProviderAliasError(Exception):
        error_code = "rate_limit_exceeded"
        status = 429
        http_status = 429

    class FakeLLM:
        async def query(self, inputs, **kwargs):
            _ = inputs, kwargs
            raise ProviderAliasError("provider alias attrs")

    client = _make_client()
    with patch.object(model_helpers, "get_registry_model", return_value=FakeLLM()):
        resp = client.post(
            "/query",
            json={
                "model": "openai/gpt-4o",
                "inputs": [{"kind": "text", "text": "hi"}],
            },
            headers=HEADERS,
        )

    assert resp.status_code == 200
    assert resp.json() == {
        "error": {
            "type": "ProviderError",
            "message": "provider alias attrs",
            "provider": "openai",
            "exception_type": "ProviderAliasError",
        }
    }


@pytest.mark.parametrize(
    ("exc_cls", "expected_message"),
    [
        pytest.param(
            "ModelNoOutputError",
            "Model failed to produce any output. This may indicate an issue with the model or input.",
            id="model-no-output",
        ),
        pytest.param(
            "MaxContextWindowExceededError",
            "Context window exceeded the maximum allowed context window. Consider reducing the context window size.",
            id="context-window",
        ),
        pytest.param(
            "MaxOutputTokensExceededError",
            "Output exceeded max tokens limit and model produced no useful content.",
            id="max-output-tokens",
        ),
        pytest.param(
            "ContentFilterError",
            "Model's content filter triggered",
            id="content-filter",
        ),
    ],
)
def test_query_provider_exception_envelope_redacts_raw_finish_reason_context(
    exc_cls: str, expected_message: str
):
    from model_library import exceptions

    exception_type = getattr(exceptions, exc_cls)

    class FakeLLM:
        async def query(self, inputs, **kwargs):
            _ = inputs, kwargs
            raise exception_type(
                "{'finish_reason': 'stop', 'response': 'SECRET_MODEL_OUTPUT_SHOULD_NOT_LEAK'}"
            )

    client = _make_client()
    with (
        patch.object(model_helpers, "get_registry_model", return_value=FakeLLM()),
        patch.object(gateway_app.telemetry, "record_exception") as record_exception,
    ):
        resp = client.post(
            "/query",
            json={
                "model": "openai/gpt-4o",
                "inputs": [{"kind": "text", "text": "hi"}],
            },
            headers=HEADERS,
        )

    assert resp.status_code == 200
    assert resp.json() == {
        "error": {
            "type": "ProviderError",
            "message": expected_message,
            "provider": "openai",
            "exception_type": exc_cls,
        }
    }
    assert "SECRET_MODEL_OUTPUT_SHOULD_NOT_LEAK" not in resp.text
    recorded_exc = record_exception.call_args.args[0]
    assert "SECRET_MODEL_OUTPUT_SHOULD_NOT_LEAK" not in str(recorded_exc)


def test_query_returns_invalid_structured_output_provider_exception_envelope():
    from model_library.exceptions import InvalidStructuredOutputError

    secret_output = "SECRET_MODEL_OUTPUT_SHOULD_NOT_LEAK"

    class FakeLLM:
        async def query(self, inputs, **kwargs):
            raise InvalidStructuredOutputError(
                f"Model produced invalid structured output: output={secret_output!r}"
            )

    client = _make_client()
    with (
        patch.object(model_helpers, "get_registry_model", return_value=FakeLLM()),
        patch.object(gateway_app.telemetry, "record_exception") as record_exception,
    ):
        resp = client.post(
            "/query",
            json={
                "model": "openai/gpt-4o",
                "inputs": [{"kind": "text", "text": "hi"}],
                "output_schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            },
            headers=HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body == {
        "error": {
            "type": "ProviderError",
            "message": InvalidStructuredOutputError.DEFAULT_MESSAGE,
            "provider": "openai",
            "exception_type": "InvalidStructuredOutputError",
        }
    }
    assert secret_output not in resp.text
    captured_exc = record_exception.call_args.args[0]
    assert secret_output not in str(captured_exc)
    error_attrs = record_exception.call_args.args[1]
    assert error_attrs["gateway.error.code"] == "provider_error"
    assert error_attrs["gateway.error.provider"] == "openai"
    assert (
        error_attrs["gateway.provider_error.exception_type"]
        == "InvalidStructuredOutputError"
    )
    assert error_attrs["http.response.status_code"] == 200
    assert "gateway.provider_error.status_code" not in error_attrs


def test_query_provider_config_for_unknown_provider_returns_invalid_model():
    client = _make_client()

    resp = client.post(
        "/query",
        json={
            "model": "unknown-provider/new-model",
            "inputs": [{"kind": "text", "text": "hi"}],
            "config": {"provider_config": {"stream_completions": False}},
        },
        headers=HEADERS,
    )

    assert resp.status_code == 400
    assert resp.json()["code"] == "invalid_model"
    assert "unknown-provider/new-model" in resp.json()["message"]


def test_query_returns_unsupported_structured_output_provider_exception_envelope():

    class FakeLLM:
        async def query(self, inputs, **kwargs):
            raise Exception("openai/gpt-4o does not support structured outputs")

    client = _make_client()
    with patch.object(model_helpers, "get_registry_model", return_value=FakeLLM()):
        resp = client.post(
            "/query",
            json={
                "model": "openai/gpt-4o",
                "inputs": [{"kind": "text", "text": "hi"}],
                "output_schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            },
            headers=HEADERS,
        )

    assert resp.status_code == 200
    assert resp.json() == {
        "error": {
            "type": "ProviderError",
            "message": "openai/gpt-4o does not support structured outputs",
            "provider": "openai",
            "exception_type": "Exception",
        }
    }


def test_query_provider_error_returns_200_error_envelope_with_searchable_phase():

    class FakeLLM:
        async def query(self, inputs, **kwargs):
            raise RuntimeError("OpenAI rate limit")

    client = _make_client()
    with (
        patch.object(model_helpers, "get_registry_model", return_value=FakeLLM()),
        patch.object(gateway_app.telemetry, "record_exception") as record_exception,
    ):
        resp = client.post(
            "/query",
            json={
                "model": "openai/gpt-4o",
                "inputs": [{"kind": "text", "text": "hi"}],
                "run_id": "run-a",
                "question_id": "q-a",
            },
            headers=HEADERS,
        )

    assert resp.status_code == 200
    assert resp.json() == {
        "error": {
            "type": "ProviderError",
            "message": "OpenAI rate limit",
            "provider": "openai",
            "exception_type": "RuntimeError",
        }
    }
    record_exception.assert_called_once()
    captured_exc = record_exception.call_args.args[0]
    assert str(captured_exc) == "Provider call failed"
    assert "OpenAI rate limit" not in str(captured_exc)
    error_attrs = record_exception.call_args.args[1]
    assert error_attrs["gateway.error.code"] == "provider_error"
    assert error_attrs["gateway.error.phase"] == "provider_call"
    assert error_attrs["gateway.error.provider"] == "openai"
    assert error_attrs["gateway.provider_error.exception_type"] == "RuntimeError"
    assert error_attrs["http.response.status_code"] == 200
    assert "gateway.provider_error.status_code" not in error_attrs


def test_query_rejects_custom_endpoint_without_custom_api_key():
    client = _make_client()
    resp = client.post(
        "/query",
        json={
            "model": "openai/gpt-4o",
            "inputs": [{"kind": "text", "text": "hi"}],
            "config": {"custom_endpoint": "https://provider.test/v1"},
        },
        headers=HEADERS,
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "custom_key_rejected"


def test_query_returns_429_when_gateway_capacity_is_full(capsys):
    from model_gateway import capacity, metrics
    from model_gateway.capacity import GatewayCapacityLimiter

    metrics.flush_metrics()
    capsys.readouterr()

    class FakeLLM:
        async def query(self, inputs, **kwargs):
            raise AssertionError("capacity rejection should happen before query")

    client = _make_client()
    limiter = GatewayCapacityLimiter(
        max_active=1,
        max_queued=0,
        queue_timeout_seconds=1,
        request_timeout_seconds=1,
    )
    limiter._active = 1  # pyright: ignore[reportPrivateUsage]
    cast(Any, client.app).state.capacity_limiter = limiter

    telemetry_attrs: list[dict[str, object | None]] = []
    with (
        patch.object(model_helpers, "get_registry_model", return_value=FakeLLM()),
        patch.object(capacity.telemetry, "is_recording", return_value=True),
        patch.object(
            capacity.telemetry,
            "set_attributes",
            side_effect=lambda attrs: telemetry_attrs.append(dict(attrs)),
        ),
        patch.object(capacity.telemetry, "record_exception") as record_exception,
    ):
        resp = client.post(
            "/query",
            json={
                "model": "openai/gpt-4o",
                "inputs": [{"kind": "text", "text": "hi"}],
                "run_id": "run-a",
                "question_id": "q-a",
                "query_id": "query-a",
            },
            headers=HEADERS,
        )

    assert resp.status_code == 429
    assert resp.json()["code"] == "gateway_overloaded"
    identity_attrs = telemetry_attrs[0]
    assert identity_attrs["run_id"] == "run-a"
    assert identity_attrs["question_id"] == "q-a"
    assert identity_attrs["query_id"] == "query-a"
    assert identity_attrs["gen_ai.request.model"] == "openai/gpt-4o"
    assert identity_attrs["gateway.operation"] == "query"
    error_attrs = next(
        attrs for attrs in telemetry_attrs if "gateway.error.code" in attrs
    )
    assert error_attrs["gateway.error.code"] == "gateway_overloaded"
    assert error_attrs["gateway.error.phase"] == "capacity"
    assert error_attrs["http.status_code"] == 429
    record_exception.assert_called_once()
    assert (
        record_exception.call_args.args[1]["gateway.error.code"] == "gateway_overloaded"
    )

    metrics.flush_metrics()
    payloads = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert any(
        payload.get("Route") == "/query"
        and payload.get("StatusCode") == "429"
        and payload.get("HttpRequestCount") == 1
        for payload in payloads
    )


def test_query_metrics_param_group_uses_bounded_config_and_request_context(capsys):
    from model_library.base.output import QueryResult
    from model_gateway import metrics

    class FakeLLM:
        async def query(self, inputs: Any, **kwargs: Any):
            return QueryResult(output_text="ok", history=cast(Any, inputs))

    client = _make_client()
    with patch.object(model_helpers, "get_registry_model", return_value=FakeLLM()):
        for question_id in ("question-a", "question-b"):
            resp = client.post(
                "/query",
                json={
                    "model": "openai/gpt-4o",
                    "inputs": [{"kind": "text", "text": "hi"}],
                    "config": {"max_tokens": 7, "temperature": 0},
                    "question_id": question_id,
                },
                headers=HEADERS,
            )
            assert resp.status_code == 200

    metrics.flush_metrics()
    payloads = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    model_payloads = [
        payload
        for payload in payloads
        if payload.get("Operation") == "query"
        and payload.get("Model") == "openai/gpt-4o"
        and "ParamGroup" in payload
    ]
    assert len(model_payloads) == 1
    assert model_payloads[0]["ModelRequestCount"] == 2


def test_query_rejects_extra_params():
    client = _make_client()

    resp = client.post(
        "/query",
        json={
            "model": "openai/gpt-4o",
            "inputs": [{"kind": "text", "text": "hi"}],
            "extra_params": {
                "temperature": 0,
                "request_id": "request-a",
            },
        },
        headers=HEADERS,
    )

    assert resp.status_code == 400
    assert resp.json()["code"] == "invalid_request"
    assert "request_id" in resp.json()["message"]


def test_query_does_not_compute_config_fingerprint_when_otel_is_not_recording():
    from model_library.base.output import QueryResult

    class FakeLLM:
        async def query(self, inputs, **kwargs):
            return QueryResult(output_text="ok", history=inputs)

    client = _make_client()
    with (
        patch.object(model_helpers, "get_registry_model", return_value=FakeLLM()),
        patch.object(gateway_app.telemetry, "is_recording", return_value=False),
        patch.object(
            gateway_app.telemetry,
            "config_fingerprint",
            side_effect=AssertionError("config fingerprint should be skipped"),
        ),
    ):
        resp = client.post(
            "/query",
            json={
                "model": "openai/gpt-4o",
                "inputs": [{"kind": "text", "text": "hi"}],
                "run_id": "run-a",
                "question_id": "q-a",
            },
            headers=HEADERS,
        )

    assert resp.status_code == 200


def test_query_telemetry_response_capture_does_not_break_success():
    from model_library.base.output import QueryResult

    class UnserializableQueryResult(QueryResult):
        def model_dump(self, *args: object, **kwargs: object) -> dict[str, object]:
            raise TypeError(
                "'MockValSer' object is not an instance of 'SchemaSerializer'"
            )

    class FakeLLM:
        async def query(self, inputs: Any, **kwargs: Any):
            return UnserializableQueryResult(
                output_text="ok", history=cast(Any, inputs)
            )

    client = _make_client()
    with (
        patch.object(model_helpers, "get_registry_model", return_value=FakeLLM()),
        patch.object(gateway_app.telemetry, "is_recording", return_value=True),
    ):
        resp = client.post(
            "/query",
            json={
                "model": "openai/gpt-4o",
                "inputs": [{"kind": "text", "text": "hi"}],
                "run_id": "run-a",
                "question_id": "q-a",
            },
            headers=HEADERS,
        )

    assert resp.status_code == 200
    assert resp.json()["output_text"] == "ok"


def _otel_attr_value(attr: Any) -> object | None:
    for field in ("string_value", "bool_value", "int_value", "double_value"):
        if attr.value.HasField(field):
            return getattr(attr.value, field)
    return None


def _otel_attrs(attributes: Any) -> dict[str, object]:
    result: dict[str, object] = {}
    for attr in attributes:
        value = _otel_attr_value(attr)
        if value is not None:
            result[attr.key] = value
    return result


def _decode_otlp_payloads(
    bodies: list[bytes],
) -> tuple[list[dict[str, object]], list[tuple[str, dict[str, object]]]]:
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest,
    )

    span_attrs: list[dict[str, object]] = []
    event_attrs: list[tuple[str, dict[str, object]]] = []
    for body in bodies:
        request = ExportTraceServiceRequest()
        request.ParseFromString(body)
        for resource_span in request.resource_spans:
            for scope_span in resource_span.scope_spans:
                for span in scope_span.spans:
                    span_attrs.append(_otel_attrs(span.attributes))
                    event_attrs.extend(
                        (event.name, _otel_attrs(event.attributes))
                        for event in span.events
                    )
    return span_attrs, event_attrs


def test_query_enabled_otel_exports_config_hash_and_redacted_lookup():
    from model_library import telemetry
    from model_library.base.output import QueryResult
    from model_gateway import main

    class ServerSettings:
        MODEL_GATEWAY_API_KEYS = "sk-test"
        MODEL_GATEWAY_HMAC_SECRET = "test-secret"

        def get(self, name: str, default: str = "") -> str:
            return getattr(self, name, default)

        def unset(self, key: str):
            pass

    seen_kwargs: dict[str, Any] = {}

    class FakeLLM:
        async def query(self, inputs: Any, **kwargs: Any):
            seen_kwargs.update(kwargs)
            return QueryResult(output_text="ok", history=cast(Any, inputs))

    bodies: list[bytes] = []

    class OtlpHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            bodies.append(self.rfile.read(length))
            self.send_response(200)
            self.end_headers()

        def log_message(self, format: str, *args: Any) -> None:
            pass

    telemetry.shutdown_telemetry()
    server = HTTPServer(("127.0.0.1", 0), OtlpHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with (
            patch.dict(
                os.environ,
                {
                    "GATEWAY_OTEL_ENABLED": "true",
                    "SENTRY_DSN": "https://public@example.com/1",
                    "SENTRY_OTLP_COLLECTOR_URL": f"http://127.0.0.1:{server.server_port}",
                    "OTEL_TRACES_SAMPLER": "always_on",
                },
                clear=True,
            ),
            patch.object(gateway_app, "model_library_settings", ServerSettings()),
            patch.object(
                gateway_app, "get_model_names", return_value=["openai/gpt-4o"]
            ),
            patch.object(model_helpers, "get_registry_model", return_value=FakeLLM()),
        ):
            app = main.create_app()
            with TestClient(app) as client:
                resp = client.post(
                    "/query",
                    json={
                        "model": "openai/gpt-4o",
                        "inputs": [{"kind": "text", "text": "hi"}],
                        "config": {
                            "max_tokens": 7,
                            "temperature": 0,
                            "provider_config": {"prompt_cache_retention": "24h"},
                        },
                        "run_id": "run-a",
                        "question_id": "question-a",
                        "query_id": "query-a",
                        "identity": {
                            "email": "user@example.com",
                            "benchmark_name": "swebench",
                            "agent_name": "swe-agent",
                        },
                    },
                    headers=HEADERS,
                )
        telemetry.shutdown_telemetry()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        telemetry.shutdown_telemetry()

    assert resp.status_code == 200
    assert seen_kwargs["query_id"] == "query-a"
    assert "identity" not in seen_kwargs
    assert bodies
    assert b"gateway.request_json" not in b"".join(bodies)
    assert b"gateway.response_json" not in b"".join(bodies)

    span_attrs, event_attrs = _decode_otlp_payloads(bodies)
    config_hashes = [
        str(attrs["model.config_hash"])
        for attrs in span_attrs
        if "model.config_hash" in attrs
    ]
    query_ids = [str(attrs["query_id"]) for attrs in span_attrs if "query_id" in attrs]
    identities = [str(attrs["identity"]) for attrs in span_attrs if "identity" in attrs]
    mode_attrs = [
        (key, str(attrs[key]))
        for attrs in span_attrs
        for key in {
            "gateway.retry_queue.mode",
            "gateway.output_schema.mode",
            "retry_queue.mode",
        }
        if key in attrs
    ]
    llm_config_attrs = [
        (key, attrs[key])
        for attrs in span_attrs
        for key in {
            "llm.config.max_tokens",
            "llm.config.temperature",
            "llm.config.provider_config.prompt_cache_retention",
        }
        if key in attrs
    ]
    param_groups = [
        str(attrs["model.param_group"])
        for attrs in span_attrs
        if "model.param_group" in attrs
    ]
    config_seen_payloads = [
        attrs for name, attrs in event_attrs if name == "model.config_seen"
    ]

    assert config_hashes
    assert seen_kwargs["query_id"] in query_ids
    assert (
        '{"agent_name":"swe-agent","benchmark_name":"swebench","email":"user@example.com"}'
        in identities
    )
    assert ("gateway.retry_queue.mode", "disabled") in mode_attrs
    assert ("gateway.output_schema.mode", "disabled") in mode_attrs
    assert ("retry_queue.mode", "disabled") in mode_attrs
    assert ("llm.config.max_tokens", "7") in llm_config_attrs
    assert ("llm.config.temperature", "0.0") in llm_config_attrs
    assert param_groups and all(group != "none" for group in param_groups)
    assert config_seen_payloads
    payload = json.loads(str(config_seen_payloads[-1]["model.config_redacted_json"]))
    assert payload["config"]["provider_config"]["prompt_cache_retention"] == "24h"
    assert payload["params"] == {
        "max_tokens": 7,
        "temperature": 0,
        "provider_config": {"prompt_cache_retention": "24h"},
    }
    assert config_seen_payloads[-1]["model.config_redacted_json_truncated"] is False


def test_query_forwards_custom_endpoint_with_custom_api_key():
    from model_library.base import LLMConfig
    from model_library.base.output import QueryResult

    seen: dict[str, object] = {}

    class FakeLLM:
        async def query(self, inputs, **kwargs):
            return QueryResult(output_text="ok", history=inputs)

    def fake_get_registry_model(model, config):
        seen["config"] = config
        return FakeLLM()

    client = _make_client()
    with patch.object(
        model_helpers, "get_registry_model", side_effect=fake_get_registry_model
    ):
        resp = client.post(
            "/query",
            json={
                "model": "openai/gpt-4o",
                "inputs": [{"kind": "text", "text": "hi"}],
                "config": {
                    "custom_endpoint": "https://provider.test/v1",
                    "custom_api_key": "sk-provider",
                },
            },
            headers=HEADERS,
        )

    assert resp.status_code == 200
    config = cast(LLMConfig, seen["config"])
    assert config.custom_endpoint == "https://provider.test/v1"
    assert config.custom_api_key is not None
    assert config.custom_api_key.get_secret_value() == "sk-provider"


def test_models_returns_list():
    client = _make_client()
    resp = client.get("/models", headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "id" in data[0]


def test_upload_file_success_uses_config_and_returns_file_id():
    from model_library.base.input import FileWithId

    seen: dict[str, object] = {}

    class FakeLLM:
        async def upload_file(
            self,
            name: str,
            mime: str,
            file_bytes: io.BytesIO,
            type: Literal["image", "file"] = "file",
        ) -> FileWithId:
            seen["name"] = name
            seen["mime"] = mime
            seen["bytes"] = file_bytes.getvalue()
            seen["type"] = type
            return FileWithId(
                type=type,
                name=name,
                mime=mime,
                file_id="file-test",
            )

    def fake_get_registry_model(model: str, config: Any) -> FakeLLM:
        seen["model"] = model
        seen["max_tokens"] = config.max_tokens
        return FakeLLM()

    client = _make_client()
    set_attributes = MagicMock()
    start_span = MagicMock(return_value=nullcontext())
    with (
        patch.object(
            model_helpers, "get_registry_model", side_effect=fake_get_registry_model
        ),
        patch.object(gateway_app.telemetry, "is_recording", return_value=True),
        patch.object(gateway_app.telemetry, "set_attributes", set_attributes),
        patch.object(gateway_app.telemetry, "start_span", start_span),
    ):
        resp = client.post(
            "/files/upload",
            json={
                "model": "openai/gpt-4o",
                "name": "doc.pdf",
                "mime": "application/pdf",
                "content_base64": base64.b64encode(b"pdf bytes").decode(),
                "type": "file",
                "config": {"max_tokens": 7},
            },
            headers=HEADERS,
        )

    assert resp.status_code == 200
    assert resp.json() == {
        "file": {
            "kind": "file_base",
            "type": "file",
            "name": "doc.pdf",
            "mime": "application/pdf",
            "append_type": "file_id",
            "file_id": "file-test",
        }
    }
    assert seen == {
        "model": "openai/gpt-4o",
        "max_tokens": 7,
        "name": "doc.pdf",
        "mime": "application/pdf",
        "bytes": b"pdf bytes",
        "type": "file",
    }
    upload_attrs = set_attributes.call_args_list[0].args[0]
    assert upload_attrs["gateway.operation"] == "files_upload"
    assert upload_attrs["gen_ai.request.model"] == "openai/gpt-4o"
    assert "gateway.request_json" not in upload_attrs
    provider_span_attrs = start_span.call_args.args[1]
    assert start_span.call_args.args[0] == "gateway.files_upload.provider_call"
    assert provider_span_attrs["gateway.operation"] == "files_upload"
    assert provider_span_attrs["gen_ai.request.model"] == "openai/gpt-4o"
    assert provider_span_attrs["model.provider"] == "openai"
    assert provider_span_attrs["model.name"] == "gpt-4o"
    assert provider_span_attrs["model.provider_endpoint"] == "default"
    assert provider_span_attrs["model.param_group"] != "none"
    assert start_span.call_args.kwargs == {"kind": "client"}


def test_upload_file_invalid_base64_returns_client_error():

    client = _make_client()
    with patch.object(gateway_app.telemetry, "record_exception") as record_exception:
        resp = client.post(
            "/files/upload",
            json={
                "model": "openai/gpt-4o",
                "name": "doc.pdf",
                "mime": "application/pdf",
                "content_base64": "not base64!",
                "type": "file",
                "config": {},
            },
            headers=HEADERS,
        )

    assert resp.status_code == 400
    assert resp.json()["code"] == "invalid_request"
    record_exception.assert_called_once()
    error_attrs = record_exception.call_args.args[1]
    assert error_attrs["gateway.error.code"] == "invalid_request"
    assert error_attrs["gateway.error.phase"] == "files_upload"
    assert error_attrs["gateway.status_code"] == 400
    assert error_attrs["http.response.status_code"] == 400


@pytest.mark.parametrize(
    ("endpoint", "request_body", "expected_message"),
    [
        (
            "/files/upload",
            {
                "model": "openai/gpt-4o",
                "name": "doc.pdf",
                "mime": "application/pdf",
                "content_base64": base64.b64encode(b"pdf bytes").decode(),
                "type": "file",
                "config": {},
            },
            "OpenAI file upload failed",
        ),
        (
            "/tokens/count",
            {
                "model": "openai/gpt-4o",
                "inputs": [{"kind": "text", "text": "hi"}],
                "tools": [],
                "config": {},
            },
            "OpenAI token count failed",
        ),
        (
            "/embeddings",
            {
                "model": "openai/gpt-4o",
                "text": "embed this",
                "embedding_model": "text-embedding-3-large",
                "config": {},
            },
            "OpenAI embedding failed",
        ),
        (
            "/moderation",
            {
                "model": "openai/moderation",
                "text": "check this",
                "config": {},
            },
            "OpenAI moderation failed",
        ),
    ],
)
def test_provider_operation_provider_errors_return_200_error_envelope(
    endpoint: str, request_body: dict[str, object], expected_message: str
):
    class FakeLLM:
        async def upload_file(
            self,
            name: str,
            mime: str,
            bytes: io.BytesIO,
            type: Literal["image", "file"] = "file",
        ):
            raise RuntimeError("OpenAI file upload failed")

        async def count_tokens(self, inputs, *, tools, **kwargs) -> int:
            raise RuntimeError("OpenAI token count failed")

        async def get_embedding(
            self, text: str, model: str = "text-embedding-3-small"
        ) -> list[float]:
            raise RuntimeError("OpenAI embedding failed")

        async def moderate_content(self, text: str) -> dict[str, Any]:
            raise RuntimeError("OpenAI moderation failed")

    client = _make_client()
    with (
        patch.object(model_helpers, "get_registry_model", return_value=FakeLLM()),
        patch.object(gateway_app.telemetry, "record_exception") as record_exception,
    ):
        resp = client.post(endpoint, json=request_body, headers=HEADERS)

    assert resp.status_code == 200
    assert resp.json() == {
        "error": {
            "type": "ProviderError",
            "message": expected_message,
            "provider": "openai",
            "exception_type": "RuntimeError",
        }
    }
    record_exception.assert_called_once()
    error_attrs = record_exception.call_args.args[1]
    assert error_attrs["gateway.error.phase"] == "provider_call"
    assert error_attrs["gateway.error.code"] == "provider_error"
    assert error_attrs["gateway.provider_error.exception_type"] == "RuntimeError"


def test_embeddings_success_uses_config_and_returns_embedding():

    seen: dict[str, object] = {}

    class FakeLLM:
        async def get_embedding(
            self, text: str, model: str = "text-embedding-3-small"
        ) -> list[float]:
            seen["text"] = text
            seen["embedding_model"] = model
            return [0.1, 0.2, 0.3]

    def fake_get_registry_model(model: str, config: Any) -> FakeLLM:
        seen["model"] = model
        seen["max_tokens"] = config.max_tokens
        return FakeLLM()

    client = _make_client()
    set_attributes = MagicMock()
    start_span = MagicMock(return_value=nullcontext())
    with (
        patch.object(
            model_helpers, "get_registry_model", side_effect=fake_get_registry_model
        ),
        patch.object(gateway_app.telemetry, "is_recording", return_value=True),
        patch.object(gateway_app.telemetry, "set_attributes", set_attributes),
        patch.object(gateway_app.telemetry, "start_span", start_span),
    ):
        resp = client.post(
            "/embeddings",
            json={
                "model": "openai/gpt-4o",
                "text": "embed this",
                "embedding_model": "text-embedding-3-large",
                "config": {"max_tokens": 7},
            },
            headers=HEADERS,
        )

    assert resp.status_code == 200
    assert resp.json() == {"embedding": [0.1, 0.2, 0.3]}
    assert seen == {
        "model": "openai/gpt-4o",
        "max_tokens": 7,
        "text": "embed this",
        "embedding_model": "text-embedding-3-large",
    }
    embedding_attrs = set_attributes.call_args_list[0].args[0]
    assert embedding_attrs["gateway.operation"] == "embeddings"
    assert embedding_attrs["gen_ai.request.model"] == "openai/gpt-4o"
    assert embedding_attrs["gateway.embedding.model"] == "text-embedding-3-large"
    assert "gateway.request_json" not in embedding_attrs
    provider_span_attrs = start_span.call_args.args[1]
    assert start_span.call_args.args[0] == "gateway.embeddings.provider_call"
    assert provider_span_attrs["gateway.operation"] == "embeddings"
    assert provider_span_attrs["gen_ai.request.model"] == "openai/gpt-4o"
    assert provider_span_attrs["model.provider"] == "openai"
    assert provider_span_attrs["model.name"] == "gpt-4o"
    assert provider_span_attrs["model.provider_endpoint"] == "default"
    assert provider_span_attrs["model.param_group"] != "none"
    assert start_span.call_args.kwargs == {"kind": "client"}


def test_moderation_success_uses_config_and_returns_response():

    seen: dict[str, object] = {}

    class FakeLLM:
        async def moderate_content(self, text: str) -> dict[str, Any]:
            seen["text"] = text
            return {"id": "modr-test", "results": [{"flagged": False}]}

    def fake_get_registry_model(model: str, config: Any) -> FakeLLM:
        seen["model"] = model
        seen["max_tokens"] = config.max_tokens
        return FakeLLM()

    client = _make_client()
    set_attributes = MagicMock()
    start_span = MagicMock(return_value=nullcontext())
    with (
        patch.object(
            model_helpers, "get_registry_model", side_effect=fake_get_registry_model
        ),
        patch.object(gateway_app.telemetry, "is_recording", return_value=True),
        patch.object(gateway_app.telemetry, "set_attributes", set_attributes),
        patch.object(gateway_app.telemetry, "start_span", start_span),
    ):
        resp = client.post(
            "/moderation",
            json={
                "model": "openai/moderation",
                "text": "check this",
                "config": {"max_tokens": 7},
            },
            headers=HEADERS,
        )

    assert resp.status_code == 200
    assert resp.json() == {
        "response": {"id": "modr-test", "results": [{"flagged": False}]}
    }
    assert seen == {
        "model": "openai/moderation",
        "max_tokens": 7,
        "text": "check this",
    }
    moderation_attrs = set_attributes.call_args_list[0].args[0]
    assert moderation_attrs["gateway.operation"] == "moderation"
    assert moderation_attrs["gen_ai.request.model"] == "openai/moderation"
    assert "gateway.request_json" not in moderation_attrs
    assert not any(
        "gateway.response_json" in call.args[0]
        for call in set_attributes.call_args_list
    )
    provider_span_attrs = start_span.call_args.args[1]
    assert start_span.call_args.args[0] == "gateway.moderation.provider_call"
    assert provider_span_attrs["gateway.operation"] == "moderation"
    assert provider_span_attrs["gen_ai.request.model"] == "openai/moderation"
    assert provider_span_attrs["model.provider"] == "openai"
    assert provider_span_attrs["model.name"] == "moderation"
    assert provider_span_attrs["model.provider_endpoint"] == "default"
    assert provider_span_attrs["model.param_group"] != "none"
    assert start_span.call_args.kwargs == {"kind": "client"}


def test_query_initializes_token_retry_on_server_side_and_reuses_token_model():
    from model_library.base.output import QueryResult

    created_models: list[Any] = []

    class FakeLLM:
        def __init__(self):
            self.token_retry_params = None
            self.init_count = 0
            self.query_count = 0

        async def init_token_retry(self, token_retry_params):
            self.token_retry_params = token_retry_params
            self.init_count += 1

        async def query(self, inputs, **kwargs):
            self.query_count += 1
            return QueryResult(output_text="pong", history=inputs)

    def fake_get_registry_model(model, config):
        llm = FakeLLM()
        created_models.append(llm)
        return llm

    client = _make_client()
    body = {
        "model": "openai/gpt-4o",
        "inputs": [{"kind": "text", "text": "hi"}],
        "token_retry_params": {
            "input_modifier": 1,
            "output_modifier": 2,
            "use_dynamic_estimate": False,
            "limit": 1000,
            "limit_refresh_seconds": 60,
        },
        "run_id": "run",
        "question_id": "question",
    }
    with patch.object(
        model_helpers, "get_registry_model", side_effect=fake_get_registry_model
    ):
        first = client.post("/query", json=body, headers=HEADERS)
        second = client.post("/query", json=body, headers=HEADERS)

    assert first.status_code == 200
    assert second.status_code == 200
    assert len(created_models) == 1
    assert created_models[0].init_count == 1
    assert created_models[0].query_count == 2
    assert created_models[0].token_retry_params.limit == 1000
    assert created_models[0].token_retry_params.output_modifier == 2
    assert created_models[0].token_retry_params.use_dynamic_estimate is False


def test_query_rejects_malformed_raw_history_with_400():

    client = _make_client()
    with patch.object(gateway_app.telemetry, "record_exception") as record_exception:
        resp = client.post(
            "/query",
            json={
                "model": "openai/gpt-4o",
                "inputs": [
                    {
                        "kind": "raw_input",
                        "input": {"messages": [{"role": "user", "content": "hi"}]},
                    }
                ],
            },
            headers=HEADERS,
        )
    assert resp.status_code == 400
    assert resp.json()["code"] == "hmac_verification_failed"
    record_exception.assert_called_once()
    error_attrs = record_exception.call_args.args[1]
    assert error_attrs["gateway.error.code"] == "hmac_verification_failed"
    assert error_attrs["gateway.error.phase"] == "restore_history"
    assert error_attrs["http.response.status_code"] == 400


def test_query_marks_sign_history_phase_when_signing_fails():
    from model_library.base.output import QueryResult, QueryResultMetadata

    class FakeLLM:
        async def query(self, inputs, **kwargs):
            return QueryResult(
                output_text="ok",
                history=[],
                metadata=QueryResultMetadata(in_tokens=1, out_tokens=2),
            )

    client = _make_client()
    with (
        patch.object(model_helpers, "get_registry_model", return_value=FakeLLM()),
        patch(
            "model_gateway.routes.query.sign_history", side_effect=RuntimeError("boom")
        ),
        patch.object(gateway_app.telemetry, "record_exception") as record_exception,
    ):
        resp = client.post(
            "/query",
            json={
                "model": "openai/gpt-4o",
                "inputs": [{"kind": "text", "text": "hi"}],
                "run_id": "run-a",
                "question_id": "q-a",
            },
            headers=HEADERS,
        )

    assert resp.status_code == 500
    assert resp.json()["code"] == "internal_error"
    record_exception.assert_called_once()
    error_attrs = record_exception.call_args.args[1]
    assert error_attrs["gateway.error.code"] == "internal_error"
    assert error_attrs["gateway.error.phase"] == "sign_history"
    assert error_attrs["http.response.status_code"] == 500


def test_query_sign_history_failure_does_not_write_usage_ledger():
    from model_library.base.output import QueryResult, QueryResultMetadata

    class FakeUsageLedger:
        enabled = True

        def __init__(self) -> None:
            self.events: list[dict[str, object]] = []

        async def start(self) -> None:
            return

        async def close(self) -> None:
            return

        async def write_success(self, event: dict[str, object]) -> None:
            self.events.append(event)

    class FakeLLM:
        async def query(self, inputs, **kwargs):
            return QueryResult(
                output_text="ok",
                history=[],
                metadata=QueryResultMetadata(in_tokens=1, out_tokens=2),
            )

    fake_ledger = FakeUsageLedger()
    with (
        patch.object(
            gateway_app, "create_usage_ledger_from_env", return_value=fake_ledger
        ),
        patch.object(model_helpers, "get_registry_model", return_value=FakeLLM()),
        patch(
            "model_gateway.routes.query.sign_history", side_effect=RuntimeError("boom")
        ),
    ):
        client = _make_client()
        resp = client.post(
            "/query",
            json={
                "model": "openai/gpt-4o",
                "inputs": [{"kind": "text", "text": "hi"}],
            },
            headers=HEADERS,
        )

    assert resp.status_code == 500
    assert resp.json()["code"] == "internal_error"
    assert fake_ledger.events == []


def test_query_success_uses_config_restores_raw_history_and_returns_metadata():
    from model_library.base.base import LLM
    from model_library.base.input import RawInput, TextInput
    from model_library.base.output import (
        FinishReason,
        FinishReasonInfo,
        QueryResult,
        QueryResultExtras,
        QueryResultMetadata,
    )

    seen: dict[str, object] = {}
    signed_inputs = LLM.serialize_input(
        [RawInput(input={"messages": [{"role": "user", "content": "hi"}]})],
        secret=b"test-secret",
    )

    class FakeLLM:
        async def query(self, inputs, **kwargs):
            seen["inputs"] = inputs
            seen["kwargs"] = kwargs
            return QueryResult(
                output_text="pong",
                history=[*inputs, TextInput(text="pong")],
                metadata=QueryResultMetadata(in_tokens=1, out_tokens=2),
                finish_reason=FinishReasonInfo(reason=FinishReason.STOP, raw="stop"),
                extras=QueryResultExtras(search_results=[{"title": "doc"}]),
            )

    def fake_get_registry_model(model, config):
        seen["model"] = model
        seen["max_tokens"] = config.max_tokens
        return FakeLLM()

    client = _make_client()
    with patch.object(
        model_helpers, "get_registry_model", side_effect=fake_get_registry_model
    ):
        resp = client.post(
            "/query",
            json={
                "model": "openai/gpt-4o",
                "inputs": json.loads(signed_inputs),
                "config": {"max_tokens": 7},
            },
            headers=HEADERS,
        )

    assert resp.status_code == 200
    data = cast(dict[str, Any], resp.json())
    assert data["output_text"] == "pong"
    assert data["metadata"]["in_tokens"] == 1
    assert data["metadata"]["out_tokens"] == 2
    assert data["finish_reason"] == {"reason": "stop", "raw": "stop"}
    assert data["extras"]["search_results"] == [{"title": "doc"}]
    assert "raw" not in data
    assert seen["model"] == "openai/gpt-4o"
    assert seen["max_tokens"] == 7
    seen_inputs = cast(list[object], seen["inputs"])
    assert isinstance(seen_inputs[0], RawInput)
    assert seen_inputs[0].input == {"messages": [{"role": "user", "content": "hi"}]}

    restored_history = LLM.deserialize_input(
        data["signed_history"], secret=b"test-secret"
    )
    assert isinstance(restored_history[0], RawInput)
    assert isinstance(restored_history[1], TextInput)
