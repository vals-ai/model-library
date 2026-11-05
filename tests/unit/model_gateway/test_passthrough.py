"""Provider pass-through authorization, forwarding, and usage recording."""

import asyncio
import gzip
import hashlib
import json
from decimal import Decimal
import logging
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID

import fakeredis.aioredis
import httpx
import pytest
import sentry_sdk
from fastapi import FastAPI
from sentry_sdk.integrations.starlette import StarletteIntegration
from sentry_sdk.transport import Transport
from starlette.testclient import WebSocketDenialResponse

import model_gateway.model_helpers as model_helpers
import model_gateway.routes.passthrough as passthrough_routes
import model_gateway.usage_ledger.schema as ledger_schema
from model_gateway.passthrough_exchanges import (
    ExchangeLog,
    ExchangeStore,
    exchange_prefix,
)
import model_library.telemetry as telemetry
from model_gateway.errors import AssignedKeyUnavailableError
from model_gateway.run_tokens import RunTokenClaims
from model_library.registry_utils import get_registry_config
from model_library.retriers.token import utils as token_utils
from tests.unit.model_gateway._support import HEADERS, _make_client

RUN_ID = "run-pass-through"
TASK_ID = "task-pass-through"
ATTEMPT_ID = "attempt-pass-through"
MODEL_KEYS = {
    "openai": "openai/gpt-4o",
    "anthropic": "anthropic/claude-sonnet-4-5-20250929",
    "google": "google/gemini-2.5-flash",
    "openrouter": "openrouter/inception/mercury-3",
}
MODELS = {
    provider: get_registry_config(key).provider_endpoint.removeprefix("models/")
    for provider, key in MODEL_KEYS.items()
}
REAL_KEYS = {
    "openai": "real-openai-key",
    "anthropic": "real-anthropic-key",
    "google": "real-google-key",
    "openrouter": "real-openrouter-key",
}
OPENROUTER_RESPONSE = (
    b'{"id":"gen-1","usage":{"prompt_tokens":10,'
    b'"prompt_tokens_details":{"cached_tokens":2},'
    b'"completion_tokens":7,"completion_tokens_details":{"reasoning_tokens":3},'
    b'"total_tokens":17,"cost":0.0000123}}'
)
OPENROUTER_STREAM = (
    b'data: {"id":"gen-1","choices":[{"delta":{"content":"hi"}}]}\n\n'
    b"data: " + OPENROUTER_RESPONSE + b"\n\n"
    b"data: [DONE]\n\n"
)
OPENAI_RESPONSE = (
    b'{"id":"resp_1","usage":{"input_tokens":10,'
    b'"input_tokens_details":{"cached_tokens":2,"cache_write_tokens":0},'
    b'"output_tokens":7,"output_tokens_details":{"reasoning_tokens":3},'
    b'"total_tokens":17}}'
)
ANTHROPIC_STREAM = (
    b"event: message_start\n"
    b'data: {"type":"message_start","message":{"usage":'
    b'{"input_tokens":17,"output_tokens":0,"cache_read_input_tokens":3}}}\n\n'
    b"event: message_delta\n"
    b'data: {"type":"message_delta","usage":{"output_tokens":9}}\n\n'
)
GOOGLE_USAGE = (
    b'{"usageMetadata":{"promptTokenCount":22,'
    b'"cachedContentTokenCount":4,"candidatesTokenCount":8,'
    b'"thoughtsTokenCount":2,"serviceTier":"standard"}}'
)
GOOGLE_STREAM = b"data: " + GOOGLE_USAGE + b"\n\n"


@pytest.fixture(autouse=True)
def redis() -> Any:
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    token_utils.set_redis_client(client)
    yield client
    token_utils.set_redis_client(None)


def _mint(
    client: Any,
    *,
    models: list[str],
    **claims: object,
) -> str:
    response = client.post(
        "/service-auth",
        headers=HEADERS,
        json={
            "run_id": RUN_ID,
            "task_id": TASK_ID,
            "attempt_id": ATTEMPT_ID,
            "allowed_models": models,
            "allowed_operations": ["passthrough"],
            "ttl_seconds": 3600,
            **claims,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["token"]


class _RecordingLedger:
    enabled = True

    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    async def write_event(self, event: dict[str, object]) -> None:
        self.events.append(event)


class _ChunkStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk

    async def aclose(self) -> None:
        return None


def _configure(
    client: Any,
    monkeypatch: pytest.MonkeyPatch,
    handler: Any,
) -> _RecordingLedger:
    settings = SimpleNamespace(
        **{f"{provider.upper()}_API_KEY": key for provider, key in REAL_KEYS.items()}
    )
    settings.get = lambda name, default=None: getattr(settings, name, default)
    monkeypatch.setattr(model_helpers, "model_library_settings", settings)
    monkeypatch.setattr(passthrough_routes, "model_library_settings", settings)
    app = client.app
    app.state.passthrough_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    )
    ledger = _RecordingLedger()
    app.state.usage_ledger = ledger
    return ledger


def _path(provider: str, model: str) -> str:
    if provider == "openai":
        return "/passthrough/openai/v1/responses"
    if provider == "anthropic":
        return "/passthrough/anthropic/v1/messages"
    if provider == "openrouter":
        return "/passthrough/openrouter/api/v1/chat/completions"
    return f"/passthrough/google/v1beta/models/{model}:generateContent"


def _provider_error_shape(provider: str, payload: dict[str, Any]) -> None:
    error = payload["error"]
    required_fields = {
        "openai": {"message", "type", "code"},
        "anthropic": {"type", "message"},
        "google": {"code", "message", "status"},
        "openrouter": {"message", "type", "code"},
    }[provider]
    assert required_fields <= set(error)
    assert provider not in ("openai", "openrouter") or set(error) == required_fields
    assert provider != "anthropic" or payload["type"] == "error"


@pytest.mark.parametrize(
    ("provider", "auth_form", "model_allowed", "expected_status"),
    [
        ("openai", "bearer", True, 200),
        ("openai", "bearer", False, 403),
        ("anthropic", "x-api-key", True, 200),
        ("anthropic", "x-api-key", False, 403),
        ("google", "x-goog-api-key", True, 200),
        ("google", "query-key", True, 401),
        ("google", "header-and-query-key", True, 200),
        ("openrouter", "bearer", True, 200),
        ("openrouter", "bearer", False, 403),
        ("openai", "static", True, 403),
        ("openai", "missing", True, 401),
    ],
)
def test_provider_auth_and_model_allowlist(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    auth_form: str,
    model_allowed: bool,
    expected_status: int,
) -> None:
    client = _make_client()
    model = MODELS[provider]
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"ok": True})

    _configure(client, monkeypatch, handler)
    token = _mint(client, models=[MODEL_KEYS[provider]])
    requested_model = model if model_allowed else f"not-{model}"
    headers: dict[str, str] = {"content-type": "application/json"}
    url = _path(provider, requested_model)
    if auth_form == "bearer":
        headers["Authorization"] = f"Bearer {token}"
    elif auth_form == "x-api-key":
        headers["x-api-key"] = token
    elif auth_form == "x-goog-api-key":
        headers["x-goog-api-key"] = token
    elif auth_form == "query-key":
        url += f"?key={token}"
    elif auth_form == "header-and-query-key":
        headers["x-goog-api-key"] = token
        url += f"?key={token}"
    elif auth_form == "static":
        headers["Authorization"] = "Bearer sk-test"

    response = client.post(
        url, headers=headers, content=json.dumps({"model": requested_model}).encode()
    )

    assert response.status_code == expected_status
    if not model_allowed:
        _provider_error_shape(provider, response.json())
    assert bool(requests) is (expected_status == 200)
    assert not any("key" in request.url.params for request in requests)


@pytest.mark.parametrize(
    ("provider", "body", "event_stream", "expected"),
    [
        pytest.param(
            "openai",
            OPENAI_RESPONSE,
            False,
            {
                "in_tokens": 8,
                "out_tokens": 4,
                "cache_read_tokens": 2,
                "cache_write_tokens": 0,
                "reasoning_tokens": 3,
            },
            id="openai-response",
        ),
        pytest.param(
            "openai",
            b"event: response.completed\ndata: "
            + OPENAI_RESPONSE.replace(
                b'{"id":"resp_1",', b'{"response":{"id":"resp_1",'
            )
            + b"}\n\n",
            True,
            {
                "in_tokens": 8,
                "out_tokens": 4,
                "cache_read_tokens": 2,
                "cache_write_tokens": 0,
                "reasoning_tokens": 3,
            },
            id="openai-stream-completed-event",
        ),
        pytest.param(
            "openai",
            OPENAI_RESPONSE.replace(b',"cache_write_tokens":0', b""),
            False,
            {},
            id="openai-usage-missing-cache-write",
        ),
        pytest.param(
            "anthropic",
            ANTHROPIC_STREAM,
            True,
            {"in_tokens": 17, "out_tokens": 9, "cache_read_tokens": 3},
            id="anthropic-delta-omits-input-tokens",
        ),
        pytest.param(
            "google",
            GOOGLE_STREAM,
            True,
            {
                "in_tokens": 18,
                "out_tokens": 8,
                "reasoning_tokens": 2,
                "cache_read_tokens": 4,
            },
            id="google-sse",
        ),
        pytest.param(
            "google",
            b"[" + GOOGLE_USAGE + b"]",
            False,
            {
                "in_tokens": 18,
                "out_tokens": 8,
                "reasoning_tokens": 2,
                "cache_read_tokens": 4,
            },
            id="google-json-array",
        ),
        pytest.param(
            "openrouter",
            OPENROUTER_RESPONSE,
            False,
            {
                "in_tokens": 8,
                "out_tokens": 4,
                "cache_read_tokens": 2,
                "cache_write_tokens": 0,
                "reasoning_tokens": 3,
            },
            id="openrouter-chat-completion",
        ),
        pytest.param(
            "openrouter",
            OPENROUTER_STREAM,
            True,
            {
                "in_tokens": 8,
                "out_tokens": 4,
                "cache_read_tokens": 2,
                "cache_write_tokens": 0,
                "reasoning_tokens": 3,
            },
            id="openrouter-stream",
        ),
    ],
)
def test_usage_fields(
    caplog: pytest.LogCaptureFixture,
    provider: str,
    body: bytes,
    event_stream: bool,
    expected: dict[str, int],
) -> None:
    upstream = passthrough_routes.UPSTREAMS[provider]

    fields, _ = passthrough_routes._usage_fields(upstream, body, event_stream)
    assert fields == expected
    if not expected:
        assert f"Pass-through {provider} usage did not parse" in caplog.text


@pytest.mark.parametrize(
    ("provider", "upstream_status"),
    [
        ("openai", 200),
        ("anthropic", 200),
        ("google", 200),
        ("openrouter", 200),
        pytest.param("openai", 429, id="upstream-429-error"),
    ],
)
def test_forwards_provider_bytes_and_records_usage(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    upstream_status: int,
) -> None:
    model = MODELS[provider]
    request_body = {
        "openai": b'{"model": "%s", "input": "hello", "stream": false}',
        "anthropic": b'{"model": "%s", "max_tokens": 12, "messages": [], "stream": true}',
        "google": b'{"contents": [{"parts": [{"text": "hello"}]}]}',
        "openrouter": b'{"model": "%s", "messages": [{"role": "user", "content": "hi"}], "stream": true}',
    }[provider].replace(b"%s", model.encode())
    content_type, response_body = {
        "openai": ("application/json", OPENAI_RESPONSE),
        "anthropic": ("text/event-stream", ANTHROPIC_STREAM),
        "google": ("text/event-stream", GOOGLE_STREAM),
        "openrouter": ("text/event-stream", OPENROUTER_STREAM),
    }[provider]
    if upstream_status == 429:
        content_type = "application/json"
        response_body = (
            b'{"error":{"message":"rate limit exceeded",'
            b'"type":"rate_limit_error","code":"rate_limit_exceeded"}}'
        )
    chunks = [response_body[:7], response_body[7:29], response_body[29:]]
    received: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        received.append(request)
        await request.aread()
        return httpx.Response(
            upstream_status,
            headers={
                "content-type": content_type,
                "content-length": str(len(response_body)),
                "set-cookie": "session=secret",
                "openai-organization": "account-a",
                "openai-project": "project-a",
                "anthropic-organization-id": "account-b",
                "x-request-id": "request-a",
                "request-id": "request-b",
            },
            stream=_ChunkStream(chunks),
        )

    client = _make_client()
    ledger = _configure(client, monkeypatch, handler)
    token = _mint(client, models=[MODEL_KEYS[provider]])
    headers = {"content-type": "application/json"}
    if provider in ("openai", "openrouter"):
        headers.update(
            {
                "Authorization": f"Bearer {token}",
                "accept": "application/json"
                if provider == "openai"
                else "text/event-stream",
            }
        )
    elif provider == "anthropic":
        headers.update(
            {
                "x-api-key": token,
                "anthropic-version": "2023-06-01",
                "accept": "text/event-stream",
            }
        )
    else:
        headers.update(
            {
                "x-goog-api-key": token,
                "accept": "text/event-stream",
            }
        )
    url = _path(provider, model)
    if provider == "google":
        url = url.removesuffix(":generateContent") + ":streamGenerateContent?alt=sse"
        url += f"&key={token}"

    response = client.post(url, headers=headers, content=request_body)

    assert response.status_code == upstream_status
    assert response.content == response_body
    assert {
        "content-length",
        "set-cookie",
        "openai-organization",
        "openai-project",
        "anthropic-organization-id",
    }.isdisjoint(response.headers)
    assert response.headers["x-request-id"] == "request-a"
    assert response.headers["request-id"] == "request-b"
    assert len(received) == 1
    upstream_request = received[0]
    assert upstream_request.content == request_body
    key_header = {
        "openai": "authorization",
        "anthropic": "x-api-key",
        "google": "x-goog-api-key",
        "openrouter": "authorization",
    }[provider]
    expected_key = REAL_KEYS[provider]
    if provider in ("openai", "openrouter"):
        expected_key = f"Bearer {expected_key}"
    assert upstream_request.headers[key_header] == expected_key
    assert token not in str(upstream_request.headers)
    assert "key" not in upstream_request.url.params
    if provider == "google":
        assert upstream_request.url.params["alt"] == "sse"
    assert len(ledger.events) == 1
    event = ledger.events[0]
    assert event["run_id"] == RUN_ID
    assert event["question_id"] == TASK_ID
    assert event["attempt_id"] == ATTEMPT_ID
    assert event["model"] == MODEL_KEYS[provider]
    if upstream_status == 429:
        assert event["outcome"] == "error"
        error = event["error"]
        assert error["status_code"] == 429
        assert error["message"] == "rate limit exceeded"
    else:
        assert (event["input_tokens"], event["output_tokens"]) == {
            "openai": (8, 4),
            "anthropic": (17, 9),
            "google": (18, 8),
            "openrouter": (8, 4),
        }[provider]


class _StallsAfterFirstEvent(httpx.AsyncByteStream):
    async def __aiter__(self) -> AsyncIterator[bytes]:
        first_event_end = ANTHROPIC_STREAM.index(b"event: message_delta")
        yield ANTHROPIC_STREAM[:first_event_end]
        await asyncio.sleep(5)
        yield ANTHROPIC_STREAM[first_event_end:]


@pytest.mark.parametrize("spec_version", ["2.0", "2.4"])
def test_records_usage_when_the_client_disconnects_mid_stream(
    monkeypatch: pytest.MonkeyPatch,
    spec_version: str,
) -> None:
    # Starlette notices a disconnect by listening on receive() before ASGI spec 2.4,
    # and by send() raising OSError from 2.4 on; both must bill what was streamed.
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_StallsAfterFirstEvent(),
        )

    client = _make_client()
    ledger = _configure(client, monkeypatch, handler)
    token = _mint(client, models=[MODEL_KEYS["anthropic"]])
    request_body = json.dumps(
        {"model": MODELS["anthropic"], "stream": True, "messages": []}
    ).encode()
    path = "/passthrough/anthropic/v1/messages"
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": spec_version},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "server": ("test", 80),
        "client": ("test", 1),
        "headers": [
            (b"x-api-key", token.encode()),
            (b"content-type", b"application/json"),
            (b"anthropic-version", b"2023-06-01"),
        ],
        "state": {},
    }
    request_sent = False
    first_event_sent = asyncio.Event()

    async def receive() -> dict[str, Any]:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": request_body, "more_body": False}
        await first_event_sent.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.body" and message.get("body"):
            first_event_sent.set()
            if spec_version == "2.4":
                raise OSError("client went away")

    async def disconnect_mid_stream() -> None:
        try:
            await client.app(scope, receive, send)
        except OSError:
            pass

    asyncio.run(disconnect_mid_stream())

    assert len(ledger.events) == 1
    event = ledger.events[0]
    assert (event["input_tokens"], event["output_tokens"]) == (17, 0)


async def test_upstream_client_holds_more_streams_than_httpx_default() -> None:
    # A stream keeps its upstream connection until it ends, so one process can hold
    # hundreds; with httpx's default of 100, call 101 waits for a free connection.
    app = FastAPI()
    passthrough_routes.register_passthrough_routes(app)
    client: httpx.AsyncClient = app.state.passthrough_client
    release = asyncio.Event()

    async def hold_open(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        await reader.readuntil(b"\r\n\r\n")
        writer.write(
            b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n"
            b"transfer-encoding: chunked\r\n\r\n"
        )
        await writer.drain()
        await release.wait()
        writer.close()

    server = await asyncio.start_server(hold_open, "127.0.0.1", 0)
    url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/"
    async with server, client:
        try:
            responses = await asyncio.wait_for(
                asyncio.gather(
                    *(
                        client.send(client.build_request("POST", url), stream=True)
                        for _ in range(150)
                    )
                ),
                timeout=10,
            )
        finally:
            release.set()
        for response in responses:
            await response.aclose()


def test_client_that_leaves_before_its_body_gets_499(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200)

    client = _make_client()
    ledger = _configure(client, monkeypatch, handler)
    token = _mint(client, models=[MODEL_KEYS["openai"]])
    path = "/passthrough/openai/v1/responses"
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "server": ("test", 80),
        "client": ("test", 1),
        "headers": [
            (b"authorization", f"Bearer {token}".encode()),
            (b"content-type", b"application/json"),
        ],
        "state": {},
    }
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    asyncio.run(client.app(scope, receive, send))

    assert sent[0]["status"] == 499
    assert not requests
    assert not ledger.events


class _RecordingS3:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    async def put_object(self, **kwargs: Any) -> None:
        assert kwargs["Bucket"] == "exchanges"
        assert kwargs["ContentEncoding"] == "gzip"
        self.objects[kwargs["Key"]] = gzip.decompress(kwargs["Body"])


def _post_saving_exchanges(
    client: Any, path: str, headers: dict[str, str], content: bytes
) -> tuple[httpx.Response, _RecordingS3]:
    """Posts on one event loop and waits for the uploads, which TestClient cancels when a call ends."""
    s3 = _RecordingS3()
    exchanges = ExchangeStore("exchanges", client=s3)
    client.app.state.passthrough_exchanges = exchanges

    async def post() -> httpx.Response:
        transport = httpx.ASGITransport(app=client.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as http:
            response = await http.post(path, headers=headers, content=content)
        await exchanges.close()
        return response

    return asyncio.run(post()), s3


def test_saves_both_bodies_and_puts_their_refs_on_the_ledger_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_body = b'{"model": "%s", "input": "hello"}' % MODELS["openai"].encode()

    async def handler(request: httpx.Request) -> httpx.Response:
        await request.aread()
        return httpx.Response(
            200,
            headers={
                "content-type": "application/json",
                "x-request-id": "request-a",
                "set-cookie": "session=secret",
            },
            content=OPENAI_RESPONSE,
        )

    client = _make_client()
    ledger = _configure(client, monkeypatch, handler)
    token = _mint(client, models=[MODEL_KEYS["openai"]])

    response, s3 = _post_saving_exchanges(
        client,
        _path("openai", MODELS["openai"]),
        {"Authorization": f"Bearer {token}", "openai-beta": "assistants=v2"},
        request_body,
    )

    assert response.status_code == 200
    details = ledger.events[0][ledger_schema.DETAILS_FIELD]
    refs = details["request"]["passthrough"]["exchange"]
    prefix = f"passthrough/{RUN_ID}/{ATTEMPT_ID}/"
    for name, body in (("request", request_body), ("response", OPENAI_RESPONSE)):
        key = refs[f"{name}_key"]
        assert key.startswith(prefix) and key.endswith(f"/{name}")
        assert s3.objects[key] == body
        assert refs[f"{name}_sha256"] == hashlib.sha256(body).hexdigest()
    headers = json.loads(s3.objects[refs["headers_key"]])
    assert ["openai-beta", "assistants=v2"] in headers["request"]
    assert ["x-request-id", "request-a"] in headers["response"]
    saved_names = {name for name, _ in headers["request"] + headers["response"]}
    assert saved_names.isdisjoint({"authorization", "set-cookie"})
    assert all(token.encode() not in body for body in s3.objects.values())


class _RecordingLogs:
    def __init__(self) -> None:
        self.streams: list[str] = []
        self.batches: list[list[dict[str, Any]]] = []

    async def create_log_stream(self, **kwargs: Any) -> None:
        assert kwargs["logGroupName"] == "exchange-logs"
        self.streams.append(kwargs["logStreamName"])

    async def put_log_events(self, **kwargs: Any) -> None:
        assert kwargs["logStreamName"] == self.streams[0]
        self.batches.append(kwargs["logEvents"])

    def messages(self) -> list[dict[str, Any]]:
        return [
            json.loads(event["message"]) for batch in self.batches for event in batch
        ]


def test_streams_request_and_response_bodies_to_cloudwatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_body = b'{"model": "%s", "input": "hello"}' % MODELS["openai"].encode()

    async def handler(request: httpx.Request) -> httpx.Response:
        await request.aread()
        return httpx.Response(
            200, headers={"content-type": "application/json"}, content=OPENAI_RESPONSE
        )

    client = _make_client()
    _configure(client, monkeypatch, handler)
    token = _mint(client, models=[MODEL_KEYS["openai"]])
    logs = _RecordingLogs()
    exchange_log = ExchangeLog("exchange-logs", client=logs)
    client.app.state.passthrough_exchange_log = exchange_log

    async def post() -> httpx.Response:
        await exchange_log.start()
        transport = httpx.ASGITransport(app=client.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as http:
            response = await http.post(
                _path("openai", MODELS["openai"]),
                headers={"Authorization": f"Bearer {token}"},
                content=request_body,
            )
        await exchange_log.close()
        return response

    response = asyncio.run(post())

    assert response.status_code == 200
    events = {event["part"]: event for event in logs.messages()}
    assert events.keys() == {"request", "response"}
    request_event, response_event = events["request"], events["response"]
    assert request_event["body"].encode() == request_body
    assert response_event["body"].encode() == OPENAI_RESPONSE
    assert response_event["status_code"] == 200
    assert "status_code" not in request_event
    assert request_event["sha256"] == hashlib.sha256(request_body).hexdigest()
    assert response_event["sha256"] == hashlib.sha256(OPENAI_RESPONSE).hexdigest()
    for event in events.values():
        assert event["chunk"] == 0 and event["chunks"] == 1
        assert event["run_id"] == RUN_ID and event["attempt_id"] == ATTEMPT_ID
        assert event["model"] == MODEL_KEYS["openai"]
        assert event["exchange_prefix"] == exchange_prefix(
            RUN_ID, ATTEMPT_ID, event["query_id"]
        )
    assert request_event["query_id"] == response_event["query_id"]
    assert token not in json.dumps(logs.batches)


def test_splits_large_request_bodies_across_cloudwatch_events() -> None:
    body = json.dumps({"input": "\u00e9\"" * 200_000}).encode() + b"\xff"
    logs = _RecordingLogs()
    exchange_log = ExchangeLog("exchange-logs", client=logs)

    async def send() -> None:
        await exchange_log.start()
        exchange_log.send({"run_id": RUN_ID}, body)
        await exchange_log.close()

    asyncio.run(send())

    events = [event for batch in logs.batches for event in batch]
    assert len(events) > 1
    assert all(len(event["message"]) + 26 <= 256 * 1024 for event in events)
    assert all(sum(len(e["message"]) + 26 for e in b) <= 1024 * 1024 for b in logs.batches)
    messages = logs.messages()
    assert [m["chunk"] for m in messages] == list(range(len(messages)))
    assert {m["chunks"] for m in messages} == {len(messages)}
    assert "".join(m["body"] for m in messages) == body.decode(
        "utf-8", "backslashreplace"
    )


_FUNCTION = {"type": "function", "name": "run", "parameters": {}}
_CLAUDE_TOOL = {"name": "run", "input_schema": {"type": "object"}}
_GEMINI_FUNCTIONS = {"functionDeclarations": [{"name": "run"}]}
_CHAT_FUNCTION = {"type": "function", "function": {"name": "run", "parameters": {}}}
_IMAGE_URL = "https://example.com/answer.png?sig=secret"
_GEMINI_FILE = "https://generativelanguage.googleapis.com/v1beta/files/abc123"


@pytest.mark.parametrize(
    ("provider", "claims", "body", "status", "forwarded"),
    [
        (
            "openai",
            {},
            {
                "tools": [_FUNCTION, {"type": "web_search"}, {"type": "mcp"}],
                "tool_choice": {"type": "web_search"},
            },
            200,
            {"tools": [_FUNCTION]},
        ),
        (
            "anthropic",
            {},
            {
                "tools": [
                    _CLAUDE_TOOL,
                    {"type": "web_fetch_20250910", "name": "web_fetch"},
                ],
                "tool_choice": {"type": "tool", "name": "web_fetch"},
                "mcp_servers": [{"type": "url", "url": "https://mcp.example.com"}],
            },
            200,
            {"tools": [_CLAUDE_TOOL]},
        ),
        (
            "google",
            {},
            {
                "tools": [
                    {**_GEMINI_FUNCTIONS, "googleSearch": {}},
                    {"urlContext": {}},
                ],
                "toolConfig": {"functionCallingConfig": {"mode": "AUTO"}},
            },
            200,
            {
                "tools": [_GEMINI_FUNCTIONS],
                "toolConfig": {"functionCallingConfig": {"mode": "AUTO"}},
            },
        ),
        (
            "openai",
            {},
            {"tools": [{"type": "web_search"}], "tool_choice": "required"},
            200,
            {},
        ),
        (
            "openai",
            {"allow_provider_tools": True},
            {"tools": [{"type": "web_search"}]},
            200,
            {"tools": [{"type": "web_search"}]},
        ),
        (
            "openai",
            {},
            {
                "input": [
                    {
                        "type": "additional_tools",
                        "role": "developer",
                        "tools": [_FUNCTION, {"type": "web_search_preview"}],
                    },
                    {
                        "type": "additional_tools",
                        "role": "developer",
                        "tools": [{"type": "mcp"}],
                    },
                ],
                "tool_choice": {"type": "function", "name": "run"},
            },
            200,
            {
                "input": [
                    {
                        "type": "additional_tools",
                        "role": "developer",
                        "tools": [_FUNCTION],
                    }
                ],
                "tool_choice": {"type": "function", "name": "run"},
            },
        ),
        (
            "openai",
            {},
            {"prompt": {"id": "pmpt_saved", "variables": {"q": "hi"}}, "input": "hi"},
            200,
            {"input": "hi"},
        ),
        (
            "openai",
            {},
            {
                "input": [
                    {
                        "role": "user",
                        "content": [{"type": "input_image", "image_url": _IMAGE_URL}],
                    }
                ]
            },
            400,
            None,
        ),
        (
            "anthropic",
            {},
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {"type": "url", "url": _IMAGE_URL},
                            }
                        ],
                    }
                ]
            },
            400,
            None,
        ),
        (
            "anthropic",
            {},
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "document",
                                "source": {
                                    "type": "content",
                                    "content": [
                                        {
                                            "type": "image",
                                            "source": {
                                                "type": "url",
                                                "url": _IMAGE_URL,
                                            },
                                        }
                                    ],
                                },
                            }
                        ],
                    }
                ]
            },
            400,
            None,
        ),
        (
            "google",
            {},
            {"contents": [{"parts": [{"fileData": {"fileUri": _IMAGE_URL}}]}]},
            400,
            None,
        ),
        (
            "google",
            {},
            {"contents": [{"parts": [{"fileData": {"fileUri": _GEMINI_FILE}}]}]},
            200,
            {"contents": [{"parts": [{"fileData": {"fileUri": _GEMINI_FILE}}]}]},
        ),
        (
            "google",
            {"allow_urls": True, "egress": ["example.com"]},
            {"contents": [{"parts": [{"fileData": {"fileUri": _IMAGE_URL}}]}]},
            200,
            {"contents": [{"parts": [{"fileData": {"fileUri": _IMAGE_URL}}]}]},
        ),
        (
            "openai",
            {},
            {
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": "data:image/png;base64,AA",
                            }
                        ],
                    }
                ]
            },
            200,
            {
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": "data:image/png;base64,AA",
                            }
                        ],
                    }
                ]
            },
        ),
        (
            "openrouter",
            {},
            {
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    _CHAT_FUNCTION,
                    {"type": "openrouter:web_search"},
                    {"type": "openrouter:web_fetch", "defer_loading": True},
                    {"type": "web_search_preview"},
                ],
                "tool_choice": "auto",
                "plugins": [{"id": "web"}, {"id": "file-parser"}],
                "web_search_options": {"search_context_size": "low"},
                "preset": "@preset/with-web-search",
                "models": ["%s:online"],
                "route": "fallback",
            },
            200,
            {
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [_CHAT_FUNCTION],
                "tool_choice": "auto",
            },
        ),
        (
            "openrouter",
            {},
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": _IMAGE_URL}}
                        ],
                    }
                ]
            },
            400,
            None,
        ),
        (
            "openrouter",
            {},
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {"data": _IMAGE_URL, "format": "wav"},
                            }
                        ],
                    }
                ]
            },
            400,
            None,
        ),
        (
            "openrouter",
            {},
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,AA"},
                            },
                            {
                                "type": "input_audio",
                                "input_audio": {"data": "AA==", "format": "wav"},
                            },
                        ],
                    }
                ]
            },
            200,
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,AA"},
                            },
                            {
                                "type": "input_audio",
                                "input_audio": {"data": "AA==", "format": "wav"},
                            },
                        ],
                    }
                ]
            },
        ),
    ],
)
def test_provider_side_fetches_follow_the_token(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    claims: dict[str, object],
    body: dict[str, object],
    status: int,
    forwarded: dict[str, object] | None,
) -> None:
    sent: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(await request.aread()))
        return httpx.Response(200, json={"ok": True})

    client = _make_client()
    _configure(client, monkeypatch, handler)
    token = _mint(client, models=[MODEL_KEYS[provider]], **claims)
    model = {} if provider == "google" else {"model": MODELS[provider]}

    response = client.post(
        _path(provider, MODELS[provider]),
        headers={"Authorization": f"Bearer {token}"},
        content=json.dumps({**model, **body}).replace("%s", MODELS[provider]).encode(),
    )

    assert response.status_code == status
    if forwarded is None:
        assert not sent
        _provider_error_shape(provider, response.json())
        assert "https://example.com/answer.png" in response.text
        assert "secret" not in response.text
    else:
        assert sent == [{**model, **forwarded}]
@pytest.mark.parametrize(
    "path",
    [
        "/passthrough/openrouter/api/v1/responses",
        "/passthrough/openrouter/api/v1/completions",
    ],
)
def test_openrouter_forwards_chat_completions_only(
    monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    sent: list[bytes] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        sent.append(await request.aread())
        return httpx.Response(200, json={"ok": True})

    client = _make_client()
    _configure(client, monkeypatch, handler)
    token = _mint(client, models=[MODEL_KEYS["openrouter"]])

    response = client.post(
        path,
        headers={"Authorization": f"Bearer {token}"},
        content=json.dumps({"model": MODELS["openrouter"], "input": "hi"}).encode(),
    )

    assert response.status_code == 404, response.text
    assert not sent


@pytest.mark.parametrize(
    ("model", "status"),
    [
        ("openai/gpt-4o-mini", 200),
        ("openai/gpt-4o-mini:online", 403),
        ("inception/mercury-3", 403),
    ],
)
def test_pattern_allowlist_admits_unregistered_models_priced_by_the_provider(
    monkeypatch: pytest.MonkeyPatch, model: str, status: int
) -> None:
    sent: list[bytes] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        sent.append(await request.aread())
        return httpx.Response(
            200, headers={"content-type": "application/json"}, content=OPENROUTER_RESPONSE
        )

    client = _make_client()
    ledger = _configure(client, monkeypatch, handler)
    token = _mint(client, models=["openrouter/openai/*", "openrouter/openrouter/*"])

    response = client.post(
        "/passthrough/openrouter/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {token}"},
        content=json.dumps({"model": model, "messages": []}).encode(),
    )

    assert response.status_code == status, response.text
    assert len(sent) == (1 if status == 200 else 0)
    if status == 200:
        event = ledger.events[0]
        assert event["model"] == f"openrouter/{model}"
        assert (event["input_tokens"], event["output_tokens"]) == (8, 4)
        assert Decimal(str(event["cost_usd"])) == Decimal("0.0000123")


def test_registry_models_keep_registry_pricing(monkeypatch: pytest.MonkeyPatch) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "application/json"}, content=OPENROUTER_RESPONSE
        )

    client = _make_client()
    ledger = _configure(client, monkeypatch, handler)
    token = _mint(client, models=[MODEL_KEYS["openrouter"]])

    response = client.post(
        "/passthrough/openrouter/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {token}"},
        content=json.dumps({"model": MODELS["openrouter"], "messages": []}).encode(),
    )

    assert response.status_code == 200, response.text
    assert Decimal(str(ledger.events[0]["cost_usd"])) != Decimal("0.0000123")


def test_mint_rejects_a_pattern_for_an_unknown_provider() -> None:
    client = _make_client()
    response = client.post(
        "/service-auth",
        headers=HEADERS,
        json={
            "run_id": RUN_ID,
            "task_id": TASK_ID,
            "attempt_id": ATTEMPT_ID,
            "allowed_models": ["nonsense/*"],
            "ttl_seconds": 3600,
        },
    )
    assert response.status_code == 400, response.text
def test_refuses_repeated_keys_the_provider_could_read_differently(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    sent: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={"ok": True})

    client = _make_client()
    _configure(client, monkeypatch, handler)
    token = _mint(client, models=[MODEL_KEYS["openai"]])
    model = MODELS["openai"]
    body = f'{{"model":"{model}","tools":[{{"type":"web_search"}}],"tools":[]}}'
    caplog.set_level(logging.INFO)

    response = client.post(
        _path("openai", model),
        headers={"Authorization": f"Bearer {token}"},
        content=body.encode(),
    )

    assert response.status_code == 400
    _provider_error_shape("openai", response.json())
    assert not sent
    assert "gateway.run_token.egress_blocked" in caplog.text


def test_saves_the_forwarded_body_when_tools_were_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        await request.aread()
        return httpx.Response(200, content=OPENAI_RESPONSE)

    client = _make_client()
    ledger = _configure(client, monkeypatch, handler)
    token = _mint(client, models=[MODEL_KEYS["openai"]])
    request_body = json.dumps(
        {"model": MODELS["openai"], "tools": [_FUNCTION, {"type": "web_search"}]}
    ).encode()

    response, s3 = _post_saving_exchanges(
        client,
        _path("openai", MODELS["openai"]),
        {"Authorization": f"Bearer {token}"},
        request_body,
    )

    assert response.status_code == 200
    passthrough = ledger.events[0][ledger_schema.DETAILS_FIELD]["request"][
        "passthrough"
    ]
    assert passthrough["removed_tools"] == ["web_search"]
    refs = passthrough["exchange"]
    assert s3.objects[refs["request_key"]] == request_body
    assert json.loads(s3.objects[refs["forwarded_key"]])["tools"] == [_FUNCTION]


class _FailingStream(httpx.AsyncByteStream):
    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b"event: message_start\ndata: {}\n\n"
        raise httpx.ReadError("connection reset")

    async def aclose(self) -> None:
        return None


class _RecordingSentryTransport(Transport):
    def __init__(self, options: dict[str, object]):
        super().__init__(options)
        self.events: list[dict[str, Any]] = []

    def capture_envelope(self, envelope: object) -> None:
        self.events.extend(
            item.get_event()  # type: ignore[attr-defined]
            for item in envelope.items  # type: ignore[attr-defined]
            if item.type == "event"
        )


@pytest.mark.parametrize(
    ("upstream", "message"),
    [
        ("connect_error", "ConnectError"),
        ("broken_stream", "Upstream stream failed"),
        ("error_event", "Overloaded"),
    ],
)
def test_failed_calls_are_recorded_as_errors(
    monkeypatch: pytest.MonkeyPatch, upstream: str, message: str
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if upstream == "connect_error":
            raise httpx.ConnectError("refused", request=request)
        headers = {"content-type": "text/event-stream"}
        if upstream == "broken_stream":
            return httpx.Response(200, headers=headers, stream=_FailingStream())
        return httpx.Response(
            200,
            headers=headers,
            content=b'event: error\ndata: {"type":"error","error":'
            b'{"type":"overloaded_error","message":"Overloaded"}}\n\n',
        )

    client = _make_client()
    ledger = _configure(client, monkeypatch, handler)
    token = _mint(client, models=[MODEL_KEYS["anthropic"]])

    path = _path("anthropic", MODELS["anthropic"])
    headers = {"x-api-key": token, "content-type": "application/json"}
    content = json.dumps({"model": MODELS["anthropic"], "stream": True}).encode()
    if upstream == "broken_stream":
        sentry_client = sentry_sdk.Client(
            dsn="https://public@example.com/1",
            transport=_RecordingSentryTransport,
            integrations=[StarletteIntegration()],
            default_integrations=False,
            before_send=telemetry._before_send,  # pyright: ignore[reportPrivateUsage]
        )
        transport = sentry_client.transport
        assert isinstance(transport, _RecordingSentryTransport)
        monkeypatch.setattr(telemetry, "_enabled", True)
        monkeypatch.setattr(
            telemetry,
            "_sentry_context",
            telemetry.ContextVar("test_passthrough_context", default={}),
        )
        try:
            with sentry_sdk.isolation_scope() as scope:
                scope.set_client(sentry_client)
                with pytest.raises(httpx.ReadError, match="connection reset"):
                    client.post(path, headers=headers, content=content)
        finally:
            sentry_client.close()
        [sentry_event] = transport.events
        tags = sentry_event["tags"]
        assert tags["gateway.error.provider"] == "anthropic"
        assert tags["gateway.error.phase"] == "passthrough_response_body"
        assert tags["gen_ai.request.model"] == MODEL_KEYS["anthropic"]
        assert tags["model.registry_key"] == MODEL_KEYS["anthropic"]
        assert tags["model.name"] == MODEL_KEYS["anthropic"].partition("/")[2]
        assert tags["run_id"] == RUN_ID
        assert tags["question_id"] == TASK_ID
        assert tags["attempt_id"] == ATTEMPT_ID
        assert tags["query_id"]
        assert "fingerprint" not in sentry_event
        assert "connection reset" not in str(tags)
        assert REAL_KEYS["anthropic"] not in str(tags)
    else:
        client.post(path, headers=headers, content=content)

    [event] = ledger.events
    assert event["outcome"] == "error"
    assert message in cast(dict[str, Any], event["error"])["message"]


def test_removed_assigned_key_gets_a_provider_shaped_409(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not reach the provider")

    async def removed_key(*_: object) -> str:
        raise AssignedKeyUnavailableError(
            key_name="OPENAI_API_KEY_2", attempt_key=None, model="openai/gpt-4o"
        )

    client = _make_client()
    _configure(client, monkeypatch, handler)
    client.app.state.key_assignments = object()
    monkeypatch.setattr(passthrough_routes, "assigned_key_setting", removed_key)
    token = _mint(client, models=[MODEL_KEYS["openai"]])

    response = client.post(
        "/passthrough/openai/v1/responses",
        headers={"Authorization": f"Bearer {token}"},
        json={"model": MODELS["openai"]},
    )

    assert response.status_code == 409
    _provider_error_shape("openai", response.json())


def test_responses_websocket_handshake_gets_426() -> None:
    client = _make_client()
    with pytest.raises(WebSocketDenialResponse) as denial:
        with client.websocket_connect("/passthrough/openai/v1/responses"):
            pass
    assert denial.value.status_code == 426


@pytest.mark.parametrize(
    ("path", "route"),
    [
        (
            "/passthrough/google/v1beta/models/gemini-2.5-flash:streamGenerateContent",
            "/passthrough/google/v1beta/models/{model}:streamGenerateContent",
        ),
        ("/passthrough/openai/v1/responses", "/passthrough/openai/v1/responses"),
    ],
)
def test_google_paths_trace_as_their_route(path: str, route: str) -> None:
    assert telemetry.http_route(path) == route
    assert telemetry.should_trace_http_route(route)


def test_exchange_prefix_keeps_each_id_in_its_own_segment() -> None:
    assert exchange_prefix("run/../other-run", None, "q") == (
        "passthrough/run%2F..%2Fother-run/-/q"
    )


def test_approved_identity_models_are_not_passthrough_targets() -> None:
    claims = RunTokenClaims(
        run_id=RUN_ID,
        task_id=TASK_ID,
        attempt_id=ATTEMPT_ID,
        minted_by="passthrough-test",
        managed_key_id=UUID(int=1),
        allowed_models=["anthropic/claude-opus-5-5-cyber", "anthropic/claude-opus-5-5"],
    )

    entries = passthrough_routes._allowed_entries("anthropic", claims)  # pyright: ignore[reportPrivateUsage]

    assert [key for key, _, _ in entries] == ["anthropic/claude-opus-5-5"]
