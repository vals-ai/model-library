import asyncio
import base64
import gzip
import importlib.metadata
import json
import logging
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import httpx
import pytest
from aiohttp import web
from google.protobuf.wrappers_pb2 import StringValue

from model_library import utils as model_utils
from model_library.base.base import client_registry
from model_library.failure_capture import core
from model_library.providers.amazon import AmazonModel
from model_library.providers.xai import XAIModel

_AMAZON_GET_CLIENT = AmazonModel.get_client


def _records(path: Path) -> list[dict[str, object]]:
    with gzip.open(path, "rt", encoding="utf-8") as capture:
        return [json.loads(line) for line in capture]


def _payload_events(
    records: list[dict[str, object]], event: str
) -> list[tuple[object, bytes]]:
    return [
        (record["attempt"], base64.b64decode(str(record["bytes_base64"])))
        for record in records
        if record["event"] == event
    ]


def _payloads(records: list[dict[str, object]], event: str) -> list[bytes]:
    return [payload for _, payload in _payload_events(records, event)]


class _Chunks(httpx.AsyncByteStream):
    def __init__(self, *chunks: bytes) -> None:
        self.chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk

    async def aclose(self) -> None:
        pass


async def test_httpx_factory_captures_streams_and_body_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared_requests: list[httpx.Request] = []
    requests: list[tuple[str, str, str, object, bytes]] = []

    async def remember_request(request: httpx.Request) -> None:
        prepared_requests.append(request)

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request is prepared_requests[-1]
        stream = cast(httpx.AsyncByteStream, request.stream)
        body = b"".join([chunk async for chunk in stream])
        requests.append(
            (
                request.method,
                str(request.url),
                request.headers["X-Distinct"],
                request.extensions["capture-marker"],
                body,
            )
        )
        if len(requests) == 1:
            return httpx.Response(
                307,
                headers={"Location": "https://provider.test/next"},
                stream=_Chunks(b"unread-redirect"),
            )
        return httpx.Response(503, stream=_Chunks(b"partial-", b"response"))

    monkeypatch.setattr(
        model_utils.httpx,
        "AsyncHTTPTransport",
        lambda **_kwargs: httpx.MockTransport(handler),
    )
    with core.capture(enabled=True) as active:
        assert active is not None
        async with model_utils.default_httpx_client() as client:
            client.event_hooks["request"].append(remember_request)
            request = client.build_request(
                "POST",
                "https://provider.test/start",
                headers={"X-Distinct": "kept"},
                content=_Chunks(b"body-", b"preserved"),
                extensions={"capture-marker": "kept"},
            )
            first = await client.send(request, stream=True)
            assert first.next_request is not None
            second = await client.send(first.next_request, stream=True)
            assert (
                b"".join([chunk async for chunk in second.aiter_raw()])
                == b"partial-response"
            )
            await first.aclose()
        artifact = active.finalize()
        assert artifact is not None
        records = _records(artifact.path)

    assert requests == [
        ("POST", "https://provider.test/start", "kept", "kept", b"body-preserved"),
        ("POST", "https://provider.test/next", "kept", "kept", b"body-preserved"),
    ]
    assert _payload_events(records, "request.bytes") == [
        (1, b"body-"),
        (1, b"preserved"),
        (2, b"body-"),
        (2, b"preserved"),
    ]
    assert _payloads(records, "response.bytes") == [b"partial-", b"response"]
    assert {
        record.get("representation")
        for record in records
        if record["event"] in {"request.bytes", "response.bytes"}
    } == {"httpx_request_entity_bytes", "httpx_response_entity_bytes"}


async def test_aiohttp_factory_captures_real_body_paths_and_failure() -> None:
    retry_calls = 0
    stall_release = asyncio.Event()

    async def redirect(_request: web.Request) -> web.Response:
        raise web.HTTPFound("/stream")

    async def stream(request: web.Request) -> web.Response:
        await request.read()
        return web.Response(body=b"first\nsecond\n")

    async def partial(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Length": "8"})
        await response.prepare(request)
        await response.write(b"part")
        assert request.transport is not None
        request.transport.close()
        return response

    async def retry(request: web.Request) -> web.Response:
        nonlocal retry_calls
        retry_calls += 1
        await request.read()
        if retry_calls == 1:
            assert request.transport is not None
            request.transport.close()
        return web.Response(body=b"retry-ok")

    async def stall(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Length": "8"})
        await response.prepare(request)
        await response.write(b"part")
        await stall_release.wait()
        return response

    app = web.Application()
    app.router.add_post("/redirect", redirect)
    app.router.add_route("*", "/stream", stream)
    app.router.add_get("/partial", partial)
    app.router.add_post("/retry", retry)
    app.router.add_get("/stall", stall)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = cast(Any, site._server).sockets[0].getsockname()[1]  # pyright: ignore[reportPrivateUsage]
    try:
        with core.capture(enabled=True) as active:
            assert active is not None
            async with model_utils.make_aiohttp_session() as client:
                async with client.post(
                    f"http://127.0.0.1:{port}/redirect", data=b"request"
                ) as response:
                    assert (
                        await response.content.readline(max_line_length=64)
                        == b"first\n"
                    )
                    assert (
                        await response.content.readline(max_line_length=64)
                        == b"second\n"
                    )
                async with client.get(f"http://127.0.0.1:{port}/stream") as response:
                    assert await response.read() == b"first\nsecond\n"
                async with client.get(f"http://127.0.0.1:{port}/stream") as response:
                    assert b"".join([chunk async for chunk in response.content]) == (
                        b"first\nsecond\n"
                    )
                async with client.get(f"http://127.0.0.1:{port}/partial") as response:
                    received = bytearray()
                    with pytest.raises(aiohttp.ClientPayloadError):
                        async for chunk in response.content.iter_chunked(2):
                            received.extend(chunk)
                async with client.get(f"http://127.0.0.1:{port}/partial") as response:
                    with pytest.raises(aiohttp.ClientPayloadError):
                        await response.content.wait_eof()
                with pytest.raises(aiohttp.ServerDisconnectedError):
                    await client.post(
                        f"http://127.0.0.1:{port}/retry", data=b"retry-one"
                    )
                async with client.post(
                    f"http://127.0.0.1:{port}/retry", data=b"retry-two"
                ) as response:
                    assert await response.read() == b"retry-ok"
                async with client.get(f"http://127.0.0.1:{port}/stall") as response:
                    chunks = response.content.iter_chunked(4).__aiter__()
                    assert await anext(chunks) == b"part"
                    pending = asyncio.create_task(anext(chunks))
                    await asyncio.sleep(0)
                    assert not pending.done()
                    pending.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await pending
                    stall_release.set()
            async with model_utils.default_aiohttp_httpx_client() as client:
                response = await client.post(
                    f"http://127.0.0.1:{port}/stream",
                    content=_Chunks(b"httpx-", b"body"),
                )
                assert await response.aread() == b"first\nsecond\n"
            artifact = active.finalize()
            assert artifact is not None
            records = _records(artifact.path)
    finally:
        stall_release.set()
        await runner.cleanup()

    assert [
        (record["attempt"], record["outcome"])
        for record in records
        if record["event"] == "attempt.end"
    ] == [
        (1, "redirect"),
        (2, "complete"),
        (3, "complete"),
        (4, "complete"),
        (5, "error"),
        (6, "error"),
        (7, "error"),
        (8, "complete"),
        (9, "error"),
        (10, "complete"),
    ]
    assert [
        record["error_type"]
        for record in records
        if record["event"] == "attempt.end" and "error_type" in record
    ] == [
        "ClientPayloadError",
        "ClientPayloadError",
        "ServerDisconnectedError",
        "CancelledError",
    ]
    assert received == b"part"
    assert (
        b"".join(
            payload
            for attempt, payload in _payload_events(records, "response.bytes")
            if attempt == 4
        )
        == b"first\nsecond\n"
    )
    assert [
        item for item in _payload_events(records, "request.bytes") if item[0] in {7, 8}
    ] == [(7, b"retry-one"), (8, b"retry-two")]
    assert b"".join(_payloads(records, "response.bytes")) == (
        b"first\nsecond\n" * 3 + b"part" + b"retry-okpartfirst\nsecond\n"
    )


def test_amazon_model_installs_hooks_and_captures_retry_then_raw_stream() -> None:
    handlers: dict[str, Callable[..., None]] = {}

    class Events:
        def register(self, name: str, handler: Callable[..., None]) -> None:
            handlers[name] = handler

    class RawStream:
        def stream(self):
            yield b"raw-one"
            yield b"raw-two"

    sdk_client = MagicMock()
    sdk_client.meta.events = Events()
    with (
        patch.dict(client_registry, {}, clear=True),
        patch("model_library.providers.amazon.boto3.client", return_value=sdk_client),
    ):
        model = AmazonModel("anthropic.claude-3-5-haiku-2024-10-22-v2:0")
        assert _AMAZON_GET_CLIENT(model, api_key="using-environment") is sdk_client

    with core.capture(enabled=True) as active:
        assert active is not None
        request = SimpleNamespace(body=b'{"messages":[]}')
        handlers["before-send.bedrock-runtime.ConverseStream"](request=request)
        handlers["needs-retry.bedrock-runtime.ConverseStream"](
            caught_exception=ConnectionError("retry")
        )
        handlers["before-send.bedrock-runtime.ConverseStream"](request=request)
        response = {"status_code": 200, "body": RawStream()}
        handlers["before-parse.bedrock-runtime.ConverseStream"](response_dict=response)
        handlers["needs-retry.bedrock-runtime.ConverseStream"](response=None)
        assert list(cast(Any, response["body"]).stream()) == [b"raw-one", b"raw-two"]
        artifact = active.finalize()
        assert artifact is not None
        records = _records(artifact.path)

    assert _payloads(records, "request.bytes") == [
        b'{"messages":[]}',
        b'{"messages":[]}',
    ]
    assert _payloads(records, "response.bytes") == [b"raw-one", b"raw-two"]
    assert [r["outcome"] for r in records if r["event"] == "attempt.end"] == [
        "error",
        "complete",
    ]



def test_amazon_model_hook_registration_failure_is_captured(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Events:
        def register(self, name: str, handler: Callable[..., None]) -> None:
            raise RuntimeError("registration failed with secret payload")

    sdk_client = MagicMock()
    sdk_client.meta.events = Events()
    model = AmazonModel("anthropic.claude-3-5-haiku-2024-10-22-v2:0")

    with (
        patch.dict(client_registry, {}, clear=True),
        patch("model_library.providers.amazon.boto3.client", return_value=sdk_client),
        core.capture(enabled=True) as active,
        caplog.at_level(logging.WARNING),
    ):
        assert active is not None
        assert _AMAZON_GET_CLIENT(model, api_key="using-environment") is sdk_client
        assert active.stats().errors == 1

    assert [record.getMessage() for record in caplog.records] == [
        "Failed to register Bedrock ConverseStream capture hook"
    ]



class _Proto:
    def __init__(self, payload: bytes = b"", *, fail: bool = False) -> None:
        self.payload = payload
        self.fail = fail

    def SerializeToString(self) -> bytes:
        if self.fail:
            raise RuntimeError("protobuf serialization failed")
        return self.payload

async def test_pinned_xai_capture() -> None:
    assert importlib.metadata.version("xai-sdk") == "1.12.2"
    model = XAIModel("grok-3-mini")

    class FailingChat:
        requested: list[int] = []

        def _make_request(self, n: int) -> _Proto:
            self.requested.append(n)
            return _Proto(b"exact-request")

        async def stream(self):
            yield (
                SimpleNamespace(),
                SimpleNamespace(
                    proto=_Proto(b"first-response"),
                    content="first",
                    reasoning_content="",
                    tool_calls=[],
                ),
            )
            raise RuntimeError("stream failed")

    chat = FailingChat()
    client = MagicMock()
    client.chat.create.return_value = chat
    with core.capture(enabled=True) as active:
        assert active is not None
        with (
            patch.object(model, "build_body", new_callable=AsyncMock, return_value={}),
            patch.object(model, "get_client", return_value=client),
            pytest.raises(RuntimeError, match="stream failed"),
        ):
            await model._query_impl(
                [], tools=[], query_logger=logging.getLogger("test")
            )
        artifact = active.finalize()
        assert artifact is not None
        records = _records(artifact.path)

    assert chat.requested == [1]
    assert _payloads(records, "request.bytes") == [b"exact-request"]
    assert _payloads(records, "response.bytes") == [b"first-response"]
    assert {
        r.get("representation")
        for r in records
        if r["event"] in {"request.bytes", "response.bytes"}
    } == {"xai_request_protobuf_message", "xai_response_protobuf_message"}


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["make_request", "request_serialization"])
async def test_xai_capture_request_failure_preserves_result(failure: str) -> None:
    model = XAIModel("grok-3-mini")

    class FailingChat:
        def _make_request(self, n: int) -> _Proto:
            if failure == "make_request":
                raise RuntimeError("request construction failed")
            return _Proto(b"request", fail=True)

        async def stream(self):
            yield (
                SimpleNamespace(
                    finish_reason="REASON_STOP",
                    usage=SimpleNamespace(
                        prompt_tokens=1,
                        cached_prompt_text_tokens=0,
                        completion_tokens=1,
                        reasoning_tokens=0,
                    ),
                    tool_calls=[],
                ),
                SimpleNamespace(
                    proto=_Proto(b"response"),
                    content="normal result",
                    reasoning_content="",
                    tool_calls=[],
                ),
            )

    chat = FailingChat()
    client = MagicMock()
    client.chat.create.return_value = chat
    with core.capture(enabled=True) as active:
        assert active is not None
        with (
            patch.object(model, "build_body", new_callable=AsyncMock, return_value={}),
            patch.object(model, "get_client", return_value=client),
        ):
            result = await model._query_impl(
                [], tools=[], query_logger=logging.getLogger("test")
            )
        assert result.output_text == "normal result"
        assert active.stats().errors == 1



@pytest.mark.asyncio
async def test_xai_capture_response_serialization_failure_preserves_result() -> None:
    model = XAIModel("grok-3-mini")

    class Chat:
        def _make_request(self, n: int) -> _Proto:
            return _Proto(b"request")

        async def stream(self):
            yield (
                SimpleNamespace(
                    finish_reason="REASON_STOP",
                    usage=SimpleNamespace(
                        prompt_tokens=1,
                        cached_prompt_text_tokens=0,
                        completion_tokens=1,
                        reasoning_tokens=0,
                    ),
                    tool_calls=[],
                ),
                SimpleNamespace(
                    proto=_Proto(fail=True),
                    content="consumed content",
                    reasoning_content="",
                    tool_calls=[],
                ),
            )

    client = MagicMock()
    client.chat.create.return_value = Chat()
    with core.capture(enabled=True) as active:
        assert active is not None
        with (
            patch.object(model, "build_body", new_callable=AsyncMock, return_value={}),
            patch.object(model, "get_client", return_value=client),
        ):
            result = await model._query_impl(
                [], tools=[], query_logger=logging.getLogger("test")
            )
        assert result.output_text == "consumed content"
        assert active.stats().errors == 1


@pytest.mark.asyncio
async def test_xai_capture_request_base_exception_propagates_unchanged() -> None:
    model = XAIModel("grok-3-mini")

    class DiagnosticFailure(BaseException):
        pass

    failure = DiagnosticFailure()

    class FailingChat:
        def _make_request(self, n: int) -> _Proto:
            raise failure

        async def stream(self):
            raise AssertionError("request failure should propagate before streaming")
            yield

    client = MagicMock()
    client.chat.create.return_value = FailingChat()
    with core.capture(enabled=True) as active:
        assert active is not None
        with (
            patch.object(model, "build_body", new_callable=AsyncMock, return_value={}),
            patch.object(model, "get_client", return_value=client),
            pytest.raises(DiagnosticFailure) as raised,
        ):
            await model._query_impl(
                [], tools=[], query_logger=logging.getLogger("test")
            )

    assert raised.value is failure


@pytest.mark.asyncio
async def test_xai_capture_disabled_skips_diagnostic_request() -> None:
    model = XAIModel("grok-3-mini")

    class Chat:
        def _make_request(self, n: int) -> _Proto:
            raise AssertionError("diagnostic request should not be built")

        async def stream(self):
            yield (
                SimpleNamespace(
                    finish_reason="REASON_STOP",
                    usage=SimpleNamespace(
                        prompt_tokens=1,
                        cached_prompt_text_tokens=0,
                        completion_tokens=1,
                        reasoning_tokens=0,
                    ),
                    tool_calls=[],
                ),
                SimpleNamespace(
                    proto=_Proto(b"response"),
                    content="normal result",
                    reasoning_content="",
                    tool_calls=[],
                ),
            )

    client = MagicMock()
    client.chat.create.return_value = Chat()
    with core.capture(enabled=False):
        with (
            patch.object(model, "build_body", new_callable=AsyncMock, return_value={}),
            patch.object(model, "get_client", return_value=client),
        ):
            result = await model._query_impl(
                [], tools=[], query_logger=logging.getLogger("test")
            )
    assert result.output_text == "normal result"



def test_redacted_protobuf_request_remains_parseable_across_chunks() -> None:
    file_base64 = base64.b64encode(b"private-image" * 20).decode()
    message = StringValue(value=f"before:{file_base64}:after")
    wire = message.SerializeToString()

    with core.capture(enabled=True) as active:
        assert active is not None
        active.register_request_payload(file_base64, "image")
        attempt = active.attempt("xai")
        split = wire.index(file_base64.encode()) + len(file_base64) // 2
        attempt.request_bytes(
            wire[:split], representation="xai_request_protobuf_message"
        )
        attempt.request_bytes(
            wire[split:], representation="xai_request_protobuf_message"
        )
        attempt.finish()
        artifact = active.finalize()
        assert artifact is not None
        records = _records(artifact.path)

    sanitized_wire = _payloads(records, "request.bytes")[0]
    sanitized = StringValue.FromString(sanitized_wire)
    assert sanitized.value.startswith("before:<image omitted;")
    assert sanitized.value.endswith(":after")
    assert file_base64 not in sanitized.value
    assert len(sanitized_wire) == len(wire)


async def test_httpx_transport_restores_request_stream_after_transport_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport_error = httpx.ConnectError("transport failed")

    class FailingTransport(httpx.AsyncBaseTransport):
        def __init__(self) -> None:
            self.request: httpx.Request | None = None

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            self.request = request
            raise transport_error

    transport = FailingTransport()
    monkeypatch.setattr(
        model_utils.httpx,
        "AsyncHTTPTransport",
        lambda **_kwargs: transport,
    )

    async with model_utils.default_httpx_client() as client:
        request = client.build_request(
            "POST", "https://provider.test/fail", content=_Chunks(b"request")
        )
        original_stream = request.stream
        with pytest.raises(httpx.ConnectError) as raised:
            with core.capture(enabled=True):
                await client.send(request)

    assert raised.value is transport_error
    assert transport.request is request
    assert request.stream is original_stream


async def test_aiohttp_capture_isolates_overlapping_request_bodies() -> None:
    file_base64 = base64.b64encode(b"private-image" * 20)
    bodies = (
        b"first:" + file_base64 + b":body",
        b"second:" + file_base64 + b":body",
    )
    received_bodies: list[bytes] = []
    both_handlers_started = asyncio.Event()
    release_handlers = asyncio.Event()
    handler_count = 0

    async def handler(request: web.Request) -> web.Response:
        nonlocal handler_count
        handler_count += 1
        if handler_count == len(bodies):
            both_handlers_started.set()
        await both_handlers_started.wait()
        await release_handlers.wait()
        received_bodies.append(await request.read())
        return web.Response(body=b"ok")

    async def body(payload: bytes) -> AsyncIterator[bytes]:
        yield payload

    app = web.Application()
    app.router.add_post("/capture", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = cast(Any, site._server).sockets[0].getsockname()[1]  # pyright: ignore[reportPrivateUsage]

    try:
        with core.capture(enabled=True) as active:
            assert active is not None
            active.register_request_payload(file_base64.decode(), "image")
            async with model_utils.make_aiohttp_session() as client:
                requests = [
                    asyncio.create_task(
                        client.post(
                            f"http://127.0.0.1:{port}/capture", data=body(payload)
                        )
                    )
                    for payload in bodies
                ]
                await both_handlers_started.wait()
                release_handlers.set()
                responses = await asyncio.gather(*requests)
                for response in responses:
                    await response.read()
                    response.release()
            artifact = active.finalize()
            assert artifact is not None
            records = _records(artifact.path)
    finally:
        await runner.cleanup()

    captured_bodies = {
        payload for _attempt, payload in _payload_events(records, "request.bytes")
    }
    assert set(received_bodies) == set(bodies)
    assert {len(body) for body in captured_bodies} == {len(body) for body in bodies}
    assert {body.split(b":", 1)[0] for body in captured_bodies} == {
        b"first",
        b"second",
    }
    assert all(b"<image omitted;" in body for body in captured_bodies)
    assert all(file_base64 not in body for body in captured_bodies)
