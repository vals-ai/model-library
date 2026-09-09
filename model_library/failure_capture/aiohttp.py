"""aiohttp request and response entity capture."""

from collections.abc import AsyncIterator, Awaitable
from contextvars import ContextVar
from typing import Any, cast

import aiohttp

from model_library.failure_capture.core import Attempt, current_capture

_aiohttp_attempt: ContextVar[Attempt | None] = ContextVar(
    "provider_exchange_aiohttp_attempt", default=None
)


class CapturingAiohttpStream:
    def __init__(self, stream: aiohttp.StreamReader, attempt: Attempt) -> None:
        self._stream = stream
        self._attempt = attempt

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._iter(self._stream.__aiter__())

    async def read(self, n: int = -1) -> bytes:
        return await self._read(self._stream.read(n))

    async def readline(self, *, max_line_length: int | None = None) -> bytes:
        return await self._read(self._stream.readline(max_line_length=max_line_length))

    def iter_chunked(self, n: int) -> AsyncIterator[bytes]:
        return self._iter(self._stream.iter_chunked(n))

    async def wait_eof(self) -> None:
        try:
            await self._stream.wait_eof()
            self._attempt.finish()
        except BaseException as exc:
            self._attempt.fail(exc)
            raise

    async def _read(self, result: Awaitable[bytes]) -> bytes:
        try:
            return self._record(await result)
        except BaseException as exc:
            self._attempt.fail(exc)
            raise

    async def _iter(self, source: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        try:
            async for chunk in source:
                yield self._record(chunk)
            self._attempt.finish()
        except BaseException as exc:
            self._attempt.fail(exc)
            raise

    def _record(self, chunk: bytes) -> bytes:
        if chunk:
            self._attempt.response_bytes(
                chunk, representation="aiohttp_response_entity_bytes"
            )
        if self._stream.at_eof():
            self._attempt.finish()
        return chunk

    def exception(self) -> BaseException | None:
        exc = self._stream.exception()
        if exc is not None:
            self._attempt.fail(exc)
        return exc

    def set_exception(self, exc: BaseException) -> None:
        self.finish("complete" if self._stream.at_eof() else "closed")
        self._stream.set_exception(exc)

    def finish(self, outcome: str) -> None:
        self._attempt.finish(outcome)


async def aiohttp_capture_middleware(
    request: aiohttp.ClientRequest,
    handler: aiohttp.ClientHandlerType,
) -> aiohttp.ClientResponse:
    active = current_capture()
    if active is None:
        return await handler(request)

    attempt = active.attempt("aiohttp")
    token = _aiohttp_attempt.set(attempt)
    try:
        response = await handler(request)
    except BaseException as exc:
        attempt.fail(exc)
        raise
    finally:
        _aiohttp_attempt.reset(token)

    attempt.response_start(response.status)
    response.content = cast(
        aiohttp.StreamReader, CapturingAiohttpStream(response.content, attempt)
    )
    return response


def aiohttp_trace_config() -> aiohttp.TraceConfig:
    trace = aiohttp.TraceConfig()

    async def request_chunk_sent(
        _session: aiohttp.ClientSession,
        _context: Any,
        params: aiohttp.TraceRequestChunkSentParams,
    ) -> None:
        attempt = _aiohttp_attempt.get()
        if attempt is not None:
            attempt.request_bytes(
                params.chunk, representation="aiohttp_request_entity_bytes"
            )

    async def request_redirect(
        _session: aiohttp.ClientSession,
        _context: Any,
        params: aiohttp.TraceRequestRedirectParams,
    ) -> None:
        stream = params.response.content
        if isinstance(stream, CapturingAiohttpStream):
            stream.finish("redirect")

    trace.on_request_chunk_sent.append(request_chunk_sent)
    trace.on_request_redirect.append(request_redirect)
    return trace
