"""HTTPX2 request and response entity capture."""

from collections.abc import AsyncIterator
from typing import cast

import httpx2

from model_library.failure_capture.core import Attempt, current_capture


class CapturingAsyncByteStream(httpx2.AsyncByteStream):
    def __init__(
        self,
        stream: httpx2.AsyncByteStream,
        *,
        attempt: Attempt,
        response: bool,
    ) -> None:
        self._stream = stream
        self._attempt = attempt
        self._response = response

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self._stream:
                if self._response:
                    self._attempt.response_bytes(
                        chunk, representation="httpx_response_entity_bytes"
                    )
                else:
                    self._attempt.request_bytes(
                        chunk, representation="httpx_request_entity_bytes"
                    )
                yield chunk
            if self._response:
                self._attempt.finish("complete")
        except GeneratorExit as exc:
            if not self._response:
                self._attempt.fail(exc)
            raise
        except BaseException as exc:
            self._attempt.fail(exc)
            raise

    async def aclose(self) -> None:
        try:
            await self._stream.aclose()
        except BaseException as exc:
            self._attempt.fail(exc)
            raise
        if self._response:
            self._attempt.finish("closed")


class CapturingAsyncHTTPTransport(httpx2.AsyncBaseTransport):
    def __init__(self, transport: httpx2.AsyncBaseTransport) -> None:
        self._transport = transport

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        active = current_capture()
        if active is None:
            return await self._transport.handle_async_request(request)

        attempt = active.attempt("httpx")
        original_stream = request.stream
        request.stream = CapturingAsyncByteStream(
            cast(httpx2.AsyncByteStream, original_stream),
            attempt=attempt,
            response=False,
        )
        try:
            response = await self._transport.handle_async_request(request)
        except BaseException as exc:
            attempt.fail(exc)
            raise
        finally:
            request.stream = original_stream
        attempt.response_start(response.status_code)
        response.stream = CapturingAsyncByteStream(
            cast(httpx2.AsyncByteStream, response.stream),
            attempt=attempt,
            response=True,
        )
        return response

    async def aclose(self) -> None:
        await self._transport.aclose()
