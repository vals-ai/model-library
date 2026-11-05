"""Write-only copies of pass-through exchanges: full exchanges in S3, bodies in CloudWatch."""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import logging
import os
import socket
import time
import uuid
from contextlib import suppress
from typing import Any, cast
from urllib.parse import quote

from aiobotocore.config import AioConfig  # pyright: ignore[reportMissingImports]
from aiobotocore.session import get_session  # pyright: ignore[reportUnknownVariableType]

from model_gateway.observability import log_failure

logger = logging.getLogger("uvicorn.error.gateway_passthrough")

PENDING_LIMIT_BYTES = 256 * 1024 * 1024


def exchange_prefix(
    run_id: str | None, attempt_id: str | None, query_id: str | None
) -> str:
    """`passthrough/<run>/<attempt>/<query>`, with each id escaped so it stays one segment.

    The ids come from run-token claims, so a `/` in one must not reach another run's prefix.
    """
    segments = (quote(id or "-", safe="") for id in (run_id, attempt_id, query_id))
    return "/".join(("passthrough", *segments))


class ExchangeStore:
    """Puts each call's objects under `passthrough/<run>/<attempt>/<query>/` in the background.

    Objects are gzip-encoded; the sha256 values on the ledger event are of the raw bytes.
    The gateway only ever puts, so a reader needs a separate role.
    """

    def __init__(self, bucket: str, client: Any = None) -> None:
        self._bucket = bucket
        self._client = client
        self._client_context: Any = None
        self._pending: set[asyncio.Task[None]] = set()
        self._pending_bytes = 0

    async def start(self) -> None:
        self._client_context = cast(Any, get_session()).create_client("s3")
        self._client = await self._client_context.__aenter__()

    async def close(self) -> None:
        await asyncio.gather(*self._pending, return_exceptions=True)
        if self._client_context is not None:
            await self._client_context.__aexit__(None, None, None)

    def save(
        self,
        run_id: str | None,
        attempt_id: str | None,
        query_id: str | None,
        objects: dict[str, bytes],
    ) -> dict[str, str] | None:
        """Returns the keys and hashes, or None when S3 is too far behind to take more."""
        prefix = exchange_prefix(run_id, attempt_id, query_id)
        size = sum(len(body) for body in objects.values())
        if self._pending_bytes + size > PENDING_LIMIT_BYTES:
            logger.warning(
                "Pass-through exchange dropped: %s pending bytes", self._pending_bytes
            )
            return None
        self._pending_bytes += size
        refs: dict[str, str] = {}
        for name, body in objects.items():
            key = f"{prefix}/{name}"
            refs[f"{name}_key"] = key
            refs[f"{name}_sha256"] = hashlib.sha256(body).hexdigest()
            task = asyncio.create_task(self._put(key, body))
            self._pending.add(task)
            task.add_done_callback(self._pending.discard)
        return refs

    async def _put(self, key: str, body: bytes) -> None:
        try:
            with log_failure(logger, "Pass-through exchange write failed"):
                await self._client.put_object(
                    Bucket=self._bucket,
                    Key=key,
                    Body=await asyncio.to_thread(gzip.compress, body),
                    ContentEncoding="gzip",
                )
        finally:
            self._pending_bytes -= len(body)


# CloudWatch Logs caps one event at 256 KiB, and counts 26 bytes of overhead per event.
_EVENT_OVERHEAD = 26
_EVENT_BYTES = 256 * 1024 - _EVENT_OVERHEAD
_BATCH_BYTES = 1024 * 1024
_BATCH_EVENTS = 10_000
_FLUSH_SECONDS = 1.0


def _body_chunks(envelope: dict[str, object], body: bytes) -> list[str]:
    """JSON events for `body`, split so each one fits a CloudWatch event.

    Every chunk repeats `envelope`, with `chunk`/`chunks` to reassemble it in order.
    """
    envelope = {**envelope, "sha256": hashlib.sha256(body).hexdigest()}
    text = body.decode("utf-8", "backslashreplace")
    budget = _EVENT_BYTES - len(
        json.dumps({**envelope, "chunk": 99_999, "chunks": 99_999, "body": ""})
    )
    pieces: list[str] = []
    start = 0
    while start < len(text) or not pieces:
        end = min(len(text), start + budget)
        while (size := len(json.dumps(text[start:end]))) > budget:
            end = start + max(1, (end - start) * budget // size)
        pieces.append(text[start:end])
        start = end
    return [
        json.dumps({**envelope, "chunk": index, "chunks": len(pieces), "body": piece})
        for index, piece in enumerate(pieces)
    ]


class ExchangeLog:
    """Streams each pass-through request body to a CloudWatch log group as it arrives.

    Events are batched per worker into one log stream. Like `ExchangeStore`, the
    gateway can only write here; reading takes the reader role.
    """

    def __init__(self, log_group: str, client: Any = None) -> None:
        self._log_group = log_group
        self._stream = f"{socket.gethostname()}/{os.getpid()}/{uuid.uuid4().hex}"
        self._client = client
        self._client_context: Any = None
        self._events: list[tuple[int, str]] = []
        self._pending_bytes = 0
        self._wake = asyncio.Event()
        self._closing = False
        self._flusher: asyncio.Task[None] | None = None
        self._encoding: set[asyncio.Task[None]] = set()

    async def start(self) -> None:
        if self._client is None:
            self._client_context = cast(Any, get_session()).create_client(
                "logs",
                config=AioConfig(retries={"mode": "standard", "max_attempts": 5}),
            )
            self._client = await self._client_context.__aenter__()
        with log_failure(logger, "Pass-through exchange log stream create failed"):
            await self._client.create_log_stream(
                logGroupName=self._log_group, logStreamName=self._stream
            )
        self._flusher = asyncio.create_task(self._flush_loop())

    async def close(self) -> None:
        await asyncio.gather(*self._encoding)
        self._closing = True
        self._wake.set()
        if self._flusher is not None:
            await self._flusher
        if self._client_context is not None:
            await self._client_context.__aexit__(None, None, None)

    def send(self, envelope: dict[str, object], body: bytes) -> None:
        """Queues `body` without blocking; hashing and encoding run in a thread."""
        if self._pending_bytes + len(body) > PENDING_LIMIT_BYTES:
            logger.warning(
                "Pass-through exchange log dropped: %s pending bytes",
                self._pending_bytes,
            )
            return
        self._pending_bytes += len(body)
        task = asyncio.create_task(self._encode(envelope, body))
        self._encoding.add(task)
        task.add_done_callback(self._encoding.discard)

    async def _encode(self, envelope: dict[str, object], body: bytes) -> None:
        try:
            events = await asyncio.to_thread(_body_chunks, envelope, body)
        finally:
            self._pending_bytes -= len(body)
        timestamp = int(time.time() * 1000)
        self._events.extend((timestamp, event) for event in events)
        self._pending_bytes += sum(len(event) + _EVENT_OVERHEAD for event in events)
        if self._pending_bytes >= _BATCH_BYTES:
            self._wake.set()

    async def _flush_loop(self) -> None:
        while not self._closing or self._events:
            with suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), _FLUSH_SECONDS)
            self._wake.clear()
            while self._events:
                await self._put(self._next_batch())

    def _next_batch(self) -> list[tuple[int, str]]:
        size = 0
        count = 0
        for _, message in self._events[:_BATCH_EVENTS]:
            event_size = len(message) + _EVENT_OVERHEAD
            if size + event_size > _BATCH_BYTES:
                break
            size += event_size
            count += 1
        batch, self._events = self._events[:count], self._events[count:]
        self._pending_bytes -= size
        return batch

    async def _put(self, batch: list[tuple[int, str]]) -> None:
        with log_failure(logger, "Pass-through exchange log write failed"):
            await self._client.put_log_events(
                logGroupName=self._log_group,
                logStreamName=self._stream,
                logEvents=[
                    {"timestamp": timestamp, "message": message}
                    for timestamp, message in batch
                ],
            )
