"""Durable successful-query usage ledger for the gateway."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
import uuid
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from types import TracebackType
from typing import Any, Literal, Protocol, cast

from aiobotocore.config import AioConfig  # pyright: ignore[reportMissingImports]
from aiobotocore.session import get_session  # pyright: ignore[reportUnknownVariableType]
from botocore.exceptions import BotoCoreError, ClientError

import model_library.telemetry as telemetry
import model_gateway.usage_ledger.schema as ledger_schema
from model_gateway.metrics import (
    param_group,
    record_usage_ledger_sqs_batch_send,
    record_usage_ledger_sqs_pending,
    record_usage_ledger_sqs_phase,
    record_usage_ledger_sqs_sdk_attempt,
    record_usage_ledger_sqs_sdk_call,
    record_usage_ledger_sqs_write,
)
from model_gateway.types import QueryRequest
from model_gateway.usage_ledger.details import build_usage_event_details
from model_gateway.usage_ledger.dynamodb_writer import (
    AsyncDynamoDbClient,
    prepare_usage_event_for_write,
    put_usage_event_async,
)
from model_library.base.output import QueryResult, QueryResultMetadata

logger = logging.getLogger("model_proxy_server.usage_ledger")

UsageLedgerMode = Literal["disabled", "shadow", "enforced"]

DEFAULT_NORMALIZATION_VERSION = "2026-05-29"
DEFAULT_SHARD_COUNT = ledger_schema.DEFAULT_SHARD_COUNT
# Keep local SQS fanout bounded. The usage ledger now queues in-process and sends
# in the background, so excess fanout should queue locally instead of creating a
# large wave of concurrent SQS connections.
DEFAULT_MAX_POOL_CONNECTIONS = 32
# Bound SQS request latency tightly. The gateway retries failed batch sends itself,
# so the AWS client should fail quickly instead of holding background sends open.
DEFAULT_SQS_CONNECT_TIMEOUT_SECONDS = 3.0
DEFAULT_SQS_READ_TIMEOUT_SECONDS = 5.0
DEFAULT_SQS_CLIENT_TOTAL_MAX_ATTEMPTS = 1
SQS_SEND_MESSAGE_BATCH_MAX_ENTRIES = 10
SQS_SEND_MESSAGE_BATCH_MAX_BYTES = 1024 * 1024
# Small delay lets concurrent query completions coalesce into full SQS batches.
SQS_BATCH_FLUSH_DELAY_SECONDS = 0.025
SQS_BATCH_SEND_CONCURRENCY = 32
SQS_MAX_PENDING_MESSAGES = 10_000
SQS_MAX_PENDING_BYTES = 256 * 1024 * 1024
# Bounded in-process retry. Usage events are idempotent downstream, but retrying
# forever in memory would let a sustained SQS outage consume the gateway process.
SQS_SEND_RETRY_DELAY_SECONDS = 10.0
SQS_MAX_SEND_RETRY_AGE_SECONDS = 20 * 60.0


class UsageLedgerWriteError(RuntimeError):
    """Raised when an enforced usage ledger write fails."""


class UsageLedger(Protocol):
    enabled: bool

    async def start(self) -> None:
        """Open any app-lifetime resources needed by the ledger."""

    async def close(self) -> None:
        """Close any app-lifetime resources opened by the ledger."""

    async def write_success(self, event: Mapping[str, object]) -> None:
        """Persist a successful query usage event."""


class NoopUsageLedger:
    enabled = False

    async def start(self) -> None:
        return

    async def close(self) -> None:
        return

    async def write_success(self, event: Mapping[str, object]) -> None:
        _ = event
        return


class _DynamoDbClientContext(Protocol):
    async def __aenter__(self) -> AsyncDynamoDbClient: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...


class _SqsClient(Protocol):
    async def send_message_batch(self, **kwargs: object) -> Mapping[str, Any]: ...


class _SqsClientContext(Protocol):
    async def __aenter__(self) -> _SqsClient: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...


@dataclass(slots=True)
class _PendingSqsMessage:
    entry_id: str
    body: str
    body_bytes: int
    enqueued_at: float
    send_attempts: int = 0
    retry_ready_at: float | None = None
    flush_requested_at: float | None = None
    dequeued_at: float | None = None
    send_started_at: float | None = None
    send_completed_at: float | None = None


@dataclass(slots=True)
class _SqsSdkAttemptTrace:
    attempt_number: int
    created_at: float
    before_send_at: float | None = None


@dataclass(slots=True)
class _SqsBatchTrace:
    attempts_started: int = 0
    attempts_recorded: int = 0
    retry_count: int | None = None
    current_attempt: _SqsSdkAttemptTrace | None = None


_CURRENT_SQS_BATCH_TRACE: ContextVar[_SqsBatchTrace | None] = ContextVar(
    "usage_ledger_sqs_batch_trace", default=None
)


class _SqsSdkMetricsAdapter:
    def __init__(self, unique_id_source: object) -> None:
        self._unique_id_source = unique_id_source

    def register(self, client: _SqsClient) -> None:
        """Register low-overhead SDK attempt metrics for SendMessageBatch."""
        client_meta = getattr(client, "meta", None)
        events = getattr(client_meta, "events", None)
        register = getattr(events, "register", None)
        if not callable(register):
            return
        service_model = getattr(client_meta, "service_model", None)
        service_id = getattr(service_model, "service_id", None)
        hyphenize = getattr(service_id, "hyphenize", None)
        service_event_name = hyphenize() if callable(hyphenize) else "sqs"
        method_mapping = getattr(client_meta, "method_to_api_mapping", {})
        operation_name = method_mapping.get("send_message_batch", "SendMessageBatch")
        unique_prefix = f"usage-ledger-sqs-metrics-{id(self._unique_id_source)}"
        register(
            f"request-created.{service_event_name}.{operation_name}",
            self._record_request_created,
            unique_id=f"{unique_prefix}-request-created",
        )
        register(
            f"before-send.{service_event_name}.{operation_name}",
            self._record_before_send,
            unique_id=f"{unique_prefix}-before-send",
        )
        register(
            f"needs-retry.{service_event_name}.{operation_name}",
            self._record_needs_retry,
            unique_id=f"{unique_prefix}-needs-retry",
        )
        register(
            f"after-call.{service_event_name}.{operation_name}",
            self._record_after_call,
            unique_id=f"{unique_prefix}-after-call",
        )
        register(
            f"after-call-error.{service_event_name}.{operation_name}",
            self._record_after_call_error,
            unique_id=f"{unique_prefix}-after-call-error",
        )

    def _record_request_created(self, *, request: Any, **_kwargs: object) -> None:
        try:
            trace = _CURRENT_SQS_BATCH_TRACE.get()
            if trace is None:
                return
            trace.attempts_started += 1
            attempt_number = trace.attempts_started
            request_context = getattr(request, "context", None)
            if isinstance(request_context, Mapping):
                request_context_mapping = cast(Mapping[str, object], request_context)
                retries = request_context_mapping.get("retries")
                if isinstance(retries, Mapping):
                    retries_mapping = cast(Mapping[str, object], retries)
                    raw_attempt = retries_mapping.get("attempt")
                    if isinstance(raw_attempt, int) and raw_attempt > 0:
                        attempt_number = raw_attempt
            trace.current_attempt = _SqsSdkAttemptTrace(
                attempt_number=attempt_number,
                created_at=time.perf_counter(),
            )
        except Exception:
            logger.debug("Failed to record SQS request-created metric", exc_info=True)

    def _record_before_send(self, *, request: Any, **_kwargs: object) -> None:
        _ = request
        try:
            trace = _CURRENT_SQS_BATCH_TRACE.get()
            if trace is None or trace.current_attempt is None:
                return
            trace.current_attempt.before_send_at = time.perf_counter()
        except Exception:
            logger.debug("Failed to record SQS before-send metric", exc_info=True)

    def _record_needs_retry(
        self,
        *,
        attempts: int,
        response: object = None,
        caught_exception: BaseException | None = None,
        **_kwargs: object,
    ) -> None:
        try:
            trace = _CURRENT_SQS_BATCH_TRACE.get()
            if trace is None:
                return
            now = time.perf_counter()
            attempt = trace.current_attempt or _SqsSdkAttemptTrace(
                attempt_number=attempts,
                created_at=now,
                before_send_at=now,
            )
            outcome, http_status_code, error_type = _sqs_sdk_attempt_outcome(
                response=response,
                caught_exception=caught_exception,
            )
            record_usage_ledger_sqs_sdk_attempt(
                attempt_number=attempt.attempt_number,
                outcome=outcome,
                http_status_code=http_status_code,
                error_type=error_type,
                latency_ms=(now - attempt.created_at) * 1000,
                wire_latency_ms=(now - (attempt.before_send_at or attempt.created_at))
                * 1000,
            )
            trace.attempts_recorded += 1
            trace.current_attempt = None
        except Exception:
            logger.debug("Failed to record SQS needs-retry metric", exc_info=True)
        return None

    def _record_after_call(
        self, *, parsed: Mapping[str, object] | None = None, **_kwargs: object
    ) -> None:
        try:
            trace = _CURRENT_SQS_BATCH_TRACE.get()
            if trace is None or parsed is None:
                return
            trace.retry_count = _sqs_retry_attempts_from_response(parsed)
        except Exception:
            logger.debug("Failed to record SQS after-call metric", exc_info=True)

    def _record_after_call_error(
        self, *, exception: BaseException | None = None, **_kwargs: object
    ) -> None:
        _ = exception
        try:
            trace = _CURRENT_SQS_BATCH_TRACE.get()
            if trace is None:
                return
            trace.retry_count = max(0, trace.attempts_recorded - 1)
        except Exception:
            logger.debug("Failed to record SQS after-call-error metric", exc_info=True)

    def record_sdk_call_from_trace(
        self, trace: _SqsBatchTrace, *, outcome: str
    ) -> None:
        retry_count = (
            trace.retry_count
            if trace.retry_count is not None
            else max(0, trace.attempts_recorded - 1)
        )
        record_usage_ledger_sqs_sdk_call(outcome=outcome, retry_count=retry_count)


class SqsUsageLedger:
    enabled = True

    def __init__(
        self,
        *,
        queue_url: str,
        mode: UsageLedgerMode,
        region_name: str | None = None,
    ) -> None:
        self._queue_url = queue_url
        self._mode = mode
        self._region_name = region_name
        self._client_context: _SqsClientContext | None = None
        self._client: _SqsClient | None = None
        self._pending: list[_PendingSqsMessage] = []
        self._retry_pending: list[_PendingSqsMessage] = []
        self._inflight_messages = 0
        self._pending_message_bytes = 0
        self._pending_lock = asyncio.Lock()
        self._flush_loop_lock = asyncio.Lock()
        self._flush_timer_task: asyncio.Task[None] | None = None
        self._retry_timer_task: asyncio.Task[None] | None = None
        self._retry_timer_ready_at: float | None = None
        self._flush_task: asyncio.Task[None] | None = None
        self._flush_requested = False
        self._closing = False
        self._next_entry_id = 0
        self._active_batch_sends = 0
        self._sqs_sdk_metrics = _SqsSdkMetricsAdapter(self)

    async def start(self) -> None:
        session = cast(Any, get_session())
        client_context = cast(
            _SqsClientContext,
            session.create_client(
                "sqs",
                region_name=self._region_name or None,
                config=AioConfig(
                    max_pool_connections=DEFAULT_MAX_POOL_CONNECTIONS,
                    connect_timeout=DEFAULT_SQS_CONNECT_TIMEOUT_SECONDS,
                    read_timeout=DEFAULT_SQS_READ_TIMEOUT_SECONDS,
                    retries={
                        "mode": "standard",
                        "total_max_attempts": DEFAULT_SQS_CLIENT_TOTAL_MAX_ATTEMPTS,
                    },
                    # Keep the aiobotocore default aiohttp-backed session explicit;
                    # this avoids runtime backend drift without exposing a knob.
                    http_session_cls=AioConfig().http_session_cls,
                ),
            ),
        )
        self._client_context = client_context
        try:
            self._client = await client_context.__aenter__()
            self._sqs_sdk_metrics.register(self._client)
            self._closing = False
        except Exception:
            self._client_context = None
            raise

    async def close(self) -> None:
        async with self._pending_lock:
            self._closing = True
            self._cancel_flush_timer_locked()
            self._cancel_retry_timer_locked()
            self._pending.extend(self._retry_pending)
            self._retry_pending.clear()
            self._retry_timer_ready_at = None
            should_flush = self._request_flush_locked()
        if should_flush:
            await self._flush_pending()
        elif self._flush_task is not None:
            await asyncio.gather(self._flush_task, return_exceptions=True)
        if self._client_context is None:
            return
        try:
            await self._client_context.__aexit__(None, None, None)
        finally:
            self._client_context = None
            self._client = None

    async def write_success(self, event: Mapping[str, object]) -> None:
        start = time.perf_counter()
        try:
            outcome, error_type = await self._write_success(event)
        except (ValueError, TypeError) as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            record_usage_ledger_sqs_write(
                outcome="failure",
                latency_ms=latency_ms,
                error_type=type(exc).__name__,
            )
            if self._mode == "shadow":
                logger.warning(
                    "Gateway usage ledger SQS enqueue failed in shadow mode: %s", exc
                )
                return
            logger.exception(
                "Gateway usage ledger SQS enqueue failed in enforced mode: %s: %s",
                type(exc).__name__,
                exc,
            )
            raise UsageLedgerWriteError("Gateway usage ledger write failed") from exc
        latency_ms = (time.perf_counter() - start) * 1000
        record_usage_ledger_sqs_write(
            outcome=outcome, latency_ms=latency_ms, error_type=error_type
        )

    async def _write_success(self, event: Mapping[str, object]) -> tuple[str, str]:
        if self._client is None:
            raise ValueError("Gateway usage ledger SQS client is not started")
        body = prepare_usage_event_for_write(event).message
        pending_message = _PendingSqsMessage(
            entry_id=str(self._next_entry_id),
            body=body,
            body_bytes=len(body.encode("utf-8")),
            enqueued_at=time.perf_counter(),
        )
        self._next_entry_id += 1
        async with self._pending_lock:
            if self._closing:
                raise ValueError("Gateway usage ledger SQS client is closing")
            queued_message_count = self._queued_message_count_locked()
            if queued_message_count >= SQS_MAX_PENDING_MESSAGES:
                logger.error(
                    "Gateway usage ledger SQS pending queue is full; dropping usage event: "
                    "pending_messages=%s max_pending_messages=%s entry_id=%s",
                    queued_message_count,
                    SQS_MAX_PENDING_MESSAGES,
                    pending_message.entry_id,
                )
                self._record_sqs_pending_locked()
                return "dropped", "PendingQueueCountFull"
            queued_message_bytes = self._queued_message_bytes_locked()
            if (
                queued_message_bytes + pending_message.body_bytes
                > SQS_MAX_PENDING_BYTES
            ):
                logger.error(
                    "Gateway usage ledger SQS pending queue bytes are full; dropping usage event: "
                    "pending_bytes=%s message_bytes=%s max_pending_bytes=%s entry_id=%s",
                    queued_message_bytes,
                    pending_message.body_bytes,
                    SQS_MAX_PENDING_BYTES,
                    pending_message.entry_id,
                )
                self._record_sqs_pending_locked()
                return "dropped", "PendingQueueBytesFull"
            self._pending.append(pending_message)
            self._pending_message_bytes += pending_message.body_bytes
            self._record_sqs_pending_locked()
            if len(self._pending) >= SQS_SEND_MESSAGE_BATCH_MAX_ENTRIES:
                self._cancel_flush_timer_locked()
                should_flush = self._request_flush_locked()
            elif self._flush_timer_task is None or self._flush_timer_task.done():
                self._flush_timer_task = asyncio.create_task(self._flush_after_delay())
                should_flush = False
            else:
                should_flush = False
        if should_flush:
            self._start_flush_task()
        return "success", "none"

    async def _flush_after_delay(self) -> None:
        try:
            await asyncio.sleep(SQS_BATCH_FLUSH_DELAY_SECONDS)
        except asyncio.CancelledError:
            return
        async with self._pending_lock:
            should_flush = self._request_flush_locked()
        if should_flush:
            self._start_flush_task()

    def _request_flush_locked(self) -> bool:
        if not self._pending or self._flush_requested:
            return False
        flush_requested_at = time.perf_counter()
        for message in self._pending:
            if message.flush_requested_at is None:
                message.flush_requested_at = flush_requested_at
        self._flush_requested = True
        return True

    def _start_flush_task(self) -> None:
        task = asyncio.create_task(self._flush_pending())
        self._flush_task = task
        task.add_done_callback(self._clear_flush_task)

    def _clear_flush_task(self, task: asyncio.Task[None]) -> None:
        if self._flush_task is task:
            self._flush_task = None

    def _cancel_flush_timer_locked(self) -> None:
        timer_task = self._flush_timer_task
        if timer_task is None or timer_task.done():
            return
        timer_task.cancel()

    def _cancel_retry_timer_locked(self) -> None:
        retry_timer_task = self._retry_timer_task
        if retry_timer_task is None or retry_timer_task.done():
            return
        retry_timer_task.cancel()
        self._retry_timer_ready_at = None

    def _queued_message_count_locked(self) -> int:
        return len(self._pending) + len(self._retry_pending) + self._inflight_messages

    def _queued_message_bytes_locked(self) -> int:
        return self._pending_message_bytes

    def _release_message_bytes_locked(self, messages: list[_PendingSqsMessage]) -> None:
        released_bytes = sum(message.body_bytes for message in messages)
        self._pending_message_bytes = max(
            0, self._pending_message_bytes - released_bytes
        )

    def _schedule_retry_timer_locked(self) -> None:
        if not self._retry_pending:
            self._retry_timer_ready_at = None
            return
        next_ready_at = min(
            message.retry_ready_at
            for message in self._retry_pending
            if message.retry_ready_at is not None
        )
        retry_timer_task = self._retry_timer_task
        if (
            retry_timer_task is not None
            and not retry_timer_task.done()
            and self._retry_timer_ready_at is not None
            and self._retry_timer_ready_at <= next_ready_at
        ):
            return
        self._cancel_retry_timer_locked()
        self._retry_timer_ready_at = next_ready_at
        delay_seconds = max(0.0, next_ready_at - time.perf_counter())
        self._retry_timer_task = asyncio.create_task(
            self._move_due_retries_after_delay(delay_seconds)
        )

    async def _move_due_retries_after_delay(self, delay_seconds: float) -> None:
        try:
            await asyncio.sleep(delay_seconds)
        except asyncio.CancelledError:
            return
        async with self._pending_lock:
            self._retry_timer_task = None
            self._retry_timer_ready_at = None
            now = time.perf_counter()
            ready_messages: list[_PendingSqsMessage] = []
            waiting_messages: list[_PendingSqsMessage] = []
            for message in self._retry_pending:
                if message.retry_ready_at is not None and message.retry_ready_at <= now:
                    message.retry_ready_at = None
                    ready_messages.append(message)
                else:
                    waiting_messages.append(message)
            self._retry_pending = waiting_messages
            self._pending.extend(ready_messages)
            self._record_sqs_pending_locked()
            should_flush = self._request_flush_locked()
            self._schedule_retry_timer_locked()
        if should_flush:
            self._start_flush_task()

    def _prepare_message_retry(
        self, message: _PendingSqsMessage, *, now: float
    ) -> None:
        message.retry_ready_at = now + _sqs_retry_delay()
        message.flush_requested_at = None
        message.dequeued_at = None
        message.send_started_at = None
        message.send_completed_at = None

    async def _finish_batch_send(
        self,
        *,
        batch: list[_PendingSqsMessage],
        retry_messages: list[_PendingSqsMessage],
    ) -> None:
        async with self._pending_lock:
            self._inflight_messages = max(0, self._inflight_messages - len(batch))
            retained_messages: list[_PendingSqsMessage] = []
            if retry_messages and self._closing:
                self._fail_batch(
                    retry_messages,
                    ValueError("Gateway usage ledger SQS client is closing"),
                )
            elif retry_messages:
                now = time.perf_counter()
                for message in retry_messages:
                    self._prepare_message_retry(message, now=now)
                self._retry_pending.extend(retry_messages)
                retained_messages = retry_messages
                self._schedule_retry_timer_locked()
            retained_message_ids = {id(message) for message in retained_messages}
            self._release_message_bytes_locked(
                [
                    message
                    for message in batch
                    if id(message) not in retained_message_ids
                ]
            )
            self._record_sqs_pending_locked()

    async def _flush_pending(self) -> None:
        async with self._flush_loop_lock:
            while True:
                batches: list[list[_PendingSqsMessage]] = []
                async with self._pending_lock:
                    for _ in range(SQS_BATCH_SEND_CONCURRENCY):
                        if not self._pending:
                            break
                        batch_size = 0
                        batch_bytes = 0
                        for message in self._pending[
                            :SQS_SEND_MESSAGE_BATCH_MAX_ENTRIES
                        ]:
                            if (
                                batch_bytes + message.body_bytes
                                > SQS_SEND_MESSAGE_BATCH_MAX_BYTES
                            ):
                                break
                            batch_size += 1
                            batch_bytes += message.body_bytes
                        batch = self._pending[:batch_size]
                        del self._pending[:batch_size]
                        self._inflight_messages += len(batch)
                        dequeued_at = time.perf_counter()
                        for message in batch:
                            message.dequeued_at = dequeued_at
                            self._record_sqs_message_phase(
                                phase="pending_queue",
                                outcome="success",
                                start_at=message.enqueued_at,
                                end_at=dequeued_at,
                            )
                            if message.flush_requested_at is not None:
                                self._record_sqs_message_phase(
                                    phase="flush_wait",
                                    outcome="success",
                                    start_at=message.flush_requested_at,
                                    end_at=dequeued_at,
                                )
                        batches.append(batch)
                    self._record_sqs_pending_locked()
                    if not batches:
                        self._flush_requested = False
                        return
                await asyncio.gather(*(self._send_batch(batch) for batch in batches))

    def _register_sqs_event_handlers(self, client: _SqsClient) -> None:
        self._sqs_sdk_metrics.register(client)

    def _record_sqs_message_phase(
        self,
        *,
        phase: str,
        outcome: str,
        start_at: float,
        end_at: float,
    ) -> None:
        record_usage_ledger_sqs_phase(
            phase=phase,
            outcome=outcome,
            latency_ms=max(0.0, (end_at - start_at) * 1000),
        )

    def _record_sqs_pending_locked(self) -> None:
        record_usage_ledger_sqs_pending(
            pending_messages=self._queued_message_count_locked(),
            pending_bytes=self._queued_message_bytes_locked(),
            active_batch_sends=self._active_batch_sends,
        )

    def _record_sqs_active_batch_sends(self, delta: int) -> None:
        self._active_batch_sends = max(0, self._active_batch_sends + delta)
        record_usage_ledger_sqs_pending(
            pending_messages=self._queued_message_count_locked(),
            pending_bytes=self._queued_message_bytes_locked(),
            active_batch_sends=self._active_batch_sends,
        )

    async def _send_batch(self, batch: list[_PendingSqsMessage]) -> None:
        client = self._client
        if client is None:
            self._fail_batch(
                batch, ValueError("Gateway usage ledger SQS client is not started")
            )
            await self._finish_batch_send(batch=batch, retry_messages=[])
            return

        entries = [
            {"Id": message.entry_id, "MessageBody": message.body} for message in batch
        ]
        send_start = time.perf_counter()
        for message in batch:
            message.send_attempts += 1
            message.send_started_at = send_start
            if message.dequeued_at is not None:
                self._record_sqs_message_phase(
                    phase="batch_start_wait",
                    outcome="success",
                    start_at=message.dequeued_at,
                    end_at=send_start,
                )
        batch_trace = _SqsBatchTrace()
        trace_token = _CURRENT_SQS_BATCH_TRACE.set(batch_trace)
        self._record_sqs_active_batch_sends(1)
        try:
            try:
                response = await client.send_message_batch(
                    QueueUrl=self._queue_url,
                    Entries=entries,
                )
            finally:
                self._record_sqs_active_batch_sends(-1)
        except Exception as exc:
            latency_ms = (time.perf_counter() - send_start) * 1000
            self._sqs_sdk_metrics.record_sdk_call_from_trace(
                batch_trace, outcome="exception"
            )
            retry_checked_at = time.perf_counter()
            retry_messages = [
                message
                for message in batch
                if _sqs_should_retry(message, now=retry_checked_at)
            ]
            terminal_messages = [
                message
                for message in batch
                if not _sqs_should_retry(message, now=retry_checked_at)
            ]
            record_usage_ledger_sqs_batch_send(
                outcome="exception",
                latency_ms=latency_ms,
                message_count=len(batch),
                failed_message_count=len(batch),
                retried_message_count=len(retry_messages),
                error_type=type(exc).__name__,
            )
            if terminal_messages:
                self._fail_batch(terminal_messages, exc)
            if retry_messages:
                logger.warning(
                    "Gateway usage ledger SQS batch send failed; scheduled background retry: %s: %s",
                    type(exc).__name__,
                    exc,
                )
            await self._finish_batch_send(
                batch=batch,
                retry_messages=retry_messages,
            )
            return
        finally:
            _CURRENT_SQS_BATCH_TRACE.reset(trace_token)
        latency_ms = (time.perf_counter() - send_start) * 1000
        send_completed_at = time.perf_counter()
        for message in batch:
            message.send_completed_at = send_completed_at
        self._sqs_sdk_metrics.record_sdk_call_from_trace(batch_trace, outcome="success")

        failed_entries = cast(list[Mapping[str, object]], response.get("Failed", []))
        failed_by_id: dict[str, Mapping[str, object]] = {}
        for entry in failed_entries:
            entry_id = entry.get("Id")
            if entry_id is not None:
                failed_by_id[str(entry_id)] = entry

        retry_messages: list[_PendingSqsMessage] = []
        for message in batch:
            failed_entry = failed_by_id.get(message.entry_id)
            if failed_entry is None:
                continue
            exc = ValueError(
                "Gateway usage ledger SQS batch entry failed: "
                f"{failed_entry.get('Code', 'Unknown')}: "
                f"{failed_entry.get('Message', 'no message')}"
            )
            retry_checked_at = time.perf_counter()
            if bool(failed_entry.get("SenderFault")) or not _sqs_should_retry(
                message, now=retry_checked_at
            ):
                self._fail_batch([message], exc)
            else:
                retry_messages.append(message)

        record_usage_ledger_sqs_batch_send(
            outcome="partial_failure" if failed_entries else "success",
            latency_ms=latency_ms,
            message_count=len(batch),
            failed_message_count=len(failed_entries),
            retried_message_count=len(retry_messages),
            error_type="batch_entry_failure" if failed_entries else "none",
        )
        if retry_messages:
            logger.warning(
                "Gateway usage ledger SQS batch returned %s retryable failures; scheduled background retry",
                len(retry_messages),
            )
        await self._finish_batch_send(batch=batch, retry_messages=retry_messages)

    def _fail_batch(self, batch: list[_PendingSqsMessage], exc: BaseException) -> None:
        logger.error(
            "Gateway usage ledger SQS batch send failed permanently: "
            "message_count=%s error_type=%s error=%s",
            len(batch),
            type(exc).__name__,
            exc,
        )


def _sqs_sdk_attempt_outcome(
    *, response: object, caught_exception: BaseException | None
) -> tuple[str, str, str]:
    if caught_exception is not None:
        return "exception", "exception", type(caught_exception).__name__

    http_response: object | None = None
    parsed_response: object | None = None
    if isinstance(response, tuple):
        response_items = cast(tuple[object, ...], response)
        if len(response_items) == 2:
            http_response, parsed_response = response_items

    raw_status_code = getattr(http_response, "status_code", None)
    http_status_code = (
        str(raw_status_code)
        if isinstance(raw_status_code, int) and not isinstance(raw_status_code, bool)
        else "unknown"
    )
    if isinstance(raw_status_code, int) and raw_status_code >= 300:
        return "http_error", http_status_code, _sqs_error_type(parsed_response)
    return "success", http_status_code, "none"


def _sqs_error_type(parsed_response: object) -> str:
    if not isinstance(parsed_response, Mapping):
        return "unknown"
    parsed_mapping = cast(Mapping[str, object], parsed_response)
    raw_error = parsed_mapping.get("Error")
    if not isinstance(raw_error, Mapping):
        return "unknown"
    error_mapping = cast(Mapping[str, object], raw_error)
    raw_code = error_mapping.get("Code")
    if isinstance(raw_code, str) and raw_code:
        return raw_code[:128]
    return "unknown"


def _sqs_retry_attempts_from_response(response: Mapping[str, object]) -> int:
    raw_metadata = response.get("ResponseMetadata")
    if not isinstance(raw_metadata, Mapping):
        return 0
    metadata = cast(Mapping[str, object], raw_metadata)
    raw_retry_attempts = metadata.get("RetryAttempts")
    if isinstance(raw_retry_attempts, int) and not isinstance(raw_retry_attempts, bool):
        return max(0, raw_retry_attempts)
    return 0


class DynamoDbUsageLedger:
    enabled = True

    def __init__(
        self,
        *,
        table_name: str,
        mode: UsageLedgerMode,
        region_name: str | None = None,
    ) -> None:
        self._table_name = table_name
        self._mode = mode
        self._region_name = region_name
        self._client_context: _DynamoDbClientContext | None = None
        self._client: AsyncDynamoDbClient | None = None

    async def start(self) -> None:
        session = cast(Any, get_session())
        client_context = cast(
            _DynamoDbClientContext,
            session.create_client(
                "dynamodb",
                region_name=self._region_name or None,
                config=AioConfig(
                    connector_args={
                        "keepalive_timeout": 60,
                        "ttl_dns_cache": 300,
                        "force_close": False,
                    },
                    max_pool_connections=DEFAULT_MAX_POOL_CONNECTIONS,
                    connect_timeout=5,
                    read_timeout=30,
                    retries={"mode": "standard", "max_attempts": 3},
                ),
            ),
        )
        self._client_context = client_context
        try:
            self._client = await client_context.__aenter__()
        except Exception:
            self._client_context = None
            raise

    async def close(self) -> None:
        if self._client_context is None:
            return
        try:
            await self._client_context.__aexit__(None, None, None)
        finally:
            self._client_context = None
            self._client = None

    async def write_success(self, event: Mapping[str, object]) -> None:
        try:
            await self._write_success(event)
        except (BotoCoreError, ClientError, ValueError, TypeError) as exc:
            if self._mode == "shadow":
                logger.warning(
                    "Gateway usage ledger write failed in shadow mode: %s", exc
                )
                return
            raise UsageLedgerWriteError("Gateway usage ledger write failed") from exc

    async def _write_success(self, event: Mapping[str, object]) -> None:
        client = self._client
        if client is None:
            raise ValueError("Gateway usage ledger DynamoDB client is not started")
        prepared = prepare_usage_event_for_write(event)
        await put_usage_event_async(
            client=client,
            table_name=self._table_name,
            event=prepared.event,
        )


def create_usage_ledger_from_env() -> UsageLedger:
    mode = _usage_ledger_mode()
    if mode == "disabled":
        return NoopUsageLedger()

    region_name = os.environ.get("GATEWAY_USAGE_LEDGER_REGION", "").strip() or None
    queue_url = os.environ.get("GATEWAY_USAGE_LEDGER_QUEUE_URL", "").strip()
    if queue_url:
        return SqsUsageLedger(
            queue_url=queue_url,
            mode=mode,
            region_name=region_name,
        )

    table_name = os.environ.get("GATEWAY_USAGE_LEDGER_TABLE_NAME", "").strip()
    if not table_name:
        if mode == "shadow":
            logger.warning(
                "GATEWAY_USAGE_LEDGER_MODE=shadow but neither "
                "GATEWAY_USAGE_LEDGER_QUEUE_URL nor GATEWAY_USAGE_LEDGER_TABLE_NAME is set"
            )
            return NoopUsageLedger()
        raise ValueError(
            "GATEWAY_USAGE_LEDGER_QUEUE_URL or GATEWAY_USAGE_LEDGER_TABLE_NAME is "
            "required when usage ledger is enforced"
        )

    return DynamoDbUsageLedger(
        table_name=table_name,
        mode=mode,
        region_name=region_name,
    )


def build_success_usage_event(
    *,
    body: QueryRequest,
    config: Mapping[str, Any],
    query_params: Mapping[str, object | None],
    dimensions: Mapping[str, str],
    result: QueryResult,
    request: Mapping[str, object],
    completed_at: datetime | None = None,
    api_key_fingerprint: str | None = None,
) -> dict[str, object]:
    completed = completed_at or datetime.now(UTC)
    completed_at_iso = _isoformat(completed)
    day = completed.strftime("%Y%m%d")
    run_id = body.run_id
    question_id = body.question_id
    query_id = str(query_params.get("query_id") or body.query_id or "")
    usage_event_id = _usage_event_id()
    shard = _shard(usage_event_id)
    query_config_params = {
        key: value
        for key, value in query_params.items()
        if value is not None
        and key not in telemetry.CONFIG_FINGERPRINT_EXCLUDED_PARAM_KEYS
    }
    config_hash, _ = telemetry.config_fingerprint(
        body.model,
        config,
        query_config_params,
        body.token_retry_params,
    )
    metadata_counts = _metadata_counts(result.metadata)

    benchmark_name = ledger_schema.identity_dimension_value(
        body.identity, ledger_schema.IDENTITY_BENCHMARK_NAME
    )
    agent_name = ledger_schema.identity_dimension_value(
        body.identity, ledger_schema.IDENTITY_AGENT_NAME
    )
    identity_email = ledger_schema.identity_email_value(body.identity)

    event: dict[str, object] = {
        ledger_schema.BASE_PK: ledger_schema.usage_day_pk(day, shard),
        ledger_schema.BASE_SK: f"TS#{completed_at_iso}#USG#{usage_event_id}",
        "entity_type": "usage_event",
        "usage_event_id": usage_event_id,
        "run_id": run_id,
        "question_id": question_id,
        "query_id": query_id,
        ledger_schema.IDENTITY_BENCHMARK_NAME: benchmark_name,
        ledger_schema.IDENTITY_AGENT_NAME: agent_name,
        ledger_schema.IDENTITY_EMAIL: identity_email,
        "api_key_fingerprint": api_key_fingerprint,
        "model": body.model,
        "provider": body.model.partition("/")[0] or "unknown",
        "provider_endpoint": "custom" if config.get("custom_endpoint") else "default",
        "param_group": dimensions.get("ParamGroup")
        or param_group(config, query_config_params, body.token_retry_params),
        "config_hash": config_hash,
        "finish_reason": result.finish_reason.reason.value,
        "finish_reason_raw": result.finish_reason.raw,
        ledger_schema.DETAILS_FIELD: build_usage_event_details(
            request=request,
            result=result,
        ),
        "completed_at": completed_at_iso,
        "day": day,
        "usage_shard": shard,
        "schema_version": ledger_schema.USAGE_EVENT_SCHEMA_VERSION,
        "normalization_version": DEFAULT_NORMALIZATION_VERSION,
        **metadata_counts,
    }
    if identity_email is None:
        del event[ledger_schema.IDENTITY_EMAIL]
    if run_id:
        event[ledger_schema.RUN_INDEX_PK] = ledger_schema.run_pk(run_id, shard)
        event[ledger_schema.RUN_INDEX_SK] = (
            f"TS#{completed_at_iso}#QUESTION#{question_id or 'none'}"
            f"#QUERY#{query_id}#USG#{usage_event_id}"
        )
    if query_id:
        event[ledger_schema.QUERY_INDEX_PK] = ledger_schema.query_pk(query_id)
        event[ledger_schema.QUERY_INDEX_SK] = f"USG#{usage_event_id}"
    if api_key_fingerprint:
        event[ledger_schema.API_KEY_DAY_INDEX_PK] = ledger_schema.api_key_day_pk(
            api_key_fingerprint, day, shard
        )
        event[ledger_schema.API_KEY_DAY_INDEX_SK] = (
            f"TS#{completed_at_iso}#USG#{usage_event_id}"
        )
    dimension_sk = ledger_schema.dimension_sort_key(
        completed_at_iso=completed_at_iso,
        run_id=body.run_id,
        question_id=body.question_id,
        query_id=query_id,
        usage_event_id=usage_event_id,
    )
    if benchmark_name:
        event[ledger_schema.BENCHMARK_INDEX_PK] = ledger_schema.benchmark_pk(
            benchmark_name, shard
        )
        event[ledger_schema.BENCHMARK_INDEX_SK] = dimension_sk
    if agent_name:
        event[ledger_schema.AGENT_INDEX_PK] = ledger_schema.agent_pk(agent_name, shard)
        event[ledger_schema.AGENT_INDEX_SK] = dimension_sk
    return event


def _usage_ledger_mode() -> UsageLedgerMode:
    raw_mode = os.environ.get("GATEWAY_USAGE_LEDGER_MODE", "shadow").strip().lower()
    if raw_mode in {"disabled", "off", "false", "0", ""}:
        return "disabled"
    if raw_mode == "shadow":
        return "shadow"
    if raw_mode in {"enforced", "true", "1", "on"}:
        return "enforced"
    raise ValueError("GATEWAY_USAGE_LEDGER_MODE must be disabled, shadow, or enforced")


def _isoformat(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _usage_event_id() -> str:
    return f"usg_{uuid.uuid4().hex[:24]}"


def _shard(usage_event_id: str) -> str:
    shard_count = _shard_count()
    value = int(hashlib.sha256(usage_event_id.encode()).hexdigest()[:8], 16)
    return f"{value % shard_count:02d}"


def _shard_count() -> int:
    try:
        shard_count = int(
            os.environ.get("GATEWAY_USAGE_LEDGER_SHARDS", DEFAULT_SHARD_COUNT)
        )
    except ValueError:
        return DEFAULT_SHARD_COUNT
    if shard_count <= 0:
        return DEFAULT_SHARD_COUNT
    return shard_count


def _sqs_retry_delay() -> float:
    return SQS_SEND_RETRY_DELAY_SECONDS


def _sqs_should_retry(message: _PendingSqsMessage, *, now: float) -> bool:
    return (now - message.enqueued_at) < SQS_MAX_SEND_RETRY_AGE_SECONDS


def _metadata_counts(metadata: QueryResultMetadata) -> dict[str, object]:
    cost_total = metadata.cost.total if metadata.cost is not None else None
    return {
        "input_tokens": metadata.in_tokens,
        "output_tokens": metadata.out_tokens,
        "reasoning_tokens": metadata.reasoning_tokens or 0,
        "cache_read_tokens": metadata.cache_read_tokens or 0,
        "cache_write_tokens": metadata.cache_write_tokens or 0,
        "total_input_tokens": metadata.total_input_tokens,
        "total_output_tokens": metadata.total_output_tokens,
        "duration_seconds": metadata.duration_seconds,
        "cost_usd": (
            Decimal("0")
            if cost_total is None
            else ledger_schema.normalize_cost_usd_decimal(Decimal(str(cost_total)))
        ),
    }
