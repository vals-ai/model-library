"""Per-worker gateway admission control for bursty model traffic."""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from typing import Any
from collections import deque
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import TypeVar, cast

from fastapi import Request

import model_library.telemetry as telemetry
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from model_gateway.metrics import record_capacity, record_rejection

DEFAULT_MAX_ACTIVE_REQUESTS = 1000
DEFAULT_MAX_QUEUED_REQUESTS = 250
DEFAULT_QUEUE_TIMEOUT_SECONDS = 30.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 540.0
MAX_CAPACITY_IDENTITY_BODY_BYTES = 64 * 1024

T = TypeVar("T")
MODEL_CALL_PATHS = frozenset({"/query", "/files/upload", "/embeddings", "/moderation"})


class CapacityRejectedError(Exception):
    """Raised when the per-worker queue is already full."""


class CapacityQueueTimeoutError(TimeoutError):
    """Raised when a request waits too long for an active slot."""


class GatewayRequestTimeoutError(TimeoutError):
    """Raised when model-call execution exceeds the gateway timeout."""


@dataclass(frozen=True)
class CapacitySnapshot:
    active: int
    queued: int
    max_active: int
    max_queued: int

    @property
    def demand(self) -> int:
        return self.active + self.queued


class GatewayCapacityLimiter:
    """Bound active provider calls and queued requests within one worker process."""

    def __init__(
        self,
        *,
        max_active: int = DEFAULT_MAX_ACTIVE_REQUESTS,
        max_queued: int = DEFAULT_MAX_QUEUED_REQUESTS,
        queue_timeout_seconds: float = DEFAULT_QUEUE_TIMEOUT_SECONDS,
        request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        if max_active < 1:
            raise ValueError("max_active must be at least 1")
        if max_queued < 0:
            raise ValueError("max_queued must be non-negative")
        if queue_timeout_seconds <= 0:
            raise ValueError("queue_timeout_seconds must be positive")
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")

        self.max_active = max_active
        self.max_queued = max_queued
        self.queue_timeout_seconds = queue_timeout_seconds
        self.request_timeout_seconds = request_timeout_seconds
        self._active = 0
        self._queued = 0
        self._waiters: deque[object] = deque()
        self._condition = asyncio.Condition()

    @classmethod
    def from_env(cls) -> "GatewayCapacityLimiter":
        return cls(
            max_active=_env_int(
                "GATEWAY_MAX_ACTIVE_REQUESTS_PER_WORKER",
                DEFAULT_MAX_ACTIVE_REQUESTS,
            ),
            max_queued=_env_int(
                "GATEWAY_MAX_QUEUED_REQUESTS_PER_WORKER",
                DEFAULT_MAX_QUEUED_REQUESTS,
            ),
            queue_timeout_seconds=_env_float(
                "GATEWAY_QUEUE_TIMEOUT_SECONDS",
                DEFAULT_QUEUE_TIMEOUT_SECONDS,
            ),
            request_timeout_seconds=_env_float(
                "GATEWAY_REQUEST_TIMEOUT_SECONDS",
                DEFAULT_REQUEST_TIMEOUT_SECONDS,
            ),
        )

    def snapshot(self) -> CapacitySnapshot:
        return CapacitySnapshot(
            active=self._active,
            queued=self._queued,
            max_active=self.max_active,
            max_queued=self.max_queued,
        )

    def record_current(self) -> None:
        snapshot = self.snapshot()
        telemetry.add_event(
            "gateway.capacity.snapshot",
            {
                "gateway.capacity.active": snapshot.active,
                "gateway.capacity.queued": snapshot.queued,
                "gateway.capacity.demand": snapshot.demand,
                "gateway.capacity.max_active": snapshot.max_active,
                "gateway.capacity.max_queued": snapshot.max_queued,
            },
        )
        record_capacity(
            active=snapshot.active,
            queued=snapshot.queued,
            max_active=snapshot.max_active,
            max_queued=snapshot.max_queued,
        )

    async def run(self, operation: Callable[[], Awaitable[T]]) -> T:
        telemetry.add_event("gateway.capacity.acquire_start")
        wait_ms = await self._acquire()
        telemetry.add_event(
            "gateway.capacity.acquire_done", {"gateway.queue_wait_ms": wait_ms}
        )
        task = asyncio.ensure_future(operation())
        try:
            done, _pending = await asyncio.wait(
                {task}, timeout=self.request_timeout_seconds
            )
            if not done:
                telemetry.add_event(
                    "gateway.capacity.request_timeout",
                    {"gateway.request_timeout_seconds": self.request_timeout_seconds},
                )
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
                raise GatewayRequestTimeoutError(
                    f"Gateway request exceeded {self.request_timeout_seconds:g}s timeout"
                )
            return await task
        finally:
            await self._release(wait_ms)

    async def _acquire(self) -> float:
        start = time.perf_counter()
        async with self._condition:
            if self._active < self.max_active and self._queued == 0:
                self._active += 1
                return self._record_acquired(start)

            if self._queued >= self.max_queued:
                telemetry.add_event(
                    "gateway.capacity.rejected",
                    {"gateway.rejection.reason": "queue_full"},
                )
                record_rejection("queue_full")
                self.record_current()
                raise CapacityRejectedError("Gateway request queue is full")

            waiter = object()
            self._waiters.append(waiter)
            self._queued += 1
            self.record_current()
            try:
                while True:
                    if self._waiters[0] is waiter and self._active < self.max_active:
                        self._waiters.popleft()
                        self._queued -= 1
                        self._active += 1
                        # More than one slot can be free; wake the next waiter too.
                        self._condition.notify_all()
                        return self._record_acquired(start)

                    elapsed = time.perf_counter() - start
                    remaining = self.queue_timeout_seconds - elapsed
                    if remaining <= 0:
                        telemetry.add_event(
                            "gateway.capacity.rejected",
                            {"gateway.rejection.reason": "queue_timeout"},
                        )
                        record_rejection("queue_timeout")
                        raise CapacityQueueTimeoutError(
                            "Timed out waiting for a gateway request slot"
                        )
                    try:
                        await asyncio.wait_for(self._condition.wait(), remaining)
                    except TimeoutError as exc:
                        telemetry.add_event(
                            "gateway.capacity.rejected",
                            {"gateway.rejection.reason": "queue_timeout"},
                        )
                        record_rejection("queue_timeout")
                        raise CapacityQueueTimeoutError(
                            "Timed out waiting for a gateway request slot"
                        ) from exc
            except BaseException:
                if self._remove_waiter(waiter):
                    self.record_current()
                    self._condition.notify_all()
                raise

    def _record_acquired(self, start: float) -> float:
        wait_ms = (time.perf_counter() - start) * 1000
        snapshot = self.snapshot()
        telemetry.add_event(
            "gateway.capacity.slot_acquired",
            {
                "gateway.queue_wait_ms": wait_ms,
                "gateway.capacity.active": snapshot.active,
                "gateway.capacity.queued": snapshot.queued,
                "gateway.capacity.demand": snapshot.demand,
            },
        )
        record_capacity(
            active=snapshot.active,
            queued=snapshot.queued,
            max_active=snapshot.max_active,
            max_queued=snapshot.max_queued,
            queue_wait_ms=wait_ms,
        )
        return wait_ms

    def _remove_waiter(self, waiter: object) -> bool:
        try:
            self._waiters.remove(waiter)
        except ValueError:
            return False
        self._queued = max(0, self._queued - 1)
        return True

    async def _release(self, _wait_ms: float) -> None:
        async with self._condition:
            self._active = max(0, self._active - 1)
            self._condition.notify_all()
            telemetry.add_event("gateway.capacity.release")
            self.record_current()


def create_capacity_middleware() -> Callable[
    [Request, RequestResponseEndpoint], Awaitable[Response]
]:
    """Create HTTP middleware that bounds model routes before body parsing."""

    async def capacity_middleware(
        request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if request.url.path not in MODEL_CALL_PATHS:
            return await call_next(request)
        limiter = cast(GatewayCapacityLimiter, request.app.state.capacity_limiter)
        try:
            return await limiter.run(lambda: call_next(request))
        except (CapacityRejectedError, CapacityQueueTimeoutError) as exc:
            await _attach_request_identity(request)
            error_attrs = {
                "gateway.error.code": "gateway_overloaded",
                "gateway.error.phase": "capacity",
                "gateway.status_code": 429,
                "http.status_code": 429,
                "http.response.status_code": 429,
            }
            telemetry.set_attributes(error_attrs)
            telemetry.record_exception(exc, error_attrs)
            telemetry.set_status_error("gateway_overloaded")
            return JSONResponse(
                status_code=429,
                content={"code": "gateway_overloaded", "message": str(exc)},
            )
        except GatewayRequestTimeoutError as exc:
            await _attach_request_identity(request)
            error_attrs = {
                "gateway.error.code": "timeout",
                "gateway.error.phase": "capacity",
                "gateway.status_code": 504,
                "http.status_code": 504,
                "http.response.status_code": 504,
            }
            telemetry.set_attributes(error_attrs)
            telemetry.record_exception(exc, error_attrs)
            telemetry.set_status_error("timeout")
            record_rejection("request_timeout")
            return JSONResponse(
                status_code=504,
                content={"code": "timeout", "message": str(exc)},
            )

    return capacity_middleware


async def _attach_request_identity(request: Request) -> None:
    if not telemetry.is_recording():
        return

    path = request.url.path
    attrs: dict[str, object | None] = {
        "gateway.route": path,
        "gateway.operation": _operation_for_path(path),
    }
    content_length = _content_length(request)
    if content_length is not None:
        attrs["http.request.body.size"] = content_length
    if content_length is not None and content_length > MAX_CAPACITY_IDENTITY_BODY_BYTES:
        telemetry.set_attributes(attrs)
        return

    try:
        body = await request.json()
    except Exception:
        telemetry.set_attributes(attrs)
        return
    if not isinstance(body, dict):
        telemetry.set_attributes(attrs)
        return

    body_mapping = cast(dict[str, Any], body)
    attrs.update(
        {
            "gen_ai.request.model": body_mapping.get("model"),
        }
    )
    run_params = {
        "run_id": body_mapping.get("run_id"),
        "question_id": body_mapping.get("question_id"),
        "query_id": body_mapping.get("query_id"),
        "identity": body_mapping.get("identity"),
        "in_agent": body_mapping.get("in_agent"),
    }
    if path == "/query":
        query_id = run_params.get("query_id")
        query_id_text = str(query_id).strip() if query_id is not None else None
        run_params["query_id"] = query_id_text or uuid.uuid4().hex[:14]
    attrs.update(telemetry.run_attributes(run_params))
    telemetry.set_attributes(attrs)


def _content_length(request: Request) -> int | None:
    raw = request.headers.get("content-length")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _operation_for_path(path: str) -> str:
    return {
        "/query": "query",
        "/files/upload": "files_upload",
        "/embeddings": "embeddings",
        "/moderation": "moderation",
    }.get(path, path)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)
