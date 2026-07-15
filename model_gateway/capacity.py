"""Per-worker gateway admission control for bursty model traffic."""

from __future__ import annotations

import asyncio
import time
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

MAX_ACTIVE_REQUESTS_PER_WORKER = 225
MAX_QUEUED_REQUESTS_PER_WORKER = 250
QUEUE_TIMEOUT_SECONDS = 10
# Leave 60 seconds after the application timeout for best-effort response handling
# before the 3,600-second target deregistration window closes.
REQUEST_TIMEOUT_SECONDS = 3540
MAX_CAPACITY_IDENTITY_BODY_BYTES = 64 * 1024

T = TypeVar("T")
MODEL_CALL_PATHS = frozenset({"/query"})


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
        max_active: int = MAX_ACTIVE_REQUESTS_PER_WORKER,
        max_queued: int = MAX_QUEUED_REQUESTS_PER_WORKER,
        queue_timeout_seconds: float = QUEUE_TIMEOUT_SECONDS,
        request_timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self.max_active = max_active
        self.max_queued = max_queued
        self.queue_timeout_seconds = queue_timeout_seconds
        self.request_timeout_seconds = request_timeout_seconds
        self._active = 0
        self._queued = 0
        self._waiters: deque[object] = deque()
        self._condition = asyncio.Condition()

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
        except BaseException:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            raise
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
