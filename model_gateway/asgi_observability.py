"""ASGI-level gateway request/response boundary diagnostics."""

from __future__ import annotations

import logging
import time
from typing import cast

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from model_gateway.observability import (
    log_gateway_event,
    request_log_fields_from_scope,
    runtime_snapshot,
)


class GatewayObservabilityMiddleware:
    """Capture request and ASGI response-boundary events without payload data."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        path = cast(str, scope["path"])
        method = cast(str, scope["method"])
        request_fields = request_log_fields_from_scope(scope)
        state = _RequestBoundaryState(start=time.perf_counter())

        def log_anomaly(
            event: str, *, level: int = logging.INFO, **fields: object
        ) -> None:
            log_gateway_event(
                event,
                level=level,
                route=path,
                method=method,
                runtime=_runtime_snapshot_for_anomaly(state),
                **request_fields,
                **fields,
            )

        async def wrapped_receive() -> Message:
            message = await receive()
            if message["type"] == "http.disconnect" and not state.client_disconnected:
                state.client_disconnected = True
                log_anomaly(
                    "gateway.request.client_disconnect",
                    response_started=state.response_started,
                )
            return message

        async def wrapped_send(message: Message) -> None:
            message_type = message["type"]
            try:
                await send(message)
            except Exception as exc:
                state.send_error = True
                log_anomaly(
                    "gateway.response.send_error",
                    level=logging.WARNING,
                    exception_type=type(exc).__name__,
                    response_started=state.response_started,
                    first_body_sent=state.first_body_sent,
                    status_code=state.status_code,
                )
                raise
            if message_type == "http.response.start":
                state.response_started = True
                state.status_code = cast(int, message["status"])
            elif message_type == "http.response.body":
                if not state.first_body_sent:
                    state.first_body_sent = True
                more_body = bool(message.get("more_body"))
                if not more_body:
                    state.response_done = True

        try:
            await self.app(scope, wrapped_receive, wrapped_send)
        except Exception as exc:
            if not state.send_error:
                log_anomaly(
                    "gateway.request.error",
                    level=logging.WARNING,
                    exception_type=type(exc).__name__,
                    response_started=state.response_started,
                    first_body_sent=state.first_body_sent,
                    status_code=state.status_code,
                )
            raise
        finally:
            duration_ms = (time.perf_counter() - state.start) * 1000
            if not state.response_started:
                log_anomaly(
                    "gateway.response.missing_start",
                    level=logging.WARNING,
                    client_disconnected=state.client_disconnected,
                )
            if _is_anomaly(state):
                log_anomaly(
                    "gateway.request.done",
                    status_code=state.status_code,
                    duration_ms=duration_ms,
                    response_started=state.response_started,
                    first_body_sent=state.first_body_sent,
                    response_done=state.response_done,
                    send_error=state.send_error,
                    client_disconnected=state.client_disconnected,
                )


def _runtime_snapshot_for_anomaly(state: "_RequestBoundaryState") -> dict[str, int]:
    if state.anomaly_runtime is None:
        state.anomaly_runtime = runtime_snapshot()
    return state.anomaly_runtime


def _is_anomaly(state: "_RequestBoundaryState") -> bool:
    return (
        state.send_error
        or state.client_disconnected
        or not state.response_done
        or (state.status_code is not None and not 200 <= state.status_code < 300)
    )


class _RequestBoundaryState:
    def __init__(self, *, start: float):
        self.start = start
        self.response_started = False
        self.first_body_sent = False
        self.response_done = False
        self.send_error = False
        self.client_disconnected = False
        self.status_code: int | None = None
        self.anomaly_runtime: dict[str, int] | None = None
