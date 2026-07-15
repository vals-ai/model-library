"""Safe gateway observability helpers for request/runtime diagnostics."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from collections.abc import Callable, MutableMapping
from pathlib import Path
from typing import Any, TypedDict

from starlette.datastructures import Headers

logger = logging.getLogger("uvicorn.error.gateway_observability")

ALB_TRACE_HEADER = "x-amzn-trace-id"
MAX_ID_LENGTH = 128
GATEWAY_PORT = 8000
DEFAULT_STAGE = "unknown"
DEFAULT_SERVICE = "gateway"


class RequestLogFields(TypedDict):
    alb_trace_id: str | None
    run_id: str | None
    question_id: str | None
    query_id: str | None


def bounded_text(value: object, *, max_length: int = MAX_ID_LENGTH) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text[:max_length]


def request_log_fields_from_scope(scope: MutableMapping[str, Any]) -> RequestLogFields:
    headers = Headers(scope=scope)
    return {
        "alb_trace_id": bounded_text(headers.get(ALB_TRACE_HEADER)),
        "run_id": bounded_text(headers.get("x-run-id")),
        "question_id": bounded_text(headers.get("x-question-id")),
        "query_id": bounded_text(headers.get("x-query-id")),
    }


def log_gateway_event(
    event: str,
    *,
    level: int = logging.INFO,
    **fields: object,
) -> None:
    payload: dict[str, object] = {
        "event": event,
        "timestamp_ms": int(time.time() * 1000),
    }
    for key, value in fields.items():
        if value is not None:
            payload[key] = value
    logger.log(level, json.dumps(payload, sort_keys=True, separators=(",", ":")))


def runtime_snapshot() -> dict[str, int]:
    try:
        task_count = len(asyncio.all_tasks())
    except RuntimeError:
        task_count = 0

    fd_path = Path("/proc/self/fd")
    try:
        open_fd_count = sum(1 for _ in fd_path.iterdir())
    except OSError:
        open_fd_count = 0

    try:
        with Path("/proc/self/statm").open() as statm:
            statm_parts = statm.read().split()
        rss_bytes = int(statm_parts[1]) * os.sysconf("SC_PAGE_SIZE")
    except (IndexError, OSError, ValueError):
        rss_bytes = 0

    inbound_socket_count = 0
    outbound_socket_count = 0
    try:
        socket_inodes: set[str] = set()
        for fd in fd_path.iterdir():
            try:
                target = fd.readlink()
            except OSError:
                continue
            text = str(target)
            if text.startswith("socket:[") and text.endswith("]"):
                socket_inodes.add(text.removeprefix("socket:[").removesuffix("]"))

        for path in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
            try:
                lines = path.read_text().splitlines()[1:]
            except FileNotFoundError:
                continue
            for line in lines:
                parts = line.split()
                if len(parts) < 10:
                    continue
                local = parts[1]
                inode = parts[9]
                if inode not in socket_inodes:
                    continue
                try:
                    local_port = int(local.rsplit(":", 1)[1], 16)
                except (IndexError, ValueError):
                    continue
                if local_port == GATEWAY_PORT:
                    inbound_socket_count += 1
                else:
                    outbound_socket_count += 1
    except OSError:
        inbound_socket_count = 0
        outbound_socket_count = 0

    return {
        "pid": os.getpid(),
        "thread_count": threading.active_count(),
        "asyncio_task_count": task_count,
        "open_fd_count": open_fd_count,
        "inbound_socket_count": inbound_socket_count,
        "outbound_socket_count": outbound_socket_count,
        "rss_bytes": rss_bytes,
    }


LoopExceptionHandler = Callable[[asyncio.AbstractEventLoop, dict[str, Any]], object]


def install_loop_exception_handler(
    loop: asyncio.AbstractEventLoop,
) -> LoopExceptionHandler | None:
    previous = loop.get_exception_handler()

    def handler(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        exc = context.get("exception")
        log_gateway_event(
            "gateway.event_loop.exception",
            exception_type=type(exc).__name__ if exc is not None else None,
            context_message=bounded_text(context.get("message"), max_length=256),
        )

        if previous is not None:
            previous(loop, context)
        else:
            loop.default_exception_handler(context)

    loop.set_exception_handler(handler)
    return previous


def log_process_lifecycle(event: str) -> None:
    log_gateway_event(
        event,
        service=os.environ.get("GATEWAY_SERVICE", DEFAULT_SERVICE),
        stage=os.environ.get("GATEWAY_STAGE", DEFAULT_STAGE),
        worker_id=os.getpid(),
    )
