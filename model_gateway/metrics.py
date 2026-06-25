"""CloudWatch EMF metrics for the gateway server.

The request path records metrics in memory and a background task flushes
aggregated EMF payloads. This avoids one CloudWatch log line per request while
still preserving high-cardinality dimensions such as model, endpoint, and
parameter group.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, cast

from fastapi import Request
from pydantic import BaseModel as PydanticBaseModel

import model_library.telemetry as telemetry
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

NAMESPACE = "ModelProxy/Gateway"
DEFAULT_SERVICE = "gateway"
DEFAULT_STAGE = "unknown"
METRICS_FLUSH_INTERVAL_SECONDS = 10

MetricValue = int | float
MetricSpec = tuple[MetricValue, str]
DimensionSet = tuple[str, ...]
MetricPublisher = Callable[[], Awaitable[None] | None]

_GAUGE_METRICS = {
    "InFlightRequests",
    "ActiveRequests",
    "QueuedRequests",
    "GatewayDemand",
    "MaxActiveRequests",
    "MaxQueuedRequests",
}
_HIGH_RESOLUTION_METRICS = {
    "InFlightRequests",
    "ActiveRequests",
    "QueuedRequests",
    "GatewayDemand",
}

_inflight_requests = 0
_inflight_lock = asyncio.Lock()
_pending_lock = Lock()
_pending: dict[
    tuple[tuple[tuple[str, str], ...], tuple[DimensionSet, ...]], "Bucket"
] = {}


@dataclass
class MetricAggregate:
    value: MetricValue = 0
    count: int = 0

    def add(self, name: str, value: MetricValue) -> None:
        if name in _GAUGE_METRICS:
            self.value = max(self.value, value)
            self.count = 1
            return
        if name.endswith("LatencyMs") or name == "QueueWaitMs":
            self.value += value
            self.count += 1
            return
        self.value += value
        self.count = 1

    def emit_value(self, name: str) -> MetricValue:
        if (name.endswith("LatencyMs") or name == "QueueWaitMs") and self.count:
            return self.value / self.count
        return self.value


@dataclass
class Bucket:
    metrics: dict[str, MetricAggregate] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)


def _stage() -> str:
    return os.environ.get("GATEWAY_STAGE", DEFAULT_STAGE)


def _service() -> str:
    return os.environ.get("GATEWAY_SERVICE", DEFAULT_SERVICE)


def _dimension_value(value: object) -> str:
    if value is None:
        return "none"
    text = str(value)
    return text[:1024]


def provider_endpoint_bucket(config: Mapping[str, Any]) -> str:
    return "custom" if config.get("custom_endpoint") else "default"


def _safe_json_value(value: Any) -> Any:
    if isinstance(value, PydanticBaseModel):
        return _safe_json_value(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        safe_mapping: dict[str, Any] = {}
        for key, item in sorted(mapping.items(), key=lambda entry: str(entry[0])):
            text_key = str(key)
            if text_key == "custom_endpoint":
                safe_mapping[text_key] = "custom" if item else "default"
            elif not _exclude_param_group_key(text_key):
                safe_mapping[text_key] = _safe_json_value(item)
        return safe_mapping
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        sequence = cast(Sequence[object], value)
        return [_safe_json_value(v) for v in sequence]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _exclude_param_group_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    if lowered in telemetry.CONTENT_SAFE_ATTRIBUTE_KEYS or lowered.endswith(
        telemetry.CONTENT_SAFE_SUFFIXES
    ):
        return False
    tokens = set(lowered.split("_"))
    content_tokens = {
        "body",
        "content",
        "input",
        "json",
        "message",
        "messages",
        "output",
        "payload",
        "raw",
        "system",
        "text",
        "user",
    }
    prompt_like = bool(
        tokens & {"prompt", "request", "response"} and tokens & content_tokens
    )
    return (
        key in telemetry.CONFIG_FINGERPRINT_EXCLUDED_PARAM_KEYS
        or telemetry.is_sensitive_key(key)
        or prompt_like
    )


def param_group(*parts: Any) -> str:
    """Group request params into a stable low-length dimension value."""
    safe_parts: list[object] = []
    for part in parts:
        if part in ({}, None):
            continue
        safe_part = _safe_json_value(part)
        if safe_part in ({}, [], None):
            continue
        safe_parts.append(safe_part)
    if not safe_parts:
        return "none"
    payload = json.dumps(safe_parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def model_dimensions(
    *,
    operation: str,
    model: str,
    config: Mapping[str, Any],
    params: Mapping[str, Any] | None = None,
    token_retry_params: Any | None = None,
) -> dict[str, str]:
    provider = model.partition("/")[0]
    return {
        "Stage": _stage(),
        "Service": _service(),
        "Operation": operation,
        "Provider": provider or "unknown",
        "Model": model,
        "ProviderEndpoint": provider_endpoint_bucket(config),
        "ParamGroup": param_group(config, params, token_retry_params),
    }


def service_dimensions() -> dict[str, str]:
    return {"Stage": _stage(), "Service": _service()}


def _normalize_dimensions(dimensions: Mapping[str, str]) -> dict[str, str]:
    return {key: _dimension_value(value) for key, value in dimensions.items()}


def emit_metrics(
    dimensions: Mapping[str, str],
    metrics: Mapping[str, MetricSpec],
    *,
    dimension_sets: Sequence[Sequence[str]] | None = None,
) -> None:
    if not metrics:
        return

    full_dimensions = _normalize_dimensions(dimensions)
    if dimension_sets is None:
        dimension_sets = [list(full_dimensions)]

    metric_defs: list[dict[str, str | int]] = []
    for name, (_value, unit) in metrics.items():
        metric_def: dict[str, str | int] = {"Name": name, "Unit": unit}
        if name in _HIGH_RESOLUTION_METRICS:
            metric_def["StorageResolution"] = 1
        metric_defs.append(metric_def)
    payload: dict[str, Any] = {
        "_aws": {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [
                {
                    "Namespace": NAMESPACE,
                    "Dimensions": [
                        list(dimension_set) for dimension_set in dimension_sets
                    ],
                    "Metrics": metric_defs,
                }
            ],
        },
        **full_dimensions,
    }
    for name, (value, _unit) in metrics.items():
        payload[name] = value

    print(json.dumps(payload, separators=(",", ":")), flush=True)


def record_metrics(
    dimensions: Mapping[str, str],
    metrics: Mapping[str, MetricSpec],
    *,
    dimension_sets: Sequence[Sequence[str]] | None = None,
) -> None:
    """Aggregate metrics for the background EMF flusher."""
    if not metrics:
        return

    full_dimensions = _normalize_dimensions(dimensions)
    if dimension_sets is None:
        dimension_sets = [list(full_dimensions)]

    key = (
        tuple(sorted(full_dimensions.items())),
        tuple(tuple(dimension_set) for dimension_set in dimension_sets),
    )
    with _pending_lock:
        bucket = _pending.setdefault(key, Bucket())
        for name, (value, unit) in metrics.items():
            bucket.metrics.setdefault(name, MetricAggregate()).add(name, value)
            bucket.units[name] = unit


def flush_metrics() -> int:
    """Emit all pending aggregate buckets. Returns emitted bucket count."""
    with _pending_lock:
        pending = dict(_pending)
        _pending.clear()

    for (dimension_items, dimension_sets), bucket in pending.items():
        emit_metrics(
            dict(dimension_items),
            {
                name: (aggregate.emit_value(name), bucket.units[name])
                for name, aggregate in bucket.metrics.items()
            },
            dimension_sets=[list(dimension_set) for dimension_set in dimension_sets],
        )
    return len(pending)


async def adjust_inflight(delta: int) -> int:
    global _inflight_requests
    async with _inflight_lock:
        _inflight_requests = max(0, _inflight_requests + delta)
        return _inflight_requests


async def get_inflight() -> int:
    async with _inflight_lock:
        return _inflight_requests


def record_inflight(value: int) -> None:
    record_metrics(
        service_dimensions(),
        {"InFlightRequests": (value, "Count")},
    )


def record_capacity(
    *,
    active: int,
    queued: int,
    max_active: int,
    max_queued: int,
    queue_wait_ms: float | None = None,
) -> None:
    """Record per-worker model-call capacity gauges."""
    capacity_dimensions = {
        **service_dimensions(),
        "WorkerId": str(os.getpid()),
    }
    capacity_metrics: dict[str, MetricSpec] = {
        "ActiveRequests": (active, "Count"),
        "QueuedRequests": (queued, "Count"),
        "GatewayDemand": (active + queued, "Count"),
        "MaxActiveRequests": (max_active, "Count"),
        "MaxQueuedRequests": (max_queued, "Count"),
    }
    if queue_wait_ms is not None:
        capacity_metrics["QueueWaitMs"] = (queue_wait_ms, "Milliseconds")
    record_metrics(
        capacity_dimensions,
        capacity_metrics,
        dimension_sets=[
            ["Stage", "Service"],
            ["Stage", "Service", "WorkerId"],
        ],
    )


def record_rejection(reason: str) -> None:
    record_metrics(
        {**service_dimensions(), "Reason": reason},
        {"RejectedRequests": (1, "Count")},
        dimension_sets=[
            ["Stage", "Service"],
            ["Stage", "Service", "Reason"],
        ],
    )


async def publish_metrics_periodically(
    stop_event: asyncio.Event,
    publishers: Sequence[MetricPublisher] = (),
) -> None:
    async def publish_once() -> None:
        record_inflight(await get_inflight())
        for publisher in publishers:
            result = publisher()
            if result is not None:
                await result
        flush_metrics()

    while not stop_event.is_set():
        await publish_once()
        try:
            await asyncio.wait_for(
                stop_event.wait(), timeout=METRICS_FLUSH_INTERVAL_SECONDS
            )
        except TimeoutError:
            pass
    await publish_once()


def create_metrics_middleware() -> Callable[
    [Request, RequestResponseEndpoint], Awaitable[Response]
]:
    async def metrics_middleware(
        request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        route = request.url.path or "/"
        method = request.method
        status_code = 500
        start = time.perf_counter()
        trace_route = telemetry.should_trace_http_route(route)
        span_context = (
            telemetry.start_span(
                telemetry.http_server_span_name(method, route),
                {
                    "http.request.method": method,
                    "url.path": route,
                    "gateway.route": route,
                },
                kind="server",
            )
            if trace_route
            else nullcontext()
        )
        with span_context:
            if trace_route:
                telemetry.add_event("gateway.http_request.start")
            record_inflight(await adjust_inflight(1))
            try:
                response = await call_next(request)
                status_code = response.status_code
                return response
            finally:
                current = await adjust_inflight(-1)
                latency_ms = (time.perf_counter() - start) * 1000
                if trace_route:
                    telemetry.set_attributes(
                        {
                            "http.response.status_code": status_code,
                            "gateway.http_latency_ms": latency_ms,
                            "gateway.inflight.current": current,
                        }
                    )
                    if status_code >= 500:
                        telemetry.set_status_error(str(status_code))
                    else:
                        telemetry.set_status_ok()
                    telemetry.add_event("gateway.http_request.done")
                dimensions = {
                    **service_dimensions(),
                    "Route": route,
                    "Method": method,
                    "StatusCode": str(status_code),
                    "StatusClass": f"{status_code // 100}xx",
                }
                record_metrics(
                    dimensions,
                    {
                        "HttpRequestCount": (1, "Count"),
                        "HttpLatencyMs": (latency_ms, "Milliseconds"),
                        "InFlightRequests": (current, "Count"),
                    },
                    dimension_sets=[
                        ["Stage", "Service", "Route", "Method", "StatusCode"],
                        ["Stage", "Service", "Route", "Method", "StatusClass"],
                        ["Stage", "Service"],
                    ],
                )

    return metrics_middleware


def emit_model_success(
    dimensions: Mapping[str, str],
    *,
    latency_ms: float,
    extra_metrics: Mapping[str, MetricSpec] | None = None,
) -> None:
    record_metrics(
        dimensions,
        {
            "ModelRequestCount": (1, "Count"),
            "ModelSuccessCount": (1, "Count"),
            "ModelLatencyMs": (latency_ms, "Milliseconds"),
            **dict(extra_metrics or {}),
        },
        dimension_sets=[
            [
                "Stage",
                "Service",
                "Operation",
                "Provider",
                "Model",
                "ProviderEndpoint",
                "ParamGroup",
            ],
            ["Stage", "Service", "Operation", "Provider", "Model"],
            ["Stage", "Service", "Operation", "Provider"],
            ["Stage", "Service"],
        ],
    )


def emit_model_error(
    dimensions: Mapping[str, str],
    *,
    error_code: str,
    latency_ms: float,
) -> None:
    error_dimensions = {**dimensions, "ErrorCode": error_code}
    record_metrics(
        error_dimensions,
        {
            "ModelRequestCount": (1, "Count"),
            "ModelErrorCount": (1, "Count"),
            "ModelLatencyMs": (latency_ms, "Milliseconds"),
        },
        dimension_sets=[
            [
                "Stage",
                "Service",
                "Operation",
                "Provider",
                "Model",
                "ProviderEndpoint",
                "ParamGroup",
                "ErrorCode",
            ],
            ["Stage", "Service", "Operation", "Provider", "Model", "ErrorCode"],
            ["Stage", "Service", "Operation", "Provider", "ErrorCode"],
            ["Stage", "Service", "ErrorCode"],
            ["Stage", "Service"],
        ],
    )
