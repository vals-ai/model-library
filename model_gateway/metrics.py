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
import model_library.telemetry as telemetry
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

NAMESPACE = "ModelProxy/Gateway"
DEFAULT_SERVICE = "gateway"
DEFAULT_STAGE = "unknown"
METRICS_FLUSH_INTERVAL_SECONDS = 10
QUERY_INFLIGHT_PATH = "/query"

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
    "UsageLedgerSqsActiveBatchSends",
    "UsageLedgerSqsPendingBytes",
    "UsageLedgerSqsPendingMessages",
}
_HIGH_RESOLUTION_METRICS = {
    "InFlightRequests",
    "ActiveRequests",
    "QueuedRequests",
    "GatewayDemand",
    "UsageLedgerSqsActiveBatchSends",
    "UsageLedgerSqsPendingBytes",
    "UsageLedgerSqsPendingMessages",
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


def _dimension_value(value: str) -> str:
    return value[:1024]


def provider_endpoint_bucket(config: Mapping[str, Any]) -> str:
    return "custom" if config.get("custom_endpoint") else "default"


def _param_group_value(value: object) -> object:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        keys = list(mapping)
        if not all(isinstance(key, str) for key in keys):
            raise TypeError("Param group mappings must use string keys")
        string_keys = cast(list[str], keys)
        safe_mapping: dict[str, object] = {}
        for key in sorted(string_keys):
            item = mapping[key]
            if key == "custom_endpoint":
                safe_mapping[key] = "custom" if item else "default"
            elif (
                key not in telemetry.CONFIG_FINGERPRINT_EXCLUDED_PARAM_KEYS
                and telemetry.is_safe_config_attribute_key(key)
            ):
                safe_mapping[key] = _param_group_value(item)
        return safe_mapping
    if isinstance(value, list):
        items = cast(list[object], value)
        return [_param_group_value(item) for item in items]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    raise TypeError(f"Unsupported param group value: {type(value).__name__}")


def param_group(*parts: object) -> str:
    """Group request params into a stable low-length dimension value."""
    safe_parts: list[object] = []
    for part in parts:
        if part is None:
            continue
        safe_part = _param_group_value(part)
        if safe_part in ({}, []):
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
    token_retry_params: Mapping[str, object] | None = None,
) -> dict[str, str]:
    provider = model.partition("/")[0]
    return {
        **service_dimensions(),
        "Operation": operation,
        "Provider": provider or "unknown",
        "Model": model,
        "ProviderEndpoint": provider_endpoint_bucket(config),
        "ParamGroup": param_group(config, params, token_retry_params),
    }


def service_dimensions() -> dict[str, str]:
    return {
        "Stage": os.environ.get("GATEWAY_STAGE", DEFAULT_STAGE),
        "Service": os.environ.get("GATEWAY_SERVICE", DEFAULT_SERVICE),
    }


def _usage_ledger_sqs_dimensions(**extra: str) -> dict[str, str]:
    return {
        **service_dimensions(),
        "Ledger": "usage_ledger",
        "Transport": "sqs",
        **extra,
    }


_SERVICE_DIMENSION_SET: DimensionSet = ("Stage", "Service")
_USAGE_LEDGER_SQS_DIMENSION_SET: DimensionSet = (
    "Stage",
    "Service",
    "Ledger",
    "Transport",
)
_USAGE_LEDGER_SQS_OPERATION_OUTCOME_DIMENSION_SET: DimensionSet = (
    "Stage",
    "Service",
    "Ledger",
    "Transport",
    "Operation",
    "Outcome",
)
_USAGE_LEDGER_SQS_OPERATION_ERROR_DIMENSION_SET: DimensionSet = (
    "Stage",
    "Service",
    "Ledger",
    "Transport",
    "Operation",
    "Outcome",
    "ErrorType",
)
_USAGE_LEDGER_SQS_PHASE_DIMENSION_SET: DimensionSet = (
    "Stage",
    "Service",
    "Ledger",
    "Transport",
    "Operation",
    "Phase",
)
_USAGE_LEDGER_SQS_PHASE_OUTCOME_DIMENSION_SET: DimensionSet = (
    "Stage",
    "Service",
    "Ledger",
    "Transport",
    "Operation",
    "Phase",
    "Outcome",
)
_USAGE_LEDGER_SQS_ATTEMPT_OUTCOME_DIMENSION_SET: DimensionSet = (
    "Stage",
    "Service",
    "Ledger",
    "Transport",
    "Operation",
    "AttemptOutcome",
)
_USAGE_LEDGER_SQS_ATTEMPT_DETAIL_DIMENSION_SET: DimensionSet = (
    "Stage",
    "Service",
    "Ledger",
    "Transport",
    "Operation",
    "AttemptNumber",
    "AttemptOutcome",
    "HttpStatusCode",
    "ErrorType",
)


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


def record_runtime(snapshot: Mapping[str, int]) -> None:
    """Record low-cardinality per-process runtime diagnostics."""
    record_metrics(
        {
            **service_dimensions(),
            "WorkerId": str(snapshot["pid"]),
        },
        {
            "ThreadCount": (snapshot["thread_count"], "Count"),
            "AsyncioTaskCount": (snapshot["asyncio_task_count"], "Count"),
            "OpenFileDescriptors": (snapshot["open_fd_count"], "Count"),
            "InboundSocketCount": (snapshot["inbound_socket_count"], "Count"),
            "OutboundSocketCount": (snapshot["outbound_socket_count"], "Count"),
            "RssBytes": (snapshot["rss_bytes"], "Bytes"),
            "EventLoopLagMs": (snapshot["event_loop_lag_ms"], "Milliseconds"),
        },
        dimension_sets=[
            ["Stage", "Service"],
            ["Stage", "Service", "WorkerId"],
        ],
    )


def record_usage_ledger_sqs_pending(
    *, pending_messages: int, pending_bytes: int, active_batch_sends: int
) -> None:
    """Record local SQS writer pressure for the usage ledger."""
    record_metrics(
        _usage_ledger_sqs_dimensions(),
        {
            "UsageLedgerSqsPendingMessages": (pending_messages, "Count"),
            "UsageLedgerSqsPendingBytes": (pending_bytes, "Bytes"),
            "UsageLedgerSqsActiveBatchSends": (active_batch_sends, "Count"),
        },
        dimension_sets=[
            _SERVICE_DIMENSION_SET,
            _USAGE_LEDGER_SQS_DIMENSION_SET,
        ],
    )


def record_usage_ledger_sqs_write(
    *, outcome: str, latency_ms: float, error_type: str = "none"
) -> None:
    """Record local enqueue/drop outcomes for SQS usage ledger writes."""
    write_metrics: dict[str, MetricSpec] = {
        "UsageLedgerSqsWriteCount": (1, "Count"),
        "UsageLedgerSqsWriteLatencyMs": (latency_ms, "Milliseconds"),
    }
    if outcome == "dropped":
        write_metrics["UsageLedgerSqsDroppedMessageCount"] = (1, "Count")
    record_metrics(
        _usage_ledger_sqs_dimensions(
            Operation="write_success",
            Outcome=outcome,
            ErrorType=error_type,
        ),
        write_metrics,
        dimension_sets=[
            _SERVICE_DIMENSION_SET,
            _USAGE_LEDGER_SQS_OPERATION_OUTCOME_DIMENSION_SET,
            _USAGE_LEDGER_SQS_OPERATION_ERROR_DIMENSION_SET,
        ],
    )


def record_usage_ledger_sqs_batch_send(
    *,
    outcome: str,
    latency_ms: float,
    message_count: int,
    failed_message_count: int,
    retried_message_count: int = 0,
    error_type: str = "none",
) -> None:
    """Record one SQS SendMessageBatch attempt for the usage ledger."""
    batch_metrics: dict[str, MetricSpec] = {
        "UsageLedgerSqsBatchSendCount": (1, "Count"),
        "UsageLedgerSqsBatchSendLatencyMs": (latency_ms, "Milliseconds"),
        "UsageLedgerSqsBatchMessageCount": (message_count, "Count"),
        "UsageLedgerSqsBatchFailedMessageCount": (failed_message_count, "Count"),
    }
    if retried_message_count:
        batch_metrics["UsageLedgerSqsRetriedMessageCount"] = (
            retried_message_count,
            "Count",
        )
    record_metrics(
        _usage_ledger_sqs_dimensions(
            Operation="send_message_batch",
            Outcome=outcome,
            ErrorType=error_type,
        ),
        batch_metrics,
        dimension_sets=[
            _SERVICE_DIMENSION_SET,
            _USAGE_LEDGER_SQS_OPERATION_OUTCOME_DIMENSION_SET,
            _USAGE_LEDGER_SQS_OPERATION_ERROR_DIMENSION_SET,
        ],
    )


def record_usage_ledger_sqs_phase(
    *, phase: str, outcome: str, latency_ms: float
) -> None:
    """Record app-level SQS writer phase timing."""
    phase_metrics: dict[str, MetricSpec] = {
        "UsageLedgerSqsPhaseCount": (1, "Count"),
        "UsageLedgerSqsPhaseLatencyMs": (latency_ms, "Milliseconds"),
    }
    record_metrics(
        _usage_ledger_sqs_dimensions(
            Operation="send_message_batch",
            Phase=phase,
            Outcome=outcome,
        ),
        phase_metrics,
        dimension_sets=[
            _SERVICE_DIMENSION_SET,
            _USAGE_LEDGER_SQS_PHASE_DIMENSION_SET,
            _USAGE_LEDGER_SQS_PHASE_OUTCOME_DIMENSION_SET,
        ],
    )


def record_usage_ledger_sqs_sdk_attempt(
    *,
    attempt_number: int,
    outcome: str,
    http_status_code: str,
    error_type: str,
    latency_ms: float,
    wire_latency_ms: float,
) -> None:
    """Record one SDK-level SQS attempt observed by botocore events."""
    record_metrics(
        _usage_ledger_sqs_dimensions(
            Operation="send_message_batch",
            AttemptNumber=str(attempt_number),
            AttemptOutcome=outcome,
            HttpStatusCode=http_status_code,
            ErrorType=error_type,
        ),
        {
            "UsageLedgerSqsSdkAttemptCount": (1, "Count"),
            "UsageLedgerSqsSdkAttemptLatencyMs": (latency_ms, "Milliseconds"),
            "UsageLedgerSqsSdkWireLatencyMs": (wire_latency_ms, "Milliseconds"),
        },
        dimension_sets=[
            _SERVICE_DIMENSION_SET,
            _USAGE_LEDGER_SQS_ATTEMPT_OUTCOME_DIMENSION_SET,
            _USAGE_LEDGER_SQS_ATTEMPT_DETAIL_DIMENSION_SET,
        ],
    )


def record_usage_ledger_sqs_sdk_call(*, outcome: str, retry_count: int) -> None:
    """Record SDK retry count for one completed SendMessageBatch call."""
    record_metrics(
        _usage_ledger_sqs_dimensions(
            Operation="send_message_batch",
            Outcome=outcome,
        ),
        {"UsageLedgerSqsSdkRetryCount": (retry_count, "Count")},
        dimension_sets=[
            _SERVICE_DIMENSION_SET,
            _USAGE_LEDGER_SQS_OPERATION_OUTCOME_DIMENSION_SET,
        ],
    )


def record_gateway_phase(
    *,
    operation: str,
    provider: str | None,
    phase: str,
    outcome: str,
    latency_ms: float,
) -> None:
    """Record low-cardinality phase latency metrics."""
    record_metrics(
        {
            **service_dimensions(),
            "Operation": operation,
            "Provider": provider or "unknown",
            "Phase": phase,
            "Outcome": outcome,
        },
        {
            "GatewayPhaseCount": (1, "Count"),
            "GatewayPhaseLatencyMs": (latency_ms, "Milliseconds"),
        },
        dimension_sets=[
            ["Stage", "Service", "Operation", "Provider", "Phase", "Outcome"],
            ["Stage", "Service", "Operation", "Phase", "Outcome"],
            ["Stage", "Service"],
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
        route = request.url.path
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
        tracks_query_inflight = route == QUERY_INFLIGHT_PATH
        current_inflight = await get_inflight()
        with span_context:
            if trace_route:
                telemetry.add_event("gateway.http_request.start")
            if tracks_query_inflight:
                current_inflight = await adjust_inflight(1)
                record_inflight(current_inflight)
            try:
                response = await call_next(request)
                status_code = response.status_code
                return response
            finally:
                if tracks_query_inflight:
                    current_inflight = await adjust_inflight(-1)
                latency_ms = (time.perf_counter() - start) * 1000
                if trace_route:
                    telemetry.set_attributes(
                        {
                            "http.response.status_code": status_code,
                            "gateway.http_latency_ms": latency_ms,
                            "gateway.inflight.current": current_inflight,
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
                http_metrics: dict[str, MetricSpec] = {
                    "HttpRequestCount": (1, "Count"),
                    "HttpLatencyMs": (latency_ms, "Milliseconds"),
                }
                if tracks_query_inflight:
                    http_metrics["InFlightRequests"] = (current_inflight, "Count")
                record_metrics(
                    dimensions,
                    http_metrics,
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
