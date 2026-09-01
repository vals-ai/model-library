import json
import logging
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from starlette.responses import Response

from model_gateway import metrics
from model_gateway.telemetry_helpers import dimension_telemetry_attributes


@pytest.fixture(autouse=True)
def reset_metrics_state():
    metrics.flush_metrics()
    yield
    metrics.flush_metrics()


def _emf_payloads(capsys: pytest.CaptureFixture[str]) -> list[dict[str, object]]:
    out = capsys.readouterr().out.strip().splitlines()
    return [json.loads(line) for line in out if line]


def _last_emf(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    payloads = _emf_payloads(capsys)
    assert payloads
    return payloads[-1]


def _metric_metadata(payload: dict[str, object]) -> dict[str, object]:
    aws_meta = payload["_aws"]
    assert isinstance(aws_meta, dict)
    cloudwatch_metrics = aws_meta["CloudWatchMetrics"]
    assert isinstance(cloudwatch_metrics, list)
    metric_meta = cloudwatch_metrics[0]
    assert isinstance(metric_meta, dict)
    return metric_meta


async def _reset_inflight() -> None:
    current = await metrics.get_inflight()
    if current:
        await metrics.adjust_inflight(-current)


def test_model_dimensions_include_model_endpoint_and_param_group(monkeypatch):
    monkeypatch.setenv("GATEWAY_STAGE", "dev")
    dims = metrics.model_dimensions(
        operation="query",
        model="openai/gpt-4o",
        config={"custom_endpoint": "https://provider.test/v1", "max_tokens": 7},
        params={"run_id": "run-a", "temperature": 0.2},
    )
    same_group = metrics.model_dimensions(
        operation="query",
        model="openai/gpt-4o",
        config={"custom_endpoint": "https://provider.test/v1", "max_tokens": 7},
        params={
            "run_id": "run-b",
            "question_id": "q-b",
            "query_id": "query-b",
            "temperature": 0.2,
        },
    )["ParamGroup"]
    other_endpoint_group = metrics.model_dimensions(
        operation="query",
        model="openai/gpt-4o",
        config={"custom_endpoint": "https://other-provider.test/v1", "max_tokens": 7},
        params={"temperature": 0.2},
    )["ParamGroup"]

    assert dims["Stage"] == "dev"
    assert dims["Provider"] == "openai"
    assert dims["Model"] == "openai/gpt-4o"
    assert dims["ProviderEndpoint"] == "custom"
    assert "https://provider.test/v1" not in dims.values()
    assert dims["ParamGroup"] == same_group
    assert dims["ParamGroup"] == other_endpoint_group
    assert dims["ParamGroup"] != "none"


def test_param_group_excludes_sensitive_and_prompt_like_keys():
    base = metrics.model_dimensions(
        operation="query",
        model="openai/gpt-4o",
        config={"max_tokens": 7},
        params={"temperature": 0.2},
    )["ParamGroup"]
    with_sensitive_values = metrics.model_dimensions(
        operation="query",
        model="openai/gpt-4o",
        config={
            "max_tokens": 7,
            "custom_api_key": "sk-secret",
            "system_prompt": "secret prompt",
            "response_text": "secret response",
        },
        params={
            "temperature": 0.2,
            "query_id": "query-a",
            "request_json": {"messages": ["secret"]},
        },
    )["ParamGroup"]

    assert with_sensitive_values == base


def test_param_group_returns_none_when_all_params_are_excluded():
    assert metrics.param_group({"system_prompt": "secret"}) == "none"
    assert metrics.param_group({"query_id": "query-a"}) == "none"
    assert (
        metrics.param_group({"messages": [{"role": "user", "content": "secret"}]})
        == "none"
    )
    assert metrics.param_group({"output": "secret"}) == "none"


def test_param_group_keeps_response_json_schema():
    group_a = metrics.model_dimensions(
        operation="query",
        model="google/gemini",
        config={"response_json_schema": {"type": "object"}},
    )["ParamGroup"]
    group_b = metrics.model_dimensions(
        operation="query",
        model="google/gemini",
        config={"response_json_schema": {"type": "array"}},
    )["ParamGroup"]

    assert group_a != group_b


def test_param_group_rejects_unsupported_values():
    with pytest.raises(TypeError, match="Unsupported param group value: object"):
        metrics.param_group({"max_tokens": object()})


def test_dimension_telemetry_attributes_use_model_dimensions():
    dimensions = metrics.model_dimensions(
        operation="query",
        model="openai/gpt-4o",
        config={"max_tokens": 7},
    )

    attrs = dimension_telemetry_attributes(dimensions)

    assert attrs == {
        "model.provider_endpoint": dimensions["ProviderEndpoint"],
        "model.param_group": dimensions["ParamGroup"],
        "gateway.operation": dimensions["Operation"],
    }


def test_record_metrics_sums_counters_and_averages_latency(capsys):
    metrics.record_metrics(
        {"Stage": "dev", "Service": "gateway", "Model": "openai/gpt-4o"},
        {"ModelRequestCount": (1, "Count"), "ModelLatencyMs": (100.0, "Milliseconds")},
        dimension_sets=[["Stage", "Service", "Model"], ["Stage", "Service"]],
    )
    metrics.record_metrics(
        {"Stage": "dev", "Service": "gateway", "Model": "openai/gpt-4o"},
        {"ModelRequestCount": (1, "Count"), "ModelLatencyMs": (23.4, "Milliseconds")},
        dimension_sets=[["Stage", "Service", "Model"], ["Stage", "Service"]],
    )

    assert metrics.flush_metrics() == 1
    payload = _last_emf(capsys)
    assert payload["Model"] == "openai/gpt-4o"
    assert payload["ModelRequestCount"] == 2
    assert payload["ModelLatencyMs"] == 61.7


def test_record_metrics_uses_max_gauge_value(capsys):
    metrics.record_metrics(
        {"Stage": "dev", "Service": "gateway"},
        {"ActiveRequests": (5, "Count"), "GatewayDemand": (7, "Count")},
    )
    metrics.record_metrics(
        {"Stage": "dev", "Service": "gateway"},
        {"ActiveRequests": (2, "Count"), "GatewayDemand": (3, "Count")},
    )

    assert metrics.flush_metrics() == 1
    payload = _last_emf(capsys)
    assert payload["ActiveRequests"] == 5
    assert payload["GatewayDemand"] == 7


def test_emit_metrics_writes_emf_with_requested_dimensions(capsys):
    metrics.emit_metrics(
        {"Stage": "dev", "Service": "gateway", "Model": "openai/gpt-4o"},
        {"ModelRequestCount": (1, "Count"), "ModelLatencyMs": (123.4, "Milliseconds")},
        dimension_sets=[["Stage", "Service", "Model"], ["Stage", "Service"]],
    )

    payload = _last_emf(capsys)
    aws_meta = payload["_aws"]
    assert isinstance(aws_meta, dict)
    metric_meta = aws_meta["CloudWatchMetrics"][0]
    assert metric_meta["Namespace"] == metrics.NAMESPACE
    assert metric_meta["Dimensions"] == [
        ["Stage", "Service", "Model"],
        ["Stage", "Service"],
    ]
    assert payload["Model"] == "openai/gpt-4o"
    assert payload["ModelRequestCount"] == 1
    assert payload["ModelLatencyMs"] == 123.4


def test_inflight_metrics_use_high_resolution(capsys):
    metrics.emit_metrics(
        {"Stage": "dev", "Service": "gateway"},
        {"InFlightRequests": (3, "Count")},
    )

    payload = _last_emf(capsys)
    aws_meta = payload["_aws"]
    assert isinstance(aws_meta, dict)
    metric_meta = aws_meta["CloudWatchMetrics"][0]
    assert metric_meta["Metrics"] == [
        {"Name": "InFlightRequests", "Unit": "Count", "StorageResolution": 1}
    ]


def test_record_capacity_uses_container_scoped_worker_id(capsys, monkeypatch):
    monkeypatch.setenv("GATEWAY_STAGE", "dev")
    monkeypatch.setenv("GATEWAY_SERVICE", "Gateway-dev-query")
    monkeypatch.setattr(
        "model_gateway.observability.socket.gethostname", lambda: "task-a"
    )
    monkeypatch.setattr("model_gateway.observability.os.getpid", lambda: 4321)

    metrics.record_capacity(
        active=2,
        queued=3,
        max_active=4,
        max_queued=5,
    )

    assert metrics.flush_metrics() == 1
    payload = _last_emf(capsys)
    assert payload["WorkerId"] == "task-a:4321"
    assert _metric_metadata(payload)["Dimensions"] == [
        ["Stage", "Service"],
        ["Stage", "Service", "WorkerId"],
    ]


def test_record_rate_limit_monitor_ownership_contract(capsys, monkeypatch):
    monkeypatch.setenv("GATEWAY_STAGE", "dev")
    monkeypatch.setenv("GATEWAY_SERVICE", "Gateway-dev-control")
    outcome = "acquired"

    metrics.record_rate_limit_monitor_ownership(outcome)

    assert metrics.flush_metrics() == 1
    payload = _last_emf(capsys)
    assert payload["Stage"] == "dev"
    assert payload["Service"] == "Gateway-dev-control"
    assert payload["Outcome"] == outcome
    assert payload["RateLimitMonitorOwnershipCount"] == 1
    assert _metric_metadata(payload)["Dimensions"] == [["Stage", "Service", "Outcome"]]
    assert _metric_metadata(payload)["Metrics"] == [
        {"Name": "RateLimitMonitorOwnershipCount", "Unit": "Count"}
    ]


def test_record_rate_limit_monitor_poll_contract(capsys, monkeypatch):
    monkeypatch.setenv("GATEWAY_STAGE", "dev")
    monkeypatch.setenv("GATEWAY_SERVICE", "Gateway-dev-control")
    outcome = "provider_error"

    metrics.record_rate_limit_monitor_poll(
        provider="anthropic",
        source="pool_1",
        outcome=outcome,
        latency_ms=12.5,
    )

    assert metrics.flush_metrics() == 1
    payload = _last_emf(capsys)
    assert payload["Provider"] == "anthropic"
    assert payload["Source"] == "pool_1"
    assert payload["Outcome"] == outcome
    assert payload["RateLimitMonitorPollCount"] == 1
    assert payload["RateLimitMonitorPollLatencyMs"] == 12.5
    assert "Model" not in payload
    metric_meta = _metric_metadata(payload)
    assert metric_meta["Dimensions"] == [
        ["Stage", "Service", "Provider", "Source", "Outcome"]
    ]
    assert metric_meta["Metrics"] == [
        {"Name": "RateLimitMonitorPollCount", "Unit": "Count"},
        {"Name": "RateLimitMonitorPollLatencyMs", "Unit": "Milliseconds"},
    ]


def test_record_rate_limit_monitor_publish_contract(capsys, monkeypatch):
    monkeypatch.setenv("GATEWAY_STAGE", "dev")
    monkeypatch.setenv("GATEWAY_SERVICE", "Gateway-dev-control")
    outcome = "accepted"

    metrics.record_rate_limit_monitor_publish(outcome)

    assert metrics.flush_metrics() == 1
    payload = _last_emf(capsys)
    assert payload["Outcome"] == outcome
    assert payload["RateLimitMonitorPublishCount"] == 1
    assert _metric_metadata(payload)["Dimensions"] == [["Stage", "Service", "Outcome"]]
    assert _metric_metadata(payload)["Metrics"] == [
        {"Name": "RateLimitMonitorPublishCount", "Unit": "Count"}
    ]


def test_record_gateway_phase_emits_without_env_gate(capsys, monkeypatch):
    monkeypatch.delenv("GATEWAY_DIAGNOSTICS_ENABLED", raising=False)

    metrics.record_gateway_phase(
        operation="query",
        provider="openai",
        phase="provider_call",
        outcome="success",
        latency_ms=12.5,
    )

    assert metrics.flush_metrics() == 1
    payload = _last_emf(capsys)
    assert payload["Operation"] == "query"
    assert payload["Provider"] == "openai"
    assert payload["Phase"] == "provider_call"
    assert payload["Outcome"] == "success"
    assert payload["GatewayPhaseCount"] == 1
    assert payload["GatewayPhaseLatencyMs"] == 12.5


def _log_record(name: str, level: int) -> logging.LogRecord:
    return logging.LogRecord(name, level, __file__, 1, "message", (), None)


def test_telemetry_delivery_handler_records_bounded_warning_and_error_metrics(
    capsys, monkeypatch
):
    monkeypatch.setenv("GATEWAY_STAGE", "dev")
    monkeypatch.setenv("GATEWAY_SERVICE", "Gateway-dev-release-control")
    handler = metrics.TelemetryDeliveryMetricHandler()

    handler.emit(_log_record("opentelemetry.exporter.otlp", logging.INFO))
    handler.emit(_log_record("mistralai.extra.observability.otel", logging.ERROR))
    handler.emit(_log_record("opentelemetry.exporter.otlp", logging.WARNING))
    handler.emit(_log_record("opentelemetry.sdk.trace.export", logging.ERROR))
    handler.emit(_log_record("opentelemetry.sdk.trace.export.batch", logging.CRITICAL))

    assert metrics.flush_metrics() == 1
    payload = _last_emf(capsys)
    assert payload["Stage"] == "dev"
    assert payload["Service"] == "Gateway-dev-release-control"
    assert payload["TelemetryDeliveryWarningCount"] == 1
    assert payload["TelemetryDeliveryErrorCount"] == 2
    aws_meta = payload["_aws"]
    assert isinstance(aws_meta, dict)
    cloudwatch_metrics = aws_meta["CloudWatchMetrics"]
    assert isinstance(cloudwatch_metrics, list)
    metric_meta = cloudwatch_metrics[0]
    assert isinstance(metric_meta, dict)
    assert metric_meta["Dimensions"] == [["Stage", "Service"], ["Stage"]]


def test_telemetry_delivery_handler_never_raises(monkeypatch):
    handler = metrics.TelemetryDeliveryMetricHandler()
    monkeypatch.setattr(
        metrics,
        "record_metrics",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("failed")),
    )

    handler.emit(_log_record("opentelemetry.exporter.otlp", logging.ERROR))


def test_telemetry_delivery_handler_installation_is_idempotent():
    root_logger = logging.getLogger()
    first = metrics.install_telemetry_delivery_metric_handler()
    try:
        second = metrics.install_telemetry_delivery_metric_handler()

        assert first is second
        assert root_logger.handlers.count(first) == 1
    finally:
        metrics.remove_telemetry_delivery_metric_handler(first)

    assert first not in root_logger.handlers


@pytest.mark.asyncio
async def test_inflight_adjustment_never_goes_negative():
    await _reset_inflight()

    assert await metrics.adjust_inflight(1) == 1
    assert await metrics.adjust_inflight(-1) == 0
    assert await metrics.adjust_inflight(-1) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/rate-limit-monitor"),
        ("POST", "/rate-limit-monitor/activate"),
    ],
)
async def test_metrics_middleware_traces_rate_limit_monitor_routes(
    monkeypatch, method, path
):
    captured_names: list[str] = []

    def start_span(
        name: str,
        _attributes: dict[str, object],
        *,
        kind: str,
    ):
        assert kind == "server"
        captured_names.append(name)
        return nullcontext()

    monkeypatch.setattr(metrics.telemetry, "start_span", start_span)
    middleware = metrics.create_metrics_middleware()
    request = SimpleNamespace(
        url=SimpleNamespace(path=path),
        method=method,
        state=SimpleNamespace(gateway_api_key_name="dashboard"),
    )

    async def call_next(_request):
        return Response(status_code=200)

    response = await middleware(request, call_next)  # pyright: ignore[reportArgumentType]

    assert response.status_code == 200
    assert captured_names == [f"{method} {path}"]


@pytest.mark.asyncio
async def test_metrics_middleware_adds_api_key_name_to_traced_span(monkeypatch):
    captured: dict[str, object] = {}

    def start_span(
        name: str,
        attributes: dict[str, object],
        *,
        kind: str,
    ):
        captured["name"] = name
        captured["attributes"] = attributes
        captured["kind"] = kind
        return nullcontext()

    monkeypatch.setattr(
        metrics.telemetry, "should_trace_http_route", lambda _route: True
    )
    monkeypatch.setattr(metrics.telemetry, "start_span", start_span)
    middleware = metrics.create_metrics_middleware()
    request = SimpleNamespace(
        url=SimpleNamespace(path="/query"),
        method="POST",
        state=SimpleNamespace(gateway_api_key_name="security-testing"),
    )

    async def call_next(_request):
        return Response(status_code=200)

    response = await middleware(request, call_next)  # pyright: ignore[reportArgumentType]

    assert response.status_code == 200
    assert captured == {
        "name": "POST /query",
        "attributes": {
            "http.request.method": "POST",
            "url.path": "/query",
            "gateway.route": "/query",
            "gateway.api_key_name": "security-testing",
        },
        "kind": "server",
    }


@pytest.mark.asyncio
async def test_metrics_middleware_counts_inflight_only_for_query(capsys, monkeypatch):
    await _reset_inflight()
    monkeypatch.setattr(
        metrics.telemetry, "should_trace_http_route", lambda _route: False
    )
    middleware = metrics.create_metrics_middleware()

    async def call_next(_request):
        return Response(status_code=200)

    token_count_request = SimpleNamespace(
        url=SimpleNamespace(path="/tokens/count"),
        method="POST",
    )
    response = await middleware(token_count_request, call_next)  # pyright: ignore[reportArgumentType]
    assert response.status_code == 200
    assert await metrics.get_inflight() == 0
    assert metrics.flush_metrics() == 1
    token_payloads = _emf_payloads(capsys)
    assert any(payload.get("Route") == "/tokens/count" for payload in token_payloads)
    assert all("InFlightRequests" not in payload for payload in token_payloads)

    query_request = SimpleNamespace(url=SimpleNamespace(path="/query"), method="POST")
    response = await middleware(query_request, call_next)  # pyright: ignore[reportArgumentType]
    assert response.status_code == 200
    assert await metrics.get_inflight() == 0
    assert metrics.flush_metrics() == 2
    query_payloads = _emf_payloads(capsys)
    assert any(
        payload.get("Route") == "/query" and "InFlightRequests" in payload
        for payload in query_payloads
    )
