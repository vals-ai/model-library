import json

import pytest

from model_gateway import metrics


@pytest.fixture(autouse=True)
def reset_metrics_state():
    metrics.flush_metrics()
    yield
    metrics.flush_metrics()


def _last_emf(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    out = capsys.readouterr().out.strip().splitlines()
    assert out
    return json.loads(out[-1])


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


@pytest.mark.asyncio
async def test_inflight_adjustment_never_goes_negative():
    current = await metrics.get_inflight()
    if current:
        await metrics.adjust_inflight(-current)

    assert await metrics.adjust_inflight(1) == 1
    assert await metrics.adjust_inflight(-1) == 0
    assert await metrics.adjust_inflight(-1) == 0
