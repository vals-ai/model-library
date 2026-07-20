import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from decimal import Decimal
from inspect import isawaitable
from typing import Any, cast
from unittest.mock import patch

import pytest
from boto3.dynamodb.types import TypeSerializer
from botocore.exceptions import ClientError
from pydantic import ValidationError
from starlette.testclient import TestClient

from model_gateway import app as gateway_app
from model_gateway import metrics
from model_gateway import model_helpers
from model_gateway.usage_ledger import store as ledger_store
from model_gateway.usage_ledger import schema as ledger_schema
from model_gateway.usage_ledger.details import snapshot_usage_request
from model_gateway.usage_ledger.message import (
    MAX_USAGE_LEDGER_MESSAGE_BYTES,
    serialize_usage_event_message,
)
from model_gateway.usage_ledger.store import (
    DynamoDbUsageLedger,
    NoopUsageLedger,
    SqsUsageLedger,
    UsageLedgerWriteError,
    build_success_usage_event,
    create_usage_ledger_from_env,
)
from model_gateway.types import QueryRequest
from model_library.base.input import TextInput, ToolBody, ToolDefinition
from model_library.base.output import (
    ProviderToolEvent,
    QueryPerformanceEvent,
    QueryPerformanceTimelineEntry,
    decompress_query_result_performance,
    QueryResult,
    QueryResultCost,
    QueryResultExtras,
    QueryResultMetadata,
    QueryResultPerformance,
    CompressedQueryResultPerformance,
)


@pytest.fixture(autouse=True)
def reset_gateway_metrics():
    metrics.flush_metrics()
    yield
    metrics.flush_metrics()


def _emf_payloads(capsys: pytest.CaptureFixture[str]) -> list[dict[str, object]]:
    return [
        cast(dict[str, object], json.loads(line))
        for line in capsys.readouterr().out.splitlines()
    ]


def _payload_with_metric(
    payloads: list[dict[str, object]], metric_name: str
) -> dict[str, object]:
    for payload in payloads:
        if metric_name in payload:
            return payload
    raise AssertionError(f"Missing EMF metric: {metric_name}")


def _payloads_with_metric(
    payloads: list[dict[str, object]], metric_name: str
) -> list[dict[str, object]]:
    return [payload for payload in payloads if metric_name in payload]


def _metric_number(payload: dict[str, object], metric_name: str) -> int | float:
    value = payload[metric_name]
    assert isinstance(value, int | float)
    return value


def _usage_event_body_bytes(event: dict[str, object]) -> int:
    return len(serialize_usage_event_message(event).encode("utf-8"))


def _content_mapping_summary(value: dict[str, object]) -> dict[str, object]:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return {
        "type": "object",
        "fields": len(value),
        "bytes": len(serialized.encode("utf-8")),
    }


async def _drain_sqs_flush(ledger: SqsUsageLedger) -> None:
    for _ in range(20):
        await _drain_ready_sqs_flush(ledger)
        retry_timer_task = ledger._retry_timer_task
        if retry_timer_task is not None and not retry_timer_task.done():
            await asyncio.wait_for(retry_timer_task, timeout=1)
        if (
            (ledger._flush_timer_task is None or ledger._flush_timer_task.done())
            and (ledger._retry_timer_task is None or ledger._retry_timer_task.done())
            and (ledger._flush_task is None or ledger._flush_task.done())
            and not ledger._pending
            and not ledger._retry_pending
            and ledger._inflight_messages == 0
        ):
            return
        await asyncio.sleep(0)
    raise AssertionError("SQS usage ledger flush did not drain")


async def _drain_ready_sqs_flush(ledger: SqsUsageLedger) -> None:
    timer_task = ledger._flush_timer_task
    if timer_task is not None and not timer_task.done():
        await asyncio.wait_for(timer_task, timeout=1)
    flush_task = ledger._flush_task
    if flush_task is not None and not flush_task.done():
        await asyncio.wait_for(flush_task, timeout=1)


class FakeUsageLedger:
    enabled = True

    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    async def start(self) -> None:
        return

    async def close(self) -> None:
        return

    async def write_success(self, event: dict[str, object]) -> None:
        self.events.append(event)


class FakeDynamoClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.get_calls: list[dict[str, object]] = []
        self.put_error: Exception | None = None
        self.get_item_response: dict[str, object] = {}

    async def put_item(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        if self.put_error is not None:
            raise self.put_error
        return {}

    async def get_item(self, **kwargs: object) -> dict[str, object]:
        self.get_calls.append(kwargs)
        return self.get_item_response


class FakeDynamoContext:
    def __init__(self, client: FakeDynamoClient) -> None:
        self.client = client
        self.entered = False
        self.exited = False

    async def __aenter__(self) -> FakeDynamoClient:
        self.entered = True
        return self.client

    async def __aexit__(self, *_args: object) -> None:
        self.exited = True


class FakeSqsServiceId:
    def hyphenize(self) -> str:
        return "sqs"


class FakeSqsServiceModel:
    service_id = FakeSqsServiceId()


class FakeSqsEvents:
    def __init__(self) -> None:
        self.handlers: dict[str, list[Callable[..., object]]] = {}

    def register(
        self, event_name: str, handler: Callable[..., object], **_kwargs: object
    ) -> None:
        self.handlers.setdefault(event_name, []).append(handler)

    async def emit(self, event_name: str, **kwargs: object) -> None:
        for handler in self.handlers.get(event_name, []):
            result = handler(**kwargs)
            if isawaitable(result):
                await cast(Awaitable[object], result)


class FakeSqsMeta:
    def __init__(self) -> None:
        self.service_model = FakeSqsServiceModel()
        self.method_to_api_mapping = {"send_message_batch": "SendMessageBatch"}
        self.events = FakeSqsEvents()


class FakeSqsRequest:
    def __init__(self) -> None:
        self.context: dict[str, object] = {}


class FakeSqsHttpResponse:
    status_code = 200


class FakeSqsClient:
    def __init__(self) -> None:
        self.batch_calls: list[dict[str, object]] = []
        self.send_error: Exception | None = None
        self.send_errors: list[Exception] = []
        self.failed_batch_entry_ids: set[str] = set()
        self.failed_batch_entry_ids_by_call: list[set[str]] = []
        self.batch_started: asyncio.Event | None = None
        self.release_batch: asyncio.Event | None = None
        self.send_delay_seconds = 0.0
        self.active_batch_sends = 0
        self.max_active_batch_sends = 0
        self.meta = FakeSqsMeta()
        self.emit_sdk_events = False

    async def send_message_batch(self, **kwargs: object) -> dict[str, object]:
        self.batch_calls.append(kwargs)
        if self.emit_sdk_events:
            await self._emit_sdk_attempt(attempt=1)
        if self.send_errors:
            raise self.send_errors.pop(0)
        if self.send_error is not None:
            raise self.send_error
        self.active_batch_sends += 1
        self.max_active_batch_sends = max(
            self.max_active_batch_sends, self.active_batch_sends
        )
        try:
            if self.batch_started is not None:
                self.batch_started.set()
            if self.release_batch is not None:
                await self.release_batch.wait()
            if self.send_delay_seconds > 0:
                await asyncio.sleep(self.send_delay_seconds)
            entries = cast(list[dict[str, object]], kwargs["Entries"])
            failed_entry_ids = (
                self.failed_batch_entry_ids_by_call.pop(0)
                if self.failed_batch_entry_ids_by_call
                else self.failed_batch_entry_ids
            )
            failed = [
                {
                    "Id": str(entry["Id"]),
                    "SenderFault": False,
                    "Code": "InternalError",
                    "Message": "temporary sqs failure",
                }
                for entry in entries
                if entry["Id"] in failed_entry_ids
            ]
            successful = [
                {"Id": str(entry["Id"]), "MessageId": f"message-{entry['Id']}"}
                for entry in entries
                if entry["Id"] not in failed_entry_ids
            ]
            response = {"Successful": successful, "Failed": failed}
            if self.emit_sdk_events:
                response["ResponseMetadata"] = {"RetryAttempts": 0}
            return response
        finally:
            self.active_batch_sends -= 1

    async def _emit_sdk_attempt(self, *, attempt: int) -> None:
        request = FakeSqsRequest()
        await self.meta.events.emit(
            "request-created.sqs.SendMessageBatch",
            request=request,
            operation_name="SendMessageBatch",
        )
        await self.meta.events.emit(
            "before-send.sqs.SendMessageBatch",
            request=request,
        )
        await self.meta.events.emit(
            "needs-retry.sqs.SendMessageBatch",
            attempts=attempt,
            response=(FakeSqsHttpResponse(), {"ResponseMetadata": {}}),
            caught_exception=None,
            request_dict={"context": request.context},
        )


class FakeSqsContext:
    def __init__(self, client: FakeSqsClient) -> None:
        self.client = client
        self.entered = False
        self.exited = False

    async def __aenter__(self) -> FakeSqsClient:
        self.entered = True
        return self.client

    async def __aexit__(self, *_args: object) -> None:
        self.exited = True


def _identity_with_compact_json_size(size: int) -> dict[str, str]:
    value_length = size - len('{"large":""}')
    assert value_length >= 0
    identity = {"large": "x" * value_length}
    assert len(json.dumps(identity, sort_keys=True, separators=(",", ":"))) == size
    return identity


def _build_success_usage_event(**kwargs: Any) -> dict[str, object]:
    request = cast(QueryRequest, kwargs["body"])
    return build_success_usage_event(
        **kwargs,
        request=snapshot_usage_request(request),
    )


def _basic_usage_event() -> dict[str, object]:
    request = QueryRequest(
        model="openai/gpt-4o",
        inputs=[TextInput(text="hello")],
        run_id="run-1",
        question_id="question-1",
        query_id="query-1",
    )
    return _build_success_usage_event(
        body=request,
        config=request.config_dict(),
        query_params={},
        dimensions={"ProviderEndpoint": "default", "ParamGroup": "pg"},
        result=QueryResult(output_text="ok", history=[]),
        completed_at=datetime(2026, 5, 29, 12, 0, tzinfo=UTC),
    )


class FailingUsageLedger:
    enabled = True

    async def start(self) -> None:
        return

    async def close(self) -> None:
        return

    async def write_success(self, event: dict[str, object]) -> None:
        _ = event
        raise RuntimeError("ledger unavailable")


def test_query_request_rejects_non_object_identity():
    with pytest.raises(ValidationError):
        QueryRequest(
            model="openai/gpt-4o",
            inputs=[TextInput(text="hello")],
            identity=[],
        )


def test_query_request_accepts_exact_identity_boundaries():
    exact_size_identity = _identity_with_compact_json_size(4096)
    exact_depth_identity = {
        "d1": {"d2": {"d3": {"d4": {"d5": {"d6": {"d7": {"d8": "ok"}}}}}}}
    }

    for identity in (exact_size_identity, exact_depth_identity):
        request = QueryRequest(
            model="openai/gpt-4o",
            inputs=[TextInput(text="hello")],
            identity=identity,
        )
        assert request.identity == identity


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_id", ""),
        ("run_id", " "),
        ("question_id", ""),
        ("question_id", "\t"),
    ],
)
def test_query_request_rejects_blank_run_and_question_ids(field: str, value: str):
    with pytest.raises(ValidationError):
        QueryRequest.model_validate(
            {
                "model": "openai/gpt-4o",
                "inputs": [TextInput(text="hello")],
                field: value,
            }
        )


@pytest.mark.parametrize("value", ["", " ", "\t"])
def test_query_request_treats_blank_query_id_as_absent(value: str):
    request = QueryRequest(
        model="openai/gpt-4o",
        inputs=[TextInput(text="hello")],
        query_id=value,
    )
    assert request.query_id is None


def test_query_request_rejects_oversized_identity():
    with pytest.raises(ValidationError):
        QueryRequest(
            model="openai/gpt-4o",
            inputs=[TextInput(text="hello")],
            identity={"large": "x" * 4097},
        )


def test_query_request_rejects_non_finite_identity_number():
    with pytest.raises(ValidationError):
        QueryRequest.model_validate_json(
            '{"model":"openai/gpt-4o","inputs":[{"kind":"text","text":"hello"}],'
            '"identity":{"nan_value":NaN}}'
        )


def test_query_request_rejects_over_depth_identity():
    with pytest.raises(ValidationError):
        QueryRequest(
            model="openai/gpt-4o",
            inputs=[TextInput(text="hello")],
            identity={
                "d1": {
                    "d2": {
                        "d3": {"d4": {"d5": {"d6": {"d7": {"d8": {"d9": "too-deep"}}}}}}
                    }
                }
            },
        )


def test_build_success_usage_event_owns_each_value_once_in_root_or_details():
    request = QueryRequest.model_validate(
        {
            "model": "openai/gpt-4o",
            "inputs": [TextInput(text="hello")],
            "config": {
                "max_tokens": 10,
                "custom_api_key": "provider-secret",
                "custom_endpoint": "https://custom.example/v1",
                "provider_config": {"prompt_cache_retention": "24h"},
            },
            "run_id": "run-1",
            "question_id": "question-1",
            "query_id": "query-1",
            "identity": {
                "email": "user@example.com",
                "benchmark_name": "swebench",
                "agent_name": "swe-agent",
            },
            "tools": [
                ToolDefinition(
                    name="search",
                    body=ToolBody(
                        name="search",
                        description="secret tool definition",
                        properties={"query": {"type": "string"}},
                        required=["query"],
                    ),
                )
            ],
        }
    )
    result = QueryResult(
        output_text="do not persist me",
        reasoning="do not persist reasoning",
        metadata=QueryResultMetadata(
            in_tokens=100,
            out_tokens=20,
            reasoning_tokens=5,
            cache_read_tokens=10,
            cache_write_tokens=3,
            cost=QueryResultCost(
                input=0.001,
                output=0.002,
                reasoning=0.0005,
                cache_read=0.0001,
                cache_write=0.0002,
            ),
            performance=QueryResultPerformance(
                timeline=[
                    QueryPerformanceTimelineEntry(
                        channel="content",
                        index=0,
                        events=[
                            QueryPerformanceEvent(
                                type="content_started", timestamp_ms=10
                            ),
                            QueryPerformanceEvent(
                                type="content_delta", timestamp_ms=25
                            ),
                            QueryPerformanceEvent(
                                type="content_finished", timestamp_ms=100
                            ),
                        ],
                    )
                ]
            ),
            extra={
                "token_metadata": {
                    "estimated": 150,
                    "actual": 125,
                },
            },
        ),
        provider_tool_events=[
            ProviderToolEvent(
                id="provider-tool-1",
                provider="openai",
                type="web_search_call",
                name="web_search",
                input="secret provider tool input",
                output={"content": "secret provider tool output"},
            )
        ],
        extras=QueryResultExtras(
            search_results={"content": "secret search result"},
            provider_response_id="provider-response-1",
            provider_request_id="provider-request-1",
        ),
    )

    event = _build_success_usage_event(
        body=request,
        config=request.config_dict(),
        query_params={},
        dimensions={"ProviderEndpoint": "default", "ParamGroup": "pg"},
        result=result,
        completed_at=datetime(2026, 5, 29, 12, 0, tzinfo=UTC),
        api_key_fingerprint="keyfingerprint",
    )

    assert event["run_id"] == "run-1"
    assert event["question_id"] == "question-1"
    assert event["query_id"] == "query-1"
    assert event[ledger_schema.IDENTITY_EMAIL] == "user@example.com"
    assert event["api_key_fingerprint"] == "keyfingerprint"
    assert event["finish_reason"] == "unknown"
    assert not event["finish_reason_raw"]
    assert event["schema_version"] == 2
    assert event["input_tokens"] == 100
    assert event["cache_read_tokens"] == 10
    assert event["total_input_tokens"] == 113
    assert event["total_output_tokens"] == 25

    shard = event["usage_shard"]
    assert event["GSI1PK"] == f"RUN#run-1#S#{shard}"
    assert event["GSI2PK"] == "QUERY#query-1"
    assert event["GSI3PK"] == f"KEY#keyfingerprint#DAY#20260529#S#{shard}"
    assert event["GSI4PK"] == f"BENCHMARK#swebench#S#{shard}"
    assert event["GSI5PK"] == f"AGENT#swe-agent#S#{shard}"
    assert event["GSI4SK"] == (
        "TS#2026-05-29T12:00:00Z#RUN#run-1#QUESTION#question-1"
        "#QUERY#query-1#USG#" + str(event["usage_event_id"])
    )
    assert event["GSI5SK"] == event["GSI4SK"]

    details = cast(dict[str, Any], event[ledger_schema.DETAILS_FIELD])
    assert set(details) == {"request", "result"}
    assert details["request"]["config"]["custom_api_key"] == "**********"
    assert (
        details["request"]["config"]["provider_config"]["prompt_cache_retention"]
        == "24h"
    )
    assert details["request"]["identity"] == request.identity
    tool = details["request"]["tools"][0]
    assert tool["name"] == "search"
    assert tool["body_length"] == len(
        json.dumps(
            request.tools[0].body.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    result_data = details["result"]
    assert result_data["output_text_length"] == len("do not persist me")
    assert result_data["reasoning_length"] == len("do not persist reasoning")
    assert "output_text" not in result_data
    assert "reasoning" not in result_data
    stored_performance = CompressedQueryResultPerformance.model_validate(
        result_data["metadata"]["performance"]
    )
    performance = decompress_query_result_performance(stored_performance)
    assert performance.timeline[0].channel == "content"
    assert performance.timeline[0].first_token_ms == 25
    assert len(performance.timeline[0].events) == 3
    assert result_data["metadata"]["extra"]["token_metadata"] == {
        "estimated": 150,
        "actual": 125,
    }
    provider_event = result_data["provider_tool_events"][0]
    assert provider_event["input_length"] == len("secret provider tool input")
    assert provider_event["output_length"] == len(
        '{"content":"secret provider tool output"}'
    )
    assert result_data["extras"]["search_results_length"] == len(
        '{"content":"secret search result"}'
    )
    assert result_data["extras"]["provider_request_id"] == "provider-request-1"
    assert result_data["extras"]["provider_response_id"] == "provider-response-1"

    serialized = json.dumps(event, default=str, sort_keys=True)
    message = serialize_usage_event_message(event)
    for secret in [
        "do not persist me",
        "do not persist reasoning",
        "provider-secret",
        "secret tool definition",
        "secret provider tool input",
        "secret provider tool output",
        "secret search result",
        "secret response",
        "do not persist",
    ]:
        assert secret not in serialized
        assert secret not in message


def test_build_success_usage_event_preserves_absent_performance_as_null():
    event = _basic_usage_event()

    details = cast(dict[str, Any], event[ledger_schema.DETAILS_FIELD])
    assert details["result"]["metadata"]["performance"] is None


def test_build_success_usage_event_normalizes_float_artifact_cost_total():
    request = QueryRequest(
        model="openai/gpt-4o",
        inputs=[TextInput(text="hello")],
        query_id="query-1",
    )
    result = QueryResult(
        output_text="ok",
        history=[],
        metadata=QueryResultMetadata(
            in_tokens=100,
            out_tokens=20,
            cost=QueryResultCost(input=0.6, output=0.7),
        ),
    )

    event = _build_success_usage_event(
        body=request,
        config=request.config_dict(),
        query_params={},
        dimensions={"ProviderEndpoint": "default", "ParamGroup": "pg"},
        result=result,
        completed_at=datetime(2026, 5, 29, 12, 0, tzinfo=UTC),
    )

    assert event["cost_usd"] == Decimal("1.3")


def test_build_success_usage_event_preserves_subcent_cost_precision_at_root():
    request = QueryRequest(
        model="openai/gpt-4o",
        inputs=[TextInput(text="hello")],
        query_id="query-1",
    )
    result = QueryResult(
        output_text="ok",
        history=[],
        metadata=QueryResultMetadata(
            in_tokens=100,
            out_tokens=20,
            cost=QueryResultCost(input=0.000000000001, output=0.0),
        ),
    )

    event = _build_success_usage_event(
        body=request,
        config=request.config_dict(),
        query_params={},
        dimensions={"ProviderEndpoint": "default", "ParamGroup": "pg"},
        result=result,
        completed_at=datetime(2026, 5, 29, 12, 0, tzinfo=UTC),
    )

    assert event["cost_usd"] == Decimal("0.000000000001")


def test_build_success_usage_event_serializes_integral_cost_without_exponent():
    request = QueryRequest(
        model="openai/gpt-4o",
        inputs=[TextInput(text="hello")],
        query_id="query-1",
    )
    result = QueryResult(
        output_text="ok",
        history=[],
        metadata=QueryResultMetadata(
            in_tokens=100,
            out_tokens=20,
            cost=QueryResultCost(input=1000.0, output=0.0),
        ),
    )

    event = _build_success_usage_event(
        body=request,
        config=request.config_dict(),
        query_params={},
        dimensions={"ProviderEndpoint": "default", "ParamGroup": "pg"},
        result=result,
        completed_at=datetime(2026, 5, 29, 12, 0, tzinfo=UTC),
    )

    assert event["cost_usd"] == Decimal("1000")


@pytest.mark.parametrize(
    ("identity", "expected_email"),
    [
        ({"email": " User@Example.COM "}, "user@example.com"),
        ({"email": "user+tag@example.co.uk"}, "user+tag@example.co.uk"),
        ({"email": f"{'x' * 500}@example.com"}, f"{'x' * 500}@example.com"),
    ],
)
def test_usage_event_extracts_canonical_identity_email(
    identity: dict[str, object], expected_email: str
):
    request = QueryRequest(
        model="openai/gpt-4o",
        inputs=[TextInput(text="hello")],
        query_id="query-1",
        identity=identity,
    )

    event = _build_success_usage_event(
        body=request,
        config=request.config_dict(),
        query_params={},
        dimensions={"ProviderEndpoint": "default", "ParamGroup": "pg"},
        result=QueryResult(output_text="ok", history=[]),
        completed_at=datetime(2026, 5, 29, 12, 0, tzinfo=UTC),
    )

    assert event[ledger_schema.IDENTITY_EMAIL] == expected_email
    assert "identity" not in event
    details = cast(dict[str, Any], event[ledger_schema.DETAILS_FIELD])
    assert details["request"]["identity"] == request.identity


@pytest.mark.parametrize(
    "identity",
    [
        {},
        {"email": ""},
        {"email": "   "},
        {"email": 123},
        {"email": "not-an-email"},
        {"email": "missing-at.example.com"},
        {"email": "user@example.com", "nested": {"email": "nested@example.com"}},
        {"email": "x" * 501 + "@example.com"},
    ],
)
def test_usage_event_omits_invalid_or_absent_identity_email(
    identity: dict[str, object],
):
    request = QueryRequest(
        model="openai/gpt-4o",
        inputs=[TextInput(text="hello")],
        query_id="query-1",
        identity=identity,
    )

    event = _build_success_usage_event(
        body=request,
        config=request.config_dict(),
        query_params={},
        dimensions={"ProviderEndpoint": "default", "ParamGroup": "pg"},
        result=QueryResult(output_text="ok", history=[]),
        completed_at=datetime(2026, 5, 29, 12, 0, tzinfo=UTC),
    )

    if identity.get("email") == "user@example.com":
        assert event[ledger_schema.IDENTITY_EMAIL] == "user@example.com"
    else:
        assert ledger_schema.IDENTITY_EMAIL not in event
    assert "identity" not in event
    details = cast(dict[str, Any], event[ledger_schema.DETAILS_FIELD])
    assert details["request"]["identity"] == request.identity


def test_usage_event_skips_dimension_indexes_for_non_string_or_blank_identity_values():
    request = QueryRequest(
        model="openai/gpt-4o",
        inputs=[TextInput(text="hello")],
        run_id="run-1",
        question_id="question-1",
        query_id="query-1",
        identity={"benchmark_name": "   ", "agent_name": 123},
    )
    event = _build_success_usage_event(
        body=request,
        config=request.config_dict(),
        query_params={},
        dimensions={"ProviderEndpoint": "default", "ParamGroup": "pg"},
        result=QueryResult(output_text="ok", history=[]),
        completed_at=datetime(2026, 5, 29, 12, 0, tzinfo=UTC),
    )

    assert "GSI4PK" not in event
    assert "GSI4SK" not in event
    assert "GSI5PK" not in event
    assert "GSI5SK" not in event


@pytest.mark.asyncio
async def test_dynamodb_usage_ledger_writes_only_event_row():
    request = QueryRequest(
        model="openai/gpt-4o",
        inputs=[TextInput(text="hello")],
        run_id="run-1",
        question_id="question-1",
        query_id="query-1",
        identity={"email": " User@Example.COM "},
    )
    result = QueryResult(
        output_text="ok",
        history=[],
        metadata=QueryResultMetadata(
            in_tokens=4,
            out_tokens=2,
            cost=QueryResultCost(input=0.001, output=0.002),
        ),
    )
    event = _build_success_usage_event(
        body=request,
        config=request.config_dict(),
        query_params={},
        dimensions={"ProviderEndpoint": "default", "ParamGroup": "pg"},
        result=result,
        completed_at=datetime(2026, 5, 29, 12, 0, tzinfo=UTC),
        api_key_fingerprint="keyfingerprint",
    )
    client = FakeDynamoClient()
    ledger = cast(Any, object.__new__(DynamoDbUsageLedger))
    ledger._table_name = "usage-table"
    ledger._serializer = TypeSerializer()
    ledger._client = client

    await ledger._write_success(event)

    assert len(client.calls) == 1
    put_item = client.calls[0]

    assert put_item["TableName"] == "usage-table"
    assert put_item["ConditionExpression"] == "attribute_not_exists(PK)"
    assert put_item["Item"]["PK"]["S"].startswith("USAGE#DAY#20260529#S#")  # pyright: ignore[reportIndexIssue]
    assert put_item["Item"]["SK"]["S"].startswith("TS#2026-05-29T12:00:00Z#USG#")  # pyright: ignore[reportIndexIssue]
    assert put_item["Item"]["entity_type"]["S"] == "usage_event"  # pyright: ignore[reportIndexIssue]
    assert put_item["Item"][ledger_schema.IDENTITY_EMAIL]["S"] == "user@example.com"  # pyright: ignore[reportIndexIssue]
    assert all(
        not str(value).startswith("AGG#")
        for attribute in cast(dict[str, object], put_item["Item"]).values()
        for value in cast(dict[str, object], attribute).values()
    )


@pytest.mark.asyncio
async def test_dynamodb_usage_ledger_writes_canonical_details_without_rewriting():
    event = _basic_usage_event()
    event["details"] = {
        "request": {"custom": {"value": "keep"}},
        "result": {"metadata": {"performance": None}},
    }
    client = FakeDynamoClient()
    ledger = cast(Any, object.__new__(DynamoDbUsageLedger))
    ledger._table_name = "usage-table"
    ledger._client = client

    await ledger._write_success(event)

    item = cast(dict[str, Any], client.calls[0]["Item"])
    details = cast(dict[str, Any], item["details"]["M"])
    assert details == {
        "request": {"M": {"custom": {"M": {"value": {"S": "keep"}}}}},
        "result": {"M": {"metadata": {"M": {"performance": {"NULL": True}}}}},
    }


@pytest.mark.asyncio
async def test_dynamodb_usage_ledger_treats_duplicate_same_usage_event_as_success():
    request = QueryRequest(
        model="openai/gpt-4o",
        inputs=[TextInput(text="hello")],
        query_id="query-1",
    )
    event = _build_success_usage_event(
        body=request,
        config=request.config_dict(),
        query_params={},
        dimensions={"ProviderEndpoint": "default", "ParamGroup": "pg"},
        result=QueryResult(output_text="ok", history=[]),
        completed_at=datetime(2026, 5, 29, 12, 0, tzinfo=UTC),
    )
    client = FakeDynamoClient()
    client.put_error = ClientError(
        {"Error": {"Code": "ConditionalCheckFailedException", "Message": "exists"}},
        "PutItem",
    )
    client.get_item_response = {
        "Item": {"usage_event_id": {"S": str(event["usage_event_id"])}}
    }
    ledger = cast(Any, object.__new__(DynamoDbUsageLedger))
    ledger._table_name = "usage-table"
    ledger._serializer = TypeSerializer()
    ledger._client = client

    await ledger._write_success(event)

    assert len(client.calls) == 1
    assert len(client.get_calls) == 1
    item = cast(dict[str, object], client.calls[0]["Item"])
    assert client.get_calls[0]["TableName"] == "usage-table"
    assert client.get_calls[0]["Key"] == {
        "PK": item["PK"],
        "SK": item["SK"],
    }
    assert client.get_calls[0]["ProjectionExpression"] == "usage_event_id"
    assert client.get_calls[0]["ConsistentRead"] is True


@pytest.mark.asyncio
async def test_dynamodb_usage_ledger_reraises_conditional_failure_for_different_event():
    request = QueryRequest(
        model="openai/gpt-4o",
        inputs=[TextInput(text="hello")],
        query_id="query-1",
    )
    event = _build_success_usage_event(
        body=request,
        config=request.config_dict(),
        query_params={},
        dimensions={"ProviderEndpoint": "default", "ParamGroup": "pg"},
        result=QueryResult(output_text="ok", history=[]),
        completed_at=datetime(2026, 5, 29, 12, 0, tzinfo=UTC),
    )
    client = FakeDynamoClient()
    client.put_error = ClientError(
        {"Error": {"Code": "ConditionalCheckFailedException", "Message": "exists"}},
        "PutItem",
    )
    client.get_item_response = {"Item": {"usage_event_id": {"S": "usg_other"}}}
    ledger = cast(Any, object.__new__(DynamoDbUsageLedger))
    ledger._table_name = "usage-table"
    ledger._serializer = TypeSerializer()
    ledger._client = client

    with pytest.raises(ClientError):
        await ledger._write_success(event)


def test_reused_query_id_records_distinct_completed_usage_events():
    request = QueryRequest(
        model="openai/gpt-4o",
        inputs=[TextInput(text="hello")],
        run_id="run-1",
        question_id="question-1",
        query_id="query-1",
    )
    result = QueryResult(output_text="ok", history=[])

    first = _build_success_usage_event(
        body=request,
        config=request.config_dict(),
        query_params={},
        dimensions={"ProviderEndpoint": "default", "ParamGroup": "pg"},
        result=result,
        completed_at=datetime(2026, 5, 29, 12, 0, tzinfo=UTC),
        api_key_fingerprint="keyfingerprint",
    )
    second = _build_success_usage_event(
        body=request,
        config=request.config_dict(),
        query_params={},
        dimensions={"ProviderEndpoint": "default", "ParamGroup": "pg"},
        result=result,
        completed_at=datetime(2026, 5, 29, 12, 0, tzinfo=UTC),
        api_key_fingerprint="keyfingerprint",
    )

    assert first["query_id"] == second["query_id"]
    assert first["completed_at"] == second["completed_at"]
    assert first["usage_event_id"] != second["usage_event_id"]
    assert first["SK"] != second["SK"]


def test_usage_ledger_env_factory_returns_noop_when_disabled():
    with patch.dict(os.environ, {"GATEWAY_USAGE_LEDGER_MODE": "disabled"}, clear=True):
        ledger = create_usage_ledger_from_env()

    assert isinstance(ledger, NoopUsageLedger)


def test_usage_ledger_env_factory_defaults_to_shadow_sqs_when_queue_url_is_set():
    env = {
        "GATEWAY_USAGE_LEDGER_QUEUE_URL": "https://sqs.us-east-1.amazonaws.com/123/ledger",
    }
    with patch.dict(os.environ, env, clear=True):
        ledger = create_usage_ledger_from_env()

    assert isinstance(ledger, SqsUsageLedger)
    assert ledger._mode == "shadow"


def test_usage_ledger_env_factory_prefers_sqs_queue_url_over_dynamodb_table():
    env = {
        "GATEWAY_USAGE_LEDGER_MODE": "enforced",
        "GATEWAY_USAGE_LEDGER_QUEUE_URL": "https://sqs.us-east-1.amazonaws.com/123/ledger",
        "GATEWAY_USAGE_LEDGER_TABLE_NAME": "usage-table",
    }
    with patch.dict(os.environ, env, clear=True):
        ledger = create_usage_ledger_from_env()

    assert isinstance(ledger, SqsUsageLedger)
    assert ledger._queue_url == env["GATEWAY_USAGE_LEDGER_QUEUE_URL"]


def test_usage_ledger_env_factory_uses_direct_dynamodb_when_only_table_is_set():
    env = {
        "GATEWAY_USAGE_LEDGER_MODE": "enforced",
        "GATEWAY_USAGE_LEDGER_TABLE_NAME": "usage-table",
    }
    with patch.dict(os.environ, env, clear=True):
        ledger = create_usage_ledger_from_env()

    assert isinstance(ledger, DynamoDbUsageLedger)
    assert ledger._table_name == "usage-table"


def test_usage_ledger_env_factory_shadow_without_destination_returns_noop():
    with patch.dict(os.environ, {"GATEWAY_USAGE_LEDGER_MODE": "shadow"}, clear=True):
        ledger = create_usage_ledger_from_env()

    assert isinstance(ledger, NoopUsageLedger)


def test_usage_ledger_env_factory_enforced_without_destination_fails():
    with (
        patch.dict(os.environ, {"GATEWAY_USAGE_LEDGER_MODE": "enforced"}, clear=True),
        pytest.raises(ValueError, match="GATEWAY_USAGE_LEDGER_QUEUE_URL"),
    ):
        create_usage_ledger_from_env()


@pytest.mark.asyncio
async def test_sqs_usage_ledger_starts_and_closes_one_async_client_context():
    client = FakeSqsClient()
    context = FakeSqsContext(client)
    ledger = SqsUsageLedger(
        queue_url="https://sqs.us-east-1.amazonaws.com/123/ledger",
        mode="enforced",
        region_name="us-east-1",
    )

    with patch("model_gateway.usage_ledger.store.get_session") as get_session:
        get_session.return_value.create_client.return_value = context
        await ledger.start()

    get_session.return_value.create_client.assert_called_once()
    assert get_session.return_value.create_client.call_args.args == ("sqs",)
    create_client_kwargs = get_session.return_value.create_client.call_args.kwargs
    assert create_client_kwargs["region_name"] == "us-east-1"
    assert "config" in create_client_kwargs
    assert context.entered
    assert ledger._client is client

    await ledger.close()

    assert context.exited
    assert ledger._client is None
    assert ledger._client_context is None


@pytest.mark.asyncio
async def test_sqs_usage_ledger_sends_serialized_usage_event_message_in_batch(
    capsys: pytest.CaptureFixture[str],
):
    client = FakeSqsClient()
    ledger = SqsUsageLedger(
        queue_url="https://sqs.us-east-1.amazonaws.com/123/ledger",
        mode="shadow",
    )
    ledger._client = client
    ledger._register_sqs_event_handlers(client)
    event = _basic_usage_event()

    await ledger.write_success(event)
    await _drain_sqs_flush(ledger)

    assert len(client.batch_calls) == 1
    assert client.batch_calls[0]["QueueUrl"] == ledger._queue_url
    entries = cast(list[dict[str, object]], client.batch_calls[0]["Entries"])
    assert len(entries) == 1
    message = json.loads(cast(str, entries[0]["MessageBody"]))
    assert message["schema_version"] == 1
    assert message["event"]["usage_event_id"] == event["usage_event_id"]

    assert metrics.flush_metrics() >= 7
    payloads = _emf_payloads(capsys)
    pending_payload = _payload_with_metric(payloads, "UsageLedgerSqsPendingMessages")
    assert pending_payload["Ledger"] == "usage_ledger"
    assert pending_payload["Transport"] == "sqs"
    assert pending_payload["UsageLedgerSqsPendingMessages"] == 1
    assert pending_payload["UsageLedgerSqsActiveBatchSends"] == 1

    batch_payload = _payload_with_metric(payloads, "UsageLedgerSqsBatchSendCount")
    assert batch_payload["Operation"] == "send_message_batch"
    assert batch_payload["Outcome"] == "success"
    assert batch_payload["ErrorType"] == "none"
    assert batch_payload["UsageLedgerSqsBatchSendCount"] == 1
    assert batch_payload["UsageLedgerSqsBatchMessageCount"] == 1
    assert batch_payload["UsageLedgerSqsBatchFailedMessageCount"] == 0
    assert isinstance(batch_payload["UsageLedgerSqsBatchSendLatencyMs"], int | float)
    assert batch_payload["UsageLedgerSqsBatchSendLatencyMs"] >= 0

    phase_payloads = _payloads_with_metric(payloads, "UsageLedgerSqsPhaseCount")
    assert {payload["Phase"] for payload in phase_payloads} >= {
        "pending_queue",
        "flush_wait",
        "batch_start_wait",
    }
    assert "future_wait" not in {payload["Phase"] for payload in phase_payloads}
    for phase_payload in phase_payloads:
        assert phase_payload["Operation"] == "send_message_batch"
        assert phase_payload["Outcome"] == "success"
        assert _metric_number(phase_payload, "UsageLedgerSqsPhaseCount") >= 1
        assert _metric_number(phase_payload, "UsageLedgerSqsPhaseLatencyMs") >= 0

    write_payload = _payload_with_metric(payloads, "UsageLedgerSqsWriteCount")
    assert write_payload["Operation"] == "write_success"
    assert write_payload["Outcome"] == "success"
    assert write_payload["ErrorType"] == "none"
    assert write_payload["UsageLedgerSqsWriteCount"] == 1
    assert isinstance(write_payload["UsageLedgerSqsWriteLatencyMs"], int | float)
    assert write_payload["UsageLedgerSqsWriteLatencyMs"] >= 0


@pytest.mark.asyncio
async def test_sqs_usage_ledger_compacts_oversized_details_before_enqueue():
    client = FakeSqsClient()
    ledger = SqsUsageLedger(
        queue_url="https://sqs.us-east-1.amazonaws.com/123/ledger",
        mode="shadow",
    )
    ledger._client = client
    event = _basic_usage_event()
    event["details"] = {"large": "x" * MAX_USAGE_LEDGER_MESSAGE_BYTES}

    await ledger.write_success(event)
    await _drain_sqs_flush(ledger)

    entries = cast(list[dict[str, object]], client.batch_calls[0]["Entries"])
    message = json.loads(cast(str, entries[0]["MessageBody"]))
    assert message["event"]["details"] == {
        "truncated": True,
        "request": {},
        "result": {"metadata": {"performance": None}},
    }


@pytest.mark.asyncio
async def test_sqs_usage_ledger_shadow_returns_before_sqs_send_completes():
    client = FakeSqsClient()
    client.batch_started = asyncio.Event()
    client.release_batch = asyncio.Event()
    ledger = SqsUsageLedger(
        queue_url="https://sqs.us-east-1.amazonaws.com/123/ledger",
        mode="shadow",
    )
    ledger._client = client

    await ledger.write_success(_basic_usage_event())

    assert len(client.batch_calls) == 0
    await asyncio.wait_for(client.batch_started.wait(), timeout=1)
    assert len(client.batch_calls) == 1

    client.release_batch.set()
    await _drain_sqs_flush(ledger)


@pytest.mark.asyncio
async def test_sqs_usage_ledger_enforced_currently_uses_background_handoff():
    client = FakeSqsClient()
    client.batch_started = asyncio.Event()
    client.release_batch = asyncio.Event()
    ledger = SqsUsageLedger(
        queue_url="https://sqs.us-east-1.amazonaws.com/123/ledger",
        mode="enforced",
    )
    ledger._client = client

    await ledger.write_success(_basic_usage_event())

    assert len(client.batch_calls) == 0
    await asyncio.wait_for(client.batch_started.wait(), timeout=1)
    assert len(client.batch_calls) == 1

    client.release_batch.set()
    await _drain_sqs_flush(ledger)


@pytest.mark.asyncio
async def test_sqs_usage_ledger_splits_eleven_concurrent_events_by_count():
    client = FakeSqsClient()
    ledger = SqsUsageLedger(
        queue_url="https://sqs.us-east-1.amazonaws.com/123/ledger",
        mode="shadow",
    )
    ledger._client = client
    events = []
    for index in range(11):
        event = _basic_usage_event()
        event["usage_event_id"] = f"usg_{index}"
        events.append(event)

    await asyncio.gather(*(ledger.write_success(event) for event in events))
    await _drain_sqs_flush(ledger)

    assert sorted(
        len(cast(list[dict[str, object]], call["Entries"]))
        for call in client.batch_calls
    ) == [1, 10]
    entries = [
        entry
        for call in client.batch_calls
        for entry in cast(list[dict[str, object]], call["Entries"])
    ]
    assert len({entry["Id"] for entry in entries}) == 11
    event_ids = {
        json.loads(cast(str, entry["MessageBody"]))["event"]["usage_event_id"]
        for entry in entries
    }
    assert event_ids == {f"usg_{index}" for index in range(11)}


async def test_sqs_usage_ledger_batches_by_aggregate_message_bytes():
    client = FakeSqsClient()
    ledger = SqsUsageLedger(
        queue_url="https://sqs.us-east-1.amazonaws.com/123/ledger",
        mode="shadow",
    )
    ledger._client = client
    events = []
    for index in range(4):
        event = _basic_usage_event()
        event["usage_event_id"] = f"large-{index}"
        event["details"] = {
            "request": {"large": ""},
            "result": {"metadata": {"performance": None}},
        }
        empty_size = len(
            serialize_usage_event_message(
                event,
                max_bytes=MAX_USAGE_LEDGER_MESSAGE_BYTES * 2,
            ).encode("utf-8")
        )
        event["details"] = {
            "request": {"large": "x" * (MAX_USAGE_LEDGER_MESSAGE_BYTES - empty_size)},
            "result": {"metadata": {"performance": None}},
        }
        assert (
            len(
                serialize_usage_event_message(
                    event,
                    max_bytes=MAX_USAGE_LEDGER_MESSAGE_BYTES,
                ).encode("utf-8")
            )
            == MAX_USAGE_LEDGER_MESSAGE_BYTES
        )
        events.append(event)

    await asyncio.gather(*(ledger.write_success(event) for event in events))
    await _drain_sqs_flush(ledger)

    assert sorted(
        len(cast(list[dict[str, object]], call["Entries"]))
        for call in client.batch_calls
    ) == [1, 3]
    all_entries: list[dict[str, object]] = []
    for call in client.batch_calls:
        entries = cast(list[dict[str, object]], call["Entries"])
        assert len(entries) <= 10
        assert (
            sum(
                len(cast(str, entry["MessageBody"]).encode("utf-8"))
                for entry in entries
            )
            <= 1_048_576
        )
        all_entries.extend(entries)
    assert {
        json.loads(cast(str, entry["MessageBody"]))["event"]["usage_event_id"]
        for entry in all_entries
    } == {f"large-{index}" for index in range(4)}


@pytest.mark.asyncio
async def test_sqs_usage_ledger_bounds_concurrent_batch_sends(
    monkeypatch: pytest.MonkeyPatch,
):
    client = FakeSqsClient()
    client.batch_started = asyncio.Event()
    client.release_batch = asyncio.Event()
    ledger = SqsUsageLedger(
        queue_url="https://sqs.us-east-1.amazonaws.com/123/ledger",
        mode="shadow",
    )
    ledger._client = client
    monkeypatch.setattr(
        "model_gateway.usage_ledger.store.SQS_BATCH_SEND_CONCURRENCY", 1
    )
    events = []
    for index in range(20):
        event = _basic_usage_event()
        event["usage_event_id"] = f"usg_{index}"
        events.append(event)

    writes = [asyncio.create_task(ledger.write_success(event)) for event in events]
    await asyncio.wait_for(client.batch_started.wait(), timeout=1)
    await asyncio.sleep(0)

    assert len(client.batch_calls) == 1
    assert client.max_active_batch_sends == 1

    assert all(write.done() for write in writes)

    client.release_batch.set()
    await _drain_sqs_flush(ledger)
    assert len(client.batch_calls) == 2
    assert client.max_active_batch_sends == 1


@pytest.mark.asyncio
async def test_sqs_usage_ledger_retries_transient_batch_send_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    client = FakeSqsClient()
    client.send_errors = [ValueError("connect timeout")]
    client.emit_sdk_events = True
    ledger = SqsUsageLedger(
        queue_url="https://sqs.us-east-1.amazonaws.com/123/ledger",
        mode="shadow",
    )
    ledger._client = client
    ledger._register_sqs_event_handlers(client)
    monkeypatch.setattr(
        "model_gateway.usage_ledger.store.SQS_SEND_RETRY_DELAY_SECONDS", 0
    )

    await ledger.write_success(_basic_usage_event())
    await _drain_sqs_flush(ledger)

    assert len(client.batch_calls) == 2
    metrics.flush_metrics()
    payloads = _emf_payloads(capsys)
    retry_payload = _payload_with_metric(payloads, "UsageLedgerSqsRetriedMessageCount")
    assert retry_payload["Operation"] == "send_message_batch"
    assert retry_payload["Outcome"] == "exception"
    assert retry_payload["ErrorType"] == "ValueError"
    assert retry_payload["UsageLedgerSqsBatchSendCount"] == 1
    assert retry_payload["UsageLedgerSqsBatchMessageCount"] == 1
    assert retry_payload["UsageLedgerSqsBatchFailedMessageCount"] == 1
    assert retry_payload["UsageLedgerSqsRetriedMessageCount"] == 1

    sdk_attempt_payload = _payload_with_metric(
        payloads, "UsageLedgerSqsSdkAttemptCount"
    )
    assert sdk_attempt_payload["Operation"] == "send_message_batch"
    assert sdk_attempt_payload["AttemptOutcome"] == "success"
    assert sdk_attempt_payload["HttpStatusCode"] == "200"
    assert sdk_attempt_payload["ErrorType"] == "none"
    assert sdk_attempt_payload["UsageLedgerSqsSdkAttemptCount"] == 2
    assert isinstance(
        sdk_attempt_payload["UsageLedgerSqsSdkAttemptLatencyMs"], int | float
    )
    assert sdk_attempt_payload["UsageLedgerSqsSdkAttemptLatencyMs"] >= 0
    assert isinstance(
        sdk_attempt_payload["UsageLedgerSqsSdkWireLatencyMs"], int | float
    )
    assert sdk_attempt_payload["UsageLedgerSqsSdkWireLatencyMs"] >= 0

    sdk_call_payloads = _payloads_with_metric(payloads, "UsageLedgerSqsSdkRetryCount")
    sdk_call_payload = next(
        payload for payload in sdk_call_payloads if payload["Outcome"] == "success"
    )
    assert sdk_call_payload["Operation"] == "send_message_batch"
    assert sdk_call_payload["UsageLedgerSqsSdkRetryCount"] == 0


@pytest.mark.asyncio
async def test_sqs_usage_ledger_requeues_partial_batch_failure_after_immediate_retries(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    client = FakeSqsClient()
    ledger = SqsUsageLedger(
        queue_url="https://sqs.us-east-1.amazonaws.com/123/ledger",
        mode="shadow",
    )
    ledger._client = client
    events = []
    for index in range(3):
        event = _basic_usage_event()
        event["usage_event_id"] = f"usg_{index}"
        events.append(event)

    client.failed_batch_entry_ids_by_call = [{"2"}, {"2"}, set()]
    monkeypatch.setattr(
        "model_gateway.usage_ledger.store.SQS_SEND_RETRY_DELAY_SECONDS", 0
    )
    results = await asyncio.gather(
        *(ledger.write_success(event) for event in events), return_exceptions=True
    )
    await _drain_sqs_flush(ledger)

    assert results == [None, None, None]
    assert len(client.batch_calls) == 3
    retried_entries = cast(list[dict[str, object]], client.batch_calls[2]["Entries"])
    assert len(retried_entries) == 1
    assert retried_entries[0]["Id"] == "2"
    metrics.flush_metrics()
    payloads = _emf_payloads(capsys)
    partial_failure_payloads = [
        payload
        for payload in _payloads_with_metric(payloads, "UsageLedgerSqsBatchSendCount")
        if payload["Outcome"] == "partial_failure"
    ]
    assert partial_failure_payloads
    partial_failure_payload = partial_failure_payloads[-1]
    assert partial_failure_payload["ErrorType"] == "batch_entry_failure"
    assert partial_failure_payload["UsageLedgerSqsBatchMessageCount"] == 4
    assert partial_failure_payload["UsageLedgerSqsBatchFailedMessageCount"] == 2
    assert partial_failure_payload["UsageLedgerSqsRetriedMessageCount"] == 2


@pytest.mark.asyncio
async def test_sqs_usage_ledger_requeues_batch_exception_after_immediate_retries(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    client = FakeSqsClient()
    client.send_errors = [ValueError("connect timeout"), ValueError("connect timeout")]
    ledger = SqsUsageLedger(
        queue_url="https://sqs.us-east-1.amazonaws.com/123/ledger",
        mode="shadow",
    )
    ledger._client = client
    monkeypatch.setattr(
        "model_gateway.usage_ledger.store.SQS_SEND_RETRY_DELAY_SECONDS", 0
    )

    await ledger.write_success(_basic_usage_event())
    await _drain_sqs_flush(ledger)

    assert len(client.batch_calls) == 3
    assert not any(
        "Gateway usage ledger SQS batch send failed permanently" in record.message
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_sqs_usage_ledger_close_does_not_cancel_low_volume_flush_in_progress():
    client = FakeSqsClient()
    client.batch_started = asyncio.Event()
    client.release_batch = asyncio.Event()
    ledger = SqsUsageLedger(
        queue_url="https://sqs.us-east-1.amazonaws.com/123/ledger",
        mode="shadow",
    )
    ledger._client = client

    write_task = asyncio.create_task(ledger.write_success(_basic_usage_event()))
    await asyncio.wait_for(client.batch_started.wait(), timeout=1)
    close_task = asyncio.create_task(ledger.close())
    await asyncio.sleep(0)

    assert write_task.done()
    assert not close_task.done()
    client.release_batch.set()
    await asyncio.wait_for(close_task, timeout=1)


@pytest.mark.asyncio
async def test_sqs_usage_ledger_shadow_logs_write_started_after_close_begins(
    caplog: pytest.LogCaptureFixture,
):
    client = FakeSqsClient()
    client.batch_started = asyncio.Event()
    client.release_batch = asyncio.Event()
    ledger = SqsUsageLedger(
        queue_url="https://sqs.us-east-1.amazonaws.com/123/ledger",
        mode="shadow",
    )
    ledger._client = client

    first_write = asyncio.create_task(ledger.write_success(_basic_usage_event()))
    await asyncio.wait_for(client.batch_started.wait(), timeout=1)
    close_task = asyncio.create_task(ledger.close())
    await asyncio.sleep(0)

    await ledger.write_success(_basic_usage_event())

    assert any(
        "Gateway usage ledger SQS enqueue failed in shadow mode" in record.message
        for record in caplog.records
    )

    client.release_batch.set()
    await asyncio.wait_for(first_write, timeout=1)
    await asyncio.wait_for(close_task, timeout=1)


@pytest.mark.asyncio
async def test_sqs_usage_ledger_delayed_retries_count_toward_pending_bound(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    client = FakeSqsClient()
    client.send_errors = [ValueError("connect timeout")]
    ledger = SqsUsageLedger(
        queue_url="https://sqs.us-east-1.amazonaws.com/123/ledger",
        mode="shadow",
    )
    ledger._client = client
    monkeypatch.setattr("model_gateway.usage_ledger.store.SQS_MAX_PENDING_MESSAGES", 1)
    monkeypatch.setattr(
        "model_gateway.usage_ledger.store.SQS_SEND_RETRY_DELAY_SECONDS", 60
    )

    await ledger.write_success(_basic_usage_event())
    await _drain_ready_sqs_flush(ledger)

    assert len(ledger._retry_pending) == 1
    dropped_event = _basic_usage_event()
    dropped_event["usage_event_id"] = "usg_dropped_while_retry_pending"
    await ledger.write_success(dropped_event)

    metrics.flush_metrics()
    payloads = _emf_payloads(capsys)
    drop_payload = _payload_with_metric(payloads, "UsageLedgerSqsDroppedMessageCount")
    assert drop_payload["Outcome"] == "dropped"
    assert drop_payload["ErrorType"] == "PendingQueueCountFull"

    await ledger.close()


@pytest.mark.asyncio
async def test_sqs_usage_ledger_delayed_retries_do_not_block_fresh_messages(
    monkeypatch: pytest.MonkeyPatch,
):
    client = FakeSqsClient()
    client.send_errors = [ValueError("connect timeout")]
    ledger = SqsUsageLedger(
        queue_url="https://sqs.us-east-1.amazonaws.com/123/ledger",
        mode="shadow",
    )
    ledger._client = client
    monkeypatch.setattr(
        "model_gateway.usage_ledger.store.SQS_SEND_RETRY_DELAY_SECONDS", 60
    )

    retry_event = _basic_usage_event()
    retry_event["usage_event_id"] = "usg_retry_later"
    await ledger.write_success(retry_event)
    await _drain_ready_sqs_flush(ledger)

    fresh_event = _basic_usage_event()
    fresh_event["usage_event_id"] = "usg_fresh"
    await ledger.write_success(fresh_event)
    await _drain_ready_sqs_flush(ledger)

    assert len(client.batch_calls) == 2
    fresh_entries = cast(list[dict[str, object]], client.batch_calls[1]["Entries"])
    assert len(fresh_entries) == 1
    fresh_message = json.loads(cast(str, fresh_entries[0]["MessageBody"]))
    assert fresh_message["event"]["usage_event_id"] == "usg_fresh"

    await ledger.close()


@pytest.mark.asyncio
async def test_sqs_usage_ledger_close_flushes_delayed_retries_without_backoff_wait(
    monkeypatch: pytest.MonkeyPatch,
):
    client = FakeSqsClient()
    client.send_errors = [ValueError("connect timeout")]
    ledger = SqsUsageLedger(
        queue_url="https://sqs.us-east-1.amazonaws.com/123/ledger",
        mode="shadow",
    )
    ledger._client = client
    monkeypatch.setattr(
        "model_gateway.usage_ledger.store.SQS_SEND_RETRY_DELAY_SECONDS", 60
    )

    await ledger.write_success(_basic_usage_event())
    await _drain_ready_sqs_flush(ledger)

    assert len(ledger._retry_pending) == 1
    await asyncio.wait_for(ledger.close(), timeout=1)

    assert len(client.batch_calls) == 2
    assert ledger._retry_timer_task is None or ledger._retry_timer_task.done()


@pytest.mark.asyncio
async def test_sqs_usage_ledger_shadow_logs_background_send_failure_after_retry_age_limit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
):
    client = FakeSqsClient()
    client.send_error = ValueError("sqs unavailable")
    ledger = SqsUsageLedger(
        queue_url="https://sqs.us-east-1.amazonaws.com/123/ledger",
        mode="shadow",
    )
    ledger._client = client
    monkeypatch.setattr(
        "model_gateway.usage_ledger.store.SQS_SEND_RETRY_DELAY_SECONDS", 60
    )
    monkeypatch.setattr(
        "model_gateway.usage_ledger.store.SQS_MAX_SEND_RETRY_AGE_SECONDS", 60
    )

    await ledger.write_success(_basic_usage_event())
    await _drain_ready_sqs_flush(ledger)

    assert len(client.batch_calls) == 1
    assert len(ledger._retry_pending) == 1
    ledger._retry_pending[0].enqueued_at = 0
    ledger._retry_pending[0].retry_ready_at = 0
    ledger._cancel_retry_timer_locked()
    await ledger._move_due_retries_after_delay(0)
    await _drain_ready_sqs_flush(ledger)

    assert len(client.batch_calls) == 2
    assert any(
        "Gateway usage ledger SQS batch send failed permanently" in record.message
        and "ValueError" in record.message
        for record in caplog.records
    )
    metrics.flush_metrics()
    payloads = _emf_payloads(capsys)
    failure_payloads = [
        payload
        for payload in _payloads_with_metric(payloads, "UsageLedgerSqsBatchSendCount")
        if payload["Outcome"] == "exception"
    ]
    assert failure_payloads
    terminal_payload = failure_payloads[-1]
    assert terminal_payload["ErrorType"] == "ValueError"
    assert terminal_payload["UsageLedgerSqsBatchSendCount"] == 2
    assert terminal_payload["UsageLedgerSqsBatchFailedMessageCount"] == 2
    assert terminal_payload["UsageLedgerSqsRetriedMessageCount"] == 1


@pytest.mark.asyncio
async def test_sqs_usage_ledger_shadow_drops_when_pending_backlog_is_full(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
):
    client = FakeSqsClient()
    client.batch_started = asyncio.Event()
    client.release_batch = asyncio.Event()
    ledger = SqsUsageLedger(
        queue_url="https://sqs.us-east-1.amazonaws.com/123/ledger",
        mode="shadow",
    )
    ledger._client = client
    monkeypatch.setattr("model_gateway.usage_ledger.store.SQS_MAX_PENDING_MESSAGES", 1)

    first_event = _basic_usage_event()
    second_event = _basic_usage_event()
    second_event["usage_event_id"] = "usg_dropped"
    await ledger.write_success(first_event)
    await ledger.write_success(second_event)

    assert len(ledger._pending) == 1
    assert any(
        "Gateway usage ledger SQS pending queue is full; dropping usage event"
        in record.message
        for record in caplog.records
    )
    metrics.flush_metrics()
    payloads = _emf_payloads(capsys)
    drop_payload = _payload_with_metric(payloads, "UsageLedgerSqsDroppedMessageCount")
    assert drop_payload["Operation"] == "write_success"
    assert drop_payload["Outcome"] == "dropped"
    assert drop_payload["ErrorType"] == "PendingQueueCountFull"
    assert drop_payload["UsageLedgerSqsDroppedMessageCount"] == 1

    client.release_batch.set()
    await _drain_sqs_flush(ledger)


@pytest.mark.asyncio
async def test_sqs_usage_ledger_shadow_drops_when_pending_bytes_are_full(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
):
    client = FakeSqsClient()
    ledger = SqsUsageLedger(
        queue_url="https://sqs.us-east-1.amazonaws.com/123/ledger",
        mode="shadow",
    )
    ledger._client = client

    first_event = _basic_usage_event()
    second_event = _basic_usage_event()
    second_event["usage_event_id"] = "usg_dropped_by_bytes"
    first_event_bytes = _usage_event_body_bytes(first_event)
    monkeypatch.setattr(ledger_store, "SQS_MAX_PENDING_MESSAGES", 10)
    monkeypatch.setattr(
        ledger_store, "SQS_MAX_PENDING_BYTES", first_event_bytes, raising=False
    )

    await ledger.write_success(first_event)
    await ledger.write_success(second_event)

    assert len(ledger._pending) == 1
    assert any(
        "Gateway usage ledger SQS pending queue bytes are full; dropping usage event"
        in record.message
        for record in caplog.records
    )
    metrics.flush_metrics()
    payloads = _emf_payloads(capsys)
    pending_payload = _payload_with_metric(payloads, "UsageLedgerSqsPendingBytes")
    assert pending_payload["UsageLedgerSqsPendingBytes"] == first_event_bytes
    drop_payload = _payload_with_metric(payloads, "UsageLedgerSqsDroppedMessageCount")
    assert drop_payload["Operation"] == "write_success"
    assert drop_payload["Outcome"] == "dropped"
    assert drop_payload["ErrorType"] == "PendingQueueBytesFull"
    assert drop_payload["UsageLedgerSqsDroppedMessageCount"] == 1

    await _drain_sqs_flush(ledger)


@pytest.mark.asyncio
async def test_sqs_usage_ledger_inflight_bytes_count_toward_pending_budget(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    client = FakeSqsClient()
    client.batch_started = asyncio.Event()
    client.release_batch = asyncio.Event()
    ledger = SqsUsageLedger(
        queue_url="https://sqs.us-east-1.amazonaws.com/123/ledger",
        mode="shadow",
    )
    ledger._client = client

    first_event = _basic_usage_event()
    second_event = _basic_usage_event()
    second_event["usage_event_id"] = "usg_dropped_while_inflight"
    first_event_bytes = _usage_event_body_bytes(first_event)
    monkeypatch.setattr(ledger_store, "SQS_MAX_PENDING_MESSAGES", 10)
    monkeypatch.setattr(
        ledger_store, "SQS_MAX_PENDING_BYTES", first_event_bytes, raising=False
    )

    await ledger.write_success(first_event)
    await asyncio.wait_for(client.batch_started.wait(), timeout=1)
    assert len(ledger._pending) == 0
    assert ledger._inflight_messages == 1

    await ledger.write_success(second_event)

    metrics.flush_metrics()
    payloads = _emf_payloads(capsys)
    drop_payload = _payload_with_metric(payloads, "UsageLedgerSqsDroppedMessageCount")
    assert drop_payload["Outcome"] == "dropped"
    assert drop_payload["ErrorType"] == "PendingQueueBytesFull"

    client.release_batch.set()
    await _drain_sqs_flush(ledger)


@pytest.mark.asyncio
async def test_sqs_usage_ledger_delayed_retry_bytes_count_toward_pending_budget(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    client = FakeSqsClient()
    client.send_errors = [ValueError("connect timeout")]
    ledger = SqsUsageLedger(
        queue_url="https://sqs.us-east-1.amazonaws.com/123/ledger",
        mode="shadow",
    )
    ledger._client = client

    retry_event = _basic_usage_event()
    fresh_event = _basic_usage_event()
    fresh_event["usage_event_id"] = "usg_dropped_while_retry_bytes_pending"
    retry_event_bytes = _usage_event_body_bytes(retry_event)
    monkeypatch.setattr(ledger_store, "SQS_MAX_PENDING_MESSAGES", 10)
    monkeypatch.setattr(ledger_store, "SQS_SEND_RETRY_DELAY_SECONDS", 60)
    monkeypatch.setattr(ledger_store, "SQS_MAX_PENDING_BYTES", retry_event_bytes)

    await ledger.write_success(retry_event)
    await _drain_ready_sqs_flush(ledger)

    assert len(ledger._retry_pending) == 1
    assert ledger._pending_message_bytes == retry_event_bytes
    await ledger.write_success(fresh_event)

    metrics.flush_metrics()
    payloads = _emf_payloads(capsys)
    drop_payload = _payload_with_metric(payloads, "UsageLedgerSqsDroppedMessageCount")
    assert drop_payload["Outcome"] == "dropped"
    assert drop_payload["ErrorType"] == "PendingQueueBytesFull"

    await ledger.close()


@pytest.mark.asyncio
async def test_sqs_usage_ledger_close_does_not_double_count_retry_bytes(
    monkeypatch: pytest.MonkeyPatch,
):
    client = FakeSqsClient()
    client.send_errors = [ValueError("connect timeout")]
    ledger = SqsUsageLedger(
        queue_url="https://sqs.us-east-1.amazonaws.com/123/ledger",
        mode="shadow",
    )
    ledger._client = client
    monkeypatch.setattr(ledger_store, "SQS_SEND_RETRY_DELAY_SECONDS", 60)

    retry_event = _basic_usage_event()
    retry_event_bytes = _usage_event_body_bytes(retry_event)
    await ledger.write_success(retry_event)
    await _drain_ready_sqs_flush(ledger)

    assert len(ledger._retry_pending) == 1
    assert ledger._pending_message_bytes == retry_event_bytes

    client.batch_started = asyncio.Event()
    client.release_batch = asyncio.Event()
    close_task = asyncio.create_task(ledger.close())
    await asyncio.wait_for(client.batch_started.wait(), timeout=1)

    assert len(ledger._pending) == 0
    assert ledger._inflight_messages == 1
    assert ledger._pending_message_bytes == retry_event_bytes

    client.release_batch.set()
    await asyncio.wait_for(close_task, timeout=1)

    assert ledger._pending_message_bytes == 0
    assert ledger._inflight_messages == 0


@pytest.mark.asyncio
async def test_sqs_usage_ledger_partial_failure_releases_successful_message_bytes(
    monkeypatch: pytest.MonkeyPatch,
):
    client = FakeSqsClient()
    client.failed_batch_entry_ids = {"1"}
    ledger = SqsUsageLedger(
        queue_url="https://sqs.us-east-1.amazonaws.com/123/ledger",
        mode="shadow",
    )
    ledger._client = client
    monkeypatch.setattr(ledger_store, "SQS_SEND_RETRY_DELAY_SECONDS", 60)

    successful_event = _basic_usage_event()
    retry_event = _basic_usage_event()
    retry_event["usage_event_id"] = "usg_retry_entry"
    retry_event_bytes = _usage_event_body_bytes(retry_event)

    await ledger.write_success(successful_event)
    await ledger.write_success(retry_event)
    await _drain_ready_sqs_flush(ledger)

    assert len(ledger._retry_pending) == 1
    assert ledger._retry_pending[0].entry_id == "1"
    assert ledger._pending_message_bytes == retry_event_bytes

    await ledger.close()


@pytest.mark.asyncio
async def test_dynamodb_usage_ledger_starts_and_closes_one_async_client_context():
    client = FakeDynamoClient()
    context = FakeDynamoContext(client)
    ledger = cast(Any, object.__new__(DynamoDbUsageLedger))
    ledger._table_name = "usage-table"
    ledger._region_name = "us-east-1"
    ledger._mode = "enforced"
    ledger._serializer = TypeSerializer()
    ledger._client_context = None
    ledger._client = None

    with patch("model_gateway.usage_ledger.store.get_session") as get_session:
        get_session.return_value.create_client.return_value = context
        await ledger.start()

    assert context.entered
    assert ledger._client is client

    await ledger.close()

    assert context.exited
    assert ledger._client is None
    assert ledger._client_context is None


@pytest.mark.asyncio
async def test_dynamodb_usage_ledger_shadow_mode_swallows_write_failures():
    async def fail(_event: dict[str, object]) -> None:
        raise ValueError("dynamodb unavailable")

    ledger = cast(Any, object.__new__(DynamoDbUsageLedger))
    ledger._mode = "shadow"
    ledger._write_success = fail

    await ledger.write_success({})


@pytest.mark.asyncio
async def test_dynamodb_usage_ledger_enforced_mode_raises_on_write_failure():
    async def fail(_event: dict[str, object]) -> None:
        raise ValueError("dynamodb unavailable")

    ledger = cast(Any, object.__new__(DynamoDbUsageLedger))
    ledger._mode = "enforced"
    ledger._write_success = fail

    with pytest.raises(UsageLedgerWriteError):
        await ledger.write_success({})


def test_query_writes_success_usage_event_when_ledger_is_configured():
    from model_gateway import main
    from model_library.base.base import LLM
    from model_library.base.input import RawInput

    class ServerSettings:
        MODEL_GATEWAY_API_KEYS = "sk-test"
        MODEL_GATEWAY_HMAC_SECRET = "test-secret"

        def get(self, name: str, default: str = "") -> str:
            return getattr(self, name, default)

        def unset(self, key: str) -> None:
            pass

    class FakeLLM:
        async def query(self, _inputs: Any, **_kwargs: Any) -> QueryResult:
            return QueryResult(
                output_text="ok",
                history=[],
                metadata=QueryResultMetadata(
                    in_tokens=12,
                    out_tokens=3,
                    cost=QueryResultCost(input=0.001, output=0.002),
                ),
            )

    signed_inputs = LLM.serialize_input(
        [RawInput(input={"messages": [{"role": "user", "content": "hi"}]})],
        secret=b"test-secret",
    )
    raw_request_inputs = json.loads(signed_inputs)
    fake_ledger = FakeUsageLedger()
    with (
        patch.object(gateway_app, "model_library_settings", ServerSettings()),
        patch.object(gateway_app, "get_model_names", return_value=["openai/gpt-4o"]),
        patch.object(model_helpers, "get_registry_model", return_value=FakeLLM()),
    ):
        app = main.create_app()
        app.state.usage_ledger = fake_ledger
        client = TestClient(app)
        response = client.post(
            "/query",
            json={
                "model": "openai/gpt-4o",
                "inputs": raw_request_inputs,
                "run_id": "run-a",
                "question_id": "question-a",
                "query_id": "query-a",
                "identity": {
                    "email": "user@example.com",
                    "benchmark_name": "swebench",
                    "agent_name": "swe-agent",
                },
                "config": {
                    "custom_endpoint": "https://private-provider.example.internal/v1",
                    "custom_api_key": "sk-provider",
                },
            },
            headers={"Authorization": "Bearer sk-test"},
        )

    assert response.status_code == 200
    assert len(fake_ledger.events) == 1
    event = fake_ledger.events[0]
    assert event["run_id"] == "run-a"
    assert event["question_id"] == "question-a"
    assert event["query_id"] == "query-a"
    assert event[ledger_schema.IDENTITY_EMAIL] == "user@example.com"
    assert "identity" not in event
    assert event["api_key_fingerprint"] == "f3abf2a6cc4f0098"
    assert event["provider_endpoint"] == "custom"
    assert event["input_tokens"] == 12
    assert event["output_tokens"] == 3

    details = cast(dict[str, Any], event[ledger_schema.DETAILS_FIELD])
    request_data = details["request"]
    result_data = details["result"]
    config = request_data["config"]
    assert config["custom_endpoint"] == "https://private-provider.example.internal/v1"
    assert config["custom_api_key"] == "**********"
    raw_input = cast(dict[str, object], raw_request_inputs[0])["input"]
    raw_input_length = len(json.dumps(raw_input, sort_keys=True, separators=(",", ":")))
    expected_request_inputs = [
        {
            "kind": "raw_input",
            "input_length": raw_input_length,
        }
    ]
    assert request_data["inputs"] == expected_request_inputs
    assert request_data["identity"] == {
        "email": "user@example.com",
        "benchmark_name": "swebench",
        "agent_name": "swe-agent",
    }
    assert result_data["output_text_length"] == 2
    assert result_data["reasoning_length"] == 0
    assert result_data["metadata"]["in_tokens"] == 12
    assert result_data["metadata"]["out_tokens"] == 3


def test_local_startup_canary_query_skips_usage_ledger():
    from model_gateway import main

    class ServerSettings:
        MODEL_GATEWAY_API_KEYS = "sk-test"
        MODEL_GATEWAY_HMAC_SECRET = "test-secret"

        def get(self, name: str, default: str = "") -> str:
            return getattr(self, name, default)

        def unset(self, key: str) -> None:
            pass

    class FakeLLM:
        async def query(self, _inputs: Any, **_kwargs: Any) -> QueryResult:
            return QueryResult(output_text="ok", history=[])

    fake_ledger = FakeUsageLedger()
    with (
        patch.object(gateway_app, "model_library_settings", ServerSettings()),
        patch.object(gateway_app, "get_model_names", return_value=["openai/gpt-4o"]),
        patch.object(model_helpers, "get_registry_model", return_value=FakeLLM()),
    ):
        app = main.create_app()
        app.state.usage_ledger = fake_ledger
        client = TestClient(app, client=("127.0.0.1", 50000))
        response = client.post(
            "/query",
            json={
                "model": "openai/gpt-4o",
                "inputs": [{"kind": "text", "text": "hi"}],
                "run_id": "gateway-startup-canary",
                "question_id": "startup-test",
            },
            headers={"Authorization": "Bearer sk-test"},
        )
        assert response.status_code == 200
        assert fake_ledger.events == []

        remote_response = TestClient(app).post(
            "/query",
            json={
                "model": "openai/gpt-4o",
                "inputs": [{"kind": "text", "text": "hi"}],
                "run_id": "gateway-startup-canary",
                "question_id": "startup-test",
            },
            headers={"Authorization": "Bearer sk-test"},
        )

    assert remote_response.status_code == 200
    assert len(fake_ledger.events) == 1


def test_query_marks_usage_ledger_phase_when_enforced_write_fails():
    from model_gateway import main

    class ServerSettings:
        MODEL_GATEWAY_API_KEYS = "sk-test"
        MODEL_GATEWAY_HMAC_SECRET = "test-secret"

        def get(self, name: str, default: str = "") -> str:
            return getattr(self, name, default)

        def unset(self, key: str) -> None:
            pass

    class FakeLLM:
        async def query(self, _inputs: Any, **_kwargs: Any) -> QueryResult:
            return QueryResult(
                output_text="ok",
                history=[],
                metadata=QueryResultMetadata(in_tokens=12, out_tokens=3),
            )

    events: list[tuple[str, dict[str, object] | None]] = []
    with (
        patch.object(gateway_app, "model_library_settings", ServerSettings()),
        patch.object(gateway_app, "get_model_names", return_value=["openai/gpt-4o"]),
        patch.object(model_helpers, "get_registry_model", return_value=FakeLLM()),
        patch.object(
            gateway_app.telemetry,
            "add_event",
            side_effect=lambda name, attrs=None: events.append((name, attrs)),
        ),
    ):
        app = main.create_app()
        app.state.usage_ledger = FailingUsageLedger()
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/query",
            json={
                "model": "openai/gpt-4o",
                "inputs": [{"kind": "text", "text": "hi"}],
                "run_id": "run-a",
                "question_id": "question-a",
            },
            headers={"Authorization": "Bearer sk-test"},
        )

    assert response.status_code == 500
    assert (
        "gateway.query.usage_ledger_error",
        {
            "gateway.error.phase": "usage_ledger",
            "exception.type": "RuntimeError",
        },
    ) in events


def test_query_does_not_write_usage_event_on_provider_error():
    from model_gateway import main

    class ServerSettings:
        MODEL_GATEWAY_API_KEYS = "sk-test"
        MODEL_GATEWAY_HMAC_SECRET = "test-secret"

        def get(self, name: str, default: str = "") -> str:
            return getattr(self, name, default)

        def unset(self, key: str) -> None:
            pass

    class FailingLLM:
        async def query(self, _inputs: Any, **_kwargs: Any) -> QueryResult:
            raise RuntimeError("provider failed")

    fake_ledger = FakeUsageLedger()
    with (
        patch.object(gateway_app, "model_library_settings", ServerSettings()),
        patch.object(gateway_app, "get_model_names", return_value=["openai/gpt-4o"]),
        patch.object(model_helpers, "get_registry_model", return_value=FailingLLM()),
    ):
        app = main.create_app()
        app.state.usage_ledger = fake_ledger
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/query",
            json={
                "model": "openai/gpt-4o",
                "inputs": [{"kind": "text", "text": "hi"}],
                "run_id": "run-a",
                "question_id": "question-a",
            },
            headers={"Authorization": "Bearer sk-test"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "error": {
            "type": "ProviderError",
            "message": "provider failed",
            "provider": "openai",
            "exception_type": "RuntimeError",
        }
    }
    assert fake_ledger.events == []
