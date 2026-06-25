import json
import os
from datetime import UTC, datetime
from typing import Any, cast
from unittest.mock import patch

import pytest
from boto3.dynamodb.types import TypeSerializer
from botocore.exceptions import ClientError
from pydantic import ValidationError
from starlette.testclient import TestClient

from model_gateway import app as gateway_app
from model_gateway import model_helpers
from model_gateway.usage_ledger.store import (
    DynamoDbUsageLedger,
    NoopUsageLedger,
    SqsUsageLedger,
    UsageLedgerWriteError,
    build_success_usage_event,
    create_usage_ledger_from_env,
)
from model_gateway.types import QueryRequest
from model_library.base.output import QueryResult, QueryResultCost, QueryResultMetadata


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


class FakeSqsClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.send_error: Exception | None = None

    async def send_message(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        if self.send_error is not None:
            raise self.send_error
        return {"MessageId": "message-1"}


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


def _basic_usage_event() -> dict[str, object]:
    request = QueryRequest(
        model="openai/gpt-4o",
        inputs=[{"kind": "text", "text": "hello"}],
        run_id="run-1",
        question_id="question-1",
        query_id="query-1",
    )
    return build_success_usage_event(
        body=request,
        config=request.config_dict(),
        query_params={"query_id": request.query_id},
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
            inputs=[{"kind": "text", "text": "hello"}],
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
            inputs=[{"kind": "text", "text": "hello"}],
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
        QueryRequest(
            model="openai/gpt-4o",
            inputs=[{"kind": "text", "text": "hello"}],
            **{field: value},
        )


@pytest.mark.parametrize("value", ["", " ", "\t"])
def test_query_request_treats_blank_query_id_as_absent(value: str):
    request = QueryRequest(
        model="openai/gpt-4o",
        inputs=[{"kind": "text", "text": "hello"}],
        query_id=value,
    )
    assert request.query_id is None


def test_query_request_rejects_oversized_identity():
    with pytest.raises(ValidationError):
        QueryRequest(
            model="openai/gpt-4o",
            inputs=[{"kind": "text", "text": "hello"}],
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
            inputs=[{"kind": "text", "text": "hello"}],
            identity={
                "d1": {
                    "d2": {
                        "d3": {"d4": {"d5": {"d6": {"d7": {"d8": {"d9": "too-deep"}}}}}}
                    }
                }
            },
        )


def test_build_success_usage_event_keeps_sanitized_config_and_metadata_without_result_payload():
    request = QueryRequest(
        model="anthropic/claude-sonnet-4.5",
        inputs=[{"kind": "text", "text": "hello"}],
        config={
            "max_tokens": 10,
            "provider_config": {
                "prompt": "secret prompt",
                "custom_api_key": "provider-secret",
                "prompt_cache_retention": "24h",
            },
        },
        run_id="run-1",
        question_id="question-1",
        query_id="query-1",
        identity={
            "email": "user@example.com",
            "benchmark_name": "swebench",
            "agent_name": "swe-agent",
        },
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
            extra={
                "provider_billable_units": 7,
                "response_text": "secret response",
                "api_key": "secret",
                "token_metadata": {
                    "estimated": 150,
                    "actual": 125,
                    "raw_response": "do not persist",
                },
            },
        ),
    )

    event = build_success_usage_event(
        body=request,
        config=request.config_dict(),
        query_params={
            "run_id": request.run_id,
            "question_id": request.question_id,
            "query_id": request.query_id,
            "in_agent": request.in_agent,
        },
        dimensions={"ProviderEndpoint": "default", "ParamGroup": "pg"},
        result=result,
        completed_at=datetime(2026, 5, 29, 12, 0, tzinfo=UTC),
        api_key_fingerprint="keyfingerprint",
    )

    config = json.loads(str(event["config_redacted_json"]))
    metadata = json.loads(str(event["metadata_json"]))

    assert event["run_id"] == "run-1"
    assert event["question_id"] == "question-1"
    assert event["query_id"] == "query-1"
    assert event["identity"] == {
        "email": "user@example.com",
        "benchmark_name": "swebench",
        "agent_name": "swe-agent",
    }
    assert event["api_key_fingerprint"] == "keyfingerprint"
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

    assert config["config"]["provider_config"]["prompt"] == "<redacted>"
    assert config["config"]["provider_config"]["custom_api_key"] == "<redacted>"
    assert config["config"]["provider_config"]["prompt_cache_retention"] == "24h"
    assert metadata["token_metadata"] == {"actual": 125, "estimated": 150}
    assert "extra" not in metadata
    assert "provider_billable_units" not in metadata
    assert "response_text" not in metadata
    assert "api_key" not in metadata
    assert "output_text" not in event
    assert "reasoning" not in event
    assert "history" not in event


def test_usage_event_skips_dimension_indexes_for_non_string_or_blank_identity_values():
    request = QueryRequest(
        model="openai/gpt-4o",
        inputs=[{"kind": "text", "text": "hello"}],
        run_id="run-1",
        question_id="question-1",
        query_id="query-1",
        identity={"benchmark_name": "   ", "agent_name": 123},
    )
    event = build_success_usage_event(
        body=request,
        config=request.config_dict(),
        query_params={"query_id": request.query_id},
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
        inputs=[{"kind": "text", "text": "hello"}],
        run_id="run-1",
        question_id="question-1",
        query_id="query-1",
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
    event = build_success_usage_event(
        body=request,
        config=request.config_dict(),
        query_params={"query_id": request.query_id},
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
    assert all(
        not str(value).startswith("AGG#")
        for attribute in cast(dict[str, object], put_item["Item"]).values()
        for value in cast(dict[str, object], attribute).values()
    )


@pytest.mark.asyncio
async def test_dynamodb_usage_ledger_treats_duplicate_same_usage_event_as_success():
    request = QueryRequest(
        model="openai/gpt-4o",
        inputs=[{"kind": "text", "text": "hello"}],
        query_id="query-1",
    )
    event = build_success_usage_event(
        body=request,
        config=request.config_dict(),
        query_params={"query_id": request.query_id},
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
    assert client.get_calls[0]["TableName"] == "usage-table"
    assert client.get_calls[0]["Key"] == {
        "PK": client.calls[0]["Item"]["PK"],
        "SK": client.calls[0]["Item"]["SK"],
    }
    assert client.get_calls[0]["ProjectionExpression"] == "usage_event_id"
    assert client.get_calls[0]["ConsistentRead"] is True


@pytest.mark.asyncio
async def test_dynamodb_usage_ledger_reraises_conditional_failure_for_different_event():
    request = QueryRequest(
        model="openai/gpt-4o",
        inputs=[{"kind": "text", "text": "hello"}],
        query_id="query-1",
    )
    event = build_success_usage_event(
        body=request,
        config=request.config_dict(),
        query_params={"query_id": request.query_id},
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
        inputs=[{"kind": "text", "text": "hello"}],
        run_id="run-1",
        question_id="question-1",
        query_id="query-1",
    )
    result = QueryResult(output_text="ok", history=[])

    first = build_success_usage_event(
        body=request,
        config=request.config_dict(),
        query_params={"query_id": request.query_id},
        dimensions={"ProviderEndpoint": "default", "ParamGroup": "pg"},
        result=result,
        completed_at=datetime(2026, 5, 29, 12, 0, tzinfo=UTC),
        api_key_fingerprint="keyfingerprint",
    )
    second = build_success_usage_event(
        body=request,
        config=request.config_dict(),
        query_params={"query_id": request.query_id},
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
    sqs_config = create_client_kwargs["config"]
    assert sqs_config.connect_timeout == 2
    assert sqs_config.read_timeout == 3
    assert sqs_config.retries["max_attempts"] == 2
    assert context.entered
    assert ledger._client is client

    await ledger.close()

    assert context.exited
    assert ledger._client is None
    assert ledger._client_context is None


@pytest.mark.asyncio
async def test_sqs_usage_ledger_sends_serialized_usage_event_message():
    client = FakeSqsClient()
    ledger = SqsUsageLedger(
        queue_url="https://sqs.us-east-1.amazonaws.com/123/ledger",
        mode="enforced",
    )
    ledger._client = client
    event = _basic_usage_event()

    await ledger.write_success(event)

    assert len(client.calls) == 1
    assert client.calls[0]["QueueUrl"] == ledger._queue_url
    message = json.loads(cast(str, client.calls[0]["MessageBody"]))
    assert message["schema_version"] == 1
    assert message["event"]["usage_event_id"] == event["usage_event_id"]


@pytest.mark.asyncio
async def test_sqs_usage_ledger_enforced_mode_raises_on_send_failure():
    client = FakeSqsClient()
    client.send_error = ValueError("sqs unavailable")
    ledger = SqsUsageLedger(
        queue_url="https://sqs.us-east-1.amazonaws.com/123/ledger",
        mode="enforced",
    )
    ledger._client = client

    with pytest.raises(UsageLedgerWriteError):
        await ledger.write_success(_basic_usage_event())


@pytest.mark.asyncio
async def test_sqs_usage_ledger_shadow_mode_swallows_send_failures():
    client = FakeSqsClient()
    client.send_error = ValueError("sqs unavailable")
    ledger = SqsUsageLedger(
        queue_url="https://sqs.us-east-1.amazonaws.com/123/ledger",
        mode="shadow",
    )
    ledger._client = client

    await ledger.write_success(_basic_usage_event())


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
                "inputs": [{"kind": "text", "text": "hi"}],
                "run_id": "run-a",
                "question_id": "question-a",
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
    assert event["identity"] == {
        "email": "user@example.com",
        "benchmark_name": "swebench",
        "agent_name": "swe-agent",
    }
    assert isinstance(event["query_id"], str)
    assert event["api_key_fingerprint"] == "f3abf2a6cc4f0098"
    assert event["provider_endpoint"] == "custom"
    assert "https://private-provider.example.internal/v1" not in json.dumps(
        event, default=str
    )
    assert event["input_tokens"] == 12
    assert event["output_tokens"] == 3


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
            "code": "internal_error",
            "message": "provider failed",
            "provider": "openai",
        }
    }
    assert fake_ledger.events == []
