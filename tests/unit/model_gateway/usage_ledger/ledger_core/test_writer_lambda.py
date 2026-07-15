import json
from datetime import UTC, datetime
from typing import Any

from botocore.exceptions import ClientError

from model_gateway.types import QueryRequest
from model_gateway.usage_ledger.message import serialize_usage_event_message
from model_gateway.usage_ledger.store import build_success_usage_event
from model_library.base.output import QueryResult


class FakeDynamoClient:
    def __init__(self) -> None:
        self.put_calls: list[dict[str, object]] = []
        self.get_calls: list[dict[str, object]] = []
        self.put_error: Exception | None = None
        self.put_errors: list[Exception] = []
        self.get_item_response: dict[str, object] = {}

    def put_item(self, **kwargs: object) -> dict[str, object]:
        self.put_calls.append(kwargs)
        if self.put_errors:
            raise self.put_errors.pop(0)
        if self.put_error is not None:
            raise self.put_error
        return {}

    def get_item(self, **kwargs: object) -> dict[str, object]:
        self.get_calls.append(kwargs)
        return self.get_item_response


def _usage_event() -> dict[str, object]:
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
        query_params={},
        dimensions={"ProviderEndpoint": "default", "ParamGroup": "pg"},
        result=QueryResult(output_text="ok", history=[]),
        completed_at=datetime(2026, 5, 29, 12, 0, tzinfo=UTC),
    )


def _sqs_record(message_id: str, body: str) -> dict[str, str]:
    return {"messageId": message_id, "body": body}


def test_usage_ledger_lambda_writes_valid_sqs_message(monkeypatch: Any):
    from model_gateway.usage_ledger.lambdas import writer as lambda_handler

    client = FakeDynamoClient()
    event = _usage_event()
    emitted: list[dict[str, tuple[float, str]]] = []
    monkeypatch.setenv("GATEWAY_USAGE_LEDGER_TABLE_NAME", "usage-table")
    monkeypatch.setattr(lambda_handler, "_dynamodb_client", lambda: client)
    monkeypatch.setattr(lambda_handler, "emit_usage_ledger_metrics", emitted.append)

    response = lambda_handler.handler(
        {"Records": [_sqs_record("message-1", serialize_usage_event_message(event))]},
        None,
    )

    assert response == {"batchItemFailures": []}
    assert len(client.put_calls) == 1
    assert client.put_calls[0]["TableName"] == "usage-table"
    assert client.put_calls[0]["ConditionExpression"] == "attribute_not_exists(PK)"
    assert emitted == [{"UsageLedgerRawRowsWritten": (1, "Count")}]


def test_usage_ledger_lambda_returns_partial_batch_failure_for_bad_message(
    monkeypatch: Any,
):
    from model_gateway.usage_ledger.lambdas import writer as lambda_handler

    client = FakeDynamoClient()
    event = _usage_event()
    monkeypatch.setenv("GATEWAY_USAGE_LEDGER_TABLE_NAME", "usage-table")
    monkeypatch.setattr(lambda_handler, "_dynamodb_client", lambda: client)

    response = lambda_handler.handler(
        {
            "Records": [
                _sqs_record("bad-message", "not-json"),
                _sqs_record("good-message", serialize_usage_event_message(event)),
            ]
        },
        None,
    )

    assert response == {"batchItemFailures": [{"itemIdentifier": "bad-message"}]}
    assert len(client.put_calls) == 1


def test_usage_ledger_lambda_fails_only_invalid_numeric_record_in_mixed_batch(
    monkeypatch: Any,
):
    from model_gateway.usage_ledger.lambdas import writer as lambda_handler

    client = FakeDynamoClient()
    bad_event = _usage_event()
    good_event = _usage_event()
    bad_envelope = json.loads(serialize_usage_event_message(bad_event))
    bad_envelope["event"]["request_count"] = "oops"
    monkeypatch.setenv("GATEWAY_USAGE_LEDGER_TABLE_NAME", "usage-table")
    monkeypatch.setattr(lambda_handler, "_dynamodb_client", lambda: client)

    response = lambda_handler.handler(
        {
            "Records": [
                _sqs_record("bad-numeric", json.dumps(bad_envelope)),
                _sqs_record("good-message", serialize_usage_event_message(good_event)),
            ]
        },
        None,
    )

    assert response == {"batchItemFailures": [{"itemIdentifier": "bad-numeric"}]}
    assert len(client.put_calls) == 1


def test_usage_ledger_lambda_fails_only_unsupported_schema_version_in_mixed_batch(
    monkeypatch: Any,
):
    from model_gateway.usage_ledger.lambdas import writer as lambda_handler

    client = FakeDynamoClient()
    event = _usage_event()
    monkeypatch.setenv("GATEWAY_USAGE_LEDGER_TABLE_NAME", "usage-table")
    monkeypatch.setattr(lambda_handler, "_dynamodb_client", lambda: client)

    response = lambda_handler.handler(
        {
            "Records": [
                _sqs_record(
                    "bad-version",
                    json.dumps({"schema_version": 999, "event": event}, default=str),
                ),
                _sqs_record("good-message", serialize_usage_event_message(event)),
            ]
        },
        None,
    )

    assert response == {"batchItemFailures": [{"itemIdentifier": "bad-version"}]}
    assert len(client.put_calls) == 1


def test_usage_ledger_lambda_rejects_json_valid_event_with_disallowed_fields(
    monkeypatch: Any,
):
    from model_gateway.usage_ledger.lambdas import writer as lambda_handler

    client = FakeDynamoClient()
    event = _usage_event()
    event["output_text"] = "must not persist"
    monkeypatch.setenv("GATEWAY_USAGE_LEDGER_TABLE_NAME", "usage-table")
    monkeypatch.setattr(lambda_handler, "_dynamodb_client", lambda: client)

    response = lambda_handler.handler(
        {"Records": [_sqs_record("bad-event", serialize_usage_event_message(event))]},
        None,
    )

    assert response == {"batchItemFailures": [{"itemIdentifier": "bad-event"}]}
    assert client.put_calls == []


def test_usage_ledger_lambda_rejects_json_valid_event_missing_required_fields(
    monkeypatch: Any,
):
    from model_gateway.usage_ledger.lambdas import writer as lambda_handler

    client = FakeDynamoClient()
    event = _usage_event()
    del event["usage_event_id"]
    monkeypatch.setenv("GATEWAY_USAGE_LEDGER_TABLE_NAME", "usage-table")
    monkeypatch.setattr(lambda_handler, "_dynamodb_client", lambda: client)

    response = lambda_handler.handler(
        {"Records": [_sqs_record("bad-event", serialize_usage_event_message(event))]},
        None,
    )

    assert response == {"batchItemFailures": [{"itemIdentifier": "bad-event"}]}
    assert client.put_calls == []


def test_usage_ledger_lambda_missing_table_name_fails_before_writes(monkeypatch: Any):
    from model_gateway.usage_ledger.lambdas import writer as lambda_handler

    client = FakeDynamoClient()
    monkeypatch.delenv("GATEWAY_USAGE_LEDGER_TABLE_NAME", raising=False)
    monkeypatch.setattr(lambda_handler, "_dynamodb_client", lambda: client)

    try:
        lambda_handler.handler(
            {
                "Records": [
                    _sqs_record(
                        "message-1", serialize_usage_event_message(_usage_event())
                    )
                ]
            },
            None,
        )
    except ValueError as exc:
        assert "GATEWAY_USAGE_LEDGER_TABLE_NAME" in str(exc)
    else:
        raise AssertionError("handler should fail without table name")
    assert client.put_calls == []


def test_usage_ledger_lambda_isolates_dynamodb_failure_in_mixed_batch(
    monkeypatch: Any,
):
    from model_gateway.usage_ledger.lambdas import writer as lambda_handler

    client = FakeDynamoClient()
    first = _usage_event()
    second = _usage_event()
    client.put_errors = [
        ClientError(
            {"Error": {"Code": "InternalServerError", "Message": "try later"}},
            "PutItem",
        )
    ]
    monkeypatch.setenv("GATEWAY_USAGE_LEDGER_TABLE_NAME", "usage-table")
    monkeypatch.setattr(lambda_handler, "_dynamodb_client", lambda: client)

    response = lambda_handler.handler(
        {
            "Records": [
                _sqs_record("ddb-failure", serialize_usage_event_message(first)),
                _sqs_record("success", serialize_usage_event_message(second)),
            ]
        },
        None,
    )

    assert response == {"batchItemFailures": [{"itemIdentifier": "ddb-failure"}]}
    assert len(client.put_calls) == 2


def test_usage_ledger_lambda_treats_duplicate_same_usage_event_as_success(
    monkeypatch: Any,
):
    from model_gateway.usage_ledger.lambdas import writer as lambda_handler

    client = FakeDynamoClient()
    event = _usage_event()
    client.put_error = ClientError(
        {"Error": {"Code": "ConditionalCheckFailedException", "Message": "exists"}},
        "PutItem",
    )
    client.get_item_response = {
        "Item": {"usage_event_id": {"S": str(event["usage_event_id"])}}
    }
    emitted: list[dict[str, tuple[float, str]]] = []
    monkeypatch.setenv("GATEWAY_USAGE_LEDGER_TABLE_NAME", "usage-table")
    monkeypatch.setattr(lambda_handler, "_dynamodb_client", lambda: client)
    monkeypatch.setattr(lambda_handler, "emit_usage_ledger_metrics", emitted.append)

    response = lambda_handler.handler(
        {"Records": [_sqs_record("message-1", serialize_usage_event_message(event))]},
        None,
    )

    assert response == {"batchItemFailures": []}
    assert len(client.put_calls) == 1
    assert len(client.get_calls) == 1
    assert client.get_calls[0]["ConsistentRead"] is True
    assert emitted == []


def test_usage_ledger_lambda_retries_conditional_failure_for_different_event(
    monkeypatch: Any,
):
    from model_gateway.usage_ledger.lambdas import writer as lambda_handler

    client = FakeDynamoClient()
    event = _usage_event()
    client.put_error = ClientError(
        {"Error": {"Code": "ConditionalCheckFailedException", "Message": "exists"}},
        "PutItem",
    )
    client.get_item_response = {"Item": {"usage_event_id": {"S": "usg_other"}}}
    monkeypatch.setenv("GATEWAY_USAGE_LEDGER_TABLE_NAME", "usage-table")
    monkeypatch.setattr(lambda_handler, "_dynamodb_client", lambda: client)

    response = lambda_handler.handler(
        {"Records": [_sqs_record("message-1", serialize_usage_event_message(event))]},
        None,
    )

    assert response == {"batchItemFailures": [{"itemIdentifier": "message-1"}]}
