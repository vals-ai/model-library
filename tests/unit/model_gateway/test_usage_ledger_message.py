import json
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from model_gateway.types import QueryRequest
from model_gateway.usage_ledger.message import (
    MAX_USAGE_LEDGER_MESSAGE_BYTES,
    UsageLedgerMessageError,
    UsageLedgerMessageTooLarge,
    deserialize_usage_event_message,
    serialize_usage_event_message,
)
from model_gateway.usage_ledger.store import build_success_usage_event
from model_library.base.output import QueryResult, QueryResultCost, QueryResultMetadata


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
        query_params={"query_id": request.query_id},
        dimensions={"ProviderEndpoint": "default", "ParamGroup": "pg"},
        result=QueryResult(
            output_text="ok",
            history=[],
            metadata=QueryResultMetadata(
                in_tokens=4,
                out_tokens=2,
                cost=QueryResultCost(input=0.001, output=0.002),
            ),
        ),
        completed_at=datetime(2026, 5, 29, 12, 0, tzinfo=UTC),
        api_key_fingerprint="keyfingerprint",
    )


def test_serialize_usage_event_message_wraps_versioned_event_and_preserves_numbers():
    event = _usage_event()

    message = serialize_usage_event_message(event)

    envelope = json.loads(message)
    assert envelope["schema_version"] == 1
    assert envelope["event"]["usage_event_id"] == event["usage_event_id"]
    assert envelope["event"]["cost_usd"] == "0.003"

    decoded = deserialize_usage_event_message(message)
    assert decoded["usage_event_id"] == event["usage_event_id"]
    assert decoded["input_tokens"] == 4
    assert decoded["cost_usd"] == Decimal("0.003")


def test_deserialize_usage_event_message_rejects_invalid_numeric_strings():
    event = _usage_event()
    message = serialize_usage_event_message(event)
    envelope = json.loads(message)
    envelope["event"]["input_tokens"] = "oops"

    with pytest.raises(UsageLedgerMessageError, match="invalid numeric field"):
        deserialize_usage_event_message(json.dumps(envelope))


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_deserialize_usage_event_message_rejects_non_finite_numeric_strings(
    value: str,
):
    event = _usage_event()
    message = serialize_usage_event_message(event)
    envelope = json.loads(message)
    envelope["event"]["input_tokens"] = value

    with pytest.raises(UsageLedgerMessageError, match="invalid numeric field"):
        deserialize_usage_event_message(json.dumps(envelope))


def test_serialize_usage_event_message_summarizes_bulky_fields_to_fit_budget():
    event = _usage_event()
    escape_heavy_json = json.dumps("\\" * 31_999)
    assert len(escape_heavy_json) == 64_000
    event["config_redacted_json"] = escape_heavy_json
    event["metadata_json"] = escape_heavy_json
    event["finish_reason_json"] = escape_heavy_json

    message = serialize_usage_event_message(event)

    assert len(message.encode("utf-8")) <= MAX_USAGE_LEDGER_MESSAGE_BYTES
    decoded = deserialize_usage_event_message(message)
    for field in ("config_redacted_json", "metadata_json", "finish_reason_json"):
        summary = json.loads(str(decoded[field]))
        assert summary["truncated"] is True
        assert summary["reason"] == "sqs_message_size"
        assert summary["field"] == field
        assert decoded[f"{field}_truncated"] is True


def test_serialize_usage_event_message_rejects_messages_above_sqs_safe_budget():
    event = _usage_event()
    event["identity"] = {"large": "x" * MAX_USAGE_LEDGER_MESSAGE_BYTES}

    with pytest.raises(UsageLedgerMessageTooLarge):
        serialize_usage_event_message(event)
