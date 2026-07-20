import json
from datetime import UTC, datetime
from decimal import Decimal
import pytest

from model_gateway.types import QueryRequest
from model_gateway.usage_ledger.details import snapshot_usage_request
from model_gateway.usage_ledger.message import (
    MAX_USAGE_LEDGER_MESSAGE_BYTES,
    UsageLedgerMessageTooLarge,
    deserialize_usage_event_message,
    prepare_usage_event_message,
    serialize_usage_event_message,
)
from model_gateway.usage_ledger.store import build_success_usage_event
from model_library.base.input import TextInput
from model_library.base.output import (
    QueryResult,
    QueryResultCost,
    QueryResultExtras,
    QueryResultMetadata,
)


def _usage_event() -> dict[str, object]:
    request = QueryRequest(
        model="openai/gpt-4o",
        inputs=[TextInput(text="hello")],
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
        extras=QueryResultExtras(
            provider_response_id="provider-response-1",
            provider_request_id="provider-request-1",
        ),
    )
    return build_success_usage_event(
        body=request,
        config=request.config_dict(),
        query_params={},
        dimensions={"ProviderEndpoint": "default", "ParamGroup": "pg"},
        result=result,
        request=snapshot_usage_request(request),
        completed_at=datetime(2026, 5, 29, 12, 0, tzinfo=UTC),
        api_key_fingerprint="keyfingerprint",
    )


def test_serialize_usage_event_message_wraps_versioned_event_and_preserves_numbers():
    event = _usage_event()

    message = serialize_usage_event_message(event)

    envelope = json.loads(message)
    assert envelope["schema_version"] == 1
    assert envelope["event"]["usage_event_id"] == event["usage_event_id"]
    assert envelope["event"]["details"] == event["details"]
    assert envelope["event"]["cost_usd"] == "0.003"
    assert "provider-response-1" in message
    assert "provider-request-1" in message

    decoded = deserialize_usage_event_message(message)
    assert decoded["usage_event_id"] == event["usage_event_id"]
    assert decoded["details"] == event["details"]
    assert decoded["input_tokens"] == 4
    assert decoded["cost_usd"] == Decimal("0.003")


def test_real_producer_event_survives_sqs_normalization() -> None:
    event = _usage_event()
    prepared = prepare_usage_event_message(event)
    decoded = deserialize_usage_event_message(prepared.message)

    assert prepared.event["details"] == event["details"]
    assert decoded == prepared.event


def test_external_non_finite_request_detail_is_reduced_before_serialization() -> None:
    request = QueryRequest.model_validate(
        json.loads(
            '{"model":"openai/gpt-4o","inputs":[{"kind":"text","text":"hi"}],'
            '"config":{"temperature":NaN}}'
        )
    )
    event = build_success_usage_event(
        body=request,
        config=request.config_dict(),
        query_params={},
        dimensions={},
        result=QueryResult(output_text="ok"),
        request=snapshot_usage_request(request),
    )

    message = serialize_usage_event_message(event)

    envelope = json.loads(message)
    assert envelope["event"]["details"]["request"]["config"]["temperature"] is None


@pytest.mark.parametrize(
    ("cost_value", "wire_cost", "decoded_cost"),
    [
        (Decimal("1000.000000000000"), "1000", Decimal("1000")),
        (Decimal("0.000000000001"), "0.000000000001", Decimal("0.000000000001")),
    ],
)
def test_serialize_usage_event_message_formats_cost_usd_extremes_and_preserves_other_decimals(
    cost_value: Decimal,
    wire_cost: str,
    decoded_cost: Decimal,
):
    event = {
        "usage_event_id": "evt-1",
        "cost_usd": cost_value,
        "duration_seconds": Decimal("1.2300"),
    }

    message = serialize_usage_event_message(event)

    envelope = json.loads(message)
    assert envelope["event"]["cost_usd"] == wire_cost
    assert envelope["event"]["duration_seconds"] == "1.2300"

    decoded = deserialize_usage_event_message(message)
    assert decoded["cost_usd"] == decoded_cost
    assert decoded["duration_seconds"] == Decimal("1.2300")


def test_complete_serialized_event_has_exact_300_000_byte_limit() -> None:
    assert MAX_USAGE_LEDGER_MESSAGE_BYTES == 300_000

    event = _usage_event()
    event["duration_seconds"] = 1e-6
    event["details"] = {
        "request": {"large": ""},
        "result": {"metadata": {"performance": None}},
    }
    empty_size = len(
        serialize_usage_event_message(
            event, max_bytes=MAX_USAGE_LEDGER_MESSAGE_BYTES * 2
        ).encode("utf-8")
    )
    event["details"] = {
        "request": {"large": "x" * (MAX_USAGE_LEDGER_MESSAGE_BYTES - empty_size)},
        "result": {"metadata": {"performance": None}},
    }

    prepared = prepare_usage_event_message(event)
    assert len(prepared.message.encode("utf-8")) == MAX_USAGE_LEDGER_MESSAGE_BYTES
    assert (
        serialize_usage_event_message(
            prepared.event, max_bytes=MAX_USAGE_LEDGER_MESSAGE_BYTES * 2
        )
        == prepared.message
    )
    event["details"] = {
        "request": {"large": "x" * (MAX_USAGE_LEDGER_MESSAGE_BYTES - empty_size + 1)},
        "result": {"metadata": {"performance": None}},
    }
    oversized = serialize_usage_event_message(
        event, max_bytes=MAX_USAGE_LEDGER_MESSAGE_BYTES + 1
    )
    assert len(oversized.encode("utf-8")) == MAX_USAGE_LEDGER_MESSAGE_BYTES + 1
    with pytest.raises(UsageLedgerMessageTooLarge):
        serialize_usage_event_message(event)
