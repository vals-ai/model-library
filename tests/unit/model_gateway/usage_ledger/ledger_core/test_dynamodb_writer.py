from decimal import Decimal

import pytest

from model_gateway.usage_ledger import schema
from model_gateway.usage_ledger.dynamodb_writer import (
    _dynamodb_item_size,
    prepare_usage_event_for_write,
)


def _event() -> dict[str, object]:
    return {
        schema.BASE_PK: "USAGE#DAY#2026-07-14#S#00",
        schema.BASE_SK: "TS#2026-07-14T12:00:00Z#REQ#evt-1",
        "entity_type": "usage_event",
        "usage_event_id": "evt-1",
        "model": "openai/gpt-4o",
        "provider": "openai",
        "provider_endpoint": "default",
        "completed_at": "2026-07-14T12:00:00Z",
        "day": "2026-07-14",
        "usage_shard": "00",
        "schema_version": schema.USAGE_EVENT_SCHEMA_VERSION,
        "normalization_version": "2026-07-14",
        "finish_reason": "stop",
        "finish_reason_raw": "stop",
        "input_tokens": 4,
        "output_tokens": 2,
        "reasoning_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "total_input_tokens": 4,
        "total_output_tokens": 2,
        "duration_seconds": Decimal("0.5"),
        "cost_usd": Decimal("0.003"),
        schema.DETAILS_FIELD: {
            "request": {},
            "result": {"metadata": {"performance": None}},
        },
    }


def test_dynamodb_item_size_uses_exact_binary_limit_and_nested_overhead() -> None:
    exact_limit = {"é": {"S": "x" * (409_600 - len("é".encode("utf-8")))}}
    one_byte_over = {"é": {"S": "x" * (409_601 - len("é".encode("utf-8")))}}

    assert _dynamodb_item_size(exact_limit) == 409_600
    assert _dynamodb_item_size(one_byte_over) == 409_601
    assert (
        _dynamodb_item_size({"m": {"M": {"b": {"BOOL": True}, "n": {"NULL": True}}}})
        == 10
    )
    assert _dynamodb_item_size({"n": {"N": "-123.00"}}) == 4
    assert _dynamodb_item_size({"l": {"L": [{"N": "1"}, {"S": "x"}]}}) == 9


def test_prepare_usage_event_preserves_additive_root_fields() -> None:
    event = _event()
    event["future_additive_field"] = "value"

    prepared = prepare_usage_event_for_write(event)

    assert prepared.event["future_additive_field"] == "value"


def test_prepare_usage_event_preserves_details_below_the_size_limits() -> None:
    event = _event()
    event[schema.DETAILS_FIELD] = {
        "request": {f"k{index}": "x" for index in range(22_000)},
        "result": {"metadata": {"performance": None}},
    }

    prepared = prepare_usage_event_for_write(event)

    assert prepared.event[schema.DETAILS_FIELD] == event[schema.DETAILS_FIELD]


@pytest.mark.parametrize("case", ["message_too_large", "item_too_large"])
def test_prepare_usage_event_replaces_only_oversized_details(case: str) -> None:
    event = _event()
    if case == "message_too_large":
        event[schema.DETAILS_FIELD] = {
            "request": {"large": "x" * 300_000},
            "result": {"metadata": {"performance": None}},
        }
    else:
        event[schema.DETAILS_FIELD] = {
            "request": {"nested": [[[[]]] for _ in range(35_000)]},
            "result": {"metadata": {"performance": None}},
        }

    prepared = prepare_usage_event_for_write(event)

    assert prepared.event[schema.DETAILS_FIELD] == {
        "truncated": True,
        "request": {},
        "result": {"metadata": {"performance": None}},
    }
