from __future__ import annotations

import base64
from datetime import UTC, date, datetime
from decimal import Decimal
import gzip
import json
from typing import cast

import pytest

from model_gateway.usage_ledger import redshift_schema
from model_gateway.usage_ledger.redshift_transform import redshift_rows_from_usage_event


_TIMELINE = {"encoding": "gzip+base64", "data": "opaque"}
_COMPRESSED_PERFORMANCE = {
    "time_to_first_token_ms": {
        "any": 7,
        "answer": 11,
        "reasoning": 7,
        "content": 11,
        "tool_call": None,
    },
    "answer_decode_ms": 900,
    "timeline": _TIMELINE,
}
_TTFT_COLUMNS = {
    "ttft_any_ms": 7,
    "ttft_answer_ms": 11,
    "ttft_reasoning_ms": 7,
    "ttft_content_ms": 11,
    "ttft_tool_call_ms": None,
}


def _details(*, performance: object = ...) -> dict[str, object]:
    if performance is ...:
        performance = _COMPRESSED_PERFORMANCE
    return {
        "request": {"model": "openai/gpt-4.1-mini"},
        "result": {"metadata": {"performance": performance}},
    }


def _usage_event(**overrides: object) -> dict[str, object]:
    event: dict[str, object] = {
        "PK": "USAGE#DAY#20260529#S#03",
        "SK": "TS#2026-05-29T12:34:56Z#USG#usage-1",
        "entity_type": "usage_event",
        "usage_event_id": "usage-1",
        "run_id": "run-1",
        "question_id": "question-1",
        "query_id": "",
        "benchmark_name": "swebench",
        "agent_name": "swe-agent",
        "identity_email": "user@example.com",
        "api_key_name": "security-testing",
        "model": "openai/gpt-4.1-mini",
        "provider": "openai",
        "provider_endpoint": "default",
        "param_group": "pg",
        "config_hash": "config-hash",
        "finish_reason": "stop",
        "finish_reason_raw": "stop",
        "details": _details(),
        "completed_at": "2026-05-29T12:34:56Z",
        "day": "20260529",
        "usage_shard": "03",
        "schema_version": 3,
        "normalization_version": "v1",
        "input_tokens": 100,
        "output_tokens": 25,
        "reasoning_tokens": 5,
        "cache_read_tokens": 10,
        "cache_write_tokens": 3,
        "total_input_tokens": 113,
        "total_output_tokens": 25,
        "duration_seconds": Decimal("1.234567"),
        "cost_usd": Decimal("0.123456789012"),
    }
    event.update(overrides)
    return event


def test_redshift_rows_from_usage_event_maps_v3_fact_and_performance_rows() -> None:
    rows = redshift_rows_from_usage_event(
        _usage_event(),
        batch_id="batch-1",
        loaded_at=datetime(2026, 5, 29, 12, 35, 1, tzinfo=UTC),
    )

    assert rows.fact == {
        "batch_id": "batch-1",
        "usage_event_id": "usage-1",
        "completed_at": datetime(2026, 5, 29, 12, 34, 56, tzinfo=UTC),
        "completed_date": date(2026, 5, 29),
        "completed_hour": datetime(2026, 5, 29, 12, 0),
        "run_id": "run-1",
        "question_id": "question-1",
        "query_id": "",
        "query_id_normalized": None,
        "requested_model_key": "openai/gpt-4.1-mini",
        "provider": "openai",
        "provider_endpoint": "default",
        "param_group": "pg",
        "config_hash": "config-hash",
        "benchmark_name": "swebench",
        "agent_name": "swe-agent",
        "identity_email": "user@example.com",
        "api_key_name": "security-testing",
        "outcome": "success",
        "error_type": None,
        "input_tokens": 100,
        "output_tokens": 25,
        "reasoning_tokens": 5,
        "cache_read_tokens": 10,
        "cache_write_tokens": 3,
        "total_input_tokens": 113,
        "total_output_tokens": 25,
        "duration_seconds": Decimal("1.234567"),
        "cost_usd": Decimal("0.123456789012"),
        "finish_reason": "stop",
        "finish_reason_raw": "stop",
        "schema_version": 3,
        "metadata_schema_version": 3,
        "normalization_version": "v1",
        "usage_shard": "03",
        "source_pk": "USAGE#DAY#20260529#S#03",
        "source_sk": "TS#2026-05-29T12:34:56Z#USG#usage-1",
        "loaded_at": datetime(2026, 5, 29, 12, 35, 1, tzinfo=UTC),
    }
    assert rows.performance == {
        "usage_event_id": "usage-1",
        **_TTFT_COLUMNS,
        "timeline": _TIMELINE,
        "performance_truncated": False,
        "loaded_at": datetime(2026, 5, 29, 12, 35, 1, tzinfo=UTC),
    }


@pytest.mark.parametrize(
    "performance",
    [
        {"timeline": _TIMELINE},
        {"time_to_first_token_ms": {}, "timeline": {"encoding": "plain", "data": ""}},
        {**_COMPRESSED_PERFORMANCE, "extra": True},
        {"encoding": "plain", "data": "opaque"},
        {"encoding": "gzip+base64", "data": 1},
        {"encoding": "gzip+base64", "data": "opaque", "extra": True},
        [],
    ],
)
def test_redshift_rows_reject_invalid_performance_envelopes(
    performance: object,
) -> None:
    event = _usage_event(details=_details(performance=performance))

    with pytest.raises(ValueError):
        redshift_rows_from_usage_event(event, batch_id="batch-1")


def test_redshift_rows_split_legacy_whole_performance_envelope() -> None:
    timeline = [
        {"channel": "reasoning", "index": 0, "first_token_ms": 5, "end_ms": 9, "events": []},
        {"channel": "content", "index": 0, "first_token_ms": 10, "end_ms": 40, "events": []},
    ]
    payload = json.dumps(
        {"time_to_first_token_ms": _COMPRESSED_PERFORMANCE["time_to_first_token_ms"], "timeline": timeline}
    ).encode()
    legacy = {
        "encoding": "gzip+base64",
        "data": base64.b64encode(gzip.compress(payload)).decode("ascii"),
    }

    rows = redshift_rows_from_usage_event(
        _usage_event(details=_details(performance=legacy)), batch_id="batch-1"
    )

    assert {column: rows.performance[column] for column in _TTFT_COLUMNS} == _TTFT_COLUMNS
    envelope = cast(dict[str, str], rows.performance["timeline"])
    assert envelope["encoding"] == "gzip+base64"
    assert json.loads(gzip.decompress(base64.b64decode(envelope["data"]))) == timeline


def test_redshift_rows_from_usage_event_marks_whole_details_replacement() -> None:
    rows = redshift_rows_from_usage_event(
        _usage_event(
            details={
                "truncated": True,
                "request": {},
                "result": {"metadata": {"performance": None}},
            }
        ),
        batch_id="batch-1",
    )

    assert rows.performance["timeline"] is None
    assert rows.performance["performance_truncated"]


def test_redshift_rows_from_usage_event_handles_optional_root_and_performance() -> None:
    event = _usage_event(
        query_id="query-1",
        identity_email=None,
        api_key_name=None,
        finish_reason="length",
        details=_details(performance=None),
    )
    del event["finish_reason_raw"]

    rows = redshift_rows_from_usage_event(
        event,
        batch_id="batch-1",
        loaded_at=datetime(2026, 5, 29, 12, 35, 1, tzinfo=UTC),
    )

    assert rows.fact["query_id_normalized"] == "query-1"
    assert rows.fact["identity_email"] is None
    assert rows.fact["api_key_name"] is None
    assert rows.fact["finish_reason"] == "length"
    assert rows.fact["finish_reason_raw"] is None
    assert rows.performance["timeline"] is None
    assert not rows.performance["performance_truncated"]


def test_redshift_rows_from_usage_event_rejects_reserved_dimension_values() -> None:
    with pytest.raises(ValueError, match=redshift_schema.MISSING_DIMENSION_VALUE):
        redshift_rows_from_usage_event(
            _usage_event(benchmark_name=redshift_schema.MISSING_DIMENSION_VALUE),
            batch_id="batch-1",
        )


def test_v4_failed_result_preserves_unknown_measurements_and_explicit_zero() -> None:
    event = _usage_event(
        schema_version=4,
        outcome="error",
        error={"exception_type": "InvalidStructuredOutputError"},
        input_tokens=0,
        cost_usd=Decimal("0"),
    )
    for field in (
        "output_tokens", "reasoning_tokens", "cache_read_tokens",
        "cache_write_tokens", "total_input_tokens", "total_output_tokens",
        "duration_seconds",
    ):
        del event[field]

    rows = redshift_rows_from_usage_event(event, batch_id="batch-1")

    assert rows.fact["outcome"] == "error"
    assert rows.fact["error_type"] == "InvalidStructuredOutputError"
    assert rows.fact["input_tokens"] == 0
    assert rows.fact["cost_usd"] == Decimal("0")
    assert all(rows.fact[field] is None for field in (
        "output_tokens", "reasoning_tokens", "cache_read_tokens",
        "cache_write_tokens", "total_input_tokens", "total_output_tokens",
        "duration_seconds",
    ))
