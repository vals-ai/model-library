from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from model_gateway.usage_ledger import redshift_schema
from model_gateway.usage_ledger.redshift_transform import (
    RedshiftTransformError,
    redshift_rows_from_usage_event,
)


def _usage_event(**overrides: object) -> dict[str, object]:
    event: dict[str, object] = {
        "PK": "USAGE#DAY#20260529#S#03",
        "SK": "TS#2026-05-29T12:34:56Z#USG#usage-1",
        "entity_type": "usage_event",
        "usage_event_id": "usage-1",
        "run_id": "run-1",
        "question_id": "question-1",
        "query_id": "",
        "identity": {
            "email": "user@example.com",
            "benchmark_name": "swebench",
            "agent_name": "swe-agent",
        },
        "benchmark_name": "swebench",
        "agent_name": "swe-agent",
        "identity_email": "user@example.com",
        "api_key_fingerprint": "keyfingerprint",
        "provider_response_id": "provider-response-1",
        "provider_request_id": "provider-request-1",
        "model": "openai/gpt-4.1-mini",
        "provider": "openai",
        "provider_endpoint": "default",
        "param_group": "pg",
        "config_hash": "config-hash",
        "config_redacted_json": '{"temperature":0.2,"secret":"<redacted>"}',
        "config_redacted_json_truncated": False,
        "metadata_json": '{"token_metadata":{"actual":125}}',
        "metadata_json_truncated": False,
        "finish_reason_json": '{"reason":"stop","raw":"stop"}',
        "finish_reason_json_truncated": False,
        "performance_json": '{"timeline":[{"channel":"content","first_token_ms":25}]}',
        "performance_json_truncated": False,
        "completed_at": "2026-05-29T12:34:56Z",
        "day": "20260529",
        "usage_shard": "03",
        "schema_version": 1,
        "metadata_schema_version": 1,
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


def test_redshift_rows_from_usage_event_maps_fact_and_debug_rows():
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
        "api_key_fingerprint": "keyfingerprint",
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
        "schema_version": 1,
        "metadata_schema_version": 1,
        "normalization_version": "v1",
        "usage_shard": "03",
        "source_pk": "USAGE#DAY#20260529#S#03",
        "source_sk": "TS#2026-05-29T12:34:56Z#USG#usage-1",
        "loaded_at": datetime(2026, 5, 29, 12, 35, 1, tzinfo=UTC),
    }
    assert rows.debug == {
        "usage_event_id": "usage-1",
        "identity_json": {
            "email": "user@example.com",
            "benchmark_name": "swebench",
            "agent_name": "swe-agent",
        },
        "provider_request_id": "provider-request-1",
        "provider_response_id": "provider-response-1",
        "config_redacted_json": {"temperature": 0.2, "secret": "<redacted>"},
        "metadata_json": {"token_metadata": {"actual": 125}},
        "finish_reason_json": {"reason": "stop", "raw": "stop"},
        "performance_json": {
            "timeline": [{"channel": "content", "first_token_ms": 25}]
        },
        "config_redacted_json_truncated": False,
        "metadata_json_truncated": False,
        "finish_reason_json_truncated": False,
        "performance_json_truncated": False,
        "loaded_at": datetime(2026, 5, 29, 12, 35, 1, tzinfo=UTC),
    }


def test_redshift_rows_from_usage_event_normalizes_optional_values_and_json():
    rows = redshift_rows_from_usage_event(
        _usage_event(
            query_id="query-1",
            identity_email=None,
            api_key_fingerprint=None,
            config_redacted_json=None,
            metadata_json="not-json",
            finish_reason_json='{"finish_reason":"length"}',
            finish_reason_json_truncated=True,
            performance_json="not-json",
            performance_json_truncated=True,
        ),
        batch_id="batch-1",
        loaded_at=datetime(2026, 5, 29, 12, 35, 1, tzinfo=UTC),
    )

    assert rows.fact["query_id_normalized"] == "query-1"
    assert rows.fact["identity_email"] is None
    assert rows.fact["api_key_fingerprint"] is None
    assert rows.fact["finish_reason"] == "length"
    assert rows.fact["finish_reason_raw"] == "length"
    assert rows.debug["config_redacted_json"] is None
    assert rows.debug["metadata_json"] == "not-json"
    assert rows.debug["finish_reason_json"] == {"finish_reason": "length"}
    assert rows.debug["finish_reason_json_truncated"] is True
    assert rows.debug["performance_json"] == "not-json"
    assert rows.debug["performance_json_truncated"] is True


def test_redshift_rows_from_usage_event_rejects_missing_required_fields():
    event = _usage_event()
    del event["model"]

    with pytest.raises(RedshiftTransformError, match="model"):
        redshift_rows_from_usage_event(event, batch_id="batch-1")


def test_redshift_rows_from_usage_event_rejects_reserved_dimension_values():
    with pytest.raises(
        RedshiftTransformError, match=redshift_schema.MISSING_DIMENSION_VALUE
    ):
        redshift_rows_from_usage_event(
            _usage_event(benchmark_name=redshift_schema.MISSING_DIMENSION_VALUE),
            batch_id="batch-1",
        )
