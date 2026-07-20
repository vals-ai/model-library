"""Transform usage ledger events into Redshift load rows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

from model_gateway.usage_ledger import redshift_schema
from model_gateway.usage_ledger import schema as ledger_schema

RedshiftRow = dict[str, object]


@dataclass(frozen=True)
class RedshiftUsageRows:
    fact: RedshiftRow
    performance: RedshiftRow


_DIMENSION_FIELDS = (
    "provider",
    "model",
    "benchmark_name",
    "agent_name",
    "identity_email",
    "api_key_fingerprint",
)


def redshift_rows_from_usage_event(
    event: Mapping[str, Any],
    *,
    batch_id: str,
    loaded_at: datetime | None = None,
) -> RedshiftUsageRows:
    _validate_reserved_dimension_values(event)

    completed_at = datetime.fromisoformat(event["completed_at"])
    loaded_at_utc = (loaded_at or datetime.now(UTC)).astimezone(UTC)
    performance_data, performance_truncated = _performance_data(
        event[ledger_schema.DETAILS_FIELD]
    )
    query_id = event.get("query_id")

    fact: RedshiftRow = {
        "batch_id": batch_id,
        "usage_event_id": event["usage_event_id"],
        "completed_at": completed_at,
        "completed_date": completed_at.date(),
        "completed_hour": completed_at.replace(
            minute=0, second=0, microsecond=0, tzinfo=None
        ),
        "run_id": event.get("run_id"),
        "question_id": event.get("question_id"),
        "query_id": query_id,
        "query_id_normalized": query_id or None,
        "requested_model_key": event["model"],
        "provider": event["provider"],
        "provider_endpoint": event["provider_endpoint"],
        "param_group": event.get("param_group"),
        "config_hash": event.get("config_hash"),
        "benchmark_name": event.get("benchmark_name"),
        "agent_name": event.get("agent_name"),
        "identity_email": event.get("identity_email"),
        "api_key_fingerprint": event.get("api_key_fingerprint"),
        "input_tokens": event["input_tokens"],
        "output_tokens": event["output_tokens"],
        "reasoning_tokens": event["reasoning_tokens"],
        "cache_read_tokens": event["cache_read_tokens"],
        "cache_write_tokens": event["cache_write_tokens"],
        "total_input_tokens": event["total_input_tokens"],
        "total_output_tokens": event["total_output_tokens"],
        "duration_seconds": event.get("duration_seconds"),
        "cost_usd": event["cost_usd"],
        "finish_reason": event["finish_reason"],
        "finish_reason_raw": event.get("finish_reason_raw"),
        "schema_version": event["schema_version"],
        "metadata_schema_version": event["schema_version"],
        "normalization_version": event["normalization_version"],
        "usage_shard": event.get("usage_shard"),
        "source_pk": event["PK"],
        "source_sk": event["SK"],
        "loaded_at": loaded_at_utc,
    }
    performance: RedshiftRow = {
        "usage_event_id": fact["usage_event_id"],
        "performance": performance_data,
        "performance_truncated": performance_truncated,
        "loaded_at": loaded_at_utc,
    }
    return RedshiftUsageRows(fact=fact, performance=performance)


def _validate_reserved_dimension_values(event: Mapping[str, Any]) -> None:
    for field in _DIMENSION_FIELDS:
        value = event.get(field)
        if value is not None and redshift_schema.is_reserved_dimension_value(value):
            raise ValueError(
                f"Usage event field {field!r} uses reserved Redshift dimension value "
                f"{value!r}"
            )


def _performance_data(
    details: Any,
) -> tuple[dict[str, object] | None, bool]:
    details_truncated = bool(details.get("truncated", False))
    performance_value: object = details["result"]["metadata"]["performance"]
    if performance_value is None:
        return None, details_truncated

    if not isinstance(performance_value, Mapping):
        raise ValueError("performance must contain a compressed envelope")
    performance = cast(Mapping[str, object], performance_value)
    if set(performance) != {"encoding", "data"}:
        raise ValueError("performance must contain only encoding and data")
    encoding = performance["encoding"]
    data = performance["data"]
    if encoding != "gzip+base64" or not isinstance(data, str):
        raise ValueError("performance must contain a gzip+base64 string envelope")
    return {"encoding": encoding, "data": data}, details_truncated
