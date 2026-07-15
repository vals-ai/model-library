"""Transform DynamoDB usage ledger events into Redshift load rows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
import json
from typing import cast

from model_gateway.usage_ledger import redshift_schema

JsonValue = dict[str, object] | list[object] | str | int | float | bool | None
RedshiftRow = dict[str, object]


class RedshiftTransformError(ValueError):
    """Raised when a raw usage ledger event cannot be transformed safely."""


@dataclass(frozen=True)
class RedshiftUsageRows:
    fact: RedshiftRow
    debug: RedshiftRow


_REQUIRED_FIELDS = frozenset(
    {
        "PK",
        "SK",
        "usage_event_id",
        "completed_at",
        "model",
        "provider",
        "provider_endpoint",
        "schema_version",
        "metadata_schema_version",
        "normalization_version",
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "total_input_tokens",
        "total_output_tokens",
        "cost_usd",
    }
)

_DIMENSION_FIELDS = (
    "provider",
    "model",
    "benchmark_name",
    "agent_name",
    "identity_email",
    "api_key_fingerprint",
)

_FINISH_REASON_SCALAR_KEYS = ("reason", "finish_reason", "type", "name")
_FINISH_REASON_RAW_KEYS = ("raw", "raw_finish_reason", "finish_reason", "reason")


def redshift_rows_from_usage_event(
    event: Mapping[str, object],
    *,
    batch_id: str,
    loaded_at: datetime | None = None,
) -> RedshiftUsageRows:
    missing = sorted(field for field in _REQUIRED_FIELDS if field not in event)
    if missing:
        raise RedshiftTransformError(
            "Usage event is missing Redshift field(s): " + ", ".join(missing)
        )

    _validate_reserved_dimension_values(event)

    completed_at = _required_datetime(event, "completed_at")
    loaded_at_utc = _normalize_datetime(loaded_at or datetime.now(UTC))
    finish_reason_json = _json_value(event.get("finish_reason_json"))
    performance_json = _json_value(event.get("performance_json"))

    fact: RedshiftRow = {
        "batch_id": batch_id,
        "usage_event_id": _required_string(event, "usage_event_id"),
        "completed_at": completed_at,
        "completed_date": completed_at.date(),
        "completed_hour": completed_at.replace(
            minute=0, second=0, microsecond=0, tzinfo=None
        ),
        "run_id": _optional_string(event.get("run_id")),
        "question_id": _optional_string(event.get("question_id")),
        "query_id": _optional_string(event.get("query_id")),
        "query_id_normalized": _normalized_query_id(event.get("query_id")),
        "requested_model_key": _required_string(event, "model"),
        "provider": _required_string(event, "provider"),
        "provider_endpoint": _required_string(event, "provider_endpoint"),
        "param_group": _optional_string(event.get("param_group")),
        "config_hash": _optional_string(event.get("config_hash")),
        "benchmark_name": _optional_string(event.get("benchmark_name")),
        "agent_name": _optional_string(event.get("agent_name")),
        "identity_email": _optional_string(event.get("identity_email")),
        "api_key_fingerprint": _optional_string(event.get("api_key_fingerprint")),
        "input_tokens": _required_int(event, "input_tokens"),
        "output_tokens": _required_int(event, "output_tokens"),
        "reasoning_tokens": _required_int(event, "reasoning_tokens"),
        "cache_read_tokens": _required_int(event, "cache_read_tokens"),
        "cache_write_tokens": _required_int(event, "cache_write_tokens"),
        "total_input_tokens": _required_int(event, "total_input_tokens"),
        "total_output_tokens": _required_int(event, "total_output_tokens"),
        "duration_seconds": _optional_decimal(event.get("duration_seconds")),
        "cost_usd": _required_decimal(event, "cost_usd"),
        "finish_reason": _finish_reason_value(
            finish_reason_json, _FINISH_REASON_SCALAR_KEYS
        ),
        "finish_reason_raw": _finish_reason_value(
            finish_reason_json, _FINISH_REASON_RAW_KEYS
        ),
        "schema_version": _required_int(event, "schema_version"),
        "metadata_schema_version": _required_int(event, "metadata_schema_version"),
        "normalization_version": _required_string(event, "normalization_version"),
        "usage_shard": _optional_string(event.get("usage_shard")),
        "source_pk": _required_string(event, "PK"),
        "source_sk": _required_string(event, "SK"),
        "loaded_at": loaded_at_utc,
    }
    debug: RedshiftRow = {
        "usage_event_id": fact["usage_event_id"],
        "identity_json": _json_value(event.get("identity")),
        "provider_request_id": _optional_string(event.get("provider_request_id")),
        "provider_response_id": _optional_string(event.get("provider_response_id")),
        "config_redacted_json": _json_value(event.get("config_redacted_json")),
        "metadata_json": _json_value(event.get("metadata_json")),
        "finish_reason_json": finish_reason_json,
        "performance_json": performance_json,
        "config_redacted_json_truncated": _optional_bool(
            event.get("config_redacted_json_truncated")
        ),
        "metadata_json_truncated": _optional_bool(event.get("metadata_json_truncated")),
        "finish_reason_json_truncated": _optional_bool(
            event.get("finish_reason_json_truncated")
        ),
        "performance_json_truncated": _optional_bool(
            event.get("performance_json_truncated")
        ),
        "loaded_at": loaded_at_utc,
    }
    return RedshiftUsageRows(fact=fact, debug=debug)


def _validate_reserved_dimension_values(event: Mapping[str, object]) -> None:
    for field in _DIMENSION_FIELDS:
        value = event.get(field)
        if isinstance(value, str) and redshift_schema.is_reserved_dimension_value(
            value
        ):
            raise RedshiftTransformError(
                f"Usage event field {field!r} uses reserved Redshift dimension value "
                f"{value!r}"
            )


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _required_datetime(event: Mapping[str, object], field: str) -> datetime:
    value = event.get(field)
    if isinstance(value, datetime):
        return _normalize_datetime(value)
    if isinstance(value, str):
        try:
            return _normalize_datetime(
                datetime.fromisoformat(value.replace("Z", "+00:00"))
            )
        except ValueError as exc:
            raise RedshiftTransformError(
                f"Usage event field {field!r} is not an ISO datetime"
            ) from exc
    raise RedshiftTransformError(f"Usage event field {field!r} is not an ISO datetime")


def _required_string(event: Mapping[str, object], field: str) -> str:
    value = event.get(field)
    if not isinstance(value, str) or not value:
        raise RedshiftTransformError(f"Usage event field {field!r} is required")
    return value


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise RedshiftTransformError(
        f"Expected optional string, got {type(value).__name__}"
    )


def _normalized_query_id(value: object) -> str | None:
    query_id = _optional_string(value)
    if query_id == "":
        return None
    return query_id


def _required_int(event: Mapping[str, object], field: str) -> int:
    value = event.get(field)
    if isinstance(value, bool):
        raise RedshiftTransformError(f"Usage event field {field!r} is not an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, Decimal) and value == value.to_integral_value():
        return int(value)
    raise RedshiftTransformError(f"Usage event field {field!r} is not an integer")


def _required_decimal(event: Mapping[str, object], field: str) -> Decimal:
    value = event.get(field)
    decimal_value = _optional_decimal(value)
    if decimal_value is None:
        raise RedshiftTransformError(f"Usage event field {field!r} is not numeric")
    return decimal_value


def _optional_decimal(value: object) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        raise RedshiftTransformError("Boolean value is not numeric")
    if isinstance(value, int | float | str):
        try:
            return Decimal(str(value))
        except Exception as exc:
            raise RedshiftTransformError(f"Value {value!r} is not numeric") from exc
    raise RedshiftTransformError(f"Value {value!r} is not numeric")


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise RedshiftTransformError(f"Expected optional bool, got {type(value).__name__}")


def _json_value(value: object) -> JsonValue:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        return cast(JsonValue, parsed)
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {str(key): item for key, item in mapping.items()}
    if isinstance(value, list):
        return cast(list[object], value)
    if isinstance(value, int | float | bool):
        return value
    return str(value)


def _finish_reason_value(value: JsonValue, keys: tuple[str, ...]) -> str | None:
    if isinstance(value, Mapping):
        value_mapping = cast(Mapping[str, object], value)
        for key in keys:
            item = value_mapping.get(key)
            if item is None:
                continue
            if isinstance(item, str):
                return item
            return str(item)
    if isinstance(value, str) and value:
        return value
    return None
