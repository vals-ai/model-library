"""Transform usage ledger events into Redshift load rows."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
import gzip
import json
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
    "api_key_name",
)

_OPTIONAL_FACT_COLUMNS = frozenset(
    {
        "run_id",
        "question_id",
        "query_id",
        "attempt_id",
        "param_group",
        "config_hash",
        "benchmark_name",
        "agent_name",
        "identity_email",
        "api_key_name",
        "duration_seconds",
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "total_input_tokens",
        "total_output_tokens",
        "cost_usd",
        "finish_reason_raw",
        "usage_shard",
    }
)

_RENAMED_FACT_SOURCES = {
    "requested_model_key": "model",
    "metadata_schema_version": "schema_version",
    "source_pk": "PK",
    "source_sk": "SK",
}

_COMPUTED_FACT_SOURCE_PATHS: dict[str, tuple[tuple[str, ...], ...]] = {
    "outcome": (("outcome",),),
    "error_type": (("error", "exception_type"),),
    "completed_at": (("completed_at",),),
    "completed_date": (("completed_at",),),
    "completed_hour": (("completed_at",),),
    "query_id_normalized": (("query_id",),),
    "loaded_at": (),
}


def redshift_export_source_paths() -> tuple[tuple[str, ...], ...]:
    """Return the raw event paths needed to build the canonical warehouse rows."""
    paths: dict[tuple[str, ...], None] = {}
    for column in redshift_schema.USAGE_EVENT_FACT_COLUMN_NAMES:
        source_paths = _COMPUTED_FACT_SOURCE_PATHS.get(column)
        if source_paths is None:
            source_paths = ((_RENAMED_FACT_SOURCES.get(column, column),),)
        for path in source_paths:
            paths[path] = None
    paths[(ledger_schema.DETAILS_FIELD, "result", "metadata", "performance")] = None
    paths[(ledger_schema.DETAILS_FIELD, "truncated")] = None
    return tuple(paths)


def redshift_rows_from_usage_event(
    event: Mapping[str, Any],
    *,
    batch_id: str,
    loaded_at: datetime | None = None,
) -> RedshiftUsageRows:
    _validate_reserved_dimension_values(event)

    completed_at = datetime.fromisoformat(event["completed_at"])
    loaded_at_utc = (loaded_at or datetime.now(UTC)).astimezone(UTC)
    performance_data = _performance_data(event[ledger_schema.DETAILS_FIELD])
    query_id = event.get("query_id")

    outcome = event["outcome"] if event["schema_version"] == 4 else "success"
    computed: RedshiftRow = {
        "outcome": outcome,
        "error_type": event["error"]["exception_type"] if outcome == "error" else None,
        "completed_at": completed_at,
        "completed_date": completed_at.date(),
        "completed_hour": completed_at.replace(
            minute=0, second=0, microsecond=0, tzinfo=None
        ),
        "query_id_normalized": query_id or None,
        "loaded_at": loaded_at_utc,
    }
    fact: RedshiftRow = {"batch_id": batch_id}
    for column in redshift_schema.USAGE_EVENT_FACT_COLUMN_NAMES:
        if column in _COMPUTED_FACT_SOURCE_PATHS:
            fact[column] = computed[column]
        else:
            source = _RENAMED_FACT_SOURCES.get(column, column)
            fact[column] = (
                event.get(source) if column in _OPTIONAL_FACT_COLUMNS else event[source]
            )
    performance: RedshiftRow = {
        "usage_event_id": fact["usage_event_id"],
        **performance_data,
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


def _performance_data(details: Any) -> RedshiftRow:
    truncated = bool(details.get("truncated", False))
    value: object = details["result"]["metadata"]["performance"]
    if value is None:
        return {
            **dict.fromkeys(redshift_schema.TIME_TO_FIRST_TOKEN_COLUMNS.values()),
            "timeline": None,
            "performance_truncated": truncated,
        }
    if not isinstance(value, Mapping):
        raise ValueError("performance must be an object")
    performance = cast(Mapping[str, Any], value)
    if "timeline" not in performance:
        performance = _split_legacy_performance(performance)
    ttft = performance.get("time_to_first_token_ms")
    if not (
        {"time_to_first_token_ms", "timeline"}
        <= set(performance)
        <= {"time_to_first_token_ms", "answer_decode_ms", "timeline"}
    ) or not isinstance(ttft, Mapping):
        raise ValueError(
            "performance must contain time_to_first_token_ms and a compressed timeline"
        )
    ttft = cast(Mapping[str, object], ttft)
    return {
        **{
            column: ttft.get(field)
            for field, column in redshift_schema.TIME_TO_FIRST_TOKEN_COLUMNS.items()
        },
        "timeline": _compressed_envelope(performance["timeline"]),
        "performance_truncated": truncated,
    }


def _split_legacy_performance(envelope: object) -> dict[str, object]:
    """Convert events written when the whole performance object was compressed."""
    data = _compressed_envelope(envelope)["data"]
    payload = json.loads(gzip.decompress(base64.b64decode(data, validate=True)))
    timeline = json.dumps(payload["timeline"], separators=(",", ":")).encode()
    return {
        "time_to_first_token_ms": payload["time_to_first_token_ms"],
        "timeline": {
            "encoding": "gzip+base64",
            "data": base64.b64encode(
                gzip.compress(timeline, compresslevel=1, mtime=0)
            ).decode("ascii"),
        },
    }


def _compressed_envelope(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("performance timeline must contain a compressed envelope")
    envelope = cast(Mapping[str, object], value)
    if set(envelope) != {"encoding", "data"}:
        raise ValueError("performance timeline must contain only encoding and data")
    encoding = envelope["encoding"]
    data = envelope["data"]
    if encoding != "gzip+base64" or not isinstance(data, str):
        raise ValueError("performance timeline must be a gzip+base64 string envelope")
    return {"encoding": "gzip+base64", "data": data}
