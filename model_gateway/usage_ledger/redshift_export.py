"""Planning helpers for exporting usage ledger partitions to Redshift landing files."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
import json
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

from model_gateway.usage_ledger import redshift_schema
from model_gateway.usage_ledger import schema as ledger_schema

_REDSHIFT_STAGING_COLUMN_NAMES = redshift_schema.USAGE_EVENTS_STAGING_COLUMN_NAMES


def _redshift_staging_arrow_schema() -> Any:
    pyarrow = cast(Any, pa)
    field_types: Mapping[str, Any] = dict(
        [
            ("batch_id", pyarrow.string()),
            ("usage_event_id", pyarrow.string()),
            ("completed_at", pyarrow.timestamp("us", tz="UTC")),
            ("completed_date", pyarrow.date32()),
            ("completed_hour", pyarrow.timestamp("us")),
            ("run_id", pyarrow.string()),
            ("question_id", pyarrow.string()),
            ("query_id", pyarrow.string()),
            ("query_id_normalized", pyarrow.string()),
            ("requested_model_key", pyarrow.string()),
            ("provider", pyarrow.string()),
            ("provider_endpoint", pyarrow.string()),
            ("param_group", pyarrow.string()),
            ("config_hash", pyarrow.string()),
            ("benchmark_name", pyarrow.string()),
            ("agent_name", pyarrow.string()),
            ("identity_email", pyarrow.string()),
            ("api_key_fingerprint", pyarrow.string()),
            ("input_tokens", pyarrow.int64()),
            ("output_tokens", pyarrow.int64()),
            ("reasoning_tokens", pyarrow.int64()),
            ("cache_read_tokens", pyarrow.int64()),
            ("cache_write_tokens", pyarrow.int64()),
            ("total_input_tokens", pyarrow.int64()),
            ("total_output_tokens", pyarrow.int64()),
            ("duration_seconds", pyarrow.decimal128(18, 6)),
            ("cost_usd", pyarrow.decimal128(38, 12)),
            ("finish_reason", pyarrow.string()),
            ("finish_reason_raw", pyarrow.string()),
            ("schema_version", pyarrow.int32()),
            ("metadata_schema_version", pyarrow.int32()),
            ("normalization_version", pyarrow.string()),
            ("usage_shard", pyarrow.string()),
            ("source_pk", pyarrow.string()),
            ("source_sk", pyarrow.string()),
            ("loaded_at", pyarrow.timestamp("us", tz="UTC")),
            ("identity_json", pyarrow.string()),
            ("provider_request_id", pyarrow.string()),
            ("provider_response_id", pyarrow.string()),
            ("config_redacted_json", pyarrow.string()),
            ("metadata_json", pyarrow.string()),
            ("finish_reason_json", pyarrow.string()),
            ("performance_json", pyarrow.string()),
            ("config_redacted_json_truncated", pyarrow.bool_()),
            ("metadata_json_truncated", pyarrow.bool_()),
            ("finish_reason_json_truncated", pyarrow.bool_()),
            ("performance_json_truncated", pyarrow.bool_()),
        ]
    )
    return pyarrow.schema(
        [
            (column, field_types[column])
            for column in redshift_schema.USAGE_EVENTS_STAGING_COLUMN_NAMES
        ]
    )


_REDSHIFT_STAGING_ARROW_SCHEMA: Any = _redshift_staging_arrow_schema()


_EXPORT_WINDOW_MINUTES = 35


@dataclass(frozen=True)
class LogicalExportWindow:
    start: datetime
    end: datetime
    logical_window_id: str


def floor_export_boundary(value: datetime, *, minutes: int = 5) -> datetime:
    value_utc = _normalize_datetime(value)
    return value_utc.replace(
        minute=value_utc.minute - value_utc.minute % minutes,
        second=0,
        microsecond=0,
    )


def plan_logical_export_window(*, now: datetime) -> LogicalExportWindow:
    end = floor_export_boundary(now)
    start = end - timedelta(minutes=_EXPORT_WINDOW_MINUTES)
    return LogicalExportWindow(
        start=start,
        end=end,
        logical_window_id=f"{_timestamp_id(start)}-{_timestamp_id(end)}",
    )


def export_batch_id(window: LogicalExportWindow, *, shard: int) -> str:
    return f"{window.logical_window_id}-s{ledger_schema.format_shard(shard)}"


def export_data_key(
    window: LogicalExportWindow,
    *,
    shard: int,
    prefix: str,
) -> str:
    return (
        f"{prefix}/window={window.logical_window_id}/"
        f"shard={ledger_schema.format_shard(shard)}/part-000.parquet"
    )


def export_manifest_key(window: LogicalExportWindow, *, prefix: str) -> str:
    return f"{prefix}/window={window.logical_window_id}/manifest.json"


def parquet_bytes_from_rows(rows: Sequence[Mapping[str, object]]) -> bytes:
    pyarrow = cast(Any, pa)
    parquet = cast(Any, pq)
    table = pyarrow.Table.from_pylist(
        [_arrow_staging_row(row) for row in rows], schema=_REDSHIFT_STAGING_ARROW_SCHEMA
    )
    sink = pyarrow.BufferOutputStream()
    parquet.write_table(table, sink, compression="snappy")
    return cast(bytes, sink.getvalue().to_pybytes())


def _arrow_staging_row(row: Mapping[str, object]) -> dict[str, object]:
    normalized = {column: row.get(column) for column in _REDSHIFT_STAGING_COLUMN_NAMES}
    for decimal_column in ("duration_seconds", "cost_usd"):
        value = normalized[decimal_column]
        if value is not None and not isinstance(value, Decimal):
            normalized[decimal_column] = Decimal(str(value))
    return normalized


def build_manifest_json(entries: Sequence[tuple[str, int]]) -> str:
    return json.dumps(
        {
            "entries": [
                {
                    "url": url,
                    "mandatory": True,
                    "meta": {"content_length": content_length},
                }
                for url, content_length in entries
            ]
        },
        separators=(",", ":"),
    )


def _timestamp_id(value: datetime) -> str:
    return _normalize_datetime(value).strftime("%Y%m%dT%H%M%SZ")


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
