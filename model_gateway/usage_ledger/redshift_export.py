"""Planning helpers for exporting usage ledger partitions to Redshift landing files."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
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
            ("performance", pyarrow.binary()),
            ("performance_truncated", pyarrow.bool_()),
        ]
    )
    return pyarrow.schema(
        [
            (column, field_types[column])
            for column in redshift_schema.USAGE_EVENTS_STAGING_COLUMN_NAMES
        ]
    )


_REDSHIFT_STAGING_ARROW_SCHEMA: Any = _redshift_staging_arrow_schema()


EXPORT_WINDOW_MINUTES = 35


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


def logical_export_window(*, start: datetime, end: datetime) -> LogicalExportWindow:
    start_utc = _normalize_datetime(start)
    end_utc = _normalize_datetime(end)
    return LogicalExportWindow(
        start=start_utc,
        end=end_utc,
        logical_window_id=f"{_timestamp_id(start_utc)}-{_timestamp_id(end_utc)}",
    )


def plan_logical_export_window(*, now: datetime) -> LogicalExportWindow:
    end = floor_export_boundary(now)
    return logical_export_window(
        start=end - timedelta(minutes=EXPORT_WINDOW_MINUTES),
        end=end,
    )


def export_batch_id(
    window: LogicalExportWindow,
    *,
    shard: int,
    namespace: str | None = None,
) -> str:
    prefix = f"{namespace}-" if namespace is not None else ""
    return f"{prefix}{window.logical_window_id}-s{ledger_schema.format_shard(shard)}"


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
    return {column: row[column] for column in _REDSHIFT_STAGING_COLUMN_NAMES}


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
    return value.astimezone(UTC)
