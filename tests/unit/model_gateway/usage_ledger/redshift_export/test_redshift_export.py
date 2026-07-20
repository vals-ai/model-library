import json
from datetime import UTC, datetime
from decimal import Decimal

import pyarrow as pa
import pyarrow.parquet as pq

from model_gateway.usage_ledger import redshift_schema
from model_gateway.usage_ledger.redshift_export import (
    build_manifest_json,
    export_batch_id,
    export_data_key,
    export_manifest_key,
    parquet_bytes_from_rows,
    plan_logical_export_window,
)


def test_parquet_bytes_from_rows_writes_exact_redshift_staging_schema() -> None:
    performance = b'{"encoding":"gzip+base64","data":"' + b"x" * 90_048 + b'"}'
    payload = parquet_bytes_from_rows(
        [
            {
                "batch_id": "batch-1",
                "usage_event_id": "usage-1",
                "completed_at": datetime(2026, 5, 29, 12, 1, tzinfo=UTC),
                "completed_date": datetime(2026, 5, 29, tzinfo=UTC).date(),
                "completed_hour": datetime(2026, 5, 29, 12),
                "run_id": None,
                "question_id": None,
                "query_id": None,
                "query_id_normalized": None,
                "requested_model_key": "openai/gpt-4.1-mini",
                "provider": "openai",
                "provider_endpoint": "default",
                "param_group": None,
                "config_hash": None,
                "benchmark_name": None,
                "agent_name": None,
                "identity_email": None,
                "api_key_fingerprint": None,
                "input_tokens": Decimal("100"),
                "output_tokens": Decimal("20"),
                "reasoning_tokens": Decimal("0"),
                "cache_read_tokens": Decimal("0"),
                "cache_write_tokens": Decimal("0"),
                "total_input_tokens": Decimal("100"),
                "total_output_tokens": Decimal("20"),
                "duration_seconds": Decimal("1.000000"),
                "cost_usd": Decimal("0.123456789012"),
                "finish_reason": "stop",
                "finish_reason_raw": None,
                "schema_version": Decimal("2"),
                "metadata_schema_version": Decimal("2"),
                "normalization_version": "v1",
                "usage_shard": None,
                "source_pk": "pk",
                "source_sk": "sk",
                "loaded_at": datetime(2026, 5, 29, 12, 2, tzinfo=UTC),
                "performance": performance,
                "performance_truncated": False,
            }
        ]
    )

    table = pq.read_table(pa.BufferReader(payload))
    assert (
        tuple(table.schema.names) == redshift_schema.USAGE_EVENTS_STAGING_COLUMN_NAMES
    )
    assert table.schema.field("run_id").type == pa.string()
    assert table.schema.field("schema_version").type == pa.int32()
    assert table.schema.field("metadata_schema_version").type == pa.int32()
    assert table.schema.field("cost_usd").type == pa.decimal128(38, 12)
    assert table.schema.field("duration_seconds").type == pa.decimal128(18, 6)
    assert table.schema.field("performance").type == pa.binary()
    assert "payload" not in table.schema.names
    assert "details" not in table.schema.names
    row = table.to_pylist()[0]
    assert row["usage_event_id"] == "usage-1"
    assert row["run_id"] is None
    assert row["input_tokens"] == 100
    assert row["schema_version"] == 2
    assert row["metadata_schema_version"] == 2
    assert row["cost_usd"] == Decimal("0.123456789012")
    assert not row["performance_truncated"]
    assert len(row["performance"]) > 65_535
    assert row["performance"] == performance


def test_window_and_artifact_identity_are_deterministic() -> None:
    window = plan_logical_export_window(
        now=datetime(2026, 5, 29, 12, 7, 42, tzinfo=UTC)
    )

    assert window.start == datetime(2026, 5, 29, 11, 30, tzinfo=UTC)
    assert window.end == datetime(2026, 5, 29, 12, 5, tzinfo=UTC)
    assert window.logical_window_id == "20260529T113000Z-20260529T120500Z"
    assert export_batch_id(window, shard=7) == ("20260529T113000Z-20260529T120500Z-s07")
    assert export_data_key(window, shard=7, prefix="gateway-usage/dev/raw") == (
        "gateway-usage/dev/raw/window=20260529T113000Z-20260529T120500Z/"
        "shard=07/part-000.parquet"
    )
    assert export_manifest_key(window, prefix="gateway-usage/dev/raw") == (
        "gateway-usage/dev/raw/window=20260529T113000Z-20260529T120500Z/manifest.json"
    )


def test_cross_midnight_remains_one_window() -> None:
    window = plan_logical_export_window(now=datetime(2026, 5, 30, 0, 7, tzinfo=UTC))

    assert window.start == datetime(2026, 5, 29, 23, 30, tzinfo=UTC)
    assert window.end == datetime(2026, 5, 30, 0, 5, tzinfo=UTC)


def test_build_manifest_json_uses_redshift_copy_shape() -> None:
    manifest = json.loads(
        build_manifest_json(
            [
                ("s3://bucket/part-0.parquet", 100),
                ("s3://bucket/part-1.parquet", 200),
            ]
        )
    )

    assert manifest == {
        "entries": [
            {
                "url": "s3://bucket/part-0.parquet",
                "mandatory": True,
                "meta": {"content_length": 100},
            },
            {
                "url": "s3://bucket/part-1.parquet",
                "mandatory": True,
                "meta": {"content_length": 200},
            },
        ]
    }
