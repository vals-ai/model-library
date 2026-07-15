from datetime import UTC, datetime

from model_gateway.usage_ledger import redshift_schema
from model_gateway.usage_ledger.redshift_load import (
    RedshiftWindowLoad,
    copy_usage_events_staging_sql,
    copy_window_statements,
    finalize_window_statements,
    prepare_window_statements,
    staging_row_from_redshift_rows,
)
from model_gateway.usage_ledger.redshift_transform import RedshiftUsageRows


def _load(
    *,
    start: datetime = datetime(2026, 5, 29, 12, 35, tzinfo=UTC),
    end: datetime = datetime(2026, 5, 29, 12, 40, tzinfo=UTC),
) -> RedshiftWindowLoad:
    return RedshiftWindowLoad(
        aggregate_start=start,
        aggregate_end=end,
        manifest_s3_uri="s3://bucket/window/manifest.json",
        batch_ids=("window-s00", "window-s01"),
    )


def _rows() -> RedshiftUsageRows:
    return RedshiftUsageRows(
        fact={
            "batch_id": "batch-1",
            "usage_event_id": "usage-1",
            "completed_at": datetime(2026, 5, 29, 12, tzinfo=UTC),
            "loaded_at": datetime(2026, 5, 29, 12, 1, tzinfo=UTC),
        },
        debug={
            "usage_event_id": "usage-1",
            "identity_json": {"email": "user@example.com"},
            "provider_request_id": "request-1",
            "provider_response_id": "response-1",
            "performance_json": {"timeline": [{"channel": "content"}]},
            "performance_json_truncated": False,
            "loaded_at": datetime(2026, 5, 29, 12, 1, tzinfo=UTC),
        },
    )


def test_staging_table_carries_exact_fact_and_debug_columns() -> None:
    ddl = redshift_schema.usage_events_staging_table_ddl()

    assert "batch_id varchar(128) not null" in ddl
    assert "usage_event_id varchar(128) not null" in ddl
    assert "identity_json varchar(65535)" in ddl
    assert "performance_json varchar(65535)" in ddl


def test_staging_row_combines_fact_and_serialized_debug_payloads() -> None:
    row = staging_row_from_redshift_rows(_rows())

    assert row["usage_event_id"] == "usage-1"
    assert row["identity_json"] == '{"email":"user@example.com"}'
    assert row["performance_json"] == '{"timeline":[{"channel":"content"}]}'
    assert row["performance_json_truncated"] is False


def test_direct_prepare_copy_and_finalize_use_batch_ids() -> None:
    load = _load()
    prepare = prepare_window_statements(load, schema="gateway_usage_dev")
    copy = copy_window_statements(
        load,
        iam_role_arn="arn:aws:iam::123456789012:role/copy",
        schema="gateway_usage_dev",
    )
    finalize = finalize_window_statements(load, schema="gateway_usage_dev")
    sql = "\n".join([*prepare, *copy, *finalize])

    assert prepare == [
        "delete from gateway_usage_dev.usage_events_staging\n"
        "where batch_id in ('window-s00', 'window-s01');"
    ]
    assert "from 's3://bucket/window/manifest.json'" in copy[0]
    assert "iam_role 'arn:aws:iam::123456789012:role/copy'" in copy[0]
    assert "on usage_events.usage_event_id = source.usage_event_id" in sql
    assert "on usage_event_debug.usage_event_id = source.usage_event_id" in sql
    assert "delete from gateway_usage_dev.usage_events_staging" in finalize[-1]
    assert "load_batches" not in sql
    assert "export_control" not in sql
    assert "accept_export_fence" not in sql


def test_finalize_refreshes_aggregates_and_dimension_values() -> None:
    sql = "\n".join(
        finalize_window_statements(
            _load(
                start=datetime(2026, 5, 29, 12, tzinfo=UTC),
                end=datetime(2026, 5, 29, 13, tzinfo=UTC),
            ),
            schema=redshift_schema.WAREHOUSE_SCHEMA,
        )
    )

    assert sql.count("delete from gateway_usage.usage_agg_") == 3
    assert sql.count("insert into gateway_usage.usage_agg_") == 3
    assert "delete from gateway_usage.usage_dimension_values;" in sql
    assert "insert into gateway_usage.usage_dimension_values" in sql


def test_finalize_refreshes_complete_coarser_buckets() -> None:
    sql = "\n".join(
        finalize_window_statements(
            _load(),
            schema=redshift_schema.WAREHOUSE_SCHEMA,
        )
    )

    assert (
        "delete from gateway_usage.usage_agg_5m\n"
        "where bucket_start_utc >= '2026-05-29T12:35:00Z'\n"
        "  and bucket_start_utc < '2026-05-29T12:40:00Z';"
    ) in sql
    assert (
        "delete from gateway_usage.usage_agg_1h\n"
        "where bucket_start_utc >= '2026-05-29T12:00:00Z'\n"
        "  and bucket_start_utc < '2026-05-29T13:00:00Z';"
    ) in sql
    assert (
        "delete from gateway_usage.usage_agg_1d\n"
        "where bucket_start_utc >= '2026-05-29T00:00:00Z'\n"
        "  and bucket_start_utc < '2026-05-30T00:00:00Z';"
    ) in sql


def test_finalize_does_not_extend_past_aligned_end() -> None:
    sql = "\n".join(
        finalize_window_statements(
            _load(
                start=datetime(2026, 5, 29, 12, 35, tzinfo=UTC),
                end=datetime(2026, 5, 29, 14, tzinfo=UTC),
            ),
            schema=redshift_schema.WAREHOUSE_SCHEMA,
        )
    )

    assert (
        "delete from gateway_usage.usage_agg_1h\n"
        "where bucket_start_utc >= '2026-05-29T12:00:00Z'\n"
        "  and bucket_start_utc < '2026-05-29T14:00:00Z';"
    ) in sql
    assert "bucket_start_utc < '2026-05-29T15:00:00Z'" not in sql


def test_copy_sql_uses_manifest_and_role_literals() -> None:
    sql = copy_usage_events_staging_sql(
        manifest_uri="s3://bucket/o'hare/manifest.json",
        iam_role_arn="arn:aws:iam::123456789012:role/o'hare",
    )

    assert "s3://bucket/o''hare/manifest.json" in sql
    assert "arn:aws:iam::123456789012:role/o''hare" in sql
