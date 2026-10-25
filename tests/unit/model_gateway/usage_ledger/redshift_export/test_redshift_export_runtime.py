import io
from datetime import UTC, datetime
from decimal import Decimal
from typing import cast

import pytest
from botocore.exceptions import ClientError
from pydantic import ValidationError

from model_gateway.budgets.schema import SpendSnapshot
from model_gateway.usage_ledger.budgets import publisher
from model_gateway.usage_ledger.dynamodb_writer import serialize_item
from model_gateway.usage_ledger.redshift_export_job import (
    RedshiftExportJobConfig,
    run_export_event,
    run_export_job,
)


class FakeDynamoDbClient:
    def __init__(self, items: list[dict[str, object]]) -> None:
        self.items = items
        self.calls: list[dict[str, object]] = []

    def query(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        return {"Items": [serialize_item(item) for item in self.items]}


class FakeS3Client:
    def __init__(self) -> None:
        self.puts: list[dict[str, object]] = []
        self.snapshot_body: bytes | None = None
        self.snapshot_metadata: dict[str, str] = {}

    def get_object(self, **kwargs: object) -> dict[str, object]:
        if self.snapshot_body is None:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "GetObject"
            )
        return {
            "Body": io.BytesIO(self.snapshot_body),
            "ETag": '"snapshot"',
            "Metadata": self.snapshot_metadata,
        }

    def put_object(self, **kwargs: object) -> dict[str, object]:
        self.puts.append(kwargs)
        if kwargs["Key"] == "gateway-usage/dev/spend/current.json":
            self.snapshot_body = bytes(kwargs["Body"])
            self.snapshot_metadata = cast(dict[str, str], kwargs["Metadata"])
        return {}


class FakeRedshiftClient:
    def __init__(self) -> None:
        self.submissions: list[dict[str, object]] = []

    def batch_execute_statement(self, **kwargs: object) -> dict[str, object]:
        self.submissions.append(kwargs)
        return {"Id": f"statement-{len(self.submissions)}"}

    def describe_statement(self, **_kwargs: object) -> dict[str, object]:
        return {"Status": "FINISHED"}

    def execute_statement(self, **kwargs: object) -> dict[str, str]:
        return {"Id": "budget-spend"}

    def get_statement_result(self, **kwargs: object) -> dict[str, object]:
        return {"Records": []}


@pytest.fixture(autouse=True)
def registry_for_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        publisher.gateway_registry,
        "fetch_live_models",
        lambda: {"billable": {"metadata": {"ignored_for_cost": False}}},
    )


def _config() -> RedshiftExportJobConfig:
    return RedshiftExportJobConfig(
        raw_table_name="usage-ledger",
        shards=1,
        redshift_workgroup_name="usage-workgroup",
        redshift_database_name="usage",
        redshift_secret_arn="arn:aws:secretsmanager:us-east-1:123456789012:secret:usage-loader",
        landing_bucket_name="usage-landing",
        copy_role_arn="arn:aws:iam::123456789012:role/copy",
        redshift_schema_name="gateway_usage_dev",
        export_prefix="gateway-usage/dev/raw",
        budget_snapshot_bucket="usage-landing",
        budget_snapshot_key="gateway-usage/dev/spend/current.json",
        stage="dev",
    )


def _raw_event() -> dict[str, object]:
    return {
        "PK": "USAGE#DAY#20260529#S#00",
        "SK": "TS#2026-05-29T12:01:00Z#USG#usage-1",
        "entity_type": "usage_event",
        "usage_event_id": "usage-1",
        "model": "openai/gpt-4.1-mini",
        "provider": "openai",
        "provider_endpoint": "default",
        "api_key_name": "default",
        "finish_reason": "stop",
        "finish_reason_raw": "stop",
        "details": {
            "request": {},
            "result": {"metadata": {"performance": None}},
        },
        "completed_at": "2026-05-29T12:01:00Z",
        "usage_shard": "00",
        "schema_version": 3,
        "normalization_version": "v1",
        "input_tokens": 100,
        "output_tokens": 20,
        "reasoning_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "total_input_tokens": 100,
        "total_output_tokens": 20,
        "cost_usd": Decimal("0.1"),
    }


def test_export_job_runs_direct_window_load() -> None:
    s3 = FakeS3Client()
    redshift = FakeRedshiftClient()

    run_export_job(
        config=_config(),
        dynamodb_client=FakeDynamoDbClient([_raw_event()]),
        s3_client=s3,
        redshift_data_client=redshift,
        now=datetime(2026, 5, 29, 12, 5, tzinfo=UTC),
    )

    assert [put["Key"] for put in s3.puts] == [
        "gateway-usage/dev/raw/window=20260529T113000Z-20260529T120500Z/"
        "shard=00/part-000.parquet",
        "gateway-usage/dev/raw/window=20260529T113000Z-20260529T120500Z/manifest.json",
        "gateway-usage/dev/spend/current.json",
    ]
    assert [submission["StatementName"] for submission in redshift.submissions] == [
        "usage-redshift-window-prepare",
        "usage-redshift-window-copy",
        "usage-redshift-window-finalize",
    ]
    assert all(
        submission["WorkgroupName"] == "usage-workgroup"
        and submission["Database"] == "usage"
        for submission in redshift.submissions
    )
    submission_sql = [
        cast(list[str], submission["Sqls"])[0] for submission in redshift.submissions
    ]
    assert submission_sql[0].startswith(
        "delete from gateway_usage_dev.usage_events_staging"
    )
    assert submission_sql[1].startswith("copy gateway_usage_dev.usage_events_staging")
    assert submission_sql[2].startswith("merge into gateway_usage_dev.usage_events")


def test_explicit_backfill_event_replays_current_schema_separately() -> None:
    dynamodb = FakeDynamoDbClient([_raw_event()])
    s3 = FakeS3Client()
    redshift = FakeRedshiftClient()

    loaded = run_export_event(
        {
            "operation": "backfill",
            "start": "2026-05-29T12:00:00Z",
            "end": "2026-05-29T12:05:00Z",
        },
        config=_config(),
        dynamodb_client=dynamodb,
        s3_client=s3,
        redshift_data_client=redshift,
    )

    assert loaded == {"loaded": True, "rows": 1}
    assert [put["Key"] for put in s3.puts] == [
        "gateway-usage/dev/raw/backfill/"
        "window=20260529T120000Z-20260529T120500Z/shard=00/part-000.parquet",
        "gateway-usage/dev/raw/backfill/"
        "window=20260529T120000Z-20260529T120500Z/manifest.json",
    ]
    prepare_sql = cast(list[str], redshift.submissions[0]["Sqls"])[0]
    assert (
        "batch_id in ('backfill-20260529T120000Z-20260529T120500Z-s00')" in prepare_sql
    )


def test_explicit_analytics_refresh_runs_once_for_complete_backfill_range() -> None:
    redshift = FakeRedshiftClient()

    result = run_export_event(
        {
            "operation": "refresh_analytics",
            "start": "2026-05-29T12:00:00Z",
            "end": "2026-05-30T12:00:00Z",
        },
        config=_config(),
        dynamodb_client=FakeDynamoDbClient([]),
        s3_client=FakeS3Client(),
        redshift_data_client=redshift,
    )

    assert result == {"refreshed": True}
    assert len(redshift.submissions) == 1
    sql = "\n".join(cast(list[str], redshift.submissions[0]["Sqls"]))
    assert sql.count("delete from gateway_usage_dev.usage_agg_") == 3
    assert sql.count("insert into gateway_usage_dev.usage_agg_") == 3
    assert "delete from gateway_usage_dev.usage_dimension_values;" in sql


def test_manual_refresh_keeps_real_scan_cutoff_until_next_scan() -> None:
    dynamodb = FakeDynamoDbClient([_raw_event()])
    s3 = FakeS3Client()
    redshift = FakeRedshiftClient()
    initial_scan = datetime(2026, 5, 29, 12, 5, tzinfo=UTC)
    run_export_job(
        config=_config(),
        dynamodb_client=dynamodb,
        s3_client=s3,
        redshift_data_client=redshift,
        now=initial_scan,
    )
    assert s3.snapshot_body is not None
    first = SpendSnapshot.model_validate_json(s3.snapshot_body)
    assert first.data_through == initial_scan

    assert run_export_event(
        {
            "operation": "refresh_analytics",
            "start": "2026-05-29T12:00:00Z",
            "end": "2026-05-30T12:00:00Z",
        },
        config=_config(),
        dynamodb_client=dynamodb,
        s3_client=s3,
        redshift_data_client=redshift,
    ) == {"refreshed": True}
    assert s3.snapshot_body is not None
    refreshed = SpendSnapshot.model_validate_json(s3.snapshot_body)
    assert refreshed.revision == first.revision + 1
    assert refreshed.data_through == initial_scan

    next_event = _raw_event()
    next_event["SK"] = "TS#2026-05-29T12:06:00Z#USG#usage-2"
    next_event["completed_at"] = "2026-05-29T12:06:00Z"
    next_event["usage_event_id"] = "usage-2"
    dynamodb.items = [next_event]
    next_scan = datetime(2026, 5, 29, 12, 10, tzinfo=UTC)
    run_export_job(
        config=_config(),
        dynamodb_client=dynamodb,
        s3_client=s3,
        redshift_data_client=redshift,
        now=next_scan,
    )
    assert s3.snapshot_body is not None
    assert SpendSnapshot.model_validate_json(s3.snapshot_body).data_through == next_scan


def test_manual_refresh_before_first_scan_waits_for_normal_export() -> None:
    dynamodb = FakeDynamoDbClient([])
    s3 = FakeS3Client()
    redshift = FakeRedshiftClient()
    assert run_export_event(
        {
            "operation": "refresh_analytics",
            "start": "2026-05-29T12:00:00Z",
            "end": "2026-05-30T12:00:00Z",
        },
        config=_config(),
        dynamodb_client=dynamodb,
        s3_client=s3,
        redshift_data_client=redshift,
    ) == {"refreshed": True}
    assert s3.snapshot_body is None
    assert s3.puts == []

    dynamodb.items = [_raw_event()]
    scan_time = datetime(2026, 5, 29, 12, 5, tzinfo=UTC)
    run_export_job(
        config=_config(),
        dynamodb_client=dynamodb,
        s3_client=s3,
        redshift_data_client=redshift,
        now=scan_time,
    )
    assert s3.snapshot_body is not None
    snapshot = SpendSnapshot.model_validate_json(s3.snapshot_body)
    assert snapshot.revision == 1
    assert snapshot.data_through == scan_time


def test_backfill_event_rejects_fractional_second_boundaries() -> None:
    with pytest.raises(ValidationError, match="whole-second"):
        run_export_event(
            {
                "operation": "backfill",
                "start": "2026-05-29T12:00:00.100000Z",
                "end": "2026-05-29T12:05:00Z",
            },
            config=_config(),
            dynamodb_client=FakeDynamoDbClient([]),
            s3_client=FakeS3Client(),
            redshift_data_client=FakeRedshiftClient(),
        )


def test_empty_window_publishes_spend_snapshot_without_raw_export() -> None:
    s3 = FakeS3Client()
    redshift = FakeRedshiftClient()

    run_export_job(
        config=_config(),
        dynamodb_client=FakeDynamoDbClient([]),
        s3_client=s3,
        redshift_data_client=redshift,
        now=datetime(2026, 5, 29, 12, 5, tzinfo=UTC),
    )

    assert [put["Key"] for put in s3.puts] == ["gateway-usage/dev/spend/current.json"]
    assert s3.snapshot_body is not None
    assert (
        SpendSnapshot.model_validate_json(s3.snapshot_body).periods["monthly"].spend_usd
        == {}
    )
    assert redshift.submissions == []
