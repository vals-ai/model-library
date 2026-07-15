from datetime import UTC, datetime
from decimal import Decimal
from typing import cast
from model_gateway.usage_ledger.dynamodb_writer import serialize_item
from model_gateway.usage_ledger.redshift_export_job import (
    RedshiftExportJobConfig,
    run_export_job,
)


class FakeDynamoDbClient:
    def __init__(self, items: list[dict[str, object]]) -> None:
        self.items = items

    def query(self, **_kwargs: object) -> dict[str, object]:
        return {"Items": [serialize_item(item) for item in self.items]}


class FakeS3Client:
    def __init__(self) -> None:
        self.puts: list[dict[str, object]] = []

    def put_object(self, **kwargs: object) -> dict[str, object]:
        self.puts.append(kwargs)
        return {}


class FakeRedshiftClient:
    def __init__(self) -> None:
        self.submissions: list[dict[str, object]] = []

    def batch_execute_statement(self, **kwargs: object) -> dict[str, object]:
        self.submissions.append(kwargs)
        return {"Id": f"statement-{len(self.submissions)}"}

    def describe_statement(self, **_kwargs: object) -> dict[str, object]:
        return {"Status": "FINISHED"}


def _config() -> RedshiftExportJobConfig:
    return RedshiftExportJobConfig(
        raw_table_name="usage-ledger",
        shards=1,
        redshift_workgroup_name="usage-workgroup",
        redshift_database_name="usage",
        landing_bucket_name="usage-landing",
        copy_role_arn="arn:aws:iam::123456789012:role/copy",
        redshift_schema_name="gateway_usage_dev",
        export_prefix="gateway-usage/dev/raw",
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
        "completed_at": "2026-05-29T12:01:00Z",
        "usage_shard": "00",
        "schema_version": 1,
        "metadata_schema_version": 1,
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


def test_export_job_returns_immediately_for_empty_window() -> None:
    s3 = FakeS3Client()
    redshift = FakeRedshiftClient()

    run_export_job(
        config=_config(),
        dynamodb_client=FakeDynamoDbClient([]),
        s3_client=s3,
        redshift_data_client=redshift,
        now=datetime(2026, 5, 29, 12, 5, tzinfo=UTC),
    )

    assert s3.puts == []
    assert redshift.submissions == []
