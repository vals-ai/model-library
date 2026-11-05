import io
import json
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from botocore.exceptions import ClientError
from pydantic import ValidationError

from model_gateway.budgets.schema import PeriodSpend, SpendSnapshot, managed_key_spend_id, period_bounds
from model_gateway.managed_auth.types import (
    HumanIdentity,
    ProjectAlertIdentity,
    ProjectLimitRequest,
    ProjectPolicy,
    ProjectReference,
)
from model_gateway.usage_ledger import redshift_export_job
from model_gateway.usage_ledger.budgets.managed_publisher import (
    ManagedSpendPublishResult,
    publish_managed_spend_snapshot,
    publish_project_alert_snapshot,
    read_alert_publication_token,
)
from model_gateway.usage_ledger import gateway_registry
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
        self.alert_body: bytes | None = None

    def get_object(self, **kwargs: object) -> dict[str, object]:
        if kwargs["Key"] == "gateway-usage/dev/spend/project-alerts-current.json":
            if self.alert_body is None:
                raise ClientError(
                    {"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "GetObject"
                )
            return {"Body": io.BytesIO(self.alert_body), "ETag": '"alert"'}
        if kwargs["Key"] != "gateway-usage/dev/managed-spend/current.json":
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "GetObject"
            )
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
        if kwargs["Key"] == "gateway-usage/dev/managed-spend/current.json":
            self.snapshot_body = bytes(kwargs["Body"])
            self.snapshot_metadata = cast(dict[str, str], kwargs["Metadata"])
        if kwargs["Key"] == "gateway-usage/dev/spend/project-alerts-current.json":
            self.alert_body = bytes(kwargs["Body"])
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
        gateway_registry,
        "fetch_live_models",
        lambda: {"billable": {"metadata": {"ignored_for_cost": False}}},
    )


@pytest.fixture(autouse=True)
def managed_spend_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_GATEWAY_AUTH_DATABASE_NAME", "gateway_auth")
    monkeypatch.setattr(
        redshift_export_job.boto3,
        "client",
        lambda service: SimpleNamespace(
            get_secret_value=lambda **kwargs: {
                "SecretString": '{"username":"endpoint","password":"endpoint","host":"localhost","port":5432}'
            }
        ),
    )
    monkeypatch.setattr(
        redshift_export_job.ManagedAuthStore,
        "list_project_allowance_sources",
        AsyncMock(return_value=((), {})),
    )
    monkeypatch.setattr(
        redshift_export_job.ManagedAuthStore,
        "list_project_alert_identities",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        redshift_export_job, "list_key_spend_sources", AsyncMock(return_value=[])
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
        managed_budget_snapshot_key="gateway-usage/dev/managed-spend/current.json",
        project_alert_snapshot_key="gateway-usage/dev/spend/project-alerts-current.json",
        managed_database_secret_arn="arn:aws:secretsmanager:us-east-1:123456789012:secret:managed-db",
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
    project_id = UUID("00000000-0000-0000-0000-000000000001")
    key_id = UUID("00000000-0000-0000-0000-000000000002")
    project = ProjectReference(project_id, "example-org", "dev")
    redshift_export_job.ManagedAuthStore.list_project_allowance_sources.return_value = (
        (ProjectPolicy(project, "monthly", Decimal("10"), 1,
                       datetime(2026, 5, 1, tzinfo=UTC),
                       datetime(2026, 5, 1, tzinfo=UTC)),), {}
    )
    redshift_export_job.ManagedAuthStore.list_project_alert_identities.return_value = {
        project_id: ProjectAlertIdentity("Project label", "project-handle")
    }
    redshift_export_job.list_key_spend_sources.return_value = [key_id]
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
        "gateway-usage/dev/managed-spend/current.json",
        "gateway-usage/dev/spend/project-alerts-current.json",
    ]
    managed = SpendSnapshot.model_validate_json(s3.puts[-2]["Body"])
    for period in ("daily", "weekly", "monthly"):
        assert managed.periods[period].spend_usd == {
            str(project_id): 0, managed_key_spend_id(key_id): 0,
        }
    assert json.loads(s3.alert_body)["projects"][0]["spend_usd"] == "0"
    assert json.loads(s3.alert_body)["projects"][0]["name"] == "Project label"
    assert json.loads(s3.alert_body)["projects"][0]["handle"] == "project-handle"
    assert managed.total_spend_usd == {str(project_id): 0}
    redshift_export_job.ManagedAuthStore.list_project_allowance_sources.assert_awaited_once()
    redshift_export_job.ManagedAuthStore.list_project_alert_identities.assert_awaited_once()
    redshift_export_job.list_key_spend_sources.assert_awaited_once()
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


def test_scheduled_export_batches_are_stage_scoped_and_retry_stable() -> None:
    redshift = FakeRedshiftClient()
    dev = _config()
    preview = replace(
        dev,
        stage="preview",
        export_prefix="gateway-usage/preview/raw",
        managed_budget_snapshot_key="gateway-usage/preview/managed-spend/current.json",
    )
    scan_time = datetime(2026, 5, 29, 12, 5, tzinfo=UTC)

    for config in (dev, preview, dev):
        run_export_job(
            config=config,
            dynamodb_client=FakeDynamoDbClient([_raw_event()]),
            s3_client=FakeS3Client(),
            redshift_data_client=redshift,
            now=scan_time,
        )

    prepare_sql = [
        cast(list[str], redshift.submissions[index]["Sqls"])[0]
        for index in (0, 3, 6)
    ]
    assert "batch_id in ('20260529T113000Z-20260529T120500Z-s00')" in prepare_sql[0]
    assert (
        "batch_id in ('preview-20260529T113000Z-20260529T120500Z-s00')"
        in prepare_sql[1]
    )
    assert prepare_sql[2] == prepare_sql[0]


def test_explicit_backfill_event_replays_current_schema_with_stage_scoped_batches() -> None:
    dynamodb = FakeDynamoDbClient([_raw_event()])
    s3 = FakeS3Client()
    redshift = FakeRedshiftClient()
    dev = _config()
    preview = replace(dev, stage="preview", export_prefix="gateway-usage/preview/raw")
    prod = replace(
        dev,
        stage="prod",
        redshift_schema_name="gateway_usage_prod",
        export_prefix="gateway-usage/prod/raw",
    )
    event = {
        "operation": "backfill",
        "start": "2026-05-29T12:00:00Z",
        "end": "2026-05-29T12:05:00Z",
    }

    for config in (dev, preview, dev, preview, prod):
        loaded = run_export_event(
            event,
            config=config,
            dynamodb_client=dynamodb,
            s3_client=s3,
            redshift_data_client=redshift,
        )
        assert loaded == {"loaded": True, "rows": 1}

    assert [put["Key"] for put in s3.puts[:2]] == [
        "gateway-usage/dev/raw/backfill/"
        "window=20260529T120000Z-20260529T120500Z/shard=00/part-000.parquet",
        "gateway-usage/dev/raw/backfill/"
        "window=20260529T120000Z-20260529T120500Z/manifest.json",
    ]
    assert [put["Key"] for put in s3.puts[2:4]] == [
        "gateway-usage/preview/raw/backfill/"
        "window=20260529T120000Z-20260529T120500Z/shard=00/part-000.parquet",
        "gateway-usage/preview/raw/backfill/"
        "window=20260529T120000Z-20260529T120500Z/manifest.json",
    ]
    prepare_sql = [
        cast(list[str], redshift.submissions[index]["Sqls"])[0]
        for index in (0, 3, 6, 9, 12)
    ]
    assert (
        "batch_id in ('backfill-20260529T120000Z-20260529T120500Z-s00')"
        in prepare_sql[0]
    )
    assert (
        "batch_id in ('preview-backfill-20260529T120000Z-20260529T120500Z-s00')"
        in prepare_sql[1]
    )
    assert prepare_sql[2] == prepare_sql[0]
    assert prepare_sql[3] == prepare_sql[1]
    assert (
        "batch_id in ('backfill-20260529T120000Z-20260529T120500Z-s00')"
        in prepare_sql[4]
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

    assert [put["Key"] for put in s3.puts] == [
        "gateway-usage/dev/managed-spend/current.json",
        "gateway-usage/dev/spend/project-alerts-current.json",
    ]
    assert json.loads(s3.alert_body)["projects"] == []
    assert s3.snapshot_body is not None
    assert (
        SpendSnapshot.model_validate_json(s3.snapshot_body).periods["monthly"].spend_usd
        == {}
    )
    assert redshift.submissions == []


def test_idle_publisher_updates_policy_and_rolls_empty_period() -> None:
    s3 = FakeS3Client()
    redshift = FakeRedshiftClient()
    project = ProjectReference(
        UUID("00000000-0000-0000-0000-000000000001"), "example-org", "dev"
    )
    created = datetime(2026, 5, 1, tzinfo=UTC)
    cutoff = datetime(2026, 5, 29, 12, 5, tzinfo=UTC)

    def publish(limit: str, as_of: datetime, *, idle: bool) -> None:
        alert_key = "gateway-usage/dev/spend/project-alerts-current.json"
        token = read_alert_publication_token(s3, "usage-landing", alert_key, "dev")
        result = publish_managed_spend_snapshot(
            s3_client=s3,
            redshift_data_client=redshift,
            project_starts={project.project_id: created},
            key_ids=[],
            stage="dev",
            bucket="usage-landing",
            key="gateway-usage/dev/managed-spend/current.json",
            workgroup="usage-workgroup",
            database="usage",
            secret_arn="local",
            schema="gateway_usage_dev",
            data_through=cutoff,
            as_of=as_of,
            idle=idle,
        )
        assert result is not None
        publish_project_alert_snapshot(
            s3_client=s3,
            bucket="usage-landing",
            key=alert_key,
            result=result,
            policies=(ProjectPolicy(project, "monthly", Decimal(limit), 1,
                                    created, created),),
            extras={},
            identities={project.project_id: ProjectAlertIdentity("Project label", None)},
            publication_token=token,
        )

    publish("10", cutoff, idle=False)
    first = json.loads(s3.alert_body)
    publish("20", datetime(2026, 6, 1, 0, 1, tzinfo=UTC), idle=True)
    second = json.loads(s3.alert_body)

    assert first["projects"][0]["start"] == "2026-05-01T00:00:00+00:00"
    assert second["projects"][0]["start"] == "2026-06-01T00:00:00+00:00"
    assert second["projects"][0]["limit_usd"] == "20"
    assert second["projects"][0]["spend_usd"] == "0"
    assert second["projects"][0]["contributors"] == []
    assert second["projects"][0]["models"] == []
    assert second["projects"][0]["benchmarks"] == []
    assert second["revision"] == first["revision"] + 1
    assert second["data_through"] == first["data_through"]
    assert s3.puts[-1]["IfMatch"] == '"alert"'
    assert len(s3.puts) == 4


@pytest.mark.parametrize("token_failure", [False, True])
def test_failed_fact_query_does_not_publish_zero_snapshots(
    monkeypatch: pytest.MonkeyPatch, token_failure: bool,
) -> None:
    warehouse_error = RuntimeError("warehouse query failed")

    class FailedRedshift(FakeRedshiftClient):
        def execute_statement(self, **kwargs: object) -> dict[str, str]:
            raise warehouse_error

    s3 = FakeS3Client()
    if token_failure:
        def fail_token(*_args: object) -> tuple[int, str | None]:
            raise RuntimeError("alert read failed")

        monkeypatch.setattr(redshift_export_job, "read_alert_publication_token", fail_token)
    with pytest.raises(RuntimeError, match="warehouse query failed") as captured:
        run_export_job(
            config=_config(),
            dynamodb_client=FakeDynamoDbClient([]),
            s3_client=s3,
            redshift_data_client=FailedRedshift(),
            now=datetime(2026, 5, 29, 12, 5, tzinfo=UTC),
        )
    assert captured.value is warehouse_error
    assert s3.puts == []


def test_stale_policy_candidate_cannot_replace_newer_project_alert() -> None:
    class ConditionalS3(FakeS3Client):
        def put_object(self, **kwargs: object) -> dict[str, object]:
            if kwargs["Key"] == "gateway-usage/dev/spend/project-alerts-current.json":
                if self.alert_body is not None or kwargs.get("IfNoneMatch") != "*":
                    raise ClientError(
                        {"Error": {"Code": "PreconditionFailed"}}, "PutObject"
                    )
            return super().put_object(**kwargs)

    s3 = ConditionalS3()
    key = "gateway-usage/dev/spend/project-alerts-current.json"
    older_token = read_alert_publication_token(s3, "usage-landing", key, "dev")
    newer_token = read_alert_publication_token(s3, "usage-landing", key, "dev")
    project = ProjectReference(
        UUID("00000000-0000-0000-0000-000000000001"), "example-org", "dev"
    )
    moment = datetime(2026, 5, 29, 12, 5, tzinfo=UTC)
    newer_moment = datetime(2026, 5, 30, 12, 5, tzinfo=UTC)
    bounds = {
        period: period_bounds(period, moment)
        for period in ("daily", "weekly", "monthly")
    }
    spend = {
        period: {str(project.project_id): Decimal("5")} for period in bounds
    }
    contributors = {
        period: {str(project.project_id): {"owner@example.com": Decimal("5")}}
        for period in bounds
    }
    models = {
        period: {str(project.project_id): {"included": Decimal("5")}}
        for period in bounds
    }
    benchmarks = {
        period: {str(project.project_id): {None: Decimal("5")}}
        for period in bounds
    }
    snapshot = SpendSnapshot(
        schema_version=1,
        stage="dev",
        revision=1,
        generated_at=moment,
        data_through=moment,
        periods={
            period: PeriodSpend(start=start, end=end, spend_usd=spend[period])
            for period, (start, end) in bounds.items()
        },
    )
    result = ManagedSpendPublishResult(snapshot, contributors, models, benchmarks)
    daily_start, daily_end = period_bounds("daily", newer_moment)
    newer_snapshot = snapshot.model_copy(update={
        "generated_at": newer_moment,
        "data_through": newer_moment,
        "periods": {
            **snapshot.periods,
            "daily": PeriodSpend(
                start=daily_start, end=daily_end, spend_usd=spend["daily"],
            ),
        },
    })
    newer_result = ManagedSpendPublishResult(
        newer_snapshot, contributors, models, benchmarks,
    )
    actor = HumanIdentity("https://issuer.example", "test-actor")
    expired = ProjectLimitRequest(
        UUID("00000000-0000-0000-0000-000000000002"), project, "temporary",
        Decimal("7"), bounds["daily"][0], daily_start, "old burst", actor,
        "approved", actor, moment, moment, None, None, None,
    )
    assert expired.period_end == newer_snapshot.periods["daily"].start
    loaded_extras = {project.project_id: (expired,)}

    def publish(
        limit: str, token: tuple[int, str | None], published: ManagedSpendPublishResult,
    ) -> None:
        publish_project_alert_snapshot(
            s3_client=s3,
            bucket="usage-landing",
            key=key,
            result=published,
            policies=(
                ProjectPolicy(project, "daily", Decimal(limit), 1, moment, moment),
            ),
            extras=loaded_extras,
            identities={project.project_id: ProjectAlertIdentity("Project label", None)},
            publication_token=token,
        )

    publish("20", newer_token, newer_result)
    publish("10", older_token, result)

    assert len(s3.puts) == 1
    assert s3.puts[0]["IfNoneMatch"] == "*"
    assert json.loads(s3.alert_body)["projects"][0]["start"] == daily_start.isoformat()
    assert json.loads(s3.alert_body)["projects"][0]["limit_usd"] == "20"
    assert json.loads(s3.alert_body)["projects"][0]["models"] == [
        {"name": "included", "spend_usd": "5"}
    ]
    assert json.loads(s3.alert_body)["projects"][0]["benchmarks"] == [
        {"name": None, "spend_usd": "5"}
    ]
    assert json.loads(s3.alert_body)["revision"] == 1


def test_project_total_counts_spend_since_the_budget_began() -> None:
    project_id = UUID("00000000-0000-0000-0000-000000000001")
    started = datetime(2026, 10, 1, tzinfo=UTC)

    class TotalRedshift(FakeRedshiftClient):
        sql = ""
        parameters: list[dict[str, str]] = []

        def execute_statement(self, **kwargs: object) -> dict[str, str]:
            TotalRedshift.sql = cast(str, kwargs["Sql"])
            TotalRedshift.parameters = cast(list[dict[str, str]], kwargs["Parameters"])
            return {"Id": "budget-spend"}

        def get_statement_result(self, **kwargs: object) -> dict[str, object]:
            return {"Records": [
                [{"stringValue": "total"}, {"stringValue": "project"},
                 {"stringValue": str(project_id)}, {"isNull": True},
                 {"stringValue": "1234.5"}],
                [{"stringValue": "monthly"}, {"stringValue": "project"},
                 {"stringValue": str(project_id)}, {"stringValue": "a@example.com"},
                 {"stringValue": "34.5"}],
            ]}

    s3 = FakeS3Client()
    result = publish_managed_spend_snapshot(
        s3_client=s3,
        redshift_data_client=TotalRedshift(),
        project_starts={project_id: started},
        key_ids=[],
        stage="dev",
        bucket="usage-landing",
        key="gateway-usage/dev/managed-spend/current.json",
        workgroup="usage-workgroup",
        database="usage",
        secret_arn="local",
        schema="gateway_usage_dev",
        data_through=datetime(2026, 11, 4, 12, tzinfo=UTC),
        as_of=datetime(2026, 11, 4, 12, tzinfo=UTC),
    )

    assert result is not None
    assert result.snapshot.total_spend_usd == {str(project_id): Decimal("1234.5")}
    assert result.snapshot.periods["monthly"].spend_usd[str(project_id)] == Decimal("34.5")
    assert "usage_events.completed_at >= starts.budget_start" in TotalRedshift.sql
    assert "2026-10-01" not in TotalRedshift.sql
    assert [item["value"] for item in TotalRedshift.parameters] == [
        "total", "project", str(project_id), "2026-10-01T00:00:00+00:00",
    ]
    # Redshift rejects untyped Data API parameters in a select list ("Invalid type: 705").
    for item in TotalRedshift.parameters:
        assert f"CAST(:{item['name']} AS" in TotalRedshift.sql
