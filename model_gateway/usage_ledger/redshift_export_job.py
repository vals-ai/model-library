from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
import os
import time
from typing import Any, Literal, Protocol, Self, cast

import boto3
from boto3.dynamodb.types import TypeDeserializer
from pydantic import AwareDatetime, BaseModel, ConfigDict, model_validator

from model_gateway.usage_ledger import schema
from model_gateway.usage_ledger.dynamodb_writer import serialize_item
from model_gateway.usage_ledger.redshift_export import (
    LogicalExportWindow,
    build_manifest_json,
    export_batch_id,
    export_data_key,
    export_manifest_key,
    logical_export_window,
    parquet_bytes_from_rows,
    plan_logical_export_window,
)
from model_gateway.usage_ledger.redshift_load import (
    RedshiftWindowLoad,
    analytics_refresh_statements,
    copy_window_statements,
    finalize_window_statements,
    prepare_window_statements,
    staging_row_from_redshift_rows,
)
from model_gateway.usage_ledger.redshift_transform import redshift_rows_from_usage_event


class DynamoDbQueryClient(Protocol):
    def query(self, **kwargs: object) -> Mapping[str, object]: ...


class S3PutObjectClient(Protocol):
    def put_object(self, **kwargs: object) -> Mapping[str, object]: ...


class RedshiftDataClient(Protocol):
    def batch_execute_statement(self, **kwargs: object) -> Mapping[str, object]: ...

    def describe_statement(self, **kwargs: object) -> Mapping[str, object]: ...


class RedshiftStatementError(RuntimeError):
    pass


class _WindowRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: AwareDatetime
    end: AwareDatetime

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        if self.start.microsecond or self.end.microsecond:
            raise ValueError("window boundaries must use whole-second timestamps")
        if self.start >= self.end:
            raise ValueError("window end must be after start")
        return self


class BackfillRequest(_WindowRequest):
    operation: Literal["backfill"]


class AnalyticsRefreshRequest(_WindowRequest):
    operation: Literal["refresh_analytics"]


@dataclass(frozen=True)
class RedshiftExportJobConfig:
    raw_table_name: str
    shards: int
    redshift_workgroup_name: str
    redshift_database_name: str
    landing_bucket_name: str
    copy_role_arn: str
    redshift_schema_name: str
    export_prefix: str

    @classmethod
    def from_env(cls) -> RedshiftExportJobConfig:
        return cls(
            raw_table_name=os.environ["GATEWAY_USAGE_LEDGER_TABLE_NAME"],
            shards=int(os.environ["GATEWAY_USAGE_LEDGER_SHARDS"]),
            redshift_workgroup_name=os.environ["GATEWAY_USAGE_REDSHIFT_WORKGROUP_NAME"],
            redshift_database_name=os.environ["GATEWAY_USAGE_REDSHIFT_DATABASE_NAME"],
            landing_bucket_name=os.environ[
                "GATEWAY_USAGE_REDSHIFT_LANDING_BUCKET_NAME"
            ],
            copy_role_arn=os.environ["GATEWAY_USAGE_REDSHIFT_COPY_ROLE_ARN"],
            redshift_schema_name=os.environ["GATEWAY_USAGE_REDSHIFT_SCHEMA_NAME"],
            export_prefix=os.environ["GATEWAY_USAGE_REDSHIFT_EXPORT_PREFIX"],
        )


@dataclass(frozen=True)
class ExportShardExtraction:
    shard: int
    batch_id: str
    rows: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class ExportWindowExtraction:
    window: LogicalExportWindow
    shards: tuple[ExportShardExtraction, ...]


def extract_export_window(
    window: LogicalExportWindow,
    *,
    config: RedshiftExportJobConfig,
    dynamodb_client: DynamoDbQueryClient,
    loaded_at: datetime,
    batch_namespace: str | None = None,
) -> ExportWindowExtraction:
    shards: list[ExportShardExtraction] = []
    for shard in range(config.shards):
        events = _query_full_raw_events_for_export(
            client=dynamodb_client,
            table_name=config.raw_table_name,
            start=window.start,
            end=window.end,
            shard=shard,
        )
        batch_id = export_batch_id(
            window,
            shard=shard,
            namespace=batch_namespace,
        )
        rows = tuple(
            staging_row_from_redshift_rows(
                redshift_rows_from_usage_event(
                    event,
                    batch_id=batch_id,
                    loaded_at=loaded_at,
                )
            )
            for event in events
        )
        shards.append(
            ExportShardExtraction(
                shard=shard,
                batch_id=batch_id,
                rows=rows,
            )
        )
    return ExportWindowExtraction(window=window, shards=tuple(shards))


def write_export_window(
    extracted: ExportWindowExtraction,
    *,
    config: RedshiftExportJobConfig,
    s3_client: S3PutObjectClient,
) -> str:
    entries: list[tuple[str, int]] = []
    for shard in extracted.shards:
        if not shard.rows:
            continue
        payload = parquet_bytes_from_rows(shard.rows)
        key = export_data_key(
            extracted.window,
            shard=shard.shard,
            prefix=config.export_prefix,
        )
        uri = f"s3://{config.landing_bucket_name}/{key}"
        s3_client.put_object(
            Bucket=config.landing_bucket_name,
            Key=key,
            Body=payload,
            ContentType="application/vnd.apache.parquet",
        )
        entries.append((uri, len(payload)))
    manifest_key = export_manifest_key(
        extracted.window,
        prefix=config.export_prefix,
    )
    manifest_uri = f"s3://{config.landing_bucket_name}/{manifest_key}"
    s3_client.put_object(
        Bucket=config.landing_bucket_name,
        Key=manifest_key,
        Body=build_manifest_json(entries).encode("utf-8"),
        ContentType="application/json",
    )
    return manifest_uri


def run_export_window(
    *,
    window: LogicalExportWindow,
    config: RedshiftExportJobConfig,
    dynamodb_client: DynamoDbQueryClient,
    s3_client: S3PutObjectClient,
    redshift_data_client: RedshiftDataClient,
    loaded_at: datetime | None = None,
    batch_namespace: str | None = None,
    refresh_analytics: bool = True,
) -> int:
    current_time = (loaded_at or datetime.now(UTC)).astimezone(UTC)
    extracted = extract_export_window(
        window,
        config=config,
        dynamodb_client=dynamodb_client,
        loaded_at=current_time,
        batch_namespace=batch_namespace,
    )
    row_count = sum(len(shard.rows) for shard in extracted.shards)
    if row_count == 0:
        return 0

    manifest_s3_uri = write_export_window(
        extracted,
        config=config,
        s3_client=s3_client,
    )
    load = RedshiftWindowLoad(
        aggregate_start=window.start,
        aggregate_end=window.end,
        manifest_s3_uri=manifest_s3_uri,
        batch_ids=tuple(shard.batch_id for shard in extracted.shards if shard.rows),
    )
    submissions = (
        (
            "usage-redshift-window-prepare",
            prepare_window_statements(load, schema=config.redshift_schema_name),
        ),
        (
            "usage-redshift-window-copy",
            copy_window_statements(
                load,
                iam_role_arn=config.copy_role_arn,
                schema=config.redshift_schema_name,
            ),
        ),
        (
            "usage-redshift-window-finalize",
            finalize_window_statements(
                load,
                schema=config.redshift_schema_name,
                refresh_analytics=refresh_analytics,
            ),
        ),
    )
    for statement_name, statements in submissions:
        _execute_submission(
            statements,
            statement_name=statement_name,
            config=config,
            redshift_data_client=redshift_data_client,
        )
    return row_count


def run_export_job(
    *,
    config: RedshiftExportJobConfig,
    dynamodb_client: DynamoDbQueryClient,
    s3_client: S3PutObjectClient,
    redshift_data_client: RedshiftDataClient,
    now: datetime | None = None,
) -> None:
    current_time = (now or datetime.now(UTC)).astimezone(UTC)
    run_export_window(
        window=plan_logical_export_window(now=current_time),
        config=config,
        dynamodb_client=dynamodb_client,
        s3_client=s3_client,
        redshift_data_client=redshift_data_client,
        loaded_at=current_time,
    )


def run_export_event(
    event: Mapping[str, object],
    *,
    config: RedshiftExportJobConfig,
    dynamodb_client: DynamoDbQueryClient,
    s3_client: S3PutObjectClient,
    redshift_data_client: RedshiftDataClient,
    now: datetime | None = None,
) -> dict[str, bool | int] | None:
    if event.get("operation") == "refresh_analytics":
        request = AnalyticsRefreshRequest.model_validate(event)
        _execute_submission(
            analytics_refresh_statements(
                start=request.start,
                end=request.end,
                schema=config.redshift_schema_name,
            ),
            statement_name="usage-redshift-backfill-refresh",
            config=config,
            redshift_data_client=redshift_data_client,
        )
        return {"refreshed": True}

    if "operation" not in event:
        run_export_job(
            config=config,
            dynamodb_client=dynamodb_client,
            s3_client=s3_client,
            redshift_data_client=redshift_data_client,
            now=now,
        )
        return None

    request = BackfillRequest.model_validate(event)
    rows = run_export_window(
        window=logical_export_window(start=request.start, end=request.end),
        config=replace(config, export_prefix=f"{config.export_prefix}/backfill"),
        dynamodb_client=dynamodb_client,
        s3_client=s3_client,
        redshift_data_client=redshift_data_client,
        loaded_at=now,
        batch_namespace="backfill",
        refresh_analytics=False,
    )
    return {"loaded": rows > 0, "rows": rows}


def _execute_submission(
    statements: list[str],
    *,
    statement_name: str,
    config: RedshiftExportJobConfig,
    redshift_data_client: RedshiftDataClient,
) -> None:
    response = redshift_data_client.batch_execute_statement(
        WorkgroupName=config.redshift_workgroup_name,
        Database=config.redshift_database_name,
        Sqls=statements,
        StatementName=statement_name,
    )
    _wait_for_statement(redshift_data_client, str(response["Id"]))


def _isoformat_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _query_full_raw_events_for_export(
    *,
    client: DynamoDbQueryClient,
    table_name: str,
    start: datetime,
    end: datetime,
    shard: int,
) -> tuple[dict[str, object], ...]:
    events: list[dict[str, object]] = []
    deserializer = cast(Any, TypeDeserializer())
    for day, day_start, day_end in _day_windows(start, end):
        sk_start = f"TS#{_raw_usage_event_sk_start(day_start)}"
        sk_end = f"TS#{_raw_usage_event_sk_end(day_end)}"
        exclusive_start_key: object | None = None
        while True:
            kwargs: dict[str, object] = {
                "TableName": table_name,
                "KeyConditionExpression": "PK = :pk AND SK BETWEEN :start AND :end",
                "FilterExpression": "schema_version = :schema_version",
                "ExpressionAttributeValues": serialize_item(
                    {
                        ":pk": schema.usage_day_pk(day, shard),
                        ":start": sk_start,
                        ":end": sk_end,
                        ":schema_version": schema.USAGE_EVENT_SCHEMA_VERSION,
                    }
                ),
                "ConsistentRead": True,
            }
            if exclusive_start_key is not None:
                kwargs["ExclusiveStartKey"] = exclusive_start_key
            response = client.query(**kwargs)
            events.extend(_deserialize_items(response.get("Items", []), deserializer))
            exclusive_start_key = response.get("LastEvaluatedKey")
            if exclusive_start_key is None:
                break
    return tuple(events)


def _deserialize_items(
    items_value: object, deserializer: Any
) -> list[dict[str, object]]:
    raw_items = cast(list[dict[str, dict[str, Any]]], items_value)
    return [
        {key: deserializer.deserialize(value) for key, value in raw_item.items()}
        for raw_item in raw_items
    ]


def _raw_usage_event_sk_start(start: datetime) -> str:
    return _isoformat_utc(start).removesuffix("Z")


def _raw_usage_event_sk_end(end: datetime) -> str:
    # Event keys append a `Z` or fractional seconds before `#USG#...`, so the
    # bare whole-second prefix makes inclusive BETWEEN end-exclusive.
    return _isoformat_utc(end.replace(microsecond=0)).removesuffix("Z")


def _day_windows(
    start: datetime, end: datetime
) -> list[tuple[str, datetime, datetime]]:
    windows: list[tuple[str, datetime, datetime]] = []
    current = start.replace(hour=0, minute=0, second=0, microsecond=0)
    while current < end:
        next_day = current + timedelta(days=1)
        window_start = max(start, current)
        window_end = min(end, next_day)
        windows.append((window_start.strftime("%Y%m%d"), window_start, window_end))
        current = next_day
    return windows


def handler(
    event: Mapping[str, object],
    _context: object,
) -> dict[str, bool | int] | None:
    boto3_client = cast(Any, boto3).client
    return run_export_event(
        event,
        config=RedshiftExportJobConfig.from_env(),
        dynamodb_client=cast(DynamoDbQueryClient, boto3_client("dynamodb")),
        s3_client=cast(S3PutObjectClient, boto3_client("s3")),
        redshift_data_client=cast(
            RedshiftDataClient,
            boto3_client("redshift-data"),
        ),
    )


def _wait_for_statement(
    client: RedshiftDataClient,
    statement_id: str,
) -> None:
    while True:
        response = client.describe_statement(Id=statement_id)
        status = str(response["Status"])
        if status == "FINISHED":
            return
        if status in {"FAILED", "ABORTED"}:
            raise RedshiftStatementError(
                str(response.get("Error", "Redshift statement failed"))
            )
        time.sleep(1)
