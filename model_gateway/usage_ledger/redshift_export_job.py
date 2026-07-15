from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import os
import time
from typing import Any, Protocol, cast

import boto3
from boto3.dynamodb.types import TypeDeserializer

from model_gateway.usage_ledger import schema
from model_gateway.usage_ledger.dynamodb_writer import serialize_item
from model_gateway.usage_ledger.redshift_export import (
    LogicalExportWindow,
    build_manifest_json,
    export_batch_id,
    export_data_key,
    export_manifest_key,
    parquet_bytes_from_rows,
    plan_logical_export_window,
)
from model_gateway.usage_ledger.redshift_load import (
    RedshiftWindowLoad,
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
        batch_id = export_batch_id(window, shard=shard)
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


def run_export_job(
    *,
    config: RedshiftExportJobConfig,
    dynamodb_client: DynamoDbQueryClient,
    s3_client: S3PutObjectClient,
    redshift_data_client: RedshiftDataClient,
    now: datetime | None = None,
) -> None:
    current_time = (now or datetime.now(UTC)).astimezone(UTC)
    window = plan_logical_export_window(now=current_time)
    extracted = extract_export_window(
        window,
        config=config,
        dynamodb_client=dynamodb_client,
        loaded_at=current_time,
    )
    if not any(shard.rows for shard in extracted.shards):
        return

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
            finalize_window_statements(load, schema=config.redshift_schema_name),
        ),
    )
    for statement_name, statements in submissions:
        _execute_submission(
            statements,
            statement_name=statement_name,
            config=config,
            redshift_data_client=redshift_data_client,
        )


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


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return value.astimezone(UTC)


def _parse_utc_iso(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    return _require_utc(datetime.fromisoformat(normalized))


def _datetime_from_object(value: object) -> datetime:
    if isinstance(value, datetime):
        return _require_utc(value)
    if isinstance(value, str):
        return _parse_utc_iso(value)
    raise TypeError("expected UTC datetime or ISO string")


def _isoformat_utc(value: datetime) -> str:
    return _require_utc(value).isoformat().replace("+00:00", "Z")


def _query_full_raw_events_for_export(
    *,
    client: DynamoDbQueryClient,
    table_name: str,
    start: datetime,
    end: datetime,
    shard: int,
) -> tuple[dict[str, object], ...]:
    start_dt = _require_utc(start)
    end_dt = _require_utc(end)
    if start_dt >= end_dt:
        raise ValueError("start must be earlier than end")

    events: list[dict[str, object]] = []
    deserializer = cast(Any, TypeDeserializer())
    for day, day_start, day_end in _day_windows(start_dt, end_dt):
        sk_start = f"TS#{_raw_usage_event_sk_start(day_start)}"
        sk_end = f"TS#{_raw_usage_event_sk_end(day_end)}\uffff"
        exclusive_start_key: object | None = None
        while True:
            kwargs: dict[str, object] = {
                "TableName": table_name,
                "KeyConditionExpression": "PK = :pk AND SK BETWEEN :start AND :end",
                "ExpressionAttributeValues": serialize_item(
                    {
                        ":pk": schema.usage_day_pk(day, shard),
                        ":start": sk_start,
                        ":end": sk_end,
                    }
                ),
                "ConsistentRead": True,
            }
            if exclusive_start_key is not None:
                kwargs["ExclusiveStartKey"] = exclusive_start_key
            response = client.query(**kwargs)
            for item in _deserialize_items(response.get("Items", []), deserializer):
                if item.get("entity_type") != "usage_event":
                    continue
                completed_at = _datetime_from_object(item.get("completed_at"))
                if start_dt <= completed_at < end_dt:
                    events.append(item)
            exclusive_start_key = response.get("LastEvaluatedKey")
            if exclusive_start_key is None:
                break
    return tuple(sorted(events, key=lambda event: str(event.get("completed_at", ""))))


def _deserialize_items(
    items_value: object, deserializer: Any
) -> list[dict[str, object]]:
    if not isinstance(items_value, list):
        return []
    items: list[dict[str, object]] = []
    for raw_item in cast(list[object], items_value):
        if not isinstance(raw_item, dict):
            continue
        item = {
            str(key): deserializer.deserialize(value)
            for key, value in cast(dict[object, object], raw_item).items()
            if isinstance(value, dict)
        }
        items.append(item)
    return items


def _raw_usage_event_sk_start(start: datetime) -> str:
    return _isoformat_utc(start).removesuffix("Z")


def _raw_usage_event_sk_end(end: datetime) -> str:
    return _isoformat_utc(end.replace(microsecond=0))


def _day_windows(
    start: datetime, end: datetime
) -> list[tuple[str, datetime, datetime]]:
    windows: list[tuple[str, datetime, datetime]] = []
    current = start.replace(hour=0, minute=0, second=0, microsecond=0)
    while current < end:
        next_day = current + timedelta(days=1)
        window_start = max(start, current)
        window_end = min(end, next_day)
        if window_start < window_end:
            windows.append((window_start.strftime("%Y%m%d"), window_start, window_end))
        current = next_day
    return windows


def handler(_event: Mapping[str, object], _context: object) -> None:
    boto3_client = cast(Any, boto3).client
    run_export_job(
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
