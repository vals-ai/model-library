from datetime import UTC, datetime
from decimal import Decimal
import json
from typing import Any, cast

from boto3.dynamodb.types import TypeDeserializer

from model_gateway.usage_ledger import schema
from model_gateway.usage_ledger.dynamodb_writer import serialize_item
from model_gateway.usage_ledger.redshift_export import plan_logical_export_window
from model_gateway.usage_ledger.redshift_export_job import (
    RedshiftExportJobConfig,
    extract_export_window,
    write_export_window,
)

_COMPRESSED_PERFORMANCE = {
    "encoding": "gzip+base64",
    "data": "opaque",
}


class FakeDynamoDbClient:
    def __init__(self, pages: list[dict[str, object]]) -> None:
        self.pages = pages
        self.calls: list[dict[str, object]] = []

    def query(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        return self.pages.pop(0)


class FakeS3Client:
    def __init__(self) -> None:
        self.puts: list[dict[str, object]] = []

    def put_object(self, **kwargs: object) -> dict[str, object]:
        self.puts.append(kwargs)
        return {}


def _expression_values(call: dict[str, object]) -> dict[str, object]:
    deserializer = TypeDeserializer()
    values = cast(dict[str, Any], call["ExpressionAttributeValues"])
    return {key: deserializer.deserialize(value) for key, value in values.items()}


def _config(*, shards: int = 1) -> RedshiftExportJobConfig:
    return RedshiftExportJobConfig(
        raw_table_name="usage-ledger",
        shards=shards,
        redshift_workgroup_name="usage-workgroup",
        redshift_database_name="usage",
        landing_bucket_name="landing-bucket",
        copy_role_arn="arn:aws:iam::123456789012:role/copy",
        redshift_schema_name="gateway_usage",
        export_prefix="gateway-usage/dev/raw",
    )


def _raw_event(
    *,
    usage_event_id: str = "usage-1",
    completed_at: str = "2026-05-29T12:01:00Z",
    shard: str = "00",
) -> dict[str, object]:
    return {
        "PK": f"USAGE#DAY#20260529#S#{shard}",
        "SK": f"TS#{completed_at}#USG#{usage_event_id}",
        "entity_type": "usage_event",
        "usage_event_id": usage_event_id,
        "model": "openai/gpt-4.1-mini",
        "provider": "openai",
        "provider_endpoint": "default",
        "finish_reason": "stop",
        "finish_reason_raw": "stop",
        "details": {
            "request": {},
            "result": {"metadata": {"performance": _COMPRESSED_PERFORMANCE}},
        },
        "completed_at": completed_at,
        "usage_shard": shard,
        "schema_version": 2,
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


def test_config_from_env_uses_only_success_path_inputs(monkeypatch: Any) -> None:
    values = {
        "GATEWAY_USAGE_LEDGER_TABLE_NAME": "usage-prod",
        "GATEWAY_USAGE_LEDGER_SHARDS": "16",
        "GATEWAY_USAGE_REDSHIFT_WORKGROUP_NAME": "usage-workgroup",
        "GATEWAY_USAGE_REDSHIFT_DATABASE_NAME": "usage",
        "GATEWAY_USAGE_REDSHIFT_LANDING_BUCKET_NAME": "landing",
        "GATEWAY_USAGE_REDSHIFT_COPY_ROLE_ARN": "arn:copy",
        "GATEWAY_USAGE_REDSHIFT_SCHEMA_NAME": "gateway_usage_prod",
        "GATEWAY_USAGE_REDSHIFT_EXPORT_PREFIX": "gateway-usage/prod/raw",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    assert RedshiftExportJobConfig.from_env() == RedshiftExportJobConfig(
        raw_table_name="usage-prod",
        shards=16,
        redshift_workgroup_name="usage-workgroup",
        redshift_database_name="usage",
        landing_bucket_name="landing",
        copy_role_arn="arn:copy",
        redshift_schema_name="gateway_usage_prod",
        export_prefix="gateway-usage/prod/raw",
    )


def test_extract_export_window_paginates_with_exclusive_upper_bound() -> None:
    first = _raw_event()
    second = _raw_event(
        usage_event_id="usage-2",
        completed_at="2026-05-29T12:02:00Z",
    )
    dynamodb = FakeDynamoDbClient(
        [
            {
                "Items": [serialize_item(first)],
                "LastEvaluatedKey": {"PK": {"S": "next"}},
            },
            {"Items": [serialize_item(second)]},
        ]
    )
    window = plan_logical_export_window(now=datetime(2026, 5, 29, 12, 5, tzinfo=UTC))

    extracted = extract_export_window(
        window,
        config=_config(),
        dynamodb_client=dynamodb,
        loaded_at=datetime(2026, 5, 29, 12, 5, tzinfo=UTC),
    )

    assert [row["usage_event_id"] for row in extracted.shards[0].rows] == [
        "usage-1",
        "usage-2",
    ]
    assert len(dynamodb.calls) == 2
    assert {call["TableName"] for call in dynamodb.calls} == {"usage-ledger"}
    assert {call["KeyConditionExpression"] for call in dynamodb.calls} == {
        "PK = :pk AND SK BETWEEN :start AND :end"
    }
    assert {call["FilterExpression"] for call in dynamodb.calls} == {
        "schema_version = :schema_version"
    }
    assert {call["ConsistentRead"] for call in dynamodb.calls} == {True}
    assert [_expression_values(call) for call in dynamodb.calls] == [
        {
            ":pk": schema.usage_day_pk("20260529", 0),
            ":start": "TS#2026-05-29T11:30:00",
            ":end": "TS#2026-05-29T12:05:00",
            ":schema_version": schema.USAGE_EVENT_SCHEMA_VERSION,
        },
        {
            ":pk": schema.usage_day_pk("20260529", 0),
            ":start": "TS#2026-05-29T11:30:00",
            ":end": "TS#2026-05-29T12:05:00",
            ":schema_version": schema.USAGE_EVENT_SCHEMA_VERSION,
        },
    ]
    assert "ExclusiveStartKey" not in dynamodb.calls[0]
    assert dynamodb.calls[1]["ExclusiveStartKey"] == {"PK": {"S": "next"}}
    end_key = _expression_values(dynamodb.calls[0])[":end"]
    boundary_event_key = _raw_event(completed_at="2026-05-29T12:05:00Z")["SK"]
    fractional_boundary_event_key = _raw_event(
        completed_at="2026-05-29T12:05:00.123456Z"
    )["SK"]
    assert isinstance(end_key, str)
    assert isinstance(boundary_event_key, str)
    assert isinstance(fractional_boundary_event_key, str)
    assert boundary_event_key > end_key
    assert fractional_boundary_event_key > end_key


def test_extract_export_window_continues_after_empty_filtered_page() -> None:
    dynamodb = FakeDynamoDbClient(
        [
            {
                "Items": [],
                "LastEvaluatedKey": {"PK": {"S": "next"}},
            },
            {"Items": [serialize_item(_raw_event())]},
        ]
    )
    window = plan_logical_export_window(now=datetime(2026, 5, 29, 12, 5, tzinfo=UTC))

    extracted = extract_export_window(
        window,
        config=_config(),
        dynamodb_client=dynamodb,
        loaded_at=datetime(2026, 5, 29, 12, 5, tzinfo=UTC),
    )

    assert [row["usage_event_id"] for row in extracted.shards[0].rows] == ["usage-1"]
    assert len(dynamodb.calls) == 2
    assert dynamodb.calls[1]["ExclusiveStartKey"] == {"PK": {"S": "next"}}


def test_write_export_window_emits_one_part_per_nonempty_shard_and_manifest() -> None:
    pages: list[dict[str, object]] = [
        {
            "Items": [
                serialize_item(
                    _raw_event(usage_event_id=f"usage-{shard}", shard=f"{shard:02d}")
                )
            ]
        }
        for shard in range(16)
    ]
    window = plan_logical_export_window(now=datetime(2026, 5, 29, 12, 5, tzinfo=UTC))
    extracted = extract_export_window(
        window,
        config=_config(shards=16),
        dynamodb_client=FakeDynamoDbClient(pages),
        loaded_at=datetime(2026, 5, 29, 12, 5, tzinfo=UTC),
    )
    s3 = FakeS3Client()

    manifest_uri = write_export_window(
        extracted, config=_config(shards=16), s3_client=s3
    )

    assert manifest_uri.endswith("/manifest.json")
    assert len(s3.puts) == 17
    assert str(s3.puts[0]["Key"]).endswith("shard=00/part-000.parquet")
    assert str(s3.puts[-2]["Key"]).endswith("shard=15/part-000.parquet")
    assert str(s3.puts[-1]["Key"]).endswith("manifest.json")
    assert json.loads(cast(bytes, s3.puts[-1]["Body"])) == {
        "entries": [
            {
                "url": f"s3://landing-bucket/{part['Key']}",
                "mandatory": True,
                "meta": {"content_length": len(cast(bytes, part["Body"]))},
            }
            for part in s3.puts[:-1]
        ]
    }


def test_window_extraction_preserves_source_order_and_transform_output() -> None:
    later = _raw_event(usage_event_id="later", completed_at="2026-05-29T12:02:00Z")
    earlier = _raw_event(usage_event_id="earlier", completed_at="2026-05-29T12:01:00Z")
    dynamodb = FakeDynamoDbClient(
        [{"Items": [serialize_item(later), serialize_item(earlier)]}]
    )
    window = plan_logical_export_window(now=datetime(2026, 5, 29, 12, 5, tzinfo=UTC))

    extracted = extract_export_window(
        window,
        config=_config(),
        dynamodb_client=dynamodb,
        loaded_at=datetime(2026, 5, 29, 12, 5, tzinfo=UTC),
    )

    assert [row["usage_event_id"] for row in extracted.shards[0].rows] == [
        "later",
        "earlier",
    ]
    assert extracted.shards[0].rows[0]["batch_id"] == (
        "20260529T113000Z-20260529T120500Z-s00"
    )
    assert extracted.shards[0].rows[0]["requested_model_key"] == ("openai/gpt-4.1-mini")


def test_cross_midnight_window_queries_each_day_for_each_shard() -> None:
    dynamodb = FakeDynamoDbClient([{"Items": []} for _ in range(4)])
    window = plan_logical_export_window(now=datetime(2026, 5, 30, 0, 5, tzinfo=UTC))

    extract_export_window(
        window,
        config=_config(shards=2),
        dynamodb_client=dynamodb,
        loaded_at=datetime(2026, 5, 30, 0, 5, tzinfo=UTC),
    )

    assert {call["TableName"] for call in dynamodb.calls} == {"usage-ledger"}
    assert {call["KeyConditionExpression"] for call in dynamodb.calls} == {
        "PK = :pk AND SK BETWEEN :start AND :end"
    }
    assert {call["FilterExpression"] for call in dynamodb.calls} == {
        "schema_version = :schema_version"
    }
    assert {call["ConsistentRead"] for call in dynamodb.calls} == {True}
    assert [_expression_values(call) for call in dynamodb.calls] == [
        {
            ":pk": schema.usage_day_pk("20260529", 0),
            ":start": "TS#2026-05-29T23:30:00",
            ":end": "TS#2026-05-30T00:00:00",
            ":schema_version": schema.USAGE_EVENT_SCHEMA_VERSION,
        },
        {
            ":pk": schema.usage_day_pk("20260530", 0),
            ":start": "TS#2026-05-30T00:00:00",
            ":end": "TS#2026-05-30T00:05:00",
            ":schema_version": schema.USAGE_EVENT_SCHEMA_VERSION,
        },
        {
            ":pk": schema.usage_day_pk("20260529", 1),
            ":start": "TS#2026-05-29T23:30:00",
            ":end": "TS#2026-05-30T00:00:00",
            ":schema_version": schema.USAGE_EVENT_SCHEMA_VERSION,
        },
        {
            ":pk": schema.usage_day_pk("20260530", 1),
            ":start": "TS#2026-05-30T00:00:00",
            ":end": "TS#2026-05-30T00:05:00",
            ":schema_version": schema.USAGE_EVENT_SCHEMA_VERSION,
        },
    ]
