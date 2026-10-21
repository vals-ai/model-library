from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from decimal import Decimal
import json
from typing import Any, cast

from boto3.dynamodb.types import TypeDeserializer
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from model_gateway.usage_ledger import schema
from model_gateway.usage_ledger.dynamodb_writer import (
    prepare_usage_event_for_write,
    serialize_item,
)
from model_gateway.usage_ledger.redshift_export import (
    parquet_bytes_from_rows,
    plan_logical_export_window,
)
from model_gateway.usage_ledger.redshift_export_job import (
    RedshiftExportJobConfig,
    extract_export_window,
    write_export_window,
)
from model_gateway.usage_ledger.redshift_load import staging_row_from_redshift_rows
from model_gateway.usage_ledger.redshift_transform import redshift_rows_from_usage_event

_TIMELINE = {"encoding": "gzip+base64", "data": "opaque"}
_COMPRESSED_PERFORMANCE = {
    "time_to_first_token_ms": {
        "any": 7,
        "answer": 11,
        "reasoning": 7,
        "content": 11,
        "tool_call": None,
    },
    "timeline": _TIMELINE,
}


class FakeDynamoDbClient:
    def __init__(self, pages: list[dict[str, object]]) -> None:
        self.pages = pages
        self.calls: list[dict[str, object]] = []
        self.responses: list[dict[str, object]] = []

    def query(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        page = self.pages.pop(0)
        paths = _projection_paths(kwargs)
        response = {
            **page,
            "Items": [
                _project_item(cast(dict[str, Any], item), paths)
                for item in cast(list[object], page.get("Items", []))
            ],
        }
        self.responses.append(response)
        return response


def _projection_paths(call: Mapping[str, object]) -> tuple[tuple[str, ...], ...]:
    expression = cast(str, call["ProjectionExpression"])
    aliases = cast(dict[str, str], call["ExpressionAttributeNames"])
    return tuple(
        tuple(aliases[segment.strip()] for segment in path.split("."))
        for path in expression.split(",")
    )


def _project_item(
    item: dict[str, Any], paths: tuple[tuple[str, ...], ...]
) -> dict[str, Any]:
    projected: dict[str, Any] = {}
    for path in paths:
        source = item
        destination = projected
        for key in path[:-1]:
            attribute = source.get(key)
            if attribute is None:
                break
            source = attribute["M"]
            destination = destination.setdefault(key, {"M": {}})["M"]
        else:
            if path[-1] in source:
                destination[path[-1]] = source[path[-1]]
    return projected


class _ProjectedAccess(Mapping[str, Any]):
    def __init__(
        self,
        source: Mapping[str, Any],
        paths: tuple[tuple[str, ...], ...],
        prefix: tuple[str, ...] = (),
    ) -> None:
        self.source = source
        self.paths = paths
        self.prefix = prefix

    def __getitem__(self, key: str) -> Any:
        path = (*self.prefix, key)
        assert any(
            selected[: len(path)] == path or path[: len(selected)] == selected
            for selected in self.paths
        ), f"Unprojected transformer access: {'.'.join(path)}"
        value = self.source[key]
        if isinstance(value, Mapping):
            return _ProjectedAccess(value, self.paths, path)
        return value

    def __iter__(self) -> Iterator[str]:
        return iter(self.source)

    def __len__(self) -> int:
        return len(self.source)


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
        redshift_secret_arn="arn:aws:secretsmanager:us-east-1:123456789012:secret:usage-loader",
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
        "api_key_name": "default",
        "finish_reason": "stop",
        "finish_reason_raw": "stop",
        "details": {
            "request": {},
            "result": {"metadata": {"performance": _COMPRESSED_PERFORMANCE}},
        },
        "completed_at": completed_at,
        "usage_shard": shard,
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


def test_extract_export_window_paginates_with_exclusive_upper_bound() -> None:
    first = _raw_event()
    second = _raw_event(
        usage_event_id="usage-2",
        completed_at="2026-05-29T12:02:00Z",
    )
    second["schema_version"] = 4
    second["outcome"] = "error"
    second["error"] = {"exception_type": "InvalidStructuredOutputError"}
    del second["output_tokens"]
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
    assert extracted.shards[0].rows[0]["outcome"] == "success"
    assert extracted.shards[0].rows[1]["outcome"] == "error"
    assert extracted.shards[0].rows[1]["error_type"] == "InvalidStructuredOutputError"
    assert extracted.shards[0].rows[1]["output_tokens"] is None
    assert len(dynamodb.calls) == 2
    assert {call["TableName"] for call in dynamodb.calls} == {"usage-ledger"}
    assert {call["KeyConditionExpression"] for call in dynamodb.calls} == {
        "PK = :pk AND SK BETWEEN :start AND :end"
    }
    assert {call["FilterExpression"] for call in dynamodb.calls} == {
        "schema_version IN (:v3, :v4)"
    }
    assert {call["ConsistentRead"] for call in dynamodb.calls} == {True}
    assert [_expression_values(call) for call in dynamodb.calls] == [
        {
            ":pk": schema.usage_day_pk("20260529", 0),
            ":start": "TS#2026-05-29T11:30:00",
            ":end": "TS#2026-05-29T12:05:00",
            ":v3": 3,
            ":v4": schema.USAGE_EVENT_SCHEMA_VERSION,
        },
        {
            ":pk": schema.usage_day_pk("20260529", 0),
            ":start": "TS#2026-05-29T11:30:00",
            ":end": "TS#2026-05-29T12:05:00",
            ":v3": 3,
            ":v4": schema.USAGE_EVENT_SCHEMA_VERSION,
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
        "schema_version IN (:v3, :v4)"
    }
    assert {call["ConsistentRead"] for call in dynamodb.calls} == {True}
    assert [_expression_values(call) for call in dynamodb.calls] == [
        {
            ":pk": schema.usage_day_pk("20260529", 0),
            ":start": "TS#2026-05-29T23:30:00",
            ":end": "TS#2026-05-30T00:00:00",
            ":v3": 3,
            ":v4": schema.USAGE_EVENT_SCHEMA_VERSION,
        },
        {
            ":pk": schema.usage_day_pk("20260530", 0),
            ":start": "TS#2026-05-30T00:00:00",
            ":end": "TS#2026-05-30T00:05:00",
            ":v3": 3,
            ":v4": schema.USAGE_EVENT_SCHEMA_VERSION,
        },
        {
            ":pk": schema.usage_day_pk("20260529", 1),
            ":start": "TS#2026-05-29T23:30:00",
            ":end": "TS#2026-05-30T00:00:00",
            ":v3": 3,
            ":v4": schema.USAGE_EVENT_SCHEMA_VERSION,
        },
        {
            ":pk": schema.usage_day_pk("20260530", 1),
            ":start": "TS#2026-05-30T00:00:00",
            ":end": "TS#2026-05-30T00:05:00",
            ":v3": 3,
            ":v4": schema.USAGE_EVENT_SCHEMA_VERSION,
        },
    ]


@pytest.mark.parametrize("shape", ["populated", "absent", "null", "truncated"])
def test_projected_query_preserves_full_staging_and_parquet_rows(shape: str) -> None:
    event = _raw_event()
    event["future_additive_field"] = "not exported"
    if shape == "populated":
        event.update(
            run_id="run-1",
            question_id="question-1",
            query_id="query-1",
            benchmark_name="benchmark-1",
            agent_name="agent-1",
            identity_email="user@example.com",
            param_group="group-1",
            config_hash="hash-1",
            duration_seconds=Decimal("1.234567"),
        )
        event["details"] = {
            "request": {"model": "unrelated request content"},
            "result": {
                "metadata": {
                    "performance": _COMPRESSED_PERFORMANCE,
                    "unrelated": "not exported",
                }
            },
        }
    elif shape == "absent":
        del event["api_key_name"]
        del event["finish_reason_raw"]
    elif shape == "null":
        event["details"] = {
            "request": {},
            "result": {"metadata": {"performance": None}},
        }
    else:
        event["details"] = {
            "truncated": True,
            "request": {},
            "result": {"metadata": {"performance": None}},
        }

    prepared = prepare_usage_event_for_write(event).event
    loaded_at = datetime(2026, 5, 29, 12, 5, tzinfo=UTC)
    window = plan_logical_export_window(now=loaded_at)
    dynamodb = FakeDynamoDbClient([{"Items": [serialize_item(prepared)]}])
    extracted = extract_export_window(
        window, config=_config(), dynamodb_client=dynamodb, loaded_at=loaded_at
    )
    baseline = staging_row_from_redshift_rows(
        redshift_rows_from_usage_event(
            prepared, batch_id=extracted.shards[0].batch_id, loaded_at=loaded_at
        )
    )

    assert extracted.shards[0].rows == (baseline,)
    projected = cast(list[dict[str, Any]], dynamodb.responses[0]["Items"])[0]
    assert "future_additive_field" not in projected
    assert "request" not in projected["details"]["M"]
    assert "unrelated" not in projected["details"]["M"]["result"]["M"]["metadata"]["M"]

    s3 = FakeS3Client()
    write_export_window(extracted, config=_config(), s3_client=s3)
    actual_table = pq.read_table(pa.BufferReader(cast(bytes, s3.puts[0]["Body"])))
    baseline_table = pq.read_table(
        pa.BufferReader(parquet_bytes_from_rows([baseline]))
    )
    assert actual_table.equals(baseline_table)

    # A missing optional value still requires its .get path in the projection.
    projected_access = _ProjectedAccess(prepared, _projection_paths(dynamodb.calls[0]))
    assert redshift_rows_from_usage_event(
        projected_access,
        batch_id=extracted.shards[0].batch_id,
        loaded_at=loaded_at,
    ) == redshift_rows_from_usage_event(
        prepared,
        batch_id=extracted.shards[0].batch_id,
        loaded_at=loaded_at,
    )
