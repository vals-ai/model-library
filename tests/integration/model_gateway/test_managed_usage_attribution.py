from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime
from decimal import Decimal

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from model_gateway.types import QueryRequest
from model_gateway.usage_ledger.attribution import ManagedUsageAttribution
from model_gateway.usage_ledger.details import snapshot_usage_request
from model_gateway.usage_ledger.dynamodb_writer import (
    prepare_usage_event_for_write,
    serialize_item,
)
from model_gateway.usage_ledger.message import deserialize_usage_event_message
from model_gateway.usage_ledger.redshift_export import parquet_bytes_from_rows
from model_gateway.usage_ledger.redshift_schema import (
    SCHEMA_COLUMN_ADDITIONS,
    USAGE_EVENTS_STAGING_COLUMN_NAMES,
    USAGE_EVENTS_STAGING_TABLE_NAME,
    usage_events_column_sql,
)
from model_gateway.usage_ledger.redshift_transform import (
    redshift_export_source_paths,
    redshift_rows_from_usage_event,
)
from model_gateway.usage_ledger.store import build_managed_usage_event, build_usage_event
from model_library.base.input import TextInput
from model_library.base.output import QueryResult, QueryResultCost, QueryResultMetadata


ATTRIBUTION_FIELDS = (
    "managed_key_id",
    "budget_id",
    "owner_issuer",
    "owner_subject",
    "service_id",
    "project_id",
    "project_org_id",
    "project_environment",
)


def test_admitted_attribution_survives_ledger_message_and_warehouse_export() -> None:
    body = QueryRequest(
        model="openai/gpt-4o",
        inputs=[TextInput(text="hello")],
        identity={"email": "caller@example.com", "project_id": "forged-project"},
    )
    result = QueryResult(
        output_text="ok",
        metadata=QueryResultMetadata(
            in_tokens=4,
            out_tokens=2,
            cost=QueryResultCost(input=0.001, output=0.002),
        ),
    )
    event_args = dict(
        body=body,
        config=body.config_dict(),
        query_params={},
        dimensions={"ProviderEndpoint": "default", "ParamGroup": "pg"},
        result=result,
        error=None,
        request=snapshot_usage_request(body),
        api_key_name="display-name",
        provider_key_name=None,
        pool_revision=None,
        completed_at=datetime(2026, 5, 29, 12, 0, tzinfo=UTC),
    )
    admitted = ManagedUsageAttribution(
        managed_key_id="key-1",
        budget_id="budget-project-1",
        owner_issuer="https://identity.example",
        owner_subject="owner-1",
        service_id=None,
        project_id="project-1",
        project_org_id="org-1",
        project_environment="prod",
    )
    with pytest.raises(FrozenInstanceError):
        admitted.budget_id = "different-budget"

    event = build_managed_usage_event(attribution=admitted, **event_args)
    old_event = build_usage_event(**event_args)
    uncapped = replace(admitted, budget_id=None, project_id=None,
                       project_org_id=None, project_environment=None)
    uncapped_event = build_managed_usage_event(attribution=uncapped, **event_args)
    body.identity = {"email": "later@example.com", "project_id": "later-project"}
    later_assignment = replace(admitted, budget_id="different-budget", project_id="later-project")
    assert later_assignment != admitted

    expected = {field: getattr(admitted, field) for field in ATTRIBUTION_FIELDS}
    assert {field: event[field] for field in ATTRIBUTION_FIELDS} == expected
    assert event["identity_email"] == "caller@example.com"
    assert event["api_key_name"] == "display-name"
    assert event["cost_usd"] == Decimal("0.003")
    assert ATTRIBUTION_FIELDS[0] not in old_event

    prepared = prepare_usage_event_for_write(event)
    restored = deserialize_usage_event_message(prepared.message)
    assert {field: restored[field] for field in ATTRIBUTION_FIELDS} == expected
    dynamo_item = serialize_item(prepared.event)
    assert dynamo_item["managed_key_id"] == {"S": "key-1"}
    assert dynamo_item["budget_id"] == {"S": "budget-project-1"}
    assert "service_id" not in dynamo_item

    assert all((field,) in redshift_export_source_paths() for field in ATTRIBUTION_FIELDS)
    managed_rows = redshift_rows_from_usage_event(restored, batch_id="batch-1")
    legacy_rows = redshift_rows_from_usage_event(old_event, batch_id="batch-1")
    assert {field: managed_rows.fact[field] for field in ATTRIBUTION_FIELDS} == expected
    assert all(legacy_rows.fact[field] is None for field in ATTRIBUTION_FIELDS)
    assert managed_rows.fact["cost_usd"] == legacy_rows.fact["cost_usd"]

    payload = parquet_bytes_from_rows(
        [{**rows.fact, **rows.performance} for rows in (managed_rows, legacy_rows)]
    )
    table = pq.read_table(pa.BufferReader(payload))
    assert tuple(table.schema.names) == USAGE_EVENTS_STAGING_COLUMN_NAMES
    assert all(table.schema.field(field).type == pa.string() for field in ATTRIBUTION_FIELDS)
    managed_export, legacy_export = table.to_pylist()
    assert {field: managed_export[field] for field in ATTRIBUTION_FIELDS} == expected
    assert all(legacy_export[field] is None for field in ATTRIBUTION_FIELDS)
    assert managed_export["cost_usd"] == legacy_export["cost_usd"]
    assert managed_export["cost_usd"] == Decimal("0.003000000000")
    uncapped_prepared = prepare_usage_event_for_write(uncapped_event)
    uncapped_item = serialize_item(uncapped_prepared.event)
    assert uncapped_item["managed_key_id"] == {"S": "key-1"}
    assert uncapped_item["owner_issuer"] == {"S": "https://identity.example"}
    assert uncapped_item["owner_subject"] == {"S": "owner-1"}
    assert "budget_id" not in uncapped_item
    uncapped_restored = deserialize_usage_event_message(uncapped_prepared.message)
    uncapped_row = redshift_rows_from_usage_event(uncapped_restored, batch_id="batch-1")
    assert uncapped_row.fact["managed_key_id"] == "key-1"
    assert uncapped_row.fact["owner_subject"] == "owner-1"
    assert uncapped_row.fact["budget_id"] is None
    assert uncapped_row.fact["identity_email"] == "caller@example.com"
    assert uncapped_row.fact["cost_usd"] == Decimal("0.003")
    ddl = usage_events_column_sql()
    for field in ATTRIBUTION_FIELDS:
        assert f"  {field} varchar(" in ddl
        assert {
            addition["TableName"]
            for addition in SCHEMA_COLUMN_ADDITIONS
            if addition["ColumnName"] == field
        } == {USAGE_EVENTS_STAGING_TABLE_NAME, "usage_events"}
