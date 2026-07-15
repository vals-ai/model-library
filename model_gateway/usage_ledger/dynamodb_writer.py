"""Dependency-light DynamoDB writer for gateway usage ledger events."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Any, Protocol, cast

from boto3.dynamodb.types import TypeSerializer
from botocore.exceptions import ClientError

import model_gateway.usage_ledger.schema as ledger_schema


class UsageLedgerEventValidationError(ValueError):
    """Raised when a usage ledger event is not safe to persist."""


class AsyncDynamoDbClient(Protocol):
    async def put_item(self, **kwargs: object) -> Mapping[str, Any]: ...

    async def get_item(self, **kwargs: object) -> Mapping[str, Any]: ...


class SyncDynamoDbClient(Protocol):
    def put_item(self, **kwargs: object) -> Mapping[str, object]: ...

    def get_item(self, **kwargs: object) -> Mapping[str, object]: ...


_serializer = cast(Any, TypeSerializer())

_REQUIRED_USAGE_EVENT_FIELDS = frozenset(
    {
        ledger_schema.BASE_PK,
        ledger_schema.BASE_SK,
        "entity_type",
        "usage_event_id",
        "model",
        "provider",
        "completed_at",
        "day",
        "usage_shard",
        "schema_version",
        "metadata_schema_version",
        "normalization_version",
        *ledger_schema.NUMERIC_LEDGER_FIELDS,
    }
)

_ALLOWED_USAGE_EVENT_FIELDS = frozenset(
    {
        *_REQUIRED_USAGE_EVENT_FIELDS,
        "run_id",
        "question_id",
        "query_id",
        "identity",
        ledger_schema.IDENTITY_BENCHMARK_NAME,
        ledger_schema.IDENTITY_AGENT_NAME,
        ledger_schema.IDENTITY_EMAIL,
        "api_key_fingerprint",
        "provider_response_id",
        "provider_request_id",
        "provider_endpoint",
        "param_group",
        "config_hash",
        "config_redacted_json",
        "config_redacted_json_truncated",
        "metadata_json",
        "metadata_json_truncated",
        "finish_reason_json",
        "finish_reason_json_truncated",
        "performance_json",
        "performance_json_truncated",
        ledger_schema.RUN_INDEX_PK,
        ledger_schema.RUN_INDEX_SK,
        ledger_schema.QUERY_INDEX_PK,
        ledger_schema.QUERY_INDEX_SK,
        ledger_schema.API_KEY_DAY_INDEX_PK,
        ledger_schema.API_KEY_DAY_INDEX_SK,
        ledger_schema.BENCHMARK_INDEX_PK,
        ledger_schema.BENCHMARK_INDEX_SK,
        ledger_schema.AGENT_INDEX_PK,
        ledger_schema.AGENT_INDEX_SK,
    }
)


async def put_usage_event_async(
    *,
    client: AsyncDynamoDbClient,
    table_name: str,
    event: Mapping[str, object],
) -> None:
    """Write a usage event asynchronously, accepting duplicate same-event writes."""
    validate_usage_event(event)
    try:
        await client.put_item(
            TableName=table_name,
            Item=serialize_item(event),
            ConditionExpression="attribute_not_exists(PK)",
        )
    except ClientError as exc:
        if is_conditional_check_failed(exc) and await is_duplicate_event_async(
            client=client,
            table_name=table_name,
            event=event,
        ):
            return
        raise


def put_usage_event_sync(
    *,
    client: SyncDynamoDbClient,
    table_name: str,
    event: Mapping[str, object],
) -> bool:
    """Write a usage event synchronously; return False for duplicate same-event writes."""
    validate_usage_event(event)
    try:
        client.put_item(
            TableName=table_name,
            Item=serialize_item(event),
            ConditionExpression="attribute_not_exists(PK)",
        )
        return True
    except ClientError as exc:
        if is_conditional_check_failed(exc) and is_duplicate_event_sync(
            client=client,
            table_name=table_name,
            event=event,
        ):
            return False
        raise


def validate_usage_event(event: Mapping[str, object]) -> None:
    """Validate that a queued usage event has the current durable ledger shape."""
    missing = sorted(
        field for field in _REQUIRED_USAGE_EVENT_FIELDS if field not in event
    )
    if missing:
        raise UsageLedgerEventValidationError(
            f"Usage ledger event is missing required field(s): {', '.join(missing)}"
        )

    unknown = sorted(set(event) - _ALLOWED_USAGE_EVENT_FIELDS)
    if unknown:
        raise UsageLedgerEventValidationError(
            f"Usage ledger event has unsupported field(s): {', '.join(unknown)}"
        )

    required_non_null = (
        ledger_schema.BASE_PK,
        ledger_schema.BASE_SK,
        "entity_type",
        "usage_event_id",
        "model",
        "provider",
        "completed_at",
        "day",
        "usage_shard",
    )
    null_required = sorted(
        field for field in required_non_null if event.get(field) is None
    )
    if null_required:
        raise UsageLedgerEventValidationError(
            f"Usage ledger event has null required field(s): {', '.join(null_required)}"
        )

    if event.get("entity_type") != "usage_event":
        raise UsageLedgerEventValidationError(
            "Usage ledger event entity_type is invalid"
        )


def serialize_item(item: Mapping[str, object]) -> dict[str, dict[str, Any]]:
    return {
        key: cast(dict[str, Any], _serializer.serialize(to_dynamodb_value(value)))
        for key, value in item.items()
        if value is not None
    }


def to_dynamodb_value(value: object) -> object:
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, object], value)
        return {key: to_dynamodb_value(item) for key, item in mapping.items()}
    if isinstance(value, list):
        items = cast(list[object], value)
        return [to_dynamodb_value(item) for item in items]
    return value


async def is_duplicate_event_async(
    *,
    client: AsyncDynamoDbClient,
    table_name: str,
    event: Mapping[str, object],
) -> bool:
    response = await client.get_item(
        TableName=table_name,
        Key=serialize_item(
            {
                ledger_schema.BASE_PK: event.get(ledger_schema.BASE_PK),
                ledger_schema.BASE_SK: event.get(ledger_schema.BASE_SK),
            }
        ),
        ProjectionExpression="usage_event_id",
        ConsistentRead=True,
    )
    return stored_usage_event_id(response) == event.get("usage_event_id")


def is_duplicate_event_sync(
    *,
    client: SyncDynamoDbClient,
    table_name: str,
    event: Mapping[str, object],
) -> bool:
    response = client.get_item(
        TableName=table_name,
        Key=serialize_item(
            {
                ledger_schema.BASE_PK: event.get(ledger_schema.BASE_PK),
                ledger_schema.BASE_SK: event.get(ledger_schema.BASE_SK),
            }
        ),
        ProjectionExpression="usage_event_id",
        ConsistentRead=True,
    )
    return stored_usage_event_id(response) == event.get("usage_event_id")


def stored_usage_event_id(response: Mapping[str, object]) -> object | None:
    item = response.get("Item")
    if not isinstance(item, Mapping):
        return None
    item_mapping = cast(Mapping[str, object], item)
    usage_event_id = item_mapping.get("usage_event_id")
    if isinstance(usage_event_id, Mapping):
        usage_event_id_mapping = cast(Mapping[str, object], usage_event_id)
        return usage_event_id_mapping.get("S")
    return usage_event_id


def is_conditional_check_failed(exc: ClientError) -> bool:
    response = cast(Mapping[str, object], exc.response)
    error = response.get("Error", {})
    if not isinstance(error, Mapping):
        return False
    error_mapping = cast(Mapping[str, object], error)
    return error_mapping.get("Code") == "ConditionalCheckFailedException"
