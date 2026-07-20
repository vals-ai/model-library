"""Dependency-light DynamoDB writer for gateway usage ledger events."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from decimal import Decimal
from typing import Any, Protocol, cast

from boto3.dynamodb.types import TypeSerializer
from botocore.exceptions import ClientError

import model_gateway.usage_ledger.schema as ledger_schema
from model_gateway.usage_ledger.message import (
    PreparedUsageEvent,
    UsageLedgerMessageTooLarge,
    prepare_usage_event_message,
)

logger = logging.getLogger("model_proxy_server.usage_ledger")


class _DynamoDbItemTooLarge(ValueError):
    """Raised when a serialized event exceeds DynamoDB's item limit."""


class AsyncDynamoDbClient(Protocol):
    async def put_item(self, **kwargs: object) -> Mapping[str, Any]: ...

    async def get_item(self, **kwargs: object) -> Mapping[str, Any]: ...


class SyncDynamoDbClient(Protocol):
    def put_item(self, **kwargs: object) -> Mapping[str, object]: ...

    def get_item(self, **kwargs: object) -> Mapping[str, Any]: ...


_serializer = cast(Any, TypeSerializer())
_DYNAMODB_ITEM_MAX_BYTES = 400 * 1024


def prepare_usage_event_for_write(
    event: Mapping[str, object],
) -> PreparedUsageEvent:
    """Normalize a producer event, truncating details only for hard size limits."""
    try:
        return _prepare_with_size_limits(event)
    except (UsageLedgerMessageTooLarge, _DynamoDbItemTooLarge):
        truncated = dict(event)
        truncated[ledger_schema.DETAILS_FIELD] = {
            "truncated": True,
            "request": {},
            "result": {"metadata": {"performance": None}},
        }
        prepared = _prepare_with_size_limits(truncated)
        logger.warning("Gateway usage ledger replaced oversized details")
        return prepared


def _prepare_with_size_limits(event: Mapping[str, object]) -> PreparedUsageEvent:
    prepared = prepare_usage_event_message(event)
    item_bytes = _dynamodb_item_size(serialize_item(prepared.event))
    if item_bytes > _DYNAMODB_ITEM_MAX_BYTES:
        raise _DynamoDbItemTooLarge(
            f"Usage ledger DynamoDB item is {item_bytes} bytes; "
            f"max is {_DYNAMODB_ITEM_MAX_BYTES} bytes"
        )
    return prepared


async def put_usage_event_async(
    *,
    client: AsyncDynamoDbClient,
    table_name: str,
    event: Mapping[str, object],
) -> None:
    """Write an event asynchronously, accepting idempotent retries."""
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
    """Write an event synchronously; return False for an idempotent retry."""
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


def serialize_item(item: Mapping[str, object]) -> dict[str, dict[str, Any]]:
    return {
        key: cast(dict[str, Any], _serializer.serialize(to_dynamodb_value(value)))
        for key, value in item.items()
        if value is not None
    }


def _dynamodb_item_size(item: Mapping[str, dict[str, Any]]) -> int:
    return sum(
        len(name.encode("utf-8")) + _dynamodb_attribute_value_size(value)
        for name, value in item.items()
    )


def _dynamodb_attribute_value_size(attribute: dict[str, Any]) -> int:
    attribute_type, value = next(iter(attribute.items()))
    if attribute_type == "S":
        return len(value.encode("utf-8"))
    if attribute_type == "N":
        digits = Decimal(value).as_tuple().digits
        first = 0
        last = len(digits)
        while first < last - 1 and digits[first] == 0:
            first += 1
        while last > first + 1 and digits[last - 1] == 0:
            last -= 1
        significant_digits = last - first
        return (significant_digits + 1) // 2 + 1
    if attribute_type in {"BOOL", "NULL"}:
        return 1
    if attribute_type == "M":
        return 3 + len(value) + _dynamodb_item_size(value)
    if attribute_type == "L":
        return (
            3 + len(value) + sum(_dynamodb_attribute_value_size(item) for item in value)
        )
    raise TypeError(f"Cannot size DynamoDB attribute type {attribute_type!r}")


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


def stored_usage_event_id(response: Mapping[str, Any]) -> object | None:
    item: dict[str, dict[str, object]] | None = response.get("Item")
    if item is None:
        return None
    return item["usage_event_id"]["S"]


def is_conditional_check_failed(exc: ClientError) -> bool:
    response = cast(Mapping[str, object], exc.response)
    error = response.get("Error", {})
    if not isinstance(error, Mapping):
        return False
    error_mapping = cast(Mapping[str, object], error)
    return error_mapping.get("Code") == "ConditionalCheckFailedException"
