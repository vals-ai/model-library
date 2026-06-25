"""AWS Lambda SQS consumer for gateway usage ledger events."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any, cast

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from model_gateway.usage_ledger.dynamodb_writer import (
    SyncDynamoDbClient,
    UsageLedgerEventValidationError,
    put_usage_event_sync,
)
from model_gateway.usage_ledger.message import (
    UsageLedgerMessageError,
    deserialize_usage_event_message,
)

logger = logging.getLogger(__name__)


_client: SyncDynamoDbClient | None = None


def handler(
    event: Mapping[str, object], _context: object
) -> dict[str, list[dict[str, str]]]:
    """Process an SQS batch and return Lambda partial batch failures."""
    failures: list[dict[str, str]] = []
    records_value = event.get("Records", [])
    if not isinstance(records_value, list):
        raise ValueError("Lambda SQS event Records must be a list")
    records = cast(list[object], records_value)

    table_name = _table_name()
    client = _dynamodb_client()
    for record in records:
        message_id = _message_id(record)
        try:
            body = _message_body(record)
            usage_event = deserialize_usage_event_message(body)
            put_usage_event_sync(
                client=client,
                table_name=table_name,
                event=usage_event,
            )
        except (
            BotoCoreError,
            ClientError,
            UsageLedgerMessageError,
            UsageLedgerEventValidationError,
            ValueError,
            TypeError,
        ):
            logger.exception(
                "Gateway usage ledger Lambda failed message %s", message_id
            )
            failures.append({"itemIdentifier": message_id})
    return {"batchItemFailures": failures}


def _dynamodb_client() -> SyncDynamoDbClient:
    global _client
    if _client is None:
        client_factory = cast(Any, boto3.client)
        _client = cast(SyncDynamoDbClient, client_factory("dynamodb"))
    return _client


def _table_name() -> str:
    table_name = os.environ.get("GATEWAY_USAGE_LEDGER_TABLE_NAME", "").strip()
    if not table_name:
        raise ValueError("GATEWAY_USAGE_LEDGER_TABLE_NAME is required")
    return table_name


def _message_id(record: object) -> str:
    if isinstance(record, Mapping):
        record_mapping = cast(Mapping[str, object], record)
        value = record_mapping.get("messageId")
        if isinstance(value, str) and value:
            return value
    return "unknown"


def _message_body(record: object) -> str:
    if not isinstance(record, Mapping):
        raise ValueError("SQS record must be an object")
    record_mapping = cast(Mapping[str, object], record)
    value = record_mapping.get("body")
    if not isinstance(value, str):
        raise ValueError("SQS record body must be a string")
    return value
