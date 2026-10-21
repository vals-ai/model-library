"""Durable record of which provider key each task attempt uses.

A task attempt keeps one provider key for its whole life. Gateway workers run in
many containers, so the first worker to pick a key writes it here and every later
request for the same attempt and model gets the same key back.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
from typing import Any, Protocol, cast

from aiobotocore.config import AioConfig  # pyright: ignore[reportMissingImports]
from aiobotocore.session import get_session  # pyright: ignore[reportUnknownVariableType]
from botocore.exceptions import ClientError

from model_gateway.usage_ledger.dynamodb_writer import (
    AsyncDynamoDbClient,
    is_conditional_check_failed,
)
from model_gateway.usage_ledger.store import DEFAULT_MAX_POOL_CONNECTIONS

# The parts that identify one attempt, e.g. ("attempt", "run-5", "attempt-A").
AttemptKey = tuple[str, ...]

logger = logging.getLogger(__name__)

TABLE_NAME_ENV = "GATEWAY_KEY_ASSIGNMENTS_TABLE_NAME"


class AssignmentStore(Protocol):
    async def get(self, attempt_key: AttemptKey, model: str) -> str | None:
        """The key name this attempt holds for this model, if any."""
        ...

    async def put_if_absent(
        self, attempt_key: AttemptKey, model: str, key_name: str
    ) -> str:
        """Assign ``key_name`` unless an assignment exists; return the one that won."""
        ...


class DynamoAssignmentStore:
    """Shared store backed by the Gateway-owned key assignments DynamoDB table."""

    def __init__(
        self,
        *,
        table_name: str,
        region_name: str | None = None,
        client: AsyncDynamoDbClient | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._table_name = table_name
        self._region_name = region_name
        self._client = client
        self._client_context: (
            AbstractAsyncContextManager[AsyncDynamoDbClient] | None
        ) = None
        self._clock = clock

    async def start(self) -> None:
        if self._client is not None:
            return
        session = cast(Any, get_session())
        client_context = cast(
            AbstractAsyncContextManager[AsyncDynamoDbClient],
            session.create_client(
                "dynamodb",
                region_name=self._region_name or None,
                config=AioConfig(
                    connector_args={
                        "keepalive_timeout": 60,
                        "ttl_dns_cache": 300,
                        "force_close": False,
                    },
                    max_pool_connections=DEFAULT_MAX_POOL_CONNECTIONS,
                    connect_timeout=3,
                    read_timeout=5,
                    # Transient-throttle retries within one request, not a key fallback.
                    retries={"mode": "standard", "max_attempts": 3},
                ),
            ),
        )
        self._client = await client_context.__aenter__()
        self._client_context = client_context

    async def close(self) -> None:
        if self._client_context is None:
            return
        try:
            await self._client_context.__aexit__(None, None, None)
        finally:
            self._client_context = None
            self._client = None

    async def get(self, attempt_key: AttemptKey, model: str) -> str | None:
        response = await self._started_client().get_item(
            TableName=self._table_name,
            Key={"pk": {"S": _pk(attempt_key, model)}},
            ConsistentRead=True,
        )
        item = response.get("Item")
        return None if item is None else _key_name(item)

    async def put_if_absent(
        self, attempt_key: AttemptKey, model: str, key_name: str
    ) -> str:
        now = self._clock()
        try:
            await self._started_client().put_item(
                TableName=self._table_name,
                Item={
                    "pk": {"S": _pk(attempt_key, model)},
                    "key_name": {"S": key_name},
                    "assigned_at": {"S": _isoformat(now)},
                },
                ConditionExpression="attribute_not_exists(pk)",
                ReturnValuesOnConditionCheckFailure="ALL_OLD",
            )
        except ClientError as exc:
            if not is_conditional_check_failed(exc):
                raise
            # ReturnValuesOnConditionCheckFailure=ALL_OLD returns the winning row.
            return _key_name(cast(Mapping[str, Any], exc.response)["Item"])
        return key_name

    def _started_client(self) -> AsyncDynamoDbClient:
        if self._client is None:
            raise ValueError("Key assignment DynamoDB client is not started")
        return self._client


def create_key_assignment_store_from_env() -> DynamoAssignmentStore | None:
    """The shared store; None (per-request pick) when no table is configured."""
    table_name = os.environ.get(TABLE_NAME_ENV, "").strip()
    if not table_name:
        logger.warning("%s is not set; key assignments are off", TABLE_NAME_ENV)
        return None
    return DynamoAssignmentStore(table_name=table_name)


def _pk(attempt_key: AttemptKey, model: str) -> str:
    # A JSON pair of (attempt parts, model) keeps ("a#b", "c") and ("a", "b#c")
    # apart; its hash keeps the partition key at 64 bytes however long the ids
    # are (DynamoDB allows 2,048).
    pair = json.dumps([attempt_key, model], separators=(",", ":"))
    return hashlib.sha256(pair.encode()).hexdigest()


def _key_name(item: Mapping[str, Any]) -> str:
    # A row without a string key_name is corrupt: raise rather than serve it.
    return cast(str, item["key_name"]["S"])


def _isoformat(epoch_seconds: float) -> str:
    return (
        datetime.fromtimestamp(int(epoch_seconds), UTC)
        .isoformat()
        .replace("+00:00", "Z")
    )
