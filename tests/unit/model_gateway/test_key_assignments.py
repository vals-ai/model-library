"""Contract for the store that pins one provider key to each task attempt."""

import asyncio
import hashlib
from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from botocore.exceptions import ClientError

from model_gateway.usage_ledger.store import DEFAULT_MAX_POOL_CONNECTIONS
from model_gateway.key_assignments import (
    AssignmentStore,
    AttemptKey,
    DynamoAssignmentStore,
)
from tests.unit.model_gateway._support import InMemoryAssignmentStore

NOW = 1_760_000_000.0


class FakeDynamoTable:
    """Honors the one condition the store relies on: write only when pk is new."""

    def __init__(self) -> None:
        self.items: dict[str, dict[str, Any]] = {}
        self.put_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []

    async def put_item(self, **kwargs: Any) -> dict[str, Any]:
        self.put_calls.append(kwargs)
        await asyncio.sleep(0)
        item = kwargs["Item"]
        pk = item["pk"]["S"]
        existing = self.items.get(pk)
        if existing is not None:
            assert kwargs["ConditionExpression"] == "attribute_not_exists(pk)"
            error: dict[str, Any] = {
                "Error": {"Code": "ConditionalCheckFailedException", "Message": ""}
            }
            if kwargs.get("ReturnValuesOnConditionCheckFailure") == "ALL_OLD":
                error["Item"] = existing
            raise ClientError(error, "PutItem")  # pyright: ignore[reportArgumentType]
        self.items[pk] = item
        return {}

    async def get_item(self, **kwargs: Any) -> dict[str, Any]:
        self.get_calls.append(kwargs)
        item = self.items.get(kwargs["Key"]["pk"]["S"])
        return {} if item is None else {"Item": item}



def _in_memory() -> AssignmentStore:
    return InMemoryAssignmentStore()


def _dynamo() -> AssignmentStore:
    return DynamoAssignmentStore(
        table_name="key-assignments",
        client=FakeDynamoTable(),
        clock=lambda: NOW,
    )


STORES: list[Callable[[], AssignmentStore]] = [_in_memory, _dynamo]


async def held(store: AssignmentStore, attempt_key: AttemptKey, model: str) -> str:
    """The key an attempt holds, read back through the one call the store has."""
    return await store.put_if_absent(attempt_key, model, "probe-key")


@pytest.mark.parametrize("make_store", STORES, ids=["memory", "dynamo"])
async def test_first_put_wins_and_later_puts_return_it(
    make_store: Callable[[], AssignmentStore],
) -> None:
    store = make_store()

    first = await store.put_if_absent(("attempt-1",), "openai/gpt-4o", "openai-key-a")
    second = await store.put_if_absent(("attempt-1",), "openai/gpt-4o", "openai-key-b")

    assert first == "openai-key-a"
    assert second == "openai-key-a"


@pytest.mark.parametrize("make_store", STORES, ids=["memory", "dynamo"])
async def test_assignments_are_scoped_by_attempt_and_model(
    make_store: Callable[[], AssignmentStore],
) -> None:
    store = make_store()

    await store.put_if_absent(("attempt-1",), "openai/gpt-4o", "openai-key-a")
    await store.put_if_absent(("attempt-2",), "openai/gpt-4o", "openai-key-b")
    await store.put_if_absent(("attempt-1",), "openai/gpt-4o-mini", "openai-key-c")

    assert await held(store, ("attempt-1",), "openai/gpt-4o") == "openai-key-a"
    assert await held(store, ("attempt-2",), "openai/gpt-4o") == "openai-key-b"
    assert await held(store, ("attempt-1",), "openai/gpt-4o-mini") == "openai-key-c"


# sha256 of the JSON pair [["attempt-1"],"openai/gpt-4o"].
ATTEMPT_1_PK = hashlib.sha256(b'[["attempt-1"],"openai/gpt-4o"]').hexdigest()


async def test_dynamo_partition_key_hashes_the_attempt_parts_as_a_json_list() -> None:
    table = FakeDynamoTable()
    store = DynamoAssignmentStore(
        table_name="key-assignments", client=table, clock=lambda: NOW
    )

    await store.put_if_absent(("attempt", "run-1", "att-1"), "openai/gpt-4o", "key-a")

    expected = b'[["attempt","run-1","att-1"],"openai/gpt-4o"]'
    pk = table.put_calls[0]["Item"]["pk"]["S"]
    assert pk == hashlib.sha256(expected).hexdigest()


async def test_dynamo_partition_key_stays_under_the_dynamodb_limit() -> None:
    table = FakeDynamoTable()
    store = DynamoAssignmentStore(
        table_name="key-assignments", client=table, clock=lambda: NOW
    )
    long_attempt = ("é" * 3000,)

    await store.put_if_absent(long_attempt, "openai/gpt-4o", "key-a")

    pk = table.put_calls[0]["Item"]["pk"]["S"]
    assert len(pk.encode()) <= 2048
    assert await held(store, long_attempt, "openai/gpt-4o") == "key-a"
    assert await held(store, (*long_attempt, "x"), "openai/gpt-4o") == "probe-key"


@pytest.mark.parametrize("make_store", STORES, ids=["memory", "dynamo"])
async def test_separator_in_an_id_keeps_pairs_apart(
    make_store: Callable[[], AssignmentStore],
) -> None:
    store = make_store()

    await store.put_if_absent("a#b", "c", "key-1")
    await store.put_if_absent("a", "b#c", "key-2")

    assert await held(store, "a#b", "c") == "key-1"
    assert await held(store, "a", "b#c") == "key-2"


@pytest.mark.parametrize("make_store", STORES, ids=["memory", "dynamo"])
async def test_concurrent_puts_all_return_the_single_winner(
    make_store: Callable[[], AssignmentStore],
) -> None:
    store = make_store()
    puts: list[Awaitable[str]] = [
        store.put_if_absent(("attempt-1",), "openai/gpt-4o", f"openai-key-{index}")
        for index in range(8)
    ]

    winners = await asyncio.gather(*puts)

    assert len(set(winners)) == 1
    assert await held(store, ("attempt-1",), "openai/gpt-4o") == winners[0]


async def test_dynamo_writes_the_documented_item_shape() -> None:
    table = FakeDynamoTable()
    store = DynamoAssignmentStore(
        table_name="key-assignments",
        client=table,
        clock=lambda: NOW,
    )

    await store.put_if_absent(("attempt-1",), "openai/gpt-4o", "openai-key-a")

    put = table.put_calls[0]
    assert put["TableName"] == "key-assignments"
    assert put["ConditionExpression"] == "attribute_not_exists(pk)"
    assert put["Item"] == {
        "pk": {"S": ATTEMPT_1_PK},
        "key_name": {"S": "openai-key-a"},
        "assigned_at": {"S": "2025-10-09T08:53:20Z"},
    }


async def test_dynamo_raises_other_client_errors() -> None:
    class ThrottledTable(FakeDynamoTable):
        async def put_item(self, **kwargs: Any) -> dict[str, Any]:
            raise ClientError(
                {"Error": {"Code": "ProvisionedThroughputExceededException"}},  # pyright: ignore[reportArgumentType]
                "PutItem",
            )

    store = DynamoAssignmentStore(
        table_name="key-assignments", client=ThrottledTable(), clock=lambda: NOW
    )

    with pytest.raises(ClientError):
        await store.put_if_absent(("attempt-1",), "openai/gpt-4o", "openai-key-a")


async def test_dynamo_start_opens_one_client_and_close_releases_it() -> None:
    table = FakeDynamoTable()
    context = AsyncMock()
    context.__aenter__.return_value = table
    store = DynamoAssignmentStore(table_name="key-assignments", clock=lambda: NOW)

    with patch("model_gateway.key_assignments.get_session") as get_session:
        get_session.return_value.create_client.return_value = context
        await store.start()
        await store.start()
        await store.put_if_absent(("attempt-1",), "openai/gpt-4o", "openai-key-a")
        await store.close()
        await store.close()

    get_session.return_value.create_client.assert_called_once()
    assert get_session.return_value.create_client.call_args.args == ("dynamodb",)
    context.__aexit__.assert_awaited_once()
    assert len(table.put_calls) == 1
    with pytest.raises(ValueError, match="not started"):
        await store.put_if_absent(("attempt-1",), "openai/gpt-4o", "openai-key-a")


async def test_dynamo_client_pool_matches_the_usage_ledger_client() -> None:
    context = AsyncMock()
    context.__aenter__.return_value = FakeDynamoTable()
    store = DynamoAssignmentStore(table_name="key-assignments")

    with patch("model_gateway.key_assignments.get_session") as get_session:
        get_session.return_value.create_client.return_value = context
        await store.start()

    config = get_session.return_value.create_client.call_args.kwargs["config"]
    assert config.max_pool_connections == DEFAULT_MAX_POOL_CONNECTIONS


async def test_dynamo_raises_on_a_row_without_a_key_name() -> None:
    table = FakeDynamoTable()
    store = DynamoAssignmentStore(
        table_name="key-assignments", client=table, clock=lambda: NOW
    )
    table.items[ATTEMPT_1_PK] = {"pk": {"S": ATTEMPT_1_PK}}

    with pytest.raises(KeyError):
        await store.put_if_absent(("attempt-1",), "openai/gpt-4o", "openai-key-a")


@pytest.mark.parametrize("make_store", STORES, ids=["memory", "dynamo"])
async def test_get_reads_the_held_key_and_none_before_one(
    make_store: Callable[[], AssignmentStore],
) -> None:
    store = make_store()

    assert await store.get(("attempt-1",), "openai/gpt-4o") is None
    await store.put_if_absent(("attempt-1",), "openai/gpt-4o", "openai-key-a")
    assert await store.get(("attempt-1",), "openai/gpt-4o") == "openai-key-a"


async def test_dynamo_reads_strongly_consistent() -> None:
    table = FakeDynamoTable()
    store = DynamoAssignmentStore(
        table_name="key-assignments", client=table, clock=lambda: NOW
    )

    await store.get(("attempt-1",), "openai/gpt-4o")

    assert table.get_calls[0]["ConsistentRead"] is True
