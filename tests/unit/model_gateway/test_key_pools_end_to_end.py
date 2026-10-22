"""Key pools, end to end: each task attempt keeps the provider key it started on.

The story runs through the real `/query` route on two Gateway app instances
(two workers) that share one assignment store, the way deployed workers share
the DynamoDB table. The pool starts as Key 1 and Key 2; Key 3 is added between
an attempt's first and second request.
"""

import asyncio
import os
from collections.abc import Awaitable, Iterator, Mapping
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from botocore.exceptions import ClientError
from starlette.testclient import TestClient

import model_gateway.app as gateway_app
import model_gateway.model_helpers as model_helpers
import model_gateway.key_assignments as key_assignments
from model_gateway.key_assignments import (
    AttemptKey,
    DynamoAssignmentStore,
)
from model_gateway.key_pool_config import PoolConfigSnapshot
from model_gateway.route_helpers import GatewayOperation
from model_library.base import LLMConfig, QueryResult
from scripts.seed_key_pools import registry_endpoints, seed_document
from tests.unit.model_gateway._seeded_pools import seeded_pools
from tests.unit.model_gateway._support import (
    HEADERS,
    InMemoryAssignmentStore,
    _make_client,
)
from tests.unit.model_gateway.test_key_assignments import FakeDynamoTable

MODEL = "anthropic/claude-opus-4-6"
RUN_ID = "run-5"

# Fake key settings: the setting name is what gets recorded, the value is the secret.
KEY_1, KEY_2, KEY_3 = "ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_2", "ANTHROPIC_API_KEY_3"
VALUES = {KEY_1: "value-of-key-1", KEY_2: "value-of-key-2", KEY_3: "value-of-key-3"}
SETTING_FOR_VALUE = {value: name for name, value in VALUES.items()}

ATTEMPTS = "ABCDEF"
# The pick as deployed, before `seeded_pools` replaces it.
_PICK_FROM_RUNTIME_VIEW = model_helpers.current_pool_config


class FakeUsageLedger:
    enabled = True

    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    async def start(self) -> None:
        return

    async def close(self) -> None:
        return

    async def write_event(self, event: dict[str, object]) -> None:
        self.events.append(event)


class FakeProviderLLM:
    """Answers with the name of the key setting it was built with."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def query(self, inputs: object, **kwargs: object) -> QueryResult:
        _ = inputs, kwargs
        return QueryResult(output_text=SETTING_FOR_VALUE[self.api_key], history=[])


BUILT_WITH: list[str] = []


def _build_provider_llm(_model: str, config: LLMConfig | None) -> FakeProviderLLM:
    assert config is not None and config.custom_api_key is not None
    BUILT_WITH.append(config.custom_api_key.get_secret_value())
    return FakeProviderLLM(config.custom_api_key.get_secret_value())


@pytest.fixture
def ledger() -> FakeUsageLedger:
    return FakeUsageLedger()


@pytest.fixture(autouse=True)
def pool_of_two(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv(KEY_1, VALUES[KEY_1])
    monkeypatch.setenv(KEY_2, VALUES[KEY_2])
    monkeypatch.delenv(KEY_3, raising=False)
    BUILT_WITH.clear()
    # The pools the seed script publishes for these keys, standing in for AppConfig.
    with (
        patch.object(
            model_helpers, "get_registry_model", side_effect=_build_provider_llm
        ),
        seeded_pools(),
    ):
        yield


def _worker(
    ledger: FakeUsageLedger, store: InMemoryAssignmentStore | None
) -> TestClient:
    """One Gateway worker; `store` is the shared table, None means no table is configured."""
    with patch.object(gateway_app, "create_usage_ledger_from_env", return_value=ledger):
        client = _make_client()
    client.app.state.key_assignments = store  # pyright: ignore[reportAttributeAccessIssue]
    return client


def _request_body(attempt: str, *, question: str | None = None) -> dict[str, Any]:
    return {
        "model": MODEL,
        "inputs": [{"kind": "text", "text": "hi"}],
        "run_id": RUN_ID,
        "question_id": f"question-{question or attempt}",
        "attempt_id": f"attempt-{attempt}",
    }


def _ask(
    worker: TestClient, attempt: str, *, question: str | None = None
) -> dict[str, Any]:
    response = worker.post(
        "/query", json=_request_body(attempt, question=question), headers=HEADERS
    )
    return {"status": response.status_code, **response.json()}


def _key_used(worker: TestClient, attempt: str, **kwargs: str) -> str:
    response = _ask(worker, attempt, **kwargs)
    assert response["status"] == 200, response
    return response["output_text"]


def test_each_attempt_keeps_its_key_across_workers_and_pool_changes(
    ledger: FakeUsageLedger, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = InMemoryAssignmentStore()
    worker_1 = _worker(ledger, store)
    worker_2 = _worker(ledger, store)

    # 1. Six attempts send their first request on worker 1, each to Key 1 or Key 2.
    first = {attempt: _key_used(worker_1, attempt) for attempt in ATTEMPTS}
    assert set(first.values()) == {KEY_1, KEY_2}

    # 2. Key 3 joins the pool.
    monkeypatch.setenv(KEY_3, VALUES[KEY_3])

    # 3. Each attempt's second request lands on worker 2 and keeps its key.
    second = {attempt: _key_used(worker_2, attempt) for attempt in ATTEMPTS}
    assert second == first

    # 4. A new attempt H may be assigned any key in the three-key pool.
    assert _key_used(worker_2, "H") == KEY_3

    # 5. A retry of question A is a new attempt, so it gets a fresh pick.
    assert first["A"] == KEY_2
    assert _key_used(worker_1, "A-retry", question="A") == KEY_3

    # 6. Key 2 is deleted; attempt F held it, so its next request fails clearly.
    assert first["F"] == KEY_2
    monkeypatch.delenv(KEY_2)
    failed = _ask(worker_2, "F")
    assert failed["status"] == 409
    assert failed["code"] == "assigned_key_unavailable"
    assert KEY_2 in failed["message"]
    assert VALUES[KEY_2] not in failed["message"]

    # 7. Every usage row names the key setting that served it, never its value.
    served = [*first.values(), *second.values(), KEY_3, KEY_3]
    assert [event["provider_key_name"] for event in ledger.events] == served
    assert not any(value in repr(ledger.events) for value in VALUES.values())


def test_the_provider_call_is_tagged_with_its_key_and_pools_revision(
    ledger: FakeUsageLedger,
) -> None:
    recorded: list[Mapping[str, object | None]] = []
    provider_call = GatewayOperation.provider_call

    async def recording_provider_call(
        self: GatewayOperation,
        awaitable: Awaitable[Any],
        *,
        span_attrs: Mapping[str, object | None],
    ) -> Any:
        recorded.append(span_attrs)
        return await provider_call(self, awaitable, span_attrs=span_attrs)

    document = seed_document(lambda key: key in VALUES, registry_endpoints())
    snapshot = PoolConfigSnapshot(
        digest="d",
        document=document.model_copy(update={"revision": 7}),
        configuration_version="7",
    )
    worker = _worker(ledger, InMemoryAssignmentStore())
    with (
        patch.object(GatewayOperation, "provider_call", recording_provider_call),
        patch.object(model_helpers, "runtime_pool_snapshot", lambda: snapshot),
        patch.object(model_helpers, "current_pool_config", _PICK_FROM_RUNTIME_VIEW),
    ):
        key = _key_used(worker, "A")

    [attrs] = recorded
    assert (attrs["gateway.provider_key_name"], attrs["gateway.pool_revision"]) == (
        key,
        7,
    )
    assert VALUES[key] not in repr(attrs)


def test_without_a_table_five_of_six_attempts_move_when_key_3_is_added(
    ledger: FakeUsageLedger, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker_1 = _worker(ledger, None)
    worker_2 = _worker(ledger, None)

    first = {attempt: _key_used(worker_1, attempt) for attempt in ATTEMPTS}
    monkeypatch.setenv(KEY_3, VALUES[KEY_3])
    second = {attempt: _key_used(worker_2, attempt) for attempt in ATTEMPTS}

    moved = [attempt for attempt in ATTEMPTS if second[attempt] != first[attempt]]
    assert moved == ["A", "B", "D", "E", "F"]
    assert all(event["provider_key_name"] is None for event in ledger.events)
    assert all(event["pool_revision"] is None for event in ledger.events)


def test_a_first_request_logs_the_pools_revision_that_picked_its_key(
    ledger: FakeUsageLedger,
) -> None:
    document = seed_document(lambda key: key in VALUES, registry_endpoints())
    snapshot = PoolConfigSnapshot(
        digest="d",
        document=document.model_copy(update={"revision": 7}),
        configuration_version="7",
    )
    # Exits before `pool_of_two` restores the real pick, so nothing leaks.
    with (
        patch.object(model_helpers, "runtime_pool_snapshot", lambda: snapshot),
        patch.object(model_helpers, "current_pool_config", _PICK_FROM_RUNTIME_VIEW),
    ):
        key = _key_used(_worker(ledger, InMemoryAssignmentStore()), "A")

    [event] = ledger.events
    assert (event["provider_key_name"], event["pool_revision"]) == (key, 7)


def test_a_held_key_is_logged_with_the_pools_revision_active_for_each_request(
    ledger: FakeUsageLedger,
) -> None:
    document = seed_document(lambda key: key in VALUES, registry_endpoints())
    active = PoolConfigSnapshot(
        digest="d7",
        document=document.model_copy(update={"revision": 7}),
        configuration_version="7",
    )
    worker = _worker(ledger, InMemoryAssignmentStore())
    with (
        patch.object(model_helpers, "runtime_pool_snapshot", lambda: active),
        patch.object(model_helpers, "current_pool_config", _PICK_FROM_RUNTIME_VIEW),
    ):
        # Revision 7 picks the key; the snapshot then refreshes to revision 8.
        first_key = _key_used(worker, "A")
        active = PoolConfigSnapshot(
            digest="d8",
            document=document.model_copy(update={"revision": 8}),
            configuration_version="8",
        )
        second_key = _key_used(worker, "A")

    assert second_key == first_key
    assert [(e["provider_key_name"], e["pool_revision"]) for e in ledger.events] == [
        (first_key, 7),
        (first_key, 8),
    ]


def test_adding_the_table_moves_nobody(ledger: FakeUsageLedger) -> None:
    without_table = _worker(ledger, None)
    with_table = _worker(ledger, InMemoryAssignmentStore())

    assert {a: _key_used(with_table, a) for a in ATTEMPTS} == {
        a: _key_used(without_table, a) for a in ATTEMPTS
    }


class _WritesTogetherStore(InMemoryAssignmentStore):
    """Holds every writer until both workers have picked, so both write a pick."""

    def __init__(self) -> None:
        super().__init__()
        self._writes = 0
        self._both_picked = asyncio.Event()

    async def put_if_absent(
        self, attempt_key: AttemptKey, model: str, key_name: str
    ) -> str:
        self._writes += 1
        if self._writes >= 2:
            self._both_picked.set()
        await self._both_picked.wait()
        return await super().put_if_absent(attempt_key, model, key_name)


async def test_two_workers_racing_on_a_first_request_use_one_key(
    ledger: FakeUsageLedger,
) -> None:
    store = _WritesTogetherStore()
    workers = [_worker(ledger, store), _worker(ledger, store)]

    async def ask(worker: TestClient) -> str:
        transport = httpx.ASGITransport(app=worker.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as http:
            response = await http.post(
                "/query", json=_request_body("X"), headers=HEADERS
            )
        assert response.status_code == 200, response.text
        return response.json()["output_text"]

    # Each worker picks a different key for attempt X before either writes.
    with patch.object(
        model_helpers, "_pick_managed_key_setting", side_effect=[KEY_1, KEY_2]
    ):
        used = await asyncio.gather(*(ask(worker) for worker in workers))

    assert used[0] == used[1]
    held = store._assignments[(("attempt", RUN_ID, "attempt-X"), MODEL)]  # pyright: ignore[reportPrivateUsage]
    assert held == used[0]


class _ThrottledTable:
    async def put_item(self, **kwargs: object) -> dict[str, Any]:
        raise ClientError(
            {"Error": {"Code": "ProvisionedThroughputExceededException"}},  # pyright: ignore[reportArgumentType]
            "PutItem",
        )


_TABLE_ENV = {"GATEWAY_KEY_ASSIGNMENTS_TABLE_NAME": "key-assignments"}


def _worker_from_env(ledger: FakeUsageLedger, table: object) -> TestClient:
    """A worker whose store comes from the real table-name env, backed by a fake table."""

    def store_for_table(*, table_name: str) -> DynamoAssignmentStore:
        return DynamoAssignmentStore(table_name=table_name, client=table)  # pyright: ignore[reportArgumentType]

    with (
        patch.dict(os.environ, _TABLE_ENV),
        patch.object(key_assignments, "DynamoAssignmentStore", store_for_table),
        patch.object(gateway_app, "create_usage_ledger_from_env", return_value=ledger),
    ):
        return _make_client()


def test_the_table_name_builds_the_store_and_a_query_saves_its_key(
    ledger: FakeUsageLedger,
) -> None:
    table = FakeDynamoTable()
    worker = _worker_from_env(ledger, table)

    key = _key_used(worker, "A")

    assert isinstance(worker.app.state.key_assignments, DynamoAssignmentStore)  # pyright: ignore[reportAttributeAccessIssue]
    assert [row["key_name"]["S"] for row in table.items.values()] == [key]


def test_an_unreachable_table_fails_the_query_and_uses_no_key(
    ledger: FakeUsageLedger,
) -> None:
    worker = _worker_from_env(ledger, _ThrottledTable())

    response = _ask(worker, "A")

    assert response["status"] == 500, response
    assert response["code"] == "internal_error", response
    assert BUILT_WITH == []
    assert ledger.events == []
