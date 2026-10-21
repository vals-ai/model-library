"""Focused cases for the sticky key lookup behind GATEWAY_KEY_ASSIGNMENTS_TABLE_NAME."""

import hashlib
import os
from unittest.mock import patch

import pytest
from pydantic import SecretStr
from starlette.testclient import TestClient

import model_gateway.app as gateway_app
import model_gateway.model_helpers as model_helpers
from model_gateway.errors import AssignedKeyUnavailableError, map_exception_to_error
from model_gateway.key_assignments import (
    AttemptKey,
    DynamoAssignmentStore,
    create_key_assignment_store_from_env,
)
from model_gateway.types import QueryRequest
from model_gateway.key_pools import ModelRule, PoolConfig
from tests.unit.model_gateway._seeded_pools import seeded_pools
from tests.unit.model_gateway._support import InMemoryAssignmentStore
from model_library.base import FileWithId, LLMConfig
from model_library import model_library_settings

MODEL = "anthropic/claude-opus-4-6"
KEY_1, KEY_2, KEY_3 = "ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_2", "ANTHROPIC_API_KEY_3"


@pytest.fixture(autouse=True)
def _pools_from_the_seed():
    """Pools as the seed script publishes them for the keys each test sets."""
    with seeded_pools():
        yield


@pytest.fixture(autouse=True)
def two_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_1, "value-of-key-1")
    monkeypatch.setenv(KEY_2, "value-of-key-2")
    monkeypatch.delenv(KEY_3, raising=False)


def _body(question: str = "question-A", **overrides: object) -> QueryRequest:
    fields: dict[str, object] = {
        "model": MODEL,
        "inputs": [],
        "run_id": "run-5",
        "question_id": question,
        "attempt_id": f"attempt-{question}",
        **overrides,
    }
    return QueryRequest.model_validate(fields)


def _todays_pick(body: QueryRequest) -> str | None:
    return model_helpers._pick_managed_key_setting(  # pyright: ignore[reportPrivateUsage]
        body, model_helpers.current_pool_config()
    )


HELD = ("attempt", "run-5", "attempt-question-A")


async def test_second_writer_of_a_race_uses_the_first_writers_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryAssignmentStore()
    body = _body("question-A", attempt_id="attempt-1")

    first = await model_helpers.assigned_key_setting(body, store)
    # Key 3 arrives between the two writes, so the losing worker would pick Key 3.
    monkeypatch.setenv(KEY_3, "value-of-key-3")
    assert _todays_pick(body) == KEY_3
    second = await model_helpers.assigned_key_setting(body, store)

    assert first == second == KEY_1


@pytest.mark.parametrize("question", [f"question-{letter}" for letter in "ABCDEFG"])
async def test_first_assignment_matches_todays_pick(question: str) -> None:
    body = _body(question)
    expected = _todays_pick(body)

    assert await model_helpers.assigned_key_setting(body, InMemoryAssignmentStore()) == (
        expected
    )


def _pick_before_attempts(body: QueryRequest) -> str:
    pool = (KEY_1, KEY_2)
    digest = hashlib.sha256(f"{body.run_id}{body.question_id}".encode()).digest()
    return pool[int.from_bytes(digest, "big") % len(pool)]


async def test_a_retry_of_the_same_question_can_land_on_another_key() -> None:
    store = InMemoryAssignmentStore()
    failed, retry = _body(attempt_id="attempt-1"), _body(attempt_id="attempt-2")
    # Run and question alone put both attempts on one key.
    assert _pick_before_attempts(failed) == _pick_before_attempts(retry) == KEY_1

    keys = [await model_helpers.assigned_key_setting(b, store) for b in (failed, retry)]

    assert keys == [KEY_1, KEY_2]


async def test_a_request_without_an_attempt_keeps_the_pick_before_attempts() -> None:
    body = _body(attempt_id=None)

    assert _todays_pick(body) == _pick_before_attempts(body)


async def test_the_same_attempt_id_in_another_run_is_another_attempt() -> None:
    store = InMemoryAssignmentStore()
    await store.put_if_absent(HELD, MODEL, KEY_2)

    await model_helpers.assigned_key_setting(_body(run_id="other-run"), store)

    other = ("attempt", "other-run", "attempt-question-A")
    assert (other, MODEL) in store._assignments  # pyright: ignore[reportPrivateUsage]
    assert store._assignments[(HELD, MODEL)] == KEY_2  # pyright: ignore[reportPrivateUsage]


async def test_attempt_id_without_a_run_id_is_no_attempt() -> None:
    store = InMemoryAssignmentStore()
    body = _body(run_id=None)

    key = await model_helpers.assigned_key_setting(body, store)

    assert key == _todays_pick(body)
    assert store._assignments == {}  # pyright: ignore[reportPrivateUsage]


async def test_request_without_an_attempt_id_gets_todays_pick_unsaved() -> None:
    store = InMemoryAssignmentStore()
    body = _body(attempt_id=None)

    key = await model_helpers.assigned_key_setting(body, store)

    assert key == _todays_pick(body)
    assert store._assignments == {}  # pyright: ignore[reportPrivateUsage]


class UnreachableStore(InMemoryAssignmentStore):
    async def get(self, attempt_key: AttemptKey, model: str) -> str | None:
        raise AssertionError("the store was used")

    async def put_if_absent(
        self, attempt_key: AttemptKey, model: str, key_name: str
    ) -> str:
        raise AssertionError("the store was used")


async def test_models_without_a_managed_pool_never_read_the_store() -> None:
    # Google resolves its own credentials, so it never has a managed pool.
    body = _body(model="google/gemini-2.5-pro")

    assert await model_helpers.assigned_key_setting(body, UnreachableStore()) is None


@pytest.mark.parametrize(
    "body",
    [
        _body(config=LLMConfig(custom_api_key=SecretStr("byok"))),
        _body(
            inputs=[
                FileWithId(
                    type="file",
                    file_id="file-1",
                    name="document.pdf",
                    mime="application/pdf",
                )
            ]
        ),
        _body(model="openai/gpt-4o"),
    ],
    ids=["custom-api-key", "file-input", "no-managed-pool"],
)
async def test_requests_outside_managed_keys_record_nothing(body: QueryRequest) -> None:
    store = InMemoryAssignmentStore()

    assert await model_helpers.assigned_key_setting(body, store) is None
    assert store._assignments == {}  # pyright: ignore[reportPrivateUsage]


async def test_deleted_held_key_fails_without_switching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryAssignmentStore()
    await store.put_if_absent(HELD, MODEL, KEY_2)
    monkeypatch.delenv(KEY_2)

    with pytest.raises(AssignedKeyUnavailableError, match=KEY_2):
        await model_helpers.assigned_key_setting(_body("question-A"), store)
    assert store._assignments[(HELD, MODEL)] == KEY_2  # pyright: ignore[reportPrivateUsage]


def test_unavailable_key_maps_to_conflict() -> None:
    error = map_exception_to_error(
        AssignedKeyUnavailableError(key_name=KEY_2, attempt_key=("a",), model=MODEL)
    )

    assert error.status_code == 409
    assert error.body.code == "assigned_key_unavailable"


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, None),
        ({"GATEWAY_KEY_ASSIGNMENTS_TABLE_NAME": "table"}, DynamoAssignmentStore),
    ],
    ids=["unset", "table"],
)
def test_store_exists_when_the_table_is_named(
    env: dict[str, str], expected: type | None
) -> None:
    with patch.dict(os.environ, env, clear=True):
        store = create_key_assignment_store_from_env()

    assert (store is None) if expected is None else isinstance(store, expected)


def test_lifespan_starts_and_closes_the_store() -> None:
    from model_gateway import main

    class ServerSettings:
        MODEL_GATEWAY_API_KEYS = '{"test":"sk-test"}'
        MODEL_GATEWAY_HMAC_SECRET = "test-secret"

        def get(self, name: str, default: str = "") -> str:
            return getattr(self, name, default)

        def unset(self, key: str) -> None:
            pass

    class FakeStore:
        started = 0
        closed = 0

        async def start(self) -> None:
            self.started += 1

        async def close(self) -> None:
            self.closed += 1

    store = FakeStore()
    with (
        patch.dict(
            os.environ,
            {"GATEWAY_RUNTIME_ROLE": "query", "GATEWAY_STARTUP_CANARY_ENABLED": "false"},
        ),
        patch.object(
            gateway_app, "create_key_assignment_store_from_env", return_value=store
        ),
        patch.object(gateway_app, "get_model_names", return_value=[MODEL]),
        patch.object(gateway_app, "model_library_settings", ServerSettings()),
    ):
        app = main.create_app()
        with TestClient(app):
            assert (store.started, store.closed) == (1, 0)

    assert (store.started, store.closed) == (1, 1)


def _weighted(monkeypatch: pytest.MonkeyPatch, weights: dict[str, float]) -> None:
    seeded = model_helpers.current_pool_config()
    weighted = PoolConfig(pools=seeded.pools, rules=seeded.rules, weights=weights)
    monkeypatch.setattr(model_helpers, "current_pool_config", lambda: weighted)


async def test_a_zero_weight_key_receives_no_new_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _weighted(monkeypatch, {KEY_1: 0})

    keys = {
        await model_helpers.assigned_key_setting(
            _body(f"question-{index}"), InMemoryAssignmentStore()
        )
        for index in range(40)
    }

    assert keys == {KEY_2}


async def test_weights_split_new_attempts_in_proportion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _weighted(monkeypatch, {KEY_1: 3, KEY_2: 1})

    picks = [
        await model_helpers.assigned_key_setting(
            _body(f"question-{index}"), InMemoryAssignmentStore()
        )
        for index in range(400)
    ]

    assert 260 <= picks.count(KEY_1) <= 340


@pytest.mark.parametrize("question", [f"question-{letter}" for letter in "ABCDEFG"])
async def test_equal_weights_pick_todays_key(
    monkeypatch: pytest.MonkeyPatch, question: str
) -> None:
    body = _body(question)
    expected = _todays_pick(body)
    _weighted(monkeypatch, {KEY_1: 2, KEY_2: 2})

    assert await model_helpers.assigned_key_setting(body, InMemoryAssignmentStore()) == (
        expected
    )


def _serve_from(monkeypatch: pytest.MonkeyPatch, config: PoolConfig) -> None:
    monkeypatch.setattr(model_helpers, "current_pool_config", lambda: config)


@pytest.mark.parametrize(
    "attempt_id", ["attempt-question-A", None], ids=["attempt", "no-attempt"]
)
async def test_a_picked_key_with_no_value_fails_clearly(
    monkeypatch: pytest.MonkeyPatch, attempt_id: str | None
) -> None:
    # The pools document names Key 3, but this stage holds no value for it.
    _serve_from(monkeypatch, PoolConfig(pools={"anthropic/default": (KEY_3,)}))

    with pytest.raises(AssignedKeyUnavailableError, match=KEY_3):
        await model_helpers.assigned_key_setting(
            _body(attempt_id=attempt_id), InMemoryAssignmentStore()
        )


async def test_an_emptied_pin_pool_fails_attempts_that_hold_a_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryAssignmentStore()
    await store.put_if_absent(HELD, MODEL, KEY_2)
    pin = ModelRule("anthropic/pin", models=frozenset({MODEL}))
    _serve_from(monkeypatch, PoolConfig(pools={"anthropic/pin": ()}, rules=(pin,)))

    with pytest.raises(KeyError, match="anthropic/pin"):
        await model_helpers.assigned_key_setting(_body("question-A"), store)
