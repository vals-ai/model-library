"""Key pools, operator path: a pool change made through the control API reaches /query.

One control app and two query workers run the real FastAPI apps. They share one
assignment store, the way deployed workers share the DynamoDB table, and one
in-memory AppConfig, the way deployed tasks share the `pools` profile.
The operator is a Dashboards-style caller holding the `dashboards` Gateway key.
"""

import asyncio
import dataclasses
import json
import os
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from starlette.testclient import TestClient

import model_gateway.app as gateway_app
import model_gateway.model_helpers as model_helpers
import model_gateway.routes.key_pools as key_pool_routes
from scripts.seed_key_pools import registry_endpoints, seed_document
from model_gateway.runtime_config import (
    AppConfigAgentClient,
    RuntimeConfigManager,
    RuntimeConfigState,
)
from model_library.base import LLMConfig, QueryResult
from tests.unit.model_gateway._support import InMemoryAssignmentStore, _make_client

MODEL = "anthropic/claude-opus-4-6"
RUN_ID = "run-7"
POOL_PROFILE = "pools"
ACCESS_PROFILE = "access"

KEY_1, KEY_2, KEY_3 = "ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_2", "ANTHROPIC_API_KEY_3"
EAP_KEY = "ANTHROPIC_API_KEY_EAP_9_18"
VALUES = {
    KEY_1: "value-of-key-1",
    KEY_2: "value-of-key-2",
    KEY_3: "value-of-key-3",
    EAP_KEY: "value-of-eap-key",
}
SETTING_FOR_VALUE = {value: name for name, value in VALUES.items()}

OPERATOR = {"Authorization": "Bearer sk-dashboards"}
HARNESS = {"Authorization": "Bearer sk-harness"}
API_KEYS = {"dashboards": "sk-dashboards", "harness": "sk-harness"}
def access_document(names: list[str]) -> bytes:
    return json.dumps({"admin_api_key_names": names}).encode()


class InMemoryAppConfig:
    """One AppConfig profile: the control app publishes, the workers' agents read."""

    def __init__(self, *, label_lags: bool = False) -> None:
        # The stage's seed, published before any task starts, as revision 0.
        self.seed = (
            seed_document(lambda key: bool(os.environ.get(key)), registry_endpoints())
            .model_copy(update={"revision": 0})
            .model_dump_json()
            .encode()
        )
        self.versions: list[bytes] = []
        # Seen live: the Agent can serve a new body under the previous label.
        self.label_lags = label_lags

    async def publish(self, content: bytes) -> str:
        self.versions.append(content)
        return str(len(self.versions))

    def _label(self) -> int:
        return max(1, len(self.versions) - 1) if self.label_lags else len(self.versions)

    def agent(self) -> AppConfigAgentClient:
        def handle(request: httpx.Request) -> httpx.Response:
            # The provider secret serves the stage's keys, the pools profile its
            # document; models stay on the bootstrap registry.
            if request.url.path.endswith("/configurations/credentials"):
                return httpx.Response(
                    200,
                    content=json.dumps(
                        {name: os.environ[name] for name in VALUES if name in os.environ}
                    ).encode(),
                    headers={"Configuration-Version": "1"},
                )
            if request.url.path.endswith(f"/configurations/{ACCESS_PROFILE}"):
                return httpx.Response(
                    200,
                    content=access_document(["dashboards"]),
                    headers={"Configuration-Version": "1"},
                )
            if not request.url.path.endswith(f"/configurations/{POOL_PROFILE}"):
                return httpx.Response(404)
            return httpx.Response(
                200,
                content=self.versions[-1] if self.versions else self.seed,
                headers={"Configuration-Version": str(self._label())},
            )

        return AppConfigAgentClient(
            base_url="http://127.0.0.1:2772",
            application="model-gateway",
            environment="test",
            access_token="local-token",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
        )


class FakeProviderLLM:
    """Answers with the name of the key setting it was built with."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def query(self, inputs: object, **kwargs: object) -> QueryResult:
        _ = inputs, kwargs
        return QueryResult(output_text=SETTING_FOR_VALUE[self.api_key], history=[])


def _build_provider_llm(_model: str, config: LLMConfig | None) -> FakeProviderLLM:
    assert config is not None and config.custom_api_key is not None
    return FakeProviderLLM(config.custom_api_key.get_secret_value())


@pytest.fixture(autouse=True)
def clock_at_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Revisions count up from the task's view unless a test moves the clock."""
    monkeypatch.setattr(key_pool_routes, "time_ns", lambda: 0)


@pytest.fixture(autouse=True)
def three_anthropic_keys(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name, value in VALUES.items():
        monkeypatch.setenv(name, value)
    with patch.object(
        model_helpers, "get_registry_model", side_effect=_build_provider_llm
    ):
        yield


def _app(
    appconfig: InMemoryAppConfig, role: str, store: InMemoryAssignmentStore | None
) -> TestClient:
    built: list[RuntimeConfigManager] = []

    def manager(state: RuntimeConfigState) -> RuntimeConfigManager:
        built.append(
            RuntimeConfigManager(
                state=state,
                agent=appconfig.agent(),
                model_profile="models",
                credential_profile="credentials",
                pool_profile=POOL_PROFILE,
                access_profile=ACCESS_PROFILE if role == "control" else None,
            )
        )
        return built[-1]

    with patch.object(gateway_app, "runtime_config_manager_from_env", manager):
        client = _make_client(runtime_role=role, api_keys=API_KEYS)
    client.app.state.runtime_config_manager = built[-1]  # pyright: ignore[reportAttributeAccessIssue]
    client.app.state.key_assignments = store  # pyright: ignore[reportAttributeAccessIssue]
    client.app.state.pool_config_publisher = appconfig  # pyright: ignore[reportAttributeAccessIssue]
    # A deployed task starts only after its first refresh serves a pools document.
    _refresh(client)
    return client


def _refresh(*apps: TestClient) -> None:
    """The refresh each task's background loop runs every five seconds."""
    for app in apps:
        manager: RuntimeConfigManager = app.app.state.runtime_config_manager  # pyright: ignore[reportAttributeAccessIssue]
        result = asyncio.run(manager.refresh_once())
        assert result.failed == {"models"} and not result.rejected


def _key_used(worker: TestClient, attempt: str, model: str = MODEL) -> str:
    response = worker.post(
        "/query",
        json={
            "model": model,
            "inputs": [{"kind": "text", "text": "hi"}],
            "run_id": RUN_ID,
            "question_id": f"question-{attempt}",
            "attempt_id": f"attempt-{attempt}",
        },
        headers=HARNESS,
    )
    assert response.status_code == 200, response.text
    return response.json()["output_text"]


def _pools(*anthropic_default: str, **extra: list[str]) -> dict[str, Any]:
    return {
        "pools": {"anthropic/default": list(anthropic_default), **extra},
        "model_pools": [],
        "weights": {},
        "descriptions": {},
    }


def test_operator_change_reaches_new_attempts_and_spares_running_ones() -> None:
    appconfig = InMemoryAppConfig()
    store = InMemoryAssignmentStore()
    control = _app(appconfig, "control", None)
    worker_1 = _app(appconfig, "query", store)
    worker_2 = _app(appconfig, "query", store)
    put = control.put("/admin/key-pools", json=_pools(KEY_1, KEY_2), headers=OPERATOR)
    assert put.status_code == 200, put.text
    _refresh(control, worker_1, worker_2)

    # a. Attempts A-C start on worker 1 and get keys from anthropic/default.
    first = {attempt: _key_used(worker_1, attempt) for attempt in "ABC"}
    assert set(first.values()) <= {KEY_1, KEY_2}

    # b. The operator adds Key 3 to anthropic/default; a new version is published.
    put = control.put(
        "/admin/key-pools", json=_pools(KEY_1, KEY_2, KEY_3), headers=OPERATOR
    )
    assert (put.status_code, put.json()) == (200, {"version": "2"})

    # c. Every task refreshes, as its background loop would within five seconds.
    _refresh(control, worker_1, worker_2)

    # d. A-C keep their keys on worker 2; a new attempt can get Key 3.
    assert {attempt: _key_used(worker_2, attempt) for attempt in "ABC"} == first
    assert KEY_3 in {_key_used(worker_2, attempt) for attempt in "DEFGHIJKL"}

    # e. Mapping a public model to a pool holding an internal key is refused,
    #    and nothing is published.
    refused = control.put(
        "/admin/key-pools",
        json={
            **_pools(KEY_1, KEY_2, KEY_3, **{"anthropic/eap": [EAP_KEY]}),
            "model_pools": [{"pool": "anthropic/eap", "models": [MODEL]}],
        },
        headers=OPERATOR,
    )
    # The Dashboards contract: 422 with only the errors list.
    assert refused.status_code == 422
    assert refused.json() == {
        "errors": [
            {
                "model": MODEL,
                "pool": "anthropic/eap",
                "reason": "internal_key_for_public_model",
            }
        ]
    }
    assert len(appconfig.versions) == 2

    # f. A pooled provider with no default pool is refused, and nothing is published.
    refused = control.put(
        "/admin/key-pools",
        json={**_pools(), "pools": {"anthropic/second": [KEY_2]}},
        headers=OPERATOR,
    )
    assert refused.status_code == 422
    assert refused.json()["errors"] == [
        {"model": "", "pool": "anthropic/default", "reason": "missing_default_pool"}
    ]
    assert len(appconfig.versions) == 2

    # g. GET shows key names, providers and internal flags, never a key value.
    current = control.get("/admin/key-pools", headers=OPERATOR)
    assert current.status_code == 200
    body = current.json()
    assert body["version"] == "2"
    assert body["pools"] == {"anthropic/default": [KEY_1, KEY_2, KEY_3]}
    keys = {key["name"]: key for key in body["keys"]}
    assert keys[KEY_3] == {"name": KEY_3, "provider": "anthropic", "internal": False}
    assert keys[EAP_KEY]["internal"] is True
    assert not any(value in current.text for value in VALUES.values())

    # Only an admin key may change pools, and only control tasks serve the route.
    assert (
        control.put("/admin/key-pools", json=_pools(KEY_1), headers=HARNESS).status_code
        == 403
    )
    assert worker_1.get("/admin/key-pools", headers=OPERATOR).status_code == 404


def test_the_seeded_document_round_trips_through_get_and_put(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("META_API_KEY", "value-of-meta-key")
    # A deployed stage sets every key its pins name.
    pinned = seed_document(lambda key: key in VALUES, registry_endpoints()).pools
    for pool, keys in pinned.items():
        if not keys:
            monkeypatch.setenv(_PIN_KEYS[pool], f"value-of-{pool}")
    appconfig = InMemoryAppConfig()
    control = _app(appconfig, "control", None)
    _refresh(control)

    body = control.get("/admin/key-pools", headers=OPERATOR).json()

    assert body["version"] == "0"
    assert body["pools"]["anthropic/default"] == [KEY_1, KEY_2, KEY_3]
    # The seeded document, sent back unchanged, passes validation.
    fields = ("pools", "model_pools", "weights", "descriptions")
    document = {field: body[field] for field in fields}
    put = control.put("/admin/key-pools", json=document, headers=OPERATOR)
    assert put.status_code == 200, put.text


def test_a_pool_naming_a_setting_that_is_not_a_key_is_refused() -> None:
    appconfig = InMemoryAppConfig()
    control = _app(appconfig, "control", None)

    secret = control.put(
        "/admin/key-pools",
        json=_pools(KEY_1, "MODEL_GATEWAY_HMAC_SECRET"),
        headers=OPERATOR,
    )

    assert {"model": "", "pool": "anthropic/default", "reason": "unknown_key"} in (
        secret.json()["errors"]
    )
    assert appconfig.versions == []


def test_descriptions_of_a_pool_and_a_key_are_served_back() -> None:
    appconfig = InMemoryAppConfig()
    control = _app(appconfig, "control", None)
    descriptions = {"anthropic/default": "every Anthropic model", KEY_1: "our main key"}

    put = control.put(
        "/admin/key-pools",
        json={**_pools(KEY_1, KEY_2), "descriptions": descriptions},
        headers=OPERATOR,
    )
    assert put.status_code == 200, put.text
    _refresh(control)

    body = control.get("/admin/key-pools", headers=OPERATOR).json()
    assert body["descriptions"] == descriptions


def test_a_put_without_descriptions_is_refused_and_keeps_them() -> None:
    appconfig = InMemoryAppConfig()
    control = _app(appconfig, "control", None)
    descriptions = {KEY_1: "our main key"}
    control.put(
        "/admin/key-pools",
        json={**_pools(KEY_1), "descriptions": descriptions},
        headers=OPERATOR,
    )
    _refresh(control)
    without = _pools(KEY_1, KEY_2)
    del without["descriptions"]

    put = control.put("/admin/key-pools", json=without, headers=OPERATOR)

    assert put.status_code == 400, put.text
    assert put.json()["code"] == "invalid_request"
    assert len(appconfig.versions) == 1
    body = control.get("/admin/key-pools", headers=OPERATOR).json()
    assert body["descriptions"] == descriptions


def test_a_description_for_an_unknown_name_is_refused() -> None:
    appconfig = InMemoryAppConfig()
    control = _app(appconfig, "control", None)

    put = control.put(
        "/admin/key-pools",
        json={**_pools(KEY_1), "descriptions": {KEY_2: "not pooled"}},
        headers=OPERATOR,
    )

    assert put.status_code == 422
    assert put.json()["errors"] == [
        {"model": "", "pool": KEY_2, "reason": "unknown_description"}
    ]
    assert appconfig.versions == []


def _revision(content: bytes) -> int:
    return json.loads(content)["revision"]


def test_each_put_publishes_the_next_revision() -> None:
    appconfig = InMemoryAppConfig()
    control = _app(appconfig, "control", None)

    first = control.put("/admin/key-pools", json=_pools(KEY_1, KEY_2), headers=OPERATOR)
    _refresh(control)
    second = control.put(
        "/admin/key-pools", json=_pools(KEY_1, KEY_2, KEY_3), headers=OPERATOR
    )

    assert [_revision(content) for content in appconfig.versions] == [1, 2]
    assert (first.json(), second.json()) == ({"version": "1"}, {"version": "2"})


def test_a_new_body_under_a_lagging_label_is_applied() -> None:
    appconfig = InMemoryAppConfig(label_lags=True)
    control = _app(appconfig, "control", None)
    worker = _app(appconfig, "query", InMemoryAssignmentStore())
    control.put("/admin/key-pools", json=_pools(KEY_1), headers=OPERATOR)
    _refresh(control, worker)

    # The second body arrives still labelled "1".
    control.put("/admin/key-pools", json=_pools(KEY_3), headers=OPERATOR)
    _refresh(control, worker)

    assert {_key_used(worker, attempt) for attempt in "ABC"} == {KEY_3}
    body = control.get("/admin/key-pools", headers=OPERATOR).json()
    assert (body["version"], body["pools"]["anthropic/default"]) == ("2", [KEY_3])


def test_an_older_revision_is_rejected() -> None:
    appconfig = InMemoryAppConfig()
    worker = _app(appconfig, "query", InMemoryAssignmentStore())
    newer = {**_pools(KEY_3), "revision": 2}
    older = {**_pools(KEY_1), "revision": 1}
    appconfig.versions.append(json.dumps(newer).encode())
    _refresh(worker)

    appconfig.versions.append(json.dumps(older).encode())
    manager: RuntimeConfigManager = worker.app.state.runtime_config_manager  # pyright: ignore[reportAttributeAccessIssue]
    result = asyncio.run(manager.refresh_once())

    assert result.rejected == {"pools"}
    assert _key_used(worker, "A") == KEY_3


def test_two_puts_before_a_refresh_publish_increasing_revisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second PUT made before the control task sees the first still outranks it."""
    ticks = iter([1_000_000_000, 2_000_000_000])
    monkeypatch.setattr(key_pool_routes, "time_ns", lambda: next(ticks))
    appconfig = InMemoryAppConfig()
    control = _app(appconfig, "control", None)

    control.put("/admin/key-pools", json=_pools(KEY_1), headers=OPERATOR)
    control.put("/admin/key-pools", json=_pools(KEY_3), headers=OPERATOR)

    first, second = (_revision(content) for content in appconfig.versions)
    assert first < second


def test_a_pool_naming_an_unset_key_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(EAP_KEY)
    appconfig = InMemoryAppConfig()
    control = _app(appconfig, "control", None)

    listed = control.get("/admin/key-pools", headers=OPERATOR).json()["keys"]
    put = control.put(
        "/admin/key-pools",
        json=_pools(KEY_1, KEY_2, **{"anthropic/eap_9_18": [EAP_KEY]}),
        headers=OPERATOR,
    )

    assert EAP_KEY not in {key["name"] for key in listed}
    assert put.status_code == 422, put.text
    assert {error["reason"] for error in put.json()["errors"]} == {"unknown_key"}
    assert appconfig.versions == []


def test_a_pool_whose_weights_are_all_zero_is_refused() -> None:
    appconfig = InMemoryAppConfig()
    control = _app(appconfig, "control", None)

    put = control.put(
        "/admin/key-pools",
        json={**_pools(KEY_1), "weights": {KEY_1: 0}},
        headers=OPERATOR,
    )

    assert put.status_code == 422, put.text
    assert put.json()["errors"] == [
        {"model": "", "pool": "anthropic/default", "reason": "empty_pool"}
    ]
    assert appconfig.versions == []


def test_both_routes_refuse_without_a_served_pools_document() -> None:
    appconfig = InMemoryAppConfig()
    control = _app(appconfig, "control", None)
    state: RuntimeConfigState = control.app.state.runtime_config_manager.state  # pyright: ignore[reportAttributeAccessIssue]
    state._view = dataclasses.replace(state.current, pools=None)  # pyright: ignore[reportPrivateUsage]

    got = control.get("/admin/key-pools", headers=OPERATOR)
    put = control.put("/admin/key-pools", json=_pools(KEY_1), headers=OPERATOR)

    assert (got.status_code, got.json()["code"]) == (503, "pool_config_unavailable")
    assert (put.status_code, put.json()["code"]) == (503, "pool_config_unavailable")
    assert appconfig.versions == []


def test_an_infinite_weight_is_refused_before_publishing() -> None:
    appconfig = InMemoryAppConfig()
    control = _app(appconfig, "control", None)

    put = control.put(
        "/admin/key-pools",
        content=b'{"pools": {"anthropic/default": ["ANTHROPIC_API_KEY"]},'
        b' "model_pools": [], "weights": {"ANTHROPIC_API_KEY": Infinity}}',
        headers={**OPERATOR, "Content-Type": "application/json"},
    )

    assert put.status_code == 400, put.text
    assert appconfig.versions == []
