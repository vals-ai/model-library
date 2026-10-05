"""Behavioral contract for sandbox-scoped gateway credentials."""

import asyncio
import hashlib
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import fakeredis.aioredis
import pytest

import model_gateway.model_helpers as model_helpers
import model_gateway.routes.query as query_routes
import model_gateway.routes.run_tokens as run_token_routes
from model_gateway.errors import RunTokenAuthorizationError
from model_gateway.run_tokens import RunTokenClaims, run_token_authorized
from model_gateway.types import EmbeddingRequest
from model_library.base import dump_llm_config
from model_library.retriers.token import utils as token_utils
from tests.unit.model_gateway._support import HEADERS, _make_client

MODEL = "openai/gpt-4o"
OTHER = "anthropic/claude-4-opus"
EMBEDDING = "text-embedding-3-small"
RUN_ID = "run-1"
TASK_ID = "task_0"


@pytest.fixture(autouse=True)
def redis() -> Any:
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    token_utils.set_redis_client(client)
    yield client
    token_utils.set_redis_client(None)


def _mint(
    client: Any, *, headers: dict[str, str] = HEADERS, **overrides: Any
) -> dict[str, Any]:
    payload = {
        "run_id": RUN_ID,
        "task_id": TASK_ID,
        "allowed_models": [MODEL],
        "ttl_seconds": 3600,
        **overrides,
    }
    response = client.post("/service-auth", headers=headers, json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _query(model: str = MODEL, **overrides: Any) -> dict[str, Any]:
    return {"model": model, "inputs": [{"kind": "text", "text": "hi"}], **overrides}


def _post_query(client: Any, token: str, **overrides: Any) -> Any:
    return client.post("/query", headers=_bearer(token), json=_query(**overrides))


def _capture_query(client: Any, token: str, **overrides: Any) -> Any:
    seen: dict[str, Any] = {}

    async def capture(_cache: Any, body: Any, **_kwargs: Any) -> Any:
        seen["body"] = body
        raise ValueError("provider intentionally unavailable")

    with patch.object(query_routes, "get_query_llm", capture):
        _post_query(client, token, **overrides)
    assert "body" in seen, "authorized request must reach provider boundary"
    return seen["body"]


def _claims(**overrides: Any) -> RunTokenClaims:
    return RunTokenClaims(
        run_id=RUN_ID,
        task_id=TASK_ID,
        allowed_models=[MODEL],
        allowed_embedding_models=[EMBEDDING],
        minted_by="executor",
        minted_with="minting-key-digest",
        **overrides,
    )


def _authorize_body(claims: RunTokenClaims, body: Any) -> Any:
    request = SimpleNamespace(state=SimpleNamespace(run_token_claims=claims))
    return asyncio.run(run_token_authorized(type(body))(request, body))


def test_minted_token_can_call_query_and_revoke_by_its_digest() -> None:
    client = _make_client()
    minted = _mint(client)
    token = minted["token"]
    assert minted["lease_id"] == hashlib.sha256(token.encode()).hexdigest()
    assert minted["expires_at"] == pytest.approx(time.time() + 3600, abs=60)
    _capture_query(client, token)

    revoked = client.post(
        "/service-auth/revoke", headers=HEADERS, json={"lease_id": minted["lease_id"]}
    )
    assert revoked.status_code == 200
    assert _post_query(client, token).status_code == 401
    assert client.post(
        "/service-auth/revoke", headers=HEADERS, json={"lease_id": minted["lease_id"]}
    ).status_code == 404


def test_revoke_that_loses_a_race_reports_not_found() -> None:
    client = _make_client()
    lease = {"lease_id": _mint(client)["lease_id"]}
    stale = asyncio.run(run_token_routes.get_run_token(lease["lease_id"]))
    assert client.post("/service-auth/revoke", headers=HEADERS, json=lease).status_code == 200

    # A concurrent revoke read the lease before the first one deleted it.
    with patch.object(run_token_routes, "get_run_token", AsyncMock(return_value=stale)):
        lost = client.post("/service-auth/revoke", headers=HEADERS, json=lease)
    assert lost.status_code == 404


def test_another_gateway_key_cannot_use_or_revoke_a_token() -> None:
    dev = _make_client(api_keys={"dev": "sk-dev"})
    prod = _make_client(api_keys={"prod": "sk-prod"})
    minted = _mint(dev, headers=_bearer("sk-dev"))
    assert _post_query(prod, minted["token"]).status_code == 401

    both = _make_client(api_keys={"dev": "sk-dev", "prod": "sk-prod"})
    denied = both.post(
        "/service-auth/revoke",
        headers=_bearer("sk-prod"),
        json={"lease_id": minted["lease_id"]},
    )
    assert denied.status_code == 403
    _capture_query(dev, minted["token"])


def test_only_assigned_models_reach_provider() -> None:
    client = _make_client()
    token = _mint(client)["token"]
    _capture_query(client, token, model=MODEL)
    denied = _post_query(client, token, model=OTHER)
    assert denied.status_code == 403
    assert denied.json()["code"] == "model_not_authorized"


def test_embedding_request_cannot_choose_an_unassigned_embedding_model() -> None:
    body = EmbeddingRequest.model_validate(
        {"model": MODEL, "text": "hi", "embedding_model": "other-embedding"}
    )
    with pytest.raises(RunTokenAuthorizationError):
        _authorize_body(_claims(), body)


def test_embedded_route_respects_model_scope() -> None:
    client = _make_client()
    token = _mint(
        client, allowed_models=[MODEL], allowed_embedding_models=[EMBEDDING]
    )["token"]
    denied = client.post(
        "/embeddings", headers=_bearer(token), json={"model": OTHER, "text": "hi"}
    )
    assert denied.status_code == 403
    assert denied.json()["code"] == "model_not_authorized"
    llm = SimpleNamespace(get_embedding=AsyncMock(return_value=[0.1, 0.2]))
    with patch.object(model_helpers, "get_cached_llm", return_value=llm):
        allowed = client.post(
            "/embeddings", headers=_bearer(token), json={"model": MODEL, "text": "hi"}
        )
    assert allowed.status_code == 200, allowed.text
    llm.get_embedding.assert_awaited_once_with("hi", model=EMBEDDING)


@pytest.mark.parametrize(
    "config",
    [
        {"reasoning_effort": "low", "supports_tools": True},
        {"provider_config": {"verbosity": "low"}},
        {"custom_api_key": "sk-caller"},
    ],
)
def test_token_allows_any_explicit_config_override(config: dict[str, Any]) -> None:
    client = _make_client()
    token = _mint(client)["token"]
    body = _capture_query(client, token, config=config)
    assert dump_llm_config(body.config).items() >= config.items()


@pytest.mark.parametrize(
    "overrides",
    [
        {"ttl_seconds": None},
        {"ttl_seconds": 0},
        {"ttl_seconds": 7 * 24 * 3600 + 1},
        {"allowed_models": ["not-a/model"]},
        {"allowed_operations": ["chat"]},
    ],
    ids=["no-ttl", "zero-ttl", "ttl-over-cap", "unknown-model", "unknown-operation"],
)
def test_mint_rejects_an_unusable_scope(overrides: dict[str, Any]) -> None:
    payload = {
        "run_id": RUN_ID,
        "task_id": TASK_ID,
        "allowed_models": [MODEL],
        "ttl_seconds": 3600,
        **overrides,
    }
    payload = {key: value for key, value in payload.items() if value is not None}
    response = _make_client().post("/service-auth", headers=HEADERS, json=payload)
    assert response.status_code == 400


def test_token_narrowed_to_an_operation_cannot_use_others() -> None:
    client = _make_client()
    token = _mint(client, allowed_operations=["embeddings"])["token"]
    assert _post_query(client, token).status_code == 403
    assert client.get("/registry", headers=_bearer(token)).status_code == 200


def test_run_revoke_ends_every_token_the_key_minted_for_the_run() -> None:
    client = _make_client(api_keys={"dev": "sk-dev", "other": "sk-other"})
    first = _mint(client, headers=_bearer("sk-dev"))["token"]
    second = _mint(client, headers=_bearer("sk-dev"), task_id="task_1")["token"]
    foreign = _mint(client, headers=_bearer("sk-other"))["token"]

    response = client.post(
        "/service-auth/revoke", headers=_bearer("sk-dev"), json={"run_id": RUN_ID}
    )
    assert response.json() == {"revoked": 2}
    assert _post_query(client, first).status_code == 401
    assert _post_query(client, second).status_code == 401
    _capture_query(client, foreign)


@pytest.mark.parametrize(
    "sent", [None, {"email": "victim@example.com"}], ids=["omitted", "forged"]
)
def test_executor_identity_replaces_sandbox_identity(
    sent: dict[str, str] | None,
) -> None:
    client = _make_client()
    identity = {"benchmark_name": "valsmith", "agent_name": "opencode"}
    token = _mint(client, identity=identity)["token"]
    body = _capture_query(client, token, identity=sent)
    assert body.identity == identity


def test_unattested_identity_is_removed_and_task_ids_are_stamped() -> None:
    client = _make_client()
    token = _mint(client)["token"]
    body = _capture_query(
        client,
        token,
        run_id="client-default",
        question_id="another-task",
        identity={"email": "victim@example.com"},
    )
    assert (body.run_id, body.question_id, body.identity) == (
        RUN_ID,
        TASK_ID,
        None,
    )


def test_token_cannot_set_retry_params() -> None:
    client = _make_client()
    token = _mint(client)["token"]
    retry = {"input_modifier": 1.0, "output_modifier": 1.0}
    denied = _post_query(client, token, token_retry_params=retry)
    assert denied.status_code == 403
    assert denied.json()["code"] == "model_not_authorized"
    assert (
        client.post(
            "/service-auth",
            headers=HEADERS,
            json={
                "run_id": RUN_ID,
                "task_id": TASK_ID,
                "allowed_models": [MODEL],
                "ttl_seconds": 3600,
                "token_retry_params": retry,
            },
        ).status_code
        == 400
    )
def test_token_can_set_provider_config_on_other_models() -> None:
    client = _make_client()
    model = "fireworks/glm-5p3-flash"
    token = _mint(client, allowed_models=[model])["token"]
    body = _capture_query(
        client, token, model=model, config={"provider_config": {"serverless": True}}
    )
    assert body.config.provider_config is not None


def test_sandbox_token_cannot_enter_control_or_model_catalog() -> None:
    client = _make_client()
    minted = _mint(client)
    token = minted["token"]
    for path, payload in (
        (
            "/service-auth",
            {"run_id": RUN_ID, "task_id": TASK_ID, "allowed_models": [MODEL]},
        ),
        ("/service-auth/revoke", {"lease_id": minted["lease_id"]}),
        ("/benchmark-runs/acquire", {"run_id": RUN_ID, "model": MODEL}),
    ):
        response = client.post(path, headers=_bearer(token), json=payload)
        assert response.status_code == 403, path
    assert client.get("/models", headers=_bearer(token)).status_code == 403


def test_registry_exposes_only_assigned_models_to_sandbox() -> None:
    client = _make_client()
    token = _mint(client)["token"]
    visible = client.get("/registry", headers=_bearer(token))
    full = client.get("/registry", headers=HEADERS)
    assert visible.status_code == full.status_code == 200
    assert set(visible.json()["models"]) == {MODEL}
    assert "alibaba/qwen3-max" in full.json()["models"]
