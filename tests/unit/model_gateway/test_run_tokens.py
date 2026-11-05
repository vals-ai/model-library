"""Managed run-token scope, isolation, and revocation behavior."""

import asyncio
import hashlib
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID

import fakeredis.aioredis
import pytest
from google.genai import types as genai_types

import model_gateway.model_helpers as model_helpers
import model_gateway.routes.query as query_routes
import model_gateway.routes.run_tokens as run_token_routes
import model_gateway.run_tokens as run_token_module
from model_gateway.egress import EgressFilter
from model_gateway.errors import RunTokenAuthorizationError
from model_gateway.managed_auth.credentials import managed_key_verifier
from model_gateway.managed_auth.runtime import RuntimeAuthority
from model_gateway.managed_auth.types import HumanIdentity, PersonalOwner, ProjectReference
from model_gateway.run_token_types import ModelPermissions
from model_gateway.run_tokens import (
    RunTokenClaims,
    run_token_authorized,
    tool_results_are_content,
)
from model_gateway.types import EmbeddingRequest
from model_library.base import LLM, RawInput, dump_llm_config
from model_library.providers.amazon.chat import AmazonModel
from model_library.providers.anthropic import AnthropicModel
from model_library.providers.google.google import GoogleModel
from model_library.providers.mistral.chat import MistralModel
from model_library.providers.openai import OpenAIModel
from model_library.retriers.token import utils as token_utils
from tests.unit.model_gateway._support import HEADERS, _make_client as _base_client

MODEL = "openai/gpt-4o"
OTHER = "anthropic/claude-4-opus"
EMBEDDING = "text-embedding-3-small"
RUN_ID = "run-1"
TASK_ID = "task_0"
ATTEMPT_ID = "attempt-1"
ROOTS = {
    "dev": "mgwk_dev",
    "prod": "mgwk_prod",
    "other": "mgwk_other",
    "test": "mgwk_test",
}
KEY_IDS = {name: UUID(int=index) for index, name in enumerate(ROOTS, 1)}
MANAGED_HEADERS = {"Authorization": "Bearer mgwk_test"}


def _make_client(
    *, managed_names: tuple[str, ...] = tuple(ROOTS)
) -> Any:
    client = _base_client()
    permitted = ModelPermissions(
        allowed_models=None,
        allowed_embedding_models=[],
        allowed_operations=None,
    )
    owner = PersonalOwner(HumanIdentity("issuer", "subject"))
    authorities = {
        managed_key_verifier(root): RuntimeAuthority(
            KEY_IDS[name],
            owner,
            permitted,
            ProjectReference(UUID(int=1), "test-organization", "test"),
            None,
            None,
        )
        for name, root in ROOTS.items()
        if name in managed_names
    }

    async def by_verifier(verifier: str) -> Any:
        return authorities.get(verifier)

    async def by_key_id(key_id: UUID) -> Any:
        return next(
            (
                authority
                for authority in authorities.values()
                if authority.key_id == key_id
            ),
            None,
        )

    client.app.state.managed_authority = SimpleNamespace(
        by_verifier=by_verifier,
        by_key_id=by_key_id,
    )
    client.app.state.budget_service = SimpleNamespace(
        blocked=Mock(return_value=False),
        managed_decision=AsyncMock(return_value="allowed"),
    )
    return client


@pytest.fixture(autouse=True)
def redis() -> Any:
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    token_utils.set_redis_client(client)
    yield client
    token_utils.set_redis_client(None)


def _mint(
    client: Any, *, headers: dict[str, str] = MANAGED_HEADERS, **overrides: Any
) -> dict[str, Any]:
    payload = {
        "run_id": RUN_ID,
        "task_id": TASK_ID,
        "attempt_id": ATTEMPT_ID,
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


def _capture_query(
    client: Any, token: str, llm: LLM | None = None, **overrides: Any
) -> Any:
    """The routed body, with inputs and tools as the provider adapter receives them."""
    seen: dict[str, Any] = {}

    async def provider_query(inputs: Any, *, tools: Any, **_kwargs: Any) -> Any:
        seen["body"] = seen["lookup_body"].model_copy(
            update={"inputs": inputs, "tools": tools}
        )
        raise ValueError("provider intentionally unavailable")

    async def lookup(_cache: Any, body: Any, **_kwargs: Any) -> Any:
        seen["lookup_body"] = body
        if llm is None:
            return SimpleNamespace(query=provider_query, delegate=None)
        llm.query = provider_query  # type: ignore[method-assign]
        return llm

    with patch.object(query_routes, "get_query_llm", lookup):
        _post_query(client, token, **overrides)
    assert "body" in seen, "authorized request must reach provider boundary"
    return seen["body"]


def _claims(**overrides: Any) -> RunTokenClaims:
    return RunTokenClaims(
        run_id=RUN_ID,
        task_id=TASK_ID,
        attempt_id=ATTEMPT_ID,
        allowed_models=[MODEL],
        allowed_embedding_models=[EMBEDDING],
        minted_by="executor",
        managed_key_id=KEY_IDS["test"],
        **overrides,
    )


def _authorize_body(claims: RunTokenClaims, body: Any) -> Any:
    request = SimpleNamespace(
        state=SimpleNamespace(run_token_claims=claims, managed_authority=None)
    )
    return asyncio.run(run_token_authorized(type(body))(request, body))


def test_minted_token_can_call_query_and_revoke_by_its_digest() -> None:
    client = _make_client()
    minted = _mint(client)
    token = minted["token"]
    assert minted["lease_id"] == hashlib.sha256(token.encode()).hexdigest()
    assert minted["expires_at"] == pytest.approx(time.time() + 3600, abs=60)
    _capture_query(client, token)

    revoked = client.post(
        "/service-auth/revoke",
        headers=MANAGED_HEADERS,
        json={"lease_id": minted["lease_id"]},
    )
    assert revoked.status_code == 200
    assert _post_query(client, token).status_code == 401
    assert (
        client.post(
            "/service-auth/revoke",
            headers=MANAGED_HEADERS,
            json={"lease_id": minted["lease_id"]},
        ).status_code
        == 404
    )


def test_revoke_that_loses_a_race_reports_not_found() -> None:
    client = _make_client()
    lease = {"lease_id": _mint(client)["lease_id"]}
    stale = asyncio.run(run_token_routes.get_run_token(lease["lease_id"]))
    assert (
        client.post(
            "/service-auth/revoke", headers=MANAGED_HEADERS, json=lease
        ).status_code
        == 200
    )

    # A concurrent revoke read the lease before the first one deleted it.
    with patch.object(run_token_routes, "get_run_token", AsyncMock(return_value=stale)):
        lost = client.post("/service-auth/revoke", headers=MANAGED_HEADERS, json=lease)
    assert lost.status_code == 404


def test_another_managed_key_cannot_use_or_revoke_a_token() -> None:
    dev = _make_client(managed_names=("dev",))
    prod = _make_client(managed_names=("prod",))
    minted = _mint(dev, headers=_bearer(ROOTS["dev"]))
    assert _post_query(prod, minted["token"]).status_code == 401

    both = _make_client(managed_names=("dev", "prod"))
    denied = both.post(
        "/service-auth/revoke",
        headers=_bearer(ROOTS["prod"]),
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




def test_managed_key_without_model_restriction_reaches_any_model() -> None:
    client = _make_client()
    unlisted = "openai/gpt-5.4-nano-2026-03-17"
    seen: dict[str, Any] = {}

    async def capture(_cache: Any, body: Any, **_kwargs: Any) -> Any:
        seen["body"] = body
        raise ValueError("provider intentionally unavailable")

    with patch.object(query_routes, "get_query_llm", capture):
        client.post("/query", headers=MANAGED_HEADERS, json=_query(model=unlisted))
    assert seen["body"].model == unlisted

    token = _mint(client, allowed_models=[unlisted])["token"]
    _capture_query(client, token, model=unlisted)
    assert _post_query(client, token, model=MODEL).status_code == 403


def test_minted_token_must_name_models() -> None:
    client = _make_client()
    for payload in ({}, {"allowed_models": None}):
        response = client.post(
            "/service-auth",
            headers=MANAGED_HEADERS,
            json={
                "run_id": RUN_ID,
                "task_id": TASK_ID,
                "attempt_id": ATTEMPT_ID,
                "ttl_seconds": 3600,
                **payload,
            },
        )
        assert response.status_code == 400




def test_embedding_request_cannot_choose_an_unassigned_embedding_model() -> None:
    body = EmbeddingRequest.model_validate(
        {"model": MODEL, "text": "hi", "embedding_model": "other-embedding"}
    )
    with pytest.raises(RunTokenAuthorizationError):
        _authorize_body(_claims(), body)


def test_managed_child_embedding_and_model_scope() -> None:
    client = _make_client()
    token = _mint(client, allowed_models=[MODEL], allowed_embedding_models=[EMBEDDING])[
        "token"
    ]
    denied = client.post(
        "/embeddings", headers=_bearer(token), json={"model": OTHER, "text": "hi"}
    )
    assert denied.status_code == 403
    assert denied.json()["code"] == "model_not_authorized"
    llm = SimpleNamespace(get_embedding=AsyncMock(return_value=[0.1, 0.2]))
    with patch.object(model_helpers, "get_cached_llm", return_value=llm):
        allowed_embedding = client.post(
            "/embeddings", headers=_bearer(token), json={"model": MODEL, "text": "hi"}
        )
    assert allowed_embedding.status_code == 200, allowed_embedding.text
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
        "attempt_id": ATTEMPT_ID,
        "allowed_models": [MODEL],
        "ttl_seconds": 3600,
        **overrides,
    }
    payload = {key: value for key, value in payload.items() if value is not None}
    response = _make_client().post(
        "/service-auth", headers=MANAGED_HEADERS, json=payload
    )
    assert response.status_code == 400


def test_token_narrowed_to_an_operation_cannot_use_others() -> None:
    client = _make_client()
    token = _mint(client, allowed_operations=["embeddings"])["token"]
    assert _post_query(client, token).status_code == 403
    assert client.get("/registry", headers=_bearer(token)).status_code == 200


def test_run_revoke_ends_every_token_the_key_minted_for_the_run() -> None:
    client = _make_client()
    first = _mint(client, headers=_bearer(ROOTS["dev"]))["token"]
    second = _mint(client, headers=_bearer(ROOTS["dev"]), task_id="task_1")["token"]
    foreign = _mint(client, headers=_bearer(ROOTS["other"]))["token"]

    response = client.post(
        "/service-auth/revoke", headers=_bearer(ROOTS["dev"]), json={"run_id": RUN_ID}
    )
    assert response.json() == {"revoked": 2}
    assert _post_query(client, first).status_code == 401
    assert _post_query(client, second).status_code == 401
    _capture_query(client, foreign)


@pytest.mark.parametrize(
    "sent", [None, {"email": "victim@example.com"}], ids=["omitted", "forged"]
)
def test_executor_identity_replaces_request_identity(
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
            headers=MANAGED_HEADERS,
            json={
                "run_id": RUN_ID,
                "task_id": TASK_ID,
                "attempt_id": ATTEMPT_ID,
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


def test_run_token_cannot_enter_control_or_model_catalog() -> None:
    client = _make_client()
    minted = _mint(client)
    token = minted["token"]
    for path, payload in (
        (
            "/service-auth",
            {
                "run_id": RUN_ID,
                "task_id": TASK_ID,
                "attempt_id": ATTEMPT_ID,
                "allowed_models": [MODEL],
            },
        ),
        ("/service-auth/revoke", {"lease_id": minted["lease_id"]}),
        ("/benchmark-runs/acquire", {"run_id": RUN_ID, "model": MODEL}),
    ):
        response = client.post(path, headers=_bearer(token), json=payload)
        assert response.status_code == 403, path
    assert client.get("/models", headers=_bearer(token)).status_code == 403


def test_registry_exposes_only_assigned_models_to_run_token() -> None:
    """Run tokens see assigned models; operational registry sees all."""
    client = _make_client()
    token = _mint(client)["token"]
    visible = client.get("/registry", headers=_bearer(token))
    full = client.get("/registry", headers=HEADERS)
    assert visible.status_code == full.status_code == 200
    assert set(visible.json()["models"]) == {MODEL}
    assert set(full.json()["models"]) > {MODEL}


def test_token_attempt_is_stamped_over_the_body_attempt() -> None:
    client = _make_client()
    token = _mint(client, attempt_id="attempt-7")["token"]
    body = _capture_query(client, token, attempt_id="client-attempt")
    assert (body.run_id, body.question_id, body.attempt_id) == (
        RUN_ID,
        TASK_ID,
        "attempt-7",
    )


def test_legacy_mint_without_an_attempt_leaves_the_body_attempt_alone() -> None:
    client = _make_client()
    payload = {
        "run_id": RUN_ID,
        "task_id": TASK_ID,
        "allowed_models": [MODEL],
        "ttl_seconds": 3600,
    }
    response = client.post("/service-auth", headers=MANAGED_HEADERS, json=payload)
    assert response.status_code == 200, response.text
    body = _capture_query(client, response.json()["token"], attempt_id="client-attempt")
    assert (body.run_id, body.question_id, body.attempt_id) == (
        RUN_ID,
        TASK_ID,
        "client-attempt",
    )


ALLOWED_URL = "https://allowed.example/data.json"


BLOCKED_URL = "https://api.github.com/repos/python/cpython/commits"


DATA_URL = "data:image/png;base64,iVBORw0KGgo="


def _file(url: str, kind: str = "file") -> dict[str, Any]:
    return {
        "kind": "file_base",
        "append_type": "url",
        "type": kind,
        "name": "f",
        "mime": "application/json",
        "url": url,
    }


def _egress_query(
    client: Any, egress: Any, allow_urls: bool = True, **overrides: Any
) -> Any:
    token = _mint(
        client, egress=egress, allow_urls=allow_urls, allow_provider_tools=True
    )["token"]
    return _capture_query(client, token, **overrides)


def test_url_inputs_outside_the_egress_list_are_stripped_with_a_notice() -> None:
    inputs = [
        {"kind": "text", "text": "hi"},
        _file(BLOCKED_URL),
        _file(ALLOWED_URL),
        _file(DATA_URL, "image"),
    ]
    body = _egress_query(_make_client(), ["https://allowed.example"], inputs=inputs)
    assert [getattr(i, "url", None) for i in body.inputs] == [
        None,
        None,
        ALLOWED_URL,
        DATA_URL,
    ]
    assert body.inputs[1].kind == "text"
    assert BLOCKED_URL in body.inputs[1].text


def test_unrestricted_token_forwards_url_inputs_and_tools_unchanged() -> None:
    tools = [{"name": "search", "body": {"type": "native_web_search"}}]
    body = _egress_query(_make_client(), "*", inputs=[_file(BLOCKED_URL)], tools=tools)
    assert [i.url for i in body.inputs] == [BLOCKED_URL]
    assert [t.name for t in body.tools] == ["search"]


def test_raw_provider_parts_outside_the_egress_list_are_stripped() -> None:
    message = {
        "role": "user",
        "content": [
            {"type": "input_text", "text": "quote the sha"},
            {"type": "input_file", "file_url": BLOCKED_URL},
            {"type": "input_image", "image_url": {"url": BLOCKED_URL}},
        ],
    }
    inputs = [
        {"kind": "raw_input", "input": message},
        {"kind": "raw_input", "input": {"type": "input_file", "file_url": BLOCKED_URL}},
        {
            "kind": "raw_input",
            "input": {"type": "file", "file": {"file_data": BLOCKED_URL}},
        },
        {
            "kind": "raw_input",
            "input": {"type": "file", "file": {"file_data": DATA_URL}},
        },
    ]
    body = _egress_query(_make_client(), [], inputs=inputs)
    assert [i.kind for i in body.inputs] == ["raw_input", "text", "raw_input"]
    assert body.inputs[0].input["content"] == [
        {"type": "input_text", "text": "quote the sha"}
    ]


def test_tool_result_parts_outside_the_egress_list_are_stripped() -> None:
    call = {"id": "c1", "call_id": "c1", "name": "look", "args": {}}
    result = [
        {"type": "input_text", "text": "ok"},
        {"type": "input_image", "image_url": BLOCKED_URL},
    ]
    inputs = [{"kind": "tool_result", "tool_call": call, "result": result}]
    body = _egress_query(_make_client(), [], inputs=inputs)
    assert body.inputs[0].result == [{"type": "input_text", "text": "ok"}]
    assert body.inputs[1].kind == "text"


def test_computer_screenshot_outside_the_egress_list_rejects_the_query() -> None:
    def reply(url: str) -> dict[str, Any]:
        return {
            "kind": "computer_call_output",
            "call_id": "c1",
            "screenshot_url": url,
        }

    client = _make_client()
    body = _egress_query(client, [], inputs=[reply(DATA_URL)])
    assert body.inputs[0].screenshot_url == DATA_URL

    token = _mint(client, egress=[], allow_urls=True, allow_provider_tools=True)["token"]
    reached: list[Any] = []

    async def lookup(_cache: Any, _body: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(query=lambda *a, **k: reached.append(a), delegate=None)

    with patch.object(query_routes, "get_query_llm", lookup):
        response = _post_query(client, token, inputs=[reply(BLOCKED_URL)])
    assert response.status_code == 403
    assert response.json()["code"] == "egress_denied"
    assert reached == []


def test_provider_side_tools_are_stripped_and_client_tools_kept() -> None:
    function = {
        "name": "bash",
        "description": "run",
        "properties": {},
        "required": [],
    }
    tools = [
        {"name": "bash", "body": function},
        {"name": "computer", "body": {"type": "computer"}},
        {"name": "search", "body": {"type": "native_web_search"}},
        {"name": "fetch", "body": {"type": "web_fetch_20250910", "name": "web_fetch"}},
        {"name": "grounding", "body": {"url_context": {}}},
        {"name": "mcp", "body": {"type": "mcp", "server_url": BLOCKED_URL}},
    ]
    inputs = [{"kind": "system", "text": "sys"}, {"kind": "text", "text": "hi"}]
    body = _egress_query(_make_client(), [], inputs=inputs, tools=tools)
    assert [t.name for t in body.tools] == ["bash", "computer"]
    assert [i.kind for i in body.inputs] == ["system", "text", "text"]
    assert body.inputs[1].text.startswith("Some request content")


def test_egress_wildcards_and_cidrs_match_like_the_sandbox() -> None:
    inputs = [
        _file("https://raw.githubusercontent.com/a"),
        _file("http://10.1.2.3:8080/x"),
        _file("https://githubusercontent.com/a"),
    ]
    body = _egress_query(
        _make_client(), ["*.githubusercontent.com", "10.0.0.0/8"], inputs=inputs
    )
    assert [getattr(i, "url", None) for i in body.inputs] == [
        "https://raw.githubusercontent.com/a",
        "http://10.1.2.3:8080/x",
        None,
    ]


def test_client_tool_url_examples_are_forwarded_without_a_notice() -> None:
    function = {
        "type": "function",
        "name": "open",
        "parameters": {"examples": [{"url": BLOCKED_URL}]},
    }
    inputs = [{"kind": "text", "text": "hi"}]
    body = _egress_query(
        _make_client(), [], inputs=inputs, tools=[{"name": "open", "body": function}]
    )
    assert body.tools[0].body == function
    assert [i.kind for i in body.inputs] == ["text"]


def test_raw_urls_without_an_authority_are_stripped() -> None:
    gemini_part = {
        "file_data": {"file_uri": "gs://bucket/key", "mime_type": "image/png"}
    }
    inputs = [
        {"kind": "raw_input", "input": gemini_part},
        {"kind": "raw_input", "input": {"type": "file", "file_id": "file-abc"}},
    ]
    body = _egress_query(_make_client(), [], inputs=inputs)
    assert [i.kind for i in body.inputs] == ["text", "raw_input"]


def _file_id(file_id: str) -> dict[str, Any]:
    return {
        "kind": "file_base",
        "append_type": "file_id",
        "type": "image",
        "name": "f",
        "mime": "image/png",
        "file_id": file_id,
    }


def test_uri_file_ids_are_held_to_the_url_policy() -> None:
    google_upload = "https://generativelanguage.googleapis.com/v1beta/files/abc"
    inputs = [
        _file_id("gs://public-bucket/image.png"),
        _file_id("file-abc"),
        _file_id(google_upload),
    ]
    body = _egress_query(_make_client(), ["allowed.example"], inputs=inputs)
    assert [getattr(i, "file_id", None) for i in body.inputs] == [
        None,
        "file-abc",
        google_upload,
    ]
    assert "gs://public-bucket/image.png" in body.inputs[0].text


def test_tool_result_json_with_a_url_field_is_kept() -> None:
    call = {"id": "c1", "call_id": "c1", "name": "search", "args": {}}
    result = {"title": "Release notes", "url": BLOCKED_URL}
    inputs = [{"kind": "tool_result", "tool_call": call, "result": result}]
    body = _egress_query(_make_client(), [], inputs=inputs)
    assert [i.kind for i in body.inputs] == ["tool_result"]
    assert body.inputs[0].result == result


LINK_DATA = {"type": "link", "image_url": BLOCKED_URL}


GEMINI_PART_DATA = {
    "file_data": {
        "file_uri": "gs://secret-bucket/private.png",
        "mime_type": "image/png",
    }
}


@pytest.mark.parametrize("result", [LINK_DATA, GEMINI_PART_DATA])
def test_tool_result_data_shaped_like_a_content_part_is_kept(
    result: dict[str, Any],
) -> None:
    call = {"id": "c1", "call_id": "c1", "name": "search", "args": {}}
    inputs = [{"kind": "tool_result", "tool_call": call, "result": result}]
    body = _egress_query(_make_client(), [], inputs=inputs)
    assert [i.kind for i in body.inputs] == ["tool_result"]
    assert body.inputs[0].result == result


CONTENT_LIST_RESULT = [{"type": "input_image", "image_url": BLOCKED_URL}]


@pytest.mark.parametrize(
    "model",
    [
        lambda: GoogleModel("gemini-test"),
        lambda: AmazonModel("amazon-test"),
        lambda: AnthropicModel("claude-test"),
    ],
)
def test_list_tool_results_are_data_for_json_and_text_adapters(
    model: Callable[[], LLM],
) -> None:
    call = {"id": "c1", "call_id": "c1", "name": "search", "args": {}}
    inputs = [{"kind": "tool_result", "tool_call": call, "result": CONTENT_LIST_RESULT}]
    body = _egress_query(_make_client(), [], llm=model(), inputs=inputs)
    assert [i.kind for i in body.inputs] == ["tool_result"]
    assert body.inputs[0].result == CONTENT_LIST_RESULT


def test_tool_result_dialect_follows_the_delegate() -> None:
    assert tool_results_are_content(OpenAIModel("gpt-test"))
    assert tool_results_are_content(MistralModel("mistral-test"))
    assert not tool_results_are_content(GoogleModel("gemini-test"))
    wrapper = OpenAIModel("gpt-test")
    wrapper.delegate = AnthropicModel("claude-test")
    assert not tool_results_are_content(wrapper)


def test_signed_gemini_function_call_arguments_are_kept() -> None:
    content = genai_types.Content(
        role="model",
        parts=[
            genai_types.Part(
                function_call=genai_types.FunctionCall(name="save", args=LINK_DATA)
            )
        ],
    )
    signed = LLM._pickle_field(content, b"test-secret")  # pyright: ignore[reportPrivateUsage]
    body = _egress_query(
        _make_client(), [], inputs=[{"kind": "raw_input", "input": signed}]
    )
    assert [i.kind for i in body.inputs] == ["raw_input"]
    assert body.inputs[0].input == content


def test_gemini_function_response_parts_are_held_to_the_policy() -> None:
    response = {
        "name": "look",
        "response": GEMINI_PART_DATA,
        "parts": [{"file_data": {"file_uri": BLOCKED_URL, "mime_type": "image/png"}}],
    }
    message = {"role": "user", "parts": [{"function_response": response}]}
    body = _egress_query(
        _make_client(), [], inputs=[{"kind": "raw_input", "input": message}]
    )
    kept = body.inputs[0].input["parts"][0]["function_response"]
    assert kept == {**response, "parts": []}
    assert BLOCKED_URL in body.inputs[1].text


@pytest.mark.parametrize("media", ["image", "document", "video"])
def test_bedrock_s3_media_is_held_to_the_policy(media: str) -> None:
    s3 = {media: {"format": "png", "source": {"s3Location": {"uri": "s3://bucket/x"}}}}
    message = {
        "role": "user",
        "content": [
            {"text": "describe"},
            s3,
            {"toolResult": {"toolUseId": "t1", "content": [{"json": LINK_DATA}, s3]}},
        ],
    }
    body = _egress_query(
        _make_client(), [], inputs=[{"kind": "raw_input", "input": message}]
    )
    assert body.inputs[0].input["content"] == [
        {"text": "describe"},
        {"toolResult": {"toolUseId": "t1", "content": [{"json": LINK_DATA}]}},
    ]
    assert "s3://bucket/x" in body.inputs[1].text


def test_raw_message_left_empty_is_dropped() -> None:
    image = {"type": "image", "source": {"type": "url", "url": BLOCKED_URL}}
    message = {"role": "user", "content": [image]}
    body = _egress_query(
        _make_client(), [], inputs=[{"kind": "raw_input", "input": message}]
    )
    assert [i.kind for i in body.inputs] == ["text"]


def test_signed_history_is_held_to_the_current_token_policy() -> None:
    message = {
        "role": "user",
        "content": [
            {"type": "input_text", "text": "hi"},
            {"type": "input_file", "file_url": BLOCKED_URL},
        ],
    }
    signed = LLM._pickle_field(message, b"test-secret")  # pyright: ignore[reportPrivateUsage]
    body = _egress_query(
        _make_client(), [], inputs=[{"kind": "raw_input", "input": signed}]
    )
    assert body.inputs[0].input["content"] == [{"type": "input_text", "text": "hi"}]
    assert BLOCKED_URL in body.inputs[1].text


def test_token_count_strips_url_inputs() -> None:
    client = _make_client()
    token = _mint(client)["token"]
    seen: dict[str, Any] = {}

    async def count(inputs: Any, *, tools: Any) -> int:
        seen["inputs"] = inputs
        return 1

    async def lookup(_cache: Any, _body: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(count_tokens=count, delegate=None)

    with patch.object(model_helpers, "get_query_llm", lookup):
        client.post(
            "/tokens/count",
            headers=_bearer(token),
            json={"model": MODEL, "inputs": [_file(BLOCKED_URL)]},
        )
    assert [i.kind for i in seen["inputs"]] == ["text"]


@pytest.mark.parametrize("egress", [["[bad"], [" "], ["::1/128"], ["https://[::1]/"]])
def test_mint_rejects_egress_the_sandbox_rejects(egress: list[str]) -> None:
    response = _make_client().post(
        "/service-auth",
        headers=MANAGED_HEADERS,
        json={
            "run_id": RUN_ID,
            "task_id": TASK_ID,
            "attempt_id": ATTEMPT_ID,
            "allowed_models": [MODEL],
            "ttl_seconds": 3600,
            "egress": egress,
        },
    )
    assert response.status_code == 400, response.text


def test_logged_urls_drop_credentials_and_query() -> None:
    egress = EgressFilter([])
    assert not egress.permits("https://user:pw@evil.example:8443/p?token=x#frag")
    assert not egress.permits("https://user:pw@[2001:db8::1]:8443/x?q=secret#frag")
    assert egress.logged_urls == [
        "https://evil.example:8443/p",
        "https://[2001:db8::1]:8443/x",
    ]


def test_removed_inputs_counts_request_inputs() -> None:
    part = {"file_data": {"file_uri": BLOCKED_URL, "mime_type": "image/png"}}
    message = {"role": "user", "parts": [part]}
    sdk = genai_types.Content.model_validate(message)
    for raw in (message, sdk):
        egress = EgressFilter([])
        kept = egress.filter_inputs([RawInput(input=raw)])
        assert [i.kind for i in kept] == ["text"]
        assert egress.removed_inputs == 1


def test_default_token_strips_url_inputs_and_provider_tools() -> None:
    client = _make_client()
    tools = [
        {"name": "search", "body": {"type": "native_web_search"}},
        {"name": "run", "body": {"type": "function", "name": "run", "parameters": {}}},
    ]
    token = _mint(client)["token"]
    inputs = [_file(BLOCKED_URL), _file(DATA_URL, "image")]
    body = _capture_query(client, token, inputs=inputs, tools=tools)
    assert [i.kind for i in body.inputs] == ["text", "file_base"]
    assert body.inputs[1].url == DATA_URL
    assert [t.name for t in body.tools] == ["run"]


def test_url_opt_out_strips_allowlisted_urls_too() -> None:
    body = _egress_query(
        _make_client(),
        ["https://allowed.example"],
        allow_urls=False,
        inputs=[_file(ALLOWED_URL)],
    )
    assert [i.kind for i in body.inputs] == ["text"]


def test_unrestricted_url_opt_in_keeps_urls_but_strips_provider_tools() -> None:
    client = _make_client()
    tools = [{"name": "search", "body": {"type": "native_web_search"}}]
    token = _mint(client, allow_urls=True)["token"]
    body = _capture_query(client, token, inputs=[_file(BLOCKED_URL)], tools=tools)
    assert [getattr(i, "url", None) for i in body.inputs] == [None, BLOCKED_URL]
    assert body.tools == []


def test_backslash_url_is_stripped_even_if_python_parses_an_allowed_host() -> None:
    url = "https://evil.example\\@allowed.example/x"
    body = _egress_query(
        _make_client(), ["https://allowed.example"], inputs=[_file(url)]
    )
    assert [i.kind for i in body.inputs] == ["text"]
