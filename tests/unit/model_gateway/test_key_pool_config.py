"""Pools come from the `pools` AppConfig profile and nowhere else."""

import json

import httpx
import jsonschema
import pytest

import model_gateway.model_helpers as model_helpers
from model_gateway.key_pool_config import (
    ModelPoolRule,
    PublishedPoolDocument,
    pool_config_from_document,
)
from model_gateway.key_pools import keys_for
from model_gateway.pool_config_schema import POOL_CONFIG_JSON_SCHEMA
from model_gateway.runtime_config import (
    AppConfigAgentClient,
    RuntimeConfigManager,
    RuntimeConfigState,
    runtime_pool_snapshot,
)
from model_gateway.types import QueryRequest

MODEL = "anthropic/claude-opus-4-6"


def _document(revision: int, *keys: str) -> bytes:
    return json.dumps(
        {
            "pools": {"anthropic/default": list(keys)},
            "model_pools": [],
            "weights": {},
            "revision": revision,
        }
    ).encode()


class PoolsAgent:
    """An AppConfig Agent that serves a fixed models and credentials pair."""

    def __init__(self, pools: bytes | None) -> None:
        self.pools = pools

    def client(self) -> AppConfigAgentClient:
        def handle(request: httpx.Request) -> httpx.Response:
            profile = request.url.path.rsplit("/", 1)[-1]
            if profile == "pools":
                if self.pools is None:
                    return httpx.Response(404)
                return httpx.Response(
                    200, content=self.pools, headers={"Configuration-Version": "1"}
                )
            # Models and credentials stay on the bootstrap registry and keys.
            return httpx.Response(503)

        return AppConfigAgentClient(
            base_url="http://127.0.0.1:2772",
            application="model-gateway",
            environment="test",
            access_token="local-token",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
        )


def _manager(agent: PoolsAgent) -> RuntimeConfigManager:
    return RuntimeConfigManager(
        state=RuntimeConfigState.bootstrap(),
        agent=agent.client(),
        model_profile="models",
        credential_profile="credentials",
        pool_profile="pools",
    )


@pytest.mark.unit
def test_appconfig_schema_accepts_the_published_pool_document():
    document = PublishedPoolDocument(
        pools={"anthropic/batch": ["ANTHROPIC_API_KEY_2"]},
        model_pools=[ModelPoolRule(pool="anthropic/batch", models=["claude-x"])],
        weights={"ANTHROPIC_API_KEY_2": 1.0},
        descriptions={"ANTHROPIC_API_KEY_2": "batch key"},
        revision=3,
    )

    jsonschema.validate(json.loads(document.model_dump_json()), POOL_CONFIG_JSON_SCHEMA)


@pytest.mark.unit
def test_a_document_without_descriptions_reads_as_none_given():
    document = PublishedPoolDocument.model_validate(
        {"pools": {}, "model_pools": [], "weights": {}, "revision": 1}
    )

    assert document.descriptions == {}


@pytest.mark.unit
def test_a_published_document_serves_only_its_own_rules():
    config = pool_config_from_document(
        PublishedPoolDocument(
            pools={
                "anthropic/default": ["ANTHROPIC_API_KEY"],
                "anthropic/pin": ["ANTHROPIC_API_KEY_2"],
            },
            model_pools=[ModelPoolRule(pool="anthropic/pin", models=["anthropic/a"])],
            weights={},
            revision=1,
        )
    )

    assert keys_for(config, "anthropic/a") == ("ANTHROPIC_API_KEY_2",)
    assert keys_for(config, "anthropic/b") == ("ANTHROPIC_API_KEY",)
    assert keys_for(config, "openai/gpt-4o") == ()


@pytest.mark.unit
async def test_a_published_document_serves_each_request():
    manager = _manager(PoolsAgent(_document(1, "ANTHROPIC_API_KEY_2")))

    result = await manager.refresh_once()

    assert "pools" not in result.failed | result.rejected
    with manager.state.capture():
        assert model_helpers.managed_api_keys(MODEL) == (
            model_helpers.ManagedApiKey("pool_1", "ANTHROPIC_API_KEY_2"),
        )


@pytest.mark.unit
async def test_a_served_document_keeps_its_descriptions():
    descriptions = {"ANTHROPIC_API_KEY": "our main key"}
    body = {**json.loads(_document(1, "ANTHROPIC_API_KEY")), "descriptions": descriptions}
    manager = _manager(PoolsAgent(json.dumps(body).encode()))

    await manager.refresh_once()

    with manager.state.capture():
        snapshot = runtime_pool_snapshot()
        assert snapshot is not None
        assert snapshot.document.descriptions == descriptions


@pytest.mark.unit
async def test_a_stage_without_a_published_document_fails_its_refresh():
    manager = _manager(PoolsAgent(None))

    result = await manager.refresh_once()

    assert "pools" in result.failed
    assert manager.state.current.pools is None


@pytest.mark.unit
async def test_an_older_revision_is_rejected():
    agent = PoolsAgent(_document(5, "ANTHROPIC_API_KEY"))
    manager = _manager(agent)
    await manager.refresh_once()

    agent.pools = _document(4, "ANTHROPIC_API_KEY_2")
    result = await manager.refresh_once()

    assert "pools" in result.rejected
    with manager.state.capture():
        assert model_helpers.current_pool_config().pools == {
            "anthropic/default": ("ANTHROPIC_API_KEY",)
        }


@pytest.mark.unit
def test_without_appconfig_the_gateway_manages_no_keys():
    body = QueryRequest(model=MODEL, inputs=[], run_id="run-a", question_id="q-1")

    assert model_helpers.managed_api_keys(MODEL) == ()
    assert model_helpers._managed_api_key(body) is None  # pyright: ignore[reportPrivateUsage]
