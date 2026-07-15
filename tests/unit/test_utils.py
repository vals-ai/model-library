"""Unit tests for model_library/utils.py"""

from pathlib import Path

import httpx
import pytest

from model_library.agent.agent import AgentResult
from model_library.agent.conductor.metadata import ConductorResult, ConductorStopReason
from model_library.agent.config import TimeLimit
from model_library.base.output import QueryResultMetadata
from model_library.retriers.token.utils import InflightRequest
from model_library import utils as model_utils
from model_library.utils import ValsModel, get_context_window_for_model


class ControlTimingModel(ValsModel):
    max_seconds: float
    tokens_per_second: float


def test_vals_model_does_not_round_unannotated_suffix_fields():
    model = ControlTimingModel(max_seconds=0.0004, tokens_per_second=98.76543)

    assert model.max_seconds == 0.0004
    assert model.tokens_per_second == 98.76543


def test_time_limit_preserves_sub_millisecond_precision():
    assert TimeLimit(max_seconds=0.0004).max_seconds == 0.0004


def test_query_result_metadata_rounds_duration_seconds():
    metadata = QueryResultMetadata(duration_seconds=1.23456)

    assert metadata.duration_seconds == 1.235


def test_agent_result_rounds_final_duration_seconds():
    result = AgentResult(
        final_answer="done",
        final_history=[],
        turns=[],
        final_duration_seconds=1.23456,
        output_dir=Path("."),
    )

    assert result.final_duration_seconds == 1.235


def test_conductor_result_rounds_total_duration_seconds():
    result = ConductorResult(
        messages=[],
        stop_reason=ConductorStopReason.AUDITOR_DONE,
        total_duration_seconds=1.23456,
        output_dir=Path("."),
    )

    assert result.total_duration_seconds == 1.235


def test_inflight_request_rounds_elapsed_seconds():
    request = InflightRequest(
        question_id="question",
        elapsed_seconds=1.23456,
        estimate_input=None,
        estimate_output=None,
        estimate_total=None,
        priority=None,
        attempts=None,
        run_id=None,
        dispatched_at=123.45678,
    )

    assert request.elapsed_seconds == 1.235
    assert request.dispatched_at == 123.45678


def test_get_context_window_for_existing_model():
    """Test that context window is correctly fetched for a model that exists."""
    context_window = get_context_window_for_model("openai/gpt-4o-mini")
    assert context_window == 128_000


def test_get_context_window_for_nonexistent_model():
    """Test that None is returned for a model that doesn't exist."""
    context_window = get_context_window_for_model("nonexistent/fake-model-xyz")
    assert context_window is None


def _assert_provider_timeout(timeout: httpx.Timeout):
    assert timeout.connect == model_utils.PROVIDER_CONNECT_TIMEOUT_SECONDS
    assert timeout.read == model_utils.PROVIDER_READ_TIMEOUT_SECONDS
    assert timeout.write == model_utils.PROVIDER_WRITE_TIMEOUT_SECONDS
    assert timeout.pool == model_utils.PROVIDER_POOL_TIMEOUT_SECONDS


async def test_default_aiohttp_httpx_client_uses_provider_timeout():
    client = model_utils.default_aiohttp_httpx_client()
    try:
        _assert_provider_timeout(client.timeout)
    finally:
        await client.aclose()


async def test_default_httpx_client_uses_provider_timeout():
    client = model_utils.default_httpx_client()
    try:
        _assert_provider_timeout(client.timeout)
    finally:
        await client.aclose()


async def test_gateway_httpx_client_uses_gateway_timeout():
    client = model_utils.gateway_httpx_client(headers={"Authorization": "Bearer test"})
    try:
        assert (
            client.timeout.connect == model_utils.GATEWAY_CLIENT_CONNECT_TIMEOUT_SECONDS
        )
        assert client.timeout.read == model_utils.GATEWAY_CLIENT_READ_TIMEOUT_SECONDS
        assert client.timeout.write == model_utils.GATEWAY_CLIENT_WRITE_TIMEOUT_SECONDS
        assert client.timeout.pool == model_utils.GATEWAY_CLIENT_POOL_TIMEOUT_SECONDS
    finally:
        await client.aclose()


async def test_openai_client_uses_provider_timeout():
    client = model_utils.create_openai_client_with_defaults(api_key="sk-test")
    try:
        assert isinstance(client.timeout, httpx.Timeout)
        _assert_provider_timeout(client.timeout)
    finally:
        await client.close()


async def test_openai_client_uses_aiohttp_httpx_client_factory(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[dict[str, str] | None] = []

    def fake_default_aiohttp_httpx_client(
        *, dns_resolve: dict[str, str] | None = None
    ) -> httpx.AsyncClient:
        calls.append(dns_resolve)
        return httpx.AsyncClient()

    monkeypatch.setattr(
        model_utils,
        "default_aiohttp_httpx_client",
        fake_default_aiohttp_httpx_client,
    )

    client = model_utils.create_openai_client_with_defaults(
        api_key="sk-test", dns_resolve={"api.openai.com": "127.0.0.1"}
    )
    try:
        assert calls == [{"api.openai.com": "127.0.0.1"}]
    finally:
        await client.close()


async def test_anthropic_client_uses_provider_timeout():
    client = model_utils.create_anthropic_client_with_defaults(api_key="sk-test")
    try:
        assert isinstance(client.timeout, httpx.Timeout)
        _assert_provider_timeout(client.timeout)
    finally:
        await client.close()


async def test_anthropic_client_uses_aiohttp_httpx_client_factory(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = 0

    def fake_default_aiohttp_httpx_client(
        *, dns_resolve: dict[str, str] | None = None
    ) -> httpx.AsyncClient:
        nonlocal calls
        assert dns_resolve is None
        calls += 1
        return httpx.AsyncClient()

    monkeypatch.setattr(
        model_utils,
        "default_aiohttp_httpx_client",
        fake_default_aiohttp_httpx_client,
    )

    client = model_utils.create_anthropic_client_with_defaults(api_key="sk-test")
    try:
        assert calls == 1
    finally:
        await client.close()
