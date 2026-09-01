"""Unit tests for model_library/utils.py"""

import socket
from collections.abc import Callable
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


_REGIONAL_NAT_IDLE_TIMEOUT_SECONDS = 350

_PROXY_ENVIRONMENT_VARIABLES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)


def _clear_proxy_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in _PROXY_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(variable, raising=False)


def test_transport_keepalive_starts_before_regional_nat_idle_timeout():
    assert model_utils.TCP_KEEPALIVE_IDLE_SECONDS < _REGIONAL_NAT_IDLE_TIMEOUT_SECONDS


def test_tcp_keepalive_socket_options_include_supported_platform_options():
    expected = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    for option_name, value in (
        ("TCP_KEEPIDLE", model_utils.TCP_KEEPALIVE_IDLE_SECONDS),
        ("TCP_KEEPINTVL", model_utils.TCP_KEEPALIVE_INTERVAL_SECONDS),
        ("TCP_KEEPCNT", model_utils.TCP_KEEPALIVE_PROBES),
    ):
        option = getattr(socket, option_name, None)
        if isinstance(option, int):
            expected.append((socket.IPPROTO_TCP, option, value))

    assert model_utils.tcp_keepalive_socket_options() == tuple(expected)


def test_tcp_keepalive_socket_applies_options(monkeypatch: pytest.MonkeyPatch):
    applied_options: list[model_utils.SocketOption] = []

    class FakeSocket:
        def setsockopt(self, level: int, option: int, value: int) -> None:
            applied_options.append((level, option, value))

    fake_socket = FakeSocket()
    created_with: dict[str, int | socket.AddressFamily | socket.SocketKind] = {}

    def create_socket(
        *,
        family: int | socket.AddressFamily,
        type: int | socket.SocketKind,
        proto: int,
    ) -> FakeSocket:
        created_with.update(family=family, type=type, proto=proto)
        return fake_socket

    monkeypatch.setattr(model_utils.socket, "socket", create_socket)
    addr_info: model_utils.SocketAddressInfo = (
        socket.AF_INET,
        socket.SOCK_STREAM,
        socket.IPPROTO_TCP,
        "",
        ("127.0.0.1", 443),
    )

    assert model_utils.tcp_keepalive_socket(addr_info) is fake_socket
    assert created_with == {
        "family": socket.AF_INET,
        "type": socket.SOCK_STREAM,
        "proto": socket.IPPROTO_TCP,
    }
    assert applied_options == list(model_utils.tcp_keepalive_socket_options())


def test_make_aiohttp_session_wires_keepalive_socket_factory(
    monkeypatch: pytest.MonkeyPatch,
):
    expected_connector = object()
    connector_kwargs: dict[str, object] = {}
    session = object()

    def create_connector(**kwargs: object) -> object:
        connector_kwargs.update(kwargs)
        return expected_connector

    def create_session(*, connector: object) -> object:
        assert connector is expected_connector
        return session

    monkeypatch.setattr(model_utils.aiohttp, "TCPConnector", create_connector)
    monkeypatch.setattr(model_utils.aiohttp, "ClientSession", create_session)

    assert model_utils.make_aiohttp_session() is session
    assert connector_kwargs["socket_factory"] is model_utils.tcp_keepalive_socket


@pytest.mark.parametrize(
    ("client_factory", "uses_http2"),
    [
        (model_utils.default_httpx_client, False),
        (model_utils.gateway_httpx_client, True),
    ],
)
async def test_plain_httpx_clients_mount_keepalive_transport_for_direct_requests(
    monkeypatch: pytest.MonkeyPatch,
    client_factory: Callable[[], httpx.AsyncClient],
    uses_http2: bool,
):
    _clear_proxy_environment(monkeypatch)
    transport_kwargs: dict[str, object] = {}

    class RecordingTransport(httpx.AsyncBaseTransport):
        pass

    keepalive_transport = RecordingTransport()

    def create_transport(**kwargs: object) -> httpx.AsyncBaseTransport:
        transport_kwargs.update(kwargs)
        return keepalive_transport

    monkeypatch.setattr(model_utils.httpx, "AsyncHTTPTransport", create_transport)

    client = client_factory()
    try:
        selected_transport = client._transport_for_url(
            httpx.URL("https://provider.example")
        )
        assert selected_transport is keepalive_transport
        assert transport_kwargs["socket_options"] == (
            model_utils.tcp_keepalive_socket_options()
        )
        limits = transport_kwargs["limits"]
        assert isinstance(limits, httpx.Limits)
        assert limits.max_connections == 2000
        assert limits.max_keepalive_connections == 300
        assert transport_kwargs.get("http2", False) is uses_http2
    finally:
        await client.aclose()


@pytest.mark.parametrize(
    "client_factory",
    [model_utils.default_httpx_client, model_utils.gateway_httpx_client],
)
@pytest.mark.parametrize(
    ("proxy_variable", "request_url"),
    [
        ("HTTP_PROXY", "http://provider.example"),
        ("HTTPS_PROXY", "https://provider.example"),
        ("ALL_PROXY", "https://provider.example"),
    ],
)
async def test_plain_httpx_clients_preserve_proxy_environment_selection(
    monkeypatch: pytest.MonkeyPatch,
    client_factory: Callable[[], httpx.AsyncClient],
    proxy_variable: str,
    request_url: str,
):
    _clear_proxy_environment(monkeypatch)
    monkeypatch.setenv(proxy_variable, "http://proxy.example:8080")

    class KeepaliveTransport(httpx.AsyncBaseTransport):
        pass

    keepalive_transport = KeepaliveTransport()
    monkeypatch.setattr(
        model_utils.httpx,
        "AsyncHTTPTransport",
        lambda **_: keepalive_transport,
    )

    client = client_factory()
    try:
        selected_transport = client._transport_for_url(httpx.URL(request_url))
        assert selected_transport is not keepalive_transport
        assert selected_transport is not client._transport
    finally:
        await client.aclose()


@pytest.mark.parametrize(
    "client_factory",
    [model_utils.default_httpx_client, model_utils.gateway_httpx_client],
)
async def test_plain_httpx_clients_preserve_no_proxy_bypass(
    monkeypatch: pytest.MonkeyPatch,
    client_factory: Callable[[], httpx.AsyncClient],
):
    _clear_proxy_environment(monkeypatch)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
    monkeypatch.setenv("NO_PROXY", "direct.example")

    class KeepaliveTransport(httpx.AsyncBaseTransport):
        pass

    keepalive_transport = KeepaliveTransport()
    monkeypatch.setattr(
        model_utils.httpx,
        "AsyncHTTPTransport",
        lambda **_: keepalive_transport,
    )

    client = client_factory()
    try:
        bypass_transport = client._transport_for_url(
            httpx.URL("https://direct.example")
        )
        proxy_transport = client._transport_for_url(
            httpx.URL("https://provider.example")
        )
        assert bypass_transport is client._transport
        assert proxy_transport is not client._transport
        assert proxy_transport is not keepalive_transport
    finally:
        await client.aclose()


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
