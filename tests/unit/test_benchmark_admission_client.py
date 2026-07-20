import json
from typing import Literal

import httpx
import pytest
from pydantic import SecretStr

import model_library
from model_gateway.benchmark_admission_types import (
    BenchmarkAdmissionOutcome,
    BenchmarkAdmissionResponse,
    BenchmarkCoordinatorError,
)
from model_library.base.base import LLMConfig, TokenRetryParams
from model_library.base.gateway import GatewayLLM
from model_library.retriers.token.benchmark_admission_client import (
    GatewayBenchmarkAdmissionClient,
)


class GatewaySettings:
    MODEL_GATEWAY_API_KEY = "gateway-key"
    MODEL_GATEWAY_URL = "https://gateway.test/"


def _response(
    state: Literal["waiting", "acquired", "released"],
    *,
    model: str = "openai/gpt-4o",
    run_id: str = "run-123",
    outcome: BenchmarkAdmissionOutcome | None = None,
) -> BenchmarkAdmissionResponse:
    return BenchmarkAdmissionResponse(
        state=state,
        model=model,
        run_id=run_id,
        effective_token_limit=10_000,
        outcome=outcome,
    )


def _model(config: LLMConfig | None = None) -> GatewayLLM:
    return GatewayLLM("gpt-4o", "openai", config=config)


def _token_retry_params() -> TokenRetryParams:
    return TokenRetryParams(
        input_modifier=1.0,
        output_modifier=1.0,
        use_dynamic_estimate=True,
        limit=10_000,
    )


async def test_client_uses_model_transport_and_run_id_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []
    registry_key = "openai/canonical-key"

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = request.url.path
        if path == "/benchmark-runs/acquire":
            return httpx.Response(
                200,
                json=_response("waiting", model=registry_key).model_dump(mode="json"),
            )
        if path == "/benchmark-runs/wait":
            return httpx.Response(
                200,
                json=_response("acquired", model=registry_key).model_dump(mode="json"),
            )
        if path == "/benchmark-runs/renew":
            return httpx.Response(
                200,
                json=_response("acquired", model=registry_key).model_dump(mode="json"),
            )
        if path == "/benchmark-runs/release":
            return httpx.Response(
                200,
                json=_response(
                    "released", model=registry_key, outcome="finished"
                ).model_dump(mode="json"),
            )
        raise AssertionError(f"Unexpected path: {path}")

    monkeypatch.setattr(model_library, "model_library_settings", GatewaySettings())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        model = _model()
        setattr(model, "_registry_key", registry_key)
        monkeypatch.setattr(model, "get_client", lambda: http_client)
        client = GatewayBenchmarkAdmissionClient(model)

        await client.acquire(
            run_id="run-123",
            token_retry_params=_token_retry_params(),
            total_requests=12,
            early_release=True,
            immediate_queue_release=True,
        )
        await client.wait(run_id="run-123")
        await client.renew(run_id=" run-123 ")
        await client.release(run_id="run-123", outcome="finished")

    assert [request.url.path for request in requests] == [
        "/benchmark-runs/acquire",
        "/benchmark-runs/wait",
        "/benchmark-runs/renew",
        "/benchmark-runs/release",
    ]
    assert [request.extensions["timeout"]["read"] for request in requests] == [
        10.0,
        30.0,
        10.0,
        10.0,
    ]
    try:
        request_payloads = [json.loads(request.content) for request in requests]
    except json.JSONDecodeError as exc:
        raise AssertionError(f"Admission request was not valid JSON: {exc}") from exc
    assert request_payloads == [
        {
            "model": "openai/canonical-key",
            "config": {},
            "run_id": "run-123",
            "token_retry_params": {
                "input_modifier": 1.0,
                "output_modifier": 1.0,
                "use_dynamic_estimate": True,
                "limit": 10_000,
                "limit_refresh_seconds": 60,
            },
            "total_requests": 12,
            "early_release": True,
            "immediate_queue_release": True,
        },
        {"model": "openai/canonical-key", "run_id": "run-123", "timeout_seconds": 25.0},
        {"model": "openai/canonical-key", "run_id": "run-123"},
        {"model": "openai/canonical-key", "run_id": "run-123", "outcome": "finished"},
    ]


async def test_acquire_serializes_raw_byok_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_payload: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        try:
            request_payload.update(json.loads(request.content))
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f"Admission request was not valid JSON: {exc}"
            ) from exc
        return httpx.Response(200, json=_response("acquired").model_dump(mode="json"))

    monkeypatch.setattr(model_library, "model_library_settings", GatewaySettings())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        model = _model(
            LLMConfig(
                custom_api_key=SecretStr("byok-secret"),
                custom_endpoint="https://provider.test/v1",
            )
        )
        monkeypatch.setattr(model, "get_client", lambda: http_client)

        await GatewayBenchmarkAdmissionClient(model).acquire(
            run_id="run-123",
            token_retry_params=_token_retry_params(),
            total_requests=None,
            early_release=True,
            immediate_queue_release=False,
        )

    assert request_payload["config"] == {
        "custom_api_key": "byok-secret",
        "custom_endpoint": "https://provider.test/v1",
    }


async def test_release_returns_authoritative_terminal_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_response("released", outcome="failed").model_dump(mode="json"),
        )

    monkeypatch.setattr(model_library, "model_library_settings", GatewaySettings())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        model = _model()
        monkeypatch.setattr(model, "get_client", lambda: http_client)

        released = await GatewayBenchmarkAdmissionClient(model).release(
            run_id="run-123",
            outcome="cancelled",
        )

    assert released.outcome == "failed"


async def test_client_rejects_mismatched_run_id_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=_response("acquired", run_id="other-run").model_dump()
        )

    monkeypatch.setattr(model_library, "model_library_settings", GatewaySettings())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        model = _model()
        monkeypatch.setattr(model, "get_client", lambda: http_client)
        client = GatewayBenchmarkAdmissionClient(model)

        with pytest.raises(BenchmarkCoordinatorError, match="identity"):
            await client.acquire(
                run_id="run-123",
                token_retry_params=_token_retry_params(),
                total_requests=None,
                early_release=True,
                immediate_queue_release=False,
            )


async def test_client_maps_gateway_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400, json={"code": "invalid", "message": "invalid admission"}
        )

    monkeypatch.setattr(model_library, "model_library_settings", GatewaySettings())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        model = _model()
        monkeypatch.setattr(model, "get_client", lambda: http_client)

        with pytest.raises(BenchmarkCoordinatorError, match="invalid admission"):
            await GatewayBenchmarkAdmissionClient(model).renew(run_id="run-123")
