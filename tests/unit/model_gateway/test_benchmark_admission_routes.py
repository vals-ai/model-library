import asyncio
import logging
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fakeredis import aioredis
from pydantic import SecretStr

import model_gateway.app as gateway_app
import model_gateway.model_helpers as model_helpers
import model_gateway.routes.benchmark_admission as benchmark_admission_routes
from model_gateway.benchmark_admission_types import BenchmarkAcquireRequest
from model_gateway.cache import ModelCache
from model_gateway.types import QueryRequest, TokenCountRequest
from tests.unit.model_gateway._support import HEADERS, _make_client
from model_library.base import (
    FileWithId,
    LLMConfig,
    ResolvedTokenRetryParams,
    TokenRetryParams,
)
from model_library.register_models import DefaultRateLimit
from model_library.retriers.token import utils as token_utils
from model_library.retriers.token.utils import set_redis_client
from model_library.settings import ModelLibrarySettings

MODEL = "openai/gpt-4o"
MODEL_KEY = ("openai.gpt-4o", "server-key-hash")

logger = logging.getLogger("test_benchmark_admission_routes")


class FakeLLM:
    def __init__(self) -> None:
        self.token_retry_params: TokenRetryParams | None = None
        self._resolved_token_retry_params: ResolvedTokenRetryParams | None = None
        self._client_registry_key_model_specific = MODEL_KEY
        self.init_calls: list[ResolvedTokenRetryParams] = []

    async def ensure_resolved_token_retry(
        self,
        params: TokenRetryParams,
        resolved_params: ResolvedTokenRetryParams,
    ) -> None:
        if self._resolved_token_retry_params != resolved_params:
            self.token_retry_params = params
            self._resolved_token_retry_params = resolved_params
            self.init_calls.append(resolved_params)


@pytest.fixture
def redis():
    client = aioredis.FakeRedis(decode_responses=True)
    set_redis_client(client)
    return client


def _token_retry_params() -> TokenRetryParams:
    return TokenRetryParams(
        input_modifier=1.0,
        output_modifier=1.0,
        limit=10_000,
    )


def _acquire_body(
    run_id: str,
    *,
    total_requests: int | None = None,
    config: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "model": MODEL,
        "config": config or {},
        "run_id": run_id,
        "token_retry_params": {
            "input_modifier": 1.0,
            "output_modifier": 1.0,
            "use_dynamic_estimate": True,
            "limit": 10_000,
            "limit_refresh_seconds": 60,
        },
        "total_requests": total_requests,
        "early_release": True,
        "immediate_queue_release": False,
    }


@pytest.mark.parametrize(
    "config",
    [
        LLMConfig(),
        LLMConfig(custom_api_key=SecretStr("byok-secret")),
        LLMConfig(
            custom_api_key=SecretStr("byok-secret"),
            custom_endpoint="https://provider.test/v1",
        ),
    ],
)
async def test_gateway_paths_share_identity_but_only_provider_ops_start_token_retry(
    config: LLMConfig,
) -> None:
    cache = ModelCache()
    llm = FakeLLM()
    params = _token_retry_params().model_copy(update={"requests_per_minute": 77})
    query = QueryRequest(
        model=MODEL,
        config=config,
        inputs=[],
        token_retry_params=params,
    )
    token_count = TokenCountRequest(
        model=MODEL,
        config=config,
        inputs=[],
        token_retry_params=params,
    )
    acquire = BenchmarkAcquireRequest(
        model=MODEL,
        config=config,
        run_id="run-1",
        token_retry_params=params,
    )
    resolved_params = model_helpers.resolve_gateway_token_retry_params(
        acquire.model,
        acquire.token_retry_params,
    )

    with patch.object(model_helpers, "get_registry_model", return_value=llm):
        admission_llm = model_helpers.get_gateway_llm(
            cache,
            acquire,
            resolved_token_retry_params=resolved_params,
        )
        assert llm.init_calls == []
        query_llm = await model_helpers.get_query_llm(cache, query)
        token_count_llm = await model_helpers.get_query_llm(cache, token_count)

    assert query_llm is token_count_llm is admission_llm
    assert llm.token_retry_params == params
    assert llm.init_calls == [resolved_params]
    assert resolved_params.requests_per_minute == 77
async def test_admission_routes_coordinate_through_shared_redis(
    redis,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = FakeLLM()
    resolve_llm = MagicMock(return_value=llm)
    monkeypatch.setenv("REDIS_URL", "redis://shared-benchmark-admission")

    with (
        patch.object(
            gateway_app.async_redis, "from_url", return_value=redis
        ) as from_url,
        patch.object(
            benchmark_admission_routes,
            "get_gateway_llm",
            resolve_llm,
        ),
        _make_client() as first_client,
        _make_client() as second_client,
    ):
        first = first_client.post(
            "/benchmark-runs/acquire",
            headers=HEADERS,
            json=_acquire_body("run-1"),
        )
        second = second_client.post(
            "/benchmark-runs/acquire",
            headers=HEADERS,
            json=_acquire_body("run-2"),
        )
        waiting = first_client.post(
            "/benchmark-runs/wait",
            headers=HEADERS,
            json={"model": MODEL, "run_id": "run-2", "timeout_seconds": 0.0},
        )
        renewed = second_client.post(
            "/benchmark-runs/renew",
            headers=HEADERS,
            json={"model": MODEL, "run_id": "run-1"},
        )
        released = first_client.post(
            "/benchmark-runs/release",
            headers=HEADERS,
            json={"model": MODEL, "run_id": "run-1", "outcome": "finished"},
        )
        acquired = second_client.post(
            "/benchmark-runs/wait",
            headers=HEADERS,
            json={"model": MODEL, "run_id": "run-2", "timeout_seconds": 0.0},
        )

    assert first.status_code == 200
    assert first.json()["state"] == "acquired"
    assert second.status_code == 200
    assert second.json()["state"] == "waiting"
    assert waiting.json()["state"] == "waiting"
    assert renewed.json()["state"] == "acquired"
    assert released.json()["outcome"] == "finished"
    assert acquired.json()["state"] == "acquired"
    assert {
        response.json()["effective_token_limit"]
        for response in [first, second, waiting, renewed, released, acquired]
    } == {10_000}
    assert [call.args[0] for call in from_url.call_args_list] == [
        "redis://shared-benchmark-admission",
        "redis://shared-benchmark-admission",
    ]
    assert [call.args[1].run_id for call in resolve_llm.call_args_list] == [
        "run-1",
        "run-2",
    ]


async def test_control_role_serves_authenticated_admission_through_lifespan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = FakeLLM()
    lifecycle_redis = aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setenv("GATEWAY_RUNTIME_ROLE", "control")
    monkeypatch.setenv("REDIS_URL", "redis://control-benchmark-admission")
    monkeypatch.setattr(token_utils, "redis_client", None)

    with (
        patch.object(
            gateway_app.async_redis,
            "from_url",
            return_value=lifecycle_redis,
        ),
        patch.object(
            benchmark_admission_routes,
            "get_gateway_llm",
            return_value=llm,
        ),
        _make_client() as client,
    ):
        response = client.post(
            "/benchmark-runs/acquire",
            headers=HEADERS,
            json=_acquire_body("control-run"),
        )

    assert response.status_code == 200
    assert response.json()["state"] == "acquired"
    assert llm.init_calls == []


@pytest.mark.parametrize(
    "path",
    [
        "/benchmark-runs/acquire",
        "/benchmark-runs/wait",
        "/benchmark-runs/renew",
        "/benchmark-runs/release",
    ],
)
def test_admission_routes_require_auth(path: str) -> None:
    response = _make_client().post(path, json={})

    assert response.status_code == 401
    assert response.json()["code"] == "unauthorized"


def test_admission_validation_and_conflict_envelopes(redis) -> None:
    llm = FakeLLM()
    with patch.object(model_helpers, "get_registry_model", return_value=llm):
        client = _make_client()
        invalid = client.post(
            "/benchmark-runs/acquire",
            headers=HEADERS,
            json={"model": MODEL, "run_id": "run-1"},
        )
        first = client.post(
            "/benchmark-runs/acquire",
            headers=HEADERS,
            json=_acquire_body("run-1", total_requests=1),
        )
        conflict = client.post(
            "/benchmark-runs/acquire",
            headers=HEADERS,
            json=_acquire_body("run-1", total_requests=2),
        )

    assert invalid.status_code == 400
    assert invalid.json()["code"] == "invalid_request"
    assert first.status_code == 200
    assert conflict.status_code == 409
    assert conflict.json()["code"] == "benchmark_admission_conflict"


def test_admission_returns_503_when_redis_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(token_utils, "redis_client", None)
    response = _make_client().post(
        "/benchmark-runs/renew",
        headers=HEADERS,
        json={"model": MODEL, "run_id": "run-1"},
    )

    assert response.status_code == 503
    assert response.json()["code"] == "benchmark_admission_unavailable"


def test_admission_returns_503_when_operation_exceeds_server_budget(
    redis,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def never_complete(_operation):
        await asyncio.Event().wait()

    monkeypatch.setattr(
        benchmark_admission_routes,
        "SHORT_OPERATION_TIMEOUT_SECONDS",
        0.001,
    )
    monkeypatch.setattr(
        benchmark_admission_routes,
        "_execute_admission_operation",
        never_complete,
    )

    response = _make_client().post(
        "/benchmark-runs/renew",
        headers=HEADERS,
        json={"model": MODEL, "run_id": "run-1"},
    )

    assert response.status_code == 503
    assert response.json()["code"] == "benchmark_admission_unavailable"


async def test_acquire_never_persists_or_logs_raw_config_secret(
    redis,
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = FakeLLM()
    secret = "raw-byok-secret"
    with patch.object(model_helpers, "get_registry_model", return_value=llm):
        with caplog.at_level(logging.DEBUG):
            response = _make_client().post(
                "/benchmark-runs/acquire",
                headers=HEADERS,
                json=_acquire_body(
                    "run-1",
                    config={
                        "custom_api_key": secret,
                        "custom_endpoint": "https://provider.test/v1",
                    },
                ),
            )

    persisted: list[str] = []
    async for key in redis.scan_iter(match="*"):
        key_type = await redis.type(key)
        if key_type == "hash":
            persisted.extend((await redis.hgetall(key)).values())
        elif key_type == "list":
            persisted.extend(await redis.lrange(key, 0, -1))
        elif key_type == "string":
            value = await redis.get(key)
            if value is not None:
                persisted.append(value)

    assert response.status_code == 200
    assert secret not in response.text
    assert secret not in caplog.text
    assert all(secret not in value for value in persisted)
