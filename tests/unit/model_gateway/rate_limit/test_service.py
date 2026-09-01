import asyncio
from unittest.mock import MagicMock, patch

import httpx
import pytest

import model_gateway.model_helpers as model_helpers
import model_gateway.routes.rate_limit as rate_limit_route
from model_gateway.cache import ModelCache
from model_gateway.types import RateLimitRequest
from model_library.rate_limits import (
    RateLimit,
    RateLimitCapacity,
    RequestRateLimit,
    TokenRateLimit,
)
from tests.unit.model_gateway._support import HEADERS, _load_json, _make_client

@pytest.mark.parametrize(
    "model",
    ("openai/gpt-4o", "anthropic/claude-sonnet-4-5-20250929"),
)
def test_rate_limit_returns_provider_limits(model: str):
    from model_library.rate_limits import RateLimit

    class FakeLLM:
        async def get_rate_limit(self):
            return RateLimit(
                unix_timestamp=1_700_000_000.0,
                requests=(RequestRateLimit(limit=4_000, remaining=3_999),),
                tokens=TokenRateLimit(
                    input=RateLimitCapacity(limit=80_000, remaining=79_000),
                    output=RateLimitCapacity(limit=16_000, remaining=15_500),
                ),
            )

    client = _make_client()
    rate_body = {"model": model, "config": {}}
    assert client.post("/rate-limit", json=rate_body).status_code == 401

    with patch.object(
        model_helpers, "get_registry_model", side_effect=lambda model, config: FakeLLM()
    ):
        resp = client.post("/rate-limit", headers=HEADERS, json=rate_body)

    assert resp.status_code == 200
    assert resp.json() == {
        "rate_limit": {
            "requests": [
                {"limit": 4_000, "remaining": 3_999, "mode": "sliding_window"}
            ],
            "tokens": {
                "mode": "token_bucket",
                "input": {"limit": 80_000, "remaining": 79_000},
                "output": {"limit": 16_000, "remaining": 15_500},
            },
            "unix_timestamp": 1_700_000_000.0,
        }
    }

def test_rate_limit_cache_evicts_oldest_entry_at_max_size():
    probes = 0

    class FakeLLM:
        async def get_rate_limit(self):
            nonlocal probes
            probes += 1
            return RateLimit(
                unix_timestamp=1_700_000_000.0,
                requests=(RequestRateLimit(limit=probes),),
            )

    client = _make_client()
    first_body = {"model": "openai/gpt-4o", "config": {}}
    second_body = {"model": "openai/gpt-4o-mini", "config": {}}

    with (
        patch.object(rate_limit_route, "RATE_LIMIT_CACHE_MAXSIZE", 1),
        patch.object(
            model_helpers,
            "get_registry_model",
            side_effect=lambda model, config: FakeLLM(),
        ),
    ):
        first = client.post("/rate-limit", headers=HEADERS, json=first_body).json()
        second = client.post("/rate-limit", headers=HEADERS, json=second_body).json()
        reprobed = client.post("/rate-limit", headers=HEADERS, json=first_body).json()

    assert probes == 3
    assert first["rate_limit"]["requests"][0]["limit"] == 1
    assert second["rate_limit"]["requests"][0]["limit"] == 2
    assert reprobed["rate_limit"]["requests"][0]["limit"] == 3

def test_managed_anthropic_rate_limit_rejects_invalid_model() -> None:
    client = _make_client()

    response = client.post(
        "/rate-limit",
        headers=HEADERS,
        json={"model": "anthropic/not-a-real-model", "config": {}},
    )

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_model"

def test_rate_limit_serves_repeat_requests_from_cache():
    from model_library.rate_limits import RateLimit

    probes = 0

    class FakeLLM:
        async def get_rate_limit(self):
            nonlocal probes
            probes += 1
            return RateLimit(
                unix_timestamp=1_700_000_000.0,
                requests=(RequestRateLimit(limit=probes),),
            )

    client = _make_client()
    body = {"model": "openai/gpt-4o", "config": {}}
    with patch.object(
        model_helpers, "get_registry_model", side_effect=lambda model, config: FakeLLM()
    ):
        first = client.post("/rate-limit", headers=HEADERS, json=body).json()
        second = client.post("/rate-limit", headers=HEADERS, json=body).json()
        other = client.post(
            "/rate-limit",
            headers=HEADERS,
            json={"model": "openai/gpt-4o-mini", "config": {}},
        ).json()

    assert probes == 2
    assert first == second
    assert other["rate_limit"]["requests"][0]["limit"] == 2

@pytest.mark.asyncio
async def test_rate_limit_cache_reprobes_after_ttl():
    probes = 0
    clock = 0.0

    class FakeLLM:
        async def get_rate_limit(self):
            nonlocal probes
            probes += 1
            return RateLimit(
                unix_timestamp=1_700_000_000.0,
                requests=(RequestRateLimit(limit=probes),),
            )

    service = rate_limit_route.RateLimitProbeService(ModelCache())
    body = RateLimitRequest.model_validate({"model": "openai/gpt-4o", "config": {}})
    with (
        patch.object(
            rate_limit_route.time,
            "monotonic",
            side_effect=lambda: clock,
        ),
        patch.object(
            model_helpers,
            "get_registry_model",
            side_effect=lambda model, config: FakeLLM(),
        ),
    ):
        first = await service.get_rate_limit(body)
        cached = await service.get_rate_limit(body)
        clock = rate_limit_route.RATE_LIMIT_CACHE_SECONDS
        refreshed = await service.get_rate_limit(body)

    assert probes == 2
    assert _load_json(first.body) == _load_json(cached.body)
    assert _load_json(refreshed.body)["rate_limit"]["requests"][0]["limit"] == 2

def test_rate_limit_serves_repeat_no_data_requests_from_cache():
    probes = 0

    class FakeLLM:
        async def get_rate_limit(self):
            nonlocal probes
            probes += 1
            return None

    client = _make_client()
    body = {"model": "openai/gpt-4o", "config": {}}
    with patch.object(
        model_helpers, "get_registry_model", side_effect=lambda model, config: FakeLLM()
    ):
        first = client.post("/rate-limit", headers=HEADERS, json=body)
        second = client.post("/rate-limit", headers=HEADERS, json=body)

    assert first.status_code == 200
    assert first.json() == {}
    assert second.json() == {}
    assert probes == 1

@pytest.mark.parametrize(
    "config",
    [
        pytest.param({"custom_api_key": "caller-key"}, id="custom-api-key"),
        pytest.param(
            {"custom_endpoint": "https://caller.example/v1"},
            id="custom-endpoint",
        ),
        pytest.param(
            {
                "custom_api_key": "caller-key",
                "custom_endpoint": "https://caller.example/v1",
            },
            id="custom-api-key-and-endpoint",
        ),
    ],
)
@pytest.mark.asyncio
async def test_rate_limit_custom_connection_returns_no_data_without_probe(
    config: dict[str, str],
) -> None:
    cache = MagicMock(spec=ModelCache)
    service = rate_limit_route.RateLimitProbeService(cache)
    body = RateLimitRequest.model_validate(
        {"model": "openai/gpt-4o", "config": config}
    )
    with (
        patch.object(
            rate_limit_route,
            "dump_llm_config",
            side_effect=AssertionError("custom config must not be serialized"),
        ) as dump_llm_config,
        patch.object(model_helpers, "get_registry_model") as get_registry_model,
    ):
        response = await service.get_rate_limit(body)

    assert response.status_code == 200
    assert _load_json(response.body) == {}
    dump_llm_config.assert_not_called()
    cache.make_key.assert_not_called()
    get_registry_model.assert_not_called()

@pytest.mark.asyncio
async def test_rate_limit_collapses_concurrent_probes_for_one_key():
    from model_library.rate_limits import RateLimit

    probes = 0

    class FakeLLM:
        async def get_rate_limit(self):
            nonlocal probes
            probes += 1
            await asyncio.sleep(0)
            return RateLimit(
                unix_timestamp=1_700_000_000.0,
                requests=(RequestRateLimit(limit=probes),),
            )

    app = _make_client().app
    body = {"model": "openai/gpt-4o", "config": {}}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=("testclient", 50000)),
        base_url="http://gateway.test",
        headers=HEADERS,
    ) as client:
        with (
            patch.object(rate_limit_route, "RATE_LIMIT_MAX_IN_FLIGHT_PROBES", 1),
            patch.object(
                model_helpers,
                "get_registry_model",
                side_effect=lambda model, config: FakeLLM(),
            ),
        ):
            first, second = await asyncio.gather(
                client.post("/rate-limit", json=body),
                client.post("/rate-limit", json=body),
            )

    assert probes == 1
    assert first.status_code == 200
    assert first.json() == second.json()

@pytest.mark.asyncio
async def test_rate_limit_waiter_cancellation_does_not_cancel_shared_probe():
    from model_library.rate_limits import RateLimit

    probes = 0
    started = asyncio.Event()
    release = asyncio.Event()
    provider_cancelled = asyncio.Event()

    class FakeLLM:
        async def get_rate_limit(self):
            nonlocal probes
            probes += 1
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                provider_cancelled.set()
                raise
            return RateLimit(
                unix_timestamp=1_700_000_000.0,
                requests=(RequestRateLimit(limit=probes),),
            )

    app = _make_client().app
    body = {"model": "openai/gpt-4o", "config": {}}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=("testclient", 50000)),
        base_url="http://gateway.test",
        headers=HEADERS,
    ) as client:
        with patch.object(
            model_helpers,
            "get_registry_model",
            side_effect=lambda model, config: FakeLLM(),
        ):
            cancelled_waiter = asyncio.create_task(
                client.post("/rate-limit", json=body)
            )
            await started.wait()
            remaining_waiter = asyncio.create_task(
                client.post("/rate-limit", json=body)
            )
            await asyncio.sleep(0)
            cancelled_waiter.cancel()
            try:
                with pytest.raises(asyncio.CancelledError):
                    await cancelled_waiter
                assert not provider_cancelled.is_set()
            finally:
                release.set()
            remaining = await remaining_waiter

    assert not provider_cancelled.is_set()
    assert remaining.status_code == 200
    assert remaining.json()["rate_limit"]["requests"][0]["limit"] == 1
    assert probes == 1

@pytest.mark.asyncio
async def test_rate_limit_service_close_cancels_and_awaits_in_flight_probe():
    provider_started = asyncio.Event()
    provider_cancelled = asyncio.Event()
    cleanup_release = asyncio.Event()
    cleanup_finished = asyncio.Event()

    class FakeLLM:
        async def get_rate_limit(self):
            provider_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                provider_cancelled.set()
                await cleanup_release.wait()
                cleanup_finished.set()
                raise

    service = rate_limit_route.RateLimitProbeService(ModelCache())
    body = RateLimitRequest.model_validate({"model": "openai/gpt-4o", "config": {}})
    with patch.object(
        model_helpers,
        "get_registry_model",
        side_effect=lambda model, config: FakeLLM(),
    ):
        waiter = asyncio.create_task(service.get_rate_limit(body))
        await provider_started.wait()
        close = asyncio.create_task(service.close())
        await provider_cancelled.wait()
        await asyncio.sleep(0)

        assert not cleanup_finished.is_set()
        assert not close.done()

        cleanup_release.set()
        await close
        with pytest.raises(asyncio.CancelledError):
            await waiter

    assert cleanup_finished.is_set()

@pytest.mark.asyncio
async def test_rate_limit_rejects_new_distinct_probe_when_capacity_is_full():
    from model_library.rate_limits import RateLimit

    probes = 0
    started = asyncio.Event()
    release = asyncio.Event()

    class FakeLLM:
        async def get_rate_limit(self):
            nonlocal probes
            probes += 1
            started.set()
            await release.wait()
            return RateLimit(
                unix_timestamp=1_700_000_000.0,
                requests=(RequestRateLimit(limit=probes),),
            )

    app = _make_client().app
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=("testclient", 50000)),
        base_url="http://gateway.test",
        headers=HEADERS,
    ) as client:
        with (
            patch.object(rate_limit_route, "RATE_LIMIT_MAX_IN_FLIGHT_PROBES", 1),
            patch.object(
                model_helpers,
                "get_registry_model",
                side_effect=lambda model, config: FakeLLM(),
            ),
        ):
            first_request = asyncio.create_task(
                client.post(
                    "/rate-limit",
                    json={"model": "openai/gpt-4o", "config": {}},
                )
            )
            await started.wait()
            try:
                overloaded = await client.post(
                    "/rate-limit",
                    json={"model": "openai/gpt-4o-mini", "config": {}},
                )
            finally:
                release.set()
            first = await first_request

    assert first.status_code == 200
    assert overloaded.status_code == 429
    assert overloaded.json() == {
        "code": "gateway_overloaded",
        "message": "Gateway rate-limit probe capacity is full",
    }
    assert probes == 1

def test_rate_limit_sanitizes_and_does_not_cache_provider_exception():
    probes = 0
    secret = "Bearer provider-secret"

    class ProviderProbeError(RuntimeError):
        code = "secret-provider-code"
        status_code = 503

    class FakeLLM:
        async def get_rate_limit(self):
            nonlocal probes
            probes += 1
            raise ProviderProbeError(secret)

    client = _make_client()
    with patch.object(
        model_helpers, "get_registry_model", side_effect=lambda model, config: FakeLLM()
    ):
        responses = [
            client.post(
                "/rate-limit",
                headers=HEADERS,
                json={"model": "openai/gpt-4o", "config": {}},
            )
            for _ in range(2)
        ]

    assert probes == 2
    for response in responses:
        assert response.status_code == 200
        assert response.json()["error"] == {
            "type": "ProviderError",
            "message": "Provider rate-limit probe failed",
            "provider": "openai",
            "exception_type": "ProviderProbeError",
            "status_code": 503,
        }
        assert secret not in response.text
        assert "secret-provider-code" not in response.text

def test_rate_limit_sanitizes_returned_provider_error():
    from model_gateway.types import ProviderError

    secret = "raw-provider-secret"

    class FakeLLM:
        async def get_rate_limit(self):
            return ProviderError(
                message=secret,
                provider=secret,
                code=secret,
                exception_type="ProviderFailure",
                status_code=429,
            )

    client = _make_client()
    with patch.object(
        model_helpers, "get_registry_model", side_effect=lambda model, config: FakeLLM()
    ):
        response = client.post(
            "/rate-limit",
            headers=HEADERS,
            json={"model": "openai/gpt-4o", "config": {}},
        )

    assert response.status_code == 200
    assert response.json()["error"] == {
        "type": "ProviderError",
        "message": "Provider rate-limit probe failed",
        "provider": "openai",
        "exception_type": "ProviderFailure",
        "status_code": 429,
    }
    assert secret not in response.text

@pytest.mark.asyncio
async def test_rate_limit_timeout_returns_error_and_releases_probe_capacity():
    probes = 0
    blocked = asyncio.Event()
    cancelled = asyncio.Event()

    class FakeLLM:
        async def get_rate_limit(self):
            nonlocal probes
            probes += 1
            if probes == 1:
                try:
                    await blocked.wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
            return None

    app = _make_client().app
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=("testclient", 50000)),
        base_url="http://gateway.test",
        headers=HEADERS,
    ) as client:
        with (
            patch.object(rate_limit_route, "RATE_LIMIT_MAX_IN_FLIGHT_PROBES", 1),
            patch.object(rate_limit_route, "RATE_LIMIT_PROBE_TIMEOUT_SECONDS", 0.01),
            patch.object(
                model_helpers,
                "get_registry_model",
                side_effect=lambda model, config: FakeLLM(),
            ),
        ):
            first = await client.post(
                "/rate-limit",
                json={"model": "openai/gpt-4o", "config": {}},
            )
            await asyncio.sleep(0)
            second = await client.post(
                "/rate-limit",
                json={"model": "openai/gpt-4o-mini", "config": {}},
            )

    assert first.status_code == 200
    assert first.json()["error"]["exception_type"] == "TimeoutError"
    assert second.status_code == 200
    assert second.json() == {}
    assert probes == 2
    assert cancelled.is_set()
