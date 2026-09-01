from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import SecretStr

from model_library.base import LLMConfig
from model_library.providers.delegates.alibaba import AlibabaConfig, AlibabaModel
from model_library.providers.delegates.fireworks import FireworksModel
from model_library.providers.delegates.kimi import KimiModel
from model_library.providers.mistral import MistralModel
from model_library.providers.xai import XAIModel
from model_library.rate_limits import (
    RateLimit,
    RateLimitCapacity,
    RequestRateLimit,
    TokenRateLimit,
)

_DATE_HEADER = "Mon, 01 Jan 2024 00:00:00 GMT"
_DATE_TIMESTAMP = 1_704_067_200.0


def _response(
    payload: object,
    *,
    status_code: int = 200,
    url: str = "https://provider.example/quota",
) -> httpx.Response:
    return httpx.Response(
        status_code,
        json=payload,
        headers={"date": _DATE_HEADER},
        request=httpx.Request("GET", url),
    )


def _client(response: httpx.Response) -> tuple[MagicMock, AsyncMock]:
    get = AsyncMock(return_value=response)
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=MagicMock(get=get))
    client.__aexit__ = AsyncMock(return_value=False)
    return client, get


@pytest.mark.parametrize(
    ("model_type", "model_name", "probe_target"),
    [
        pytest.param(
            AlibabaModel,
            "qwen3.8-max",
            "model_library.providers.delegates.alibaba.default_httpx_client",
            id="alibaba",
        ),
        pytest.param(
            KimiModel,
            "kimi-k3",
            "model_library.providers.delegates.kimi.default_httpx_client",
            id="kimi",
        ),
        pytest.param(
            MistralModel,
            "mistral-medium-latest",
            "model_library.providers.mistral.probe_chat_completions_rate_limit",
            id="mistral",
        ),
        pytest.param(
            XAIModel,
            "grok-3-mini",
            "model_library.providers.xai.probe_chat_completions_rate_limit",
            id="xai",
        ),
    ],
)
@pytest.mark.parametrize("custom_field", ["custom_api_key", "custom_endpoint"])
async def test_custom_connection_skips_default_account_probe(
    model_type,
    model_name: str,
    probe_target: str,
    custom_field: str,
    monkeypatch,
) -> None:
    monkeypatch.setattr(model_type, "_get_default_api_key", lambda _: "default-key")
    config = (
        LLMConfig(custom_api_key=SecretStr("custom-key"))
        if custom_field == "custom_api_key"
        else LLMConfig(custom_endpoint="https://custom.example/v1")
    )
    model = model_type(model_name, config=config)

    with patch(
        probe_target,
        side_effect=AssertionError("unexpected default probe"),
    ) as probe:
        assert await model.get_rate_limit() is None

    probe.assert_not_called()


async def test_mistral_probe_uses_default_provider_config(monkeypatch):
    monkeypatch.setattr(MistralModel, "_get_default_api_key", lambda _: "default-key")
    with patch(
        "model_library.providers.mistral.probe_chat_completions_rate_limit",
        new_callable=AsyncMock,
    ) as probe:
        model = MistralModel("mistral-medium-latest")
        await model.get_rate_limit()

    probe.assert_awaited_once_with(
        base_url="https://api.mistral.ai/v1",
        api_key="default-key",
        model_name="mistral-medium-latest",
    )


async def test_xai_native_probe_uses_default_provider_config(monkeypatch):
    monkeypatch.setattr(XAIModel, "_get_default_api_key", lambda _: "default-key")
    with patch(
        "model_library.providers.xai.probe_chat_completions_rate_limit",
        new_callable=AsyncMock,
    ) as probe:
        model = XAIModel("grok-3-mini")
        assert model.delegate is None
        await model.get_rate_limit()

    probe.assert_awaited_once_with(
        base_url="https://api.x.ai/v1",
        api_key="default-key",
        model_name="grok-3-mini",
    )


def _fireworks_model() -> FireworksModel:
    model = FireworksModel(
        "test-model",
        config=LLMConfig(custom_api_key=SecretStr("test-key")),
    )
    assert model.delegate is not None
    return model


async def test_fireworks_scales_typed_directional_limits() -> None:
    delegated = RateLimit(
        requests=(RequestRateLimit(limit=11, remaining=7),),
        tokens=TokenRateLimit(
            input=RateLimitCapacity(limit=100, remaining=90),
            uncached_input=RateLimitCapacity(limit=40),
            output=RateLimitCapacity(limit=20, remaining=10),
        ),
        scope="api_key",
        unix_timestamp=_DATE_TIMESTAMP,
    )
    model = _fireworks_model()
    assert model.delegate is not None
    model.delegate.get_rate_limit = AsyncMock(return_value=delegated)

    rate_limit = await model.get_rate_limit()

    assert rate_limit is not None
    assert rate_limit.requests == delegated.requests
    assert rate_limit.tokens == TokenRateLimit(
        input=RateLimitCapacity(limit=6_000, remaining=5_400),
        uncached_input=RateLimitCapacity(limit=2_400),
        output=RateLimitCapacity(limit=1_200, remaining=600),
    )
    assert rate_limit.scope == "api_key"
    assert rate_limit.unix_timestamp == _DATE_TIMESTAMP


async def test_fireworks_keeps_uncached_input_optional() -> None:
    delegated = RateLimit(
        tokens=TokenRateLimit(
            input=RateLimitCapacity(limit=3, remaining=2),
            output=RateLimitCapacity(limit=1),
        ),
        unix_timestamp=_DATE_TIMESTAMP,
    )
    model = _fireworks_model()
    assert model.delegate is not None
    model.delegate.get_rate_limit = AsyncMock(return_value=delegated)

    rate_limit = await model.get_rate_limit()

    assert rate_limit is not None
    assert rate_limit.tokens == TokenRateLimit(
        input=RateLimitCapacity(limit=180, remaining=120),
        output=RateLimitCapacity(limit=60),
    )


async def test_fireworks_returns_non_directional_limits_unchanged() -> None:
    delegated = RateLimit(
        tokens=TokenRateLimit(total=RateLimitCapacity(limit=100, remaining=50)),
        unix_timestamp=_DATE_TIMESTAMP,
    )
    model = _fireworks_model()
    assert model.delegate is not None
    model.delegate.get_rate_limit = AsyncMock(return_value=delegated)

    assert await model.get_rate_limit() is delegated


@pytest.mark.parametrize("reverse", [False, True])
async def test_alibaba_selects_tightest_normalized_limits_independent_of_order(
    reverse: bool,
    monkeypatch,
):
    monkeypatch.setattr(AlibabaModel, "_default_api_key", lambda _: "default-key")
    model = AlibabaModel("qwen3.8-max")
    matching = [
        {
            "model": "qwen3.8-max",
            "model_limit": {
                "request_limit": 0,
                "request_limit_period": 1,
                "usage_limit": 500_000,
                "usage_limit_field": "total_tokens",
                "usage_limit_period": 6,
            },
        },
        {
            "model": "qwen3.8-max",
            "model_limit": {
                "request_limit": 1_500,
                "request_limit_period": 6,
                "usage_limit": 300_000,
                "usage_limit_field": "total_tokens",
                "usage_limit_period": 60,
            },
        },
        {
            "model": "qwen3.8-max",
            "model_limit": {
                "request_limit": 12,
                "request_limit_period": 60,
                "usage_limit": 12,
                "usage_limit_field": "characters",
                "usage_limit_period": 60,
            },
        },
    ]
    if reverse:
        matching.reverse()
    client, get = _client(
        _response(
            {
                "request_id": "safe-extra-field",
                "output": {
                    "quotas": [
                        *matching,
                        {
                            "model": "another-model",
                            "model_limit": {
                                "request_limit": 1,
                                "request_limit_period": 60,
                            },
                        },
                    ]
                },
            }
        )
    )

    with patch(
        "model_library.providers.delegates.alibaba.default_httpx_client",
        return_value=client,
    ):
        rate_limit = await model.get_rate_limit()

    assert rate_limit is not None
    assert rate_limit.requests[0].limit == 0
    assert rate_limit.requests[0].remaining is None
    assert rate_limit.tokens is not None
    assert rate_limit.tokens.total is not None
    assert rate_limit.tokens.total.limit == 300_000
    assert rate_limit.tokens.total.remaining is None
    assert rate_limit.scope == "api_key"
    assert rate_limit.model_dump(mode="json", exclude_none=True) == {
        "requests": [{"limit": 0, "mode": "sliding_window"}],
        "tokens": {"mode": "token_bucket", "total": {"limit": 300_000}},
        "scope": "api_key",
        "unix_timestamp": _DATE_TIMESTAMP,
    }
    get.assert_awaited_once_with(
        "https://dashscope-intl.aliyuncs.com/api/v1/quotas",
        headers={"Authorization": "Bearer default-key"},
        params={"model": "qwen3.8-max"},
    )


async def test_alibaba_uses_mainland_endpoint_and_returns_none_without_match(
    monkeypatch,
):
    monkeypatch.setattr(AlibabaModel, "_default_api_key", lambda _: "mainland-key")
    model = AlibabaModel(
        "qwen3.8-max-cn",
        config=LLMConfig(provider_config=AlibabaConfig(mainland=True)),
    )
    client, get = _client(
        _response(
            {
                "output": {
                    "quotas": [
                        {
                            "model": "another-model",
                            "model_limit": {
                                "request_limit": 10,
                                "request_limit_period": 1,
                            },
                        }
                    ]
                }
            }
        )
    )

    with patch(
        "model_library.providers.delegates.alibaba.default_httpx_client",
        return_value=client,
    ):
        assert await model.get_rate_limit() is None

    assert get.await_count == 1
    awaited = get.await_args
    assert awaited is not None
    assert awaited.args == ("https://dashscope.aliyuncs.com/api/v1/quotas",)
    assert awaited.kwargs == {
        "headers": {"Authorization": "Bearer mainland-key"},
        "params": {"model": "qwen3.8-max-cn"},
    }


async def test_alibaba_propagates_authentication_failure():
    model = AlibabaModel("qwen3.8-max")
    client, _ = _client(_response({}, status_code=401))

    with (
        patch(
            "model_library.providers.delegates.alibaba.default_httpx_client",
            return_value=client,
        ),
        pytest.raises(httpx.HTTPStatusError),
    ):
        await model.get_rate_limit()


async def test_kimi_reports_rpm_and_concurrency_and_omits_lifetime_quota(
    monkeypatch,
):
    monkeypatch.setattr(KimiModel, "_default_api_key", lambda _: "default-key")
    model = KimiModel("kimi-k3")
    client, get = _client(
        _response(
            {
                "data": {
                    "organization": {
                        "max_request_per_minute": 0,
                        "max_token_per_minute": 10_000_000,
                        "max_concurrency": 7,
                        "max_token_quota": 24_720_425_350,
                    },
                    "organization_usage": {"cur_token_usage": 123},
                },
                "status": "ok",
            }
        )
    )

    with patch(
        "model_library.providers.delegates.kimi.default_httpx_client",
        return_value=client,
    ):
        rate_limit = await model.get_rate_limit()

    assert rate_limit is not None
    assert [(limit.limit, limit.mode) for limit in rate_limit.requests] == [
        (0, "sliding_window"),
        (7, "concurrency"),
    ]
    assert rate_limit.tokens is not None
    assert rate_limit.tokens.total is not None
    assert rate_limit.tokens.total.limit == 10_000_000
    assert rate_limit.scope == "shared"
    assert "24_720_425_350" not in str(rate_limit.model_dump(mode="json"))
    get.assert_awaited_once_with(
        "https://api.moonshot.ai/v1/users/me",
        headers={"Authorization": "Bearer default-key"},
    )


async def test_kimi_uses_concurrency_only_when_rpm_is_absent():
    model = KimiModel("kimi-k3")
    client, _ = _client(
        _response(
            {
                "data": {
                    "organization": {
                        "max_concurrency": 7,
                    }
                }
            }
        )
    )

    with patch(
        "model_library.providers.delegates.kimi.default_httpx_client",
        return_value=client,
    ):
        rate_limit = await model.get_rate_limit()

    assert rate_limit is not None
    assert rate_limit.requests[0].limit == 7
    assert rate_limit.requests[0].mode == "concurrency"
    assert rate_limit.tokens is None


async def test_kimi_returns_none_when_organization_has_no_supported_limits():
    model = KimiModel("kimi-k3")
    client, _ = _client(_response({"data": {"organization": {}}}))

    with patch(
        "model_library.providers.delegates.kimi.default_httpx_client",
        return_value=client,
    ):
        assert await model.get_rate_limit() is None


async def test_kimi_propagates_http_failure():
    model = KimiModel("kimi-k3")
    client, _ = _client(_response({}, status_code=503))

    with (
        patch(
            "model_library.providers.delegates.kimi.default_httpx_client",
            return_value=client,
        ),
        pytest.raises(httpx.HTTPStatusError),
    ):
        await model.get_rate_limit()
