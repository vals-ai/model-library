from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr, ValidationError

from model_library.base import LLMConfig
from model_library.rate_limits import (
    RateLimit,
    RateLimitCapacity,
    RequestRateLimit,
    TokenRateLimit,
    rate_limit_from_headers,
)
from model_library.providers.anthropic import AnthropicModel
from model_library.providers.delegates.fireworks import FireworksModel
from model_library.providers.mistral import MistralModel
from model_library.providers.openai import OpenAIModel
from model_library.providers.xai import XAIModel

_DATE_HEADER = "Mon, 01 Jan 2024 00:00:00 GMT"
_DATE_TIMESTAMP = 1_704_067_200.0


def _config() -> LLMConfig:
    return LLMConfig(custom_api_key=SecretStr("test-key"))


def _client_with_headers(attr_path: str, headers: dict[str, str]) -> MagicMock:
    client = MagicMock()
    create = AsyncMock(return_value=SimpleNamespace(headers=headers))
    target = client
    *parents, leaf = attr_path.split(".")
    for parent in parents:
        target = getattr(target, parent)
    setattr(target, leaf, create)
    return client


def _probe_client(headers: dict[str, str]) -> tuple[MagicMock, AsyncMock]:
    post = AsyncMock(
        return_value=SimpleNamespace(
            headers=headers,
            raise_for_status=lambda: None,
        )
    )
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=SimpleNamespace(post=post))
    client.__aexit__ = AsyncMock(return_value=False)
    return client, post


async def test_anthropic_prefers_split_input_output_token_limits() -> None:
    model = AnthropicModel("claude-sonnet-4-5-20250929", config=_config())
    client = _client_with_headers(
        "messages.with_raw_response.create",
        {
            "date": _DATE_HEADER,
            "anthropic-ratelimit-requests-limit": "4000",
            "anthropic-ratelimit-requests-remaining": "3999",
            "anthropic-ratelimit-tokens-limit": "300000",
            "anthropic-ratelimit-tokens-remaining": "250000",
            "anthropic-ratelimit-input-tokens-limit": "200000",
            "anthropic-ratelimit-input-tokens-remaining": "180000",
            "anthropic-ratelimit-output-tokens-limit": "80000",
            "anthropic-ratelimit-output-tokens-remaining": "60000",
        },
    )

    with patch.object(model, "get_client", return_value=client):
        rate_limit = await model.get_rate_limit()

    assert rate_limit is not None
    assert rate_limit.unix_timestamp == _DATE_TIMESTAMP
    assert rate_limit.requests == (RequestRateLimit(limit=4_000, remaining=3_999),)
    assert rate_limit.tokens == TokenRateLimit(
        input=RateLimitCapacity(limit=200_000, remaining=180_000),
        output=RateLimitCapacity(limit=80_000, remaining=60_000),
    )
    assert rate_limit.token_limit_total == 280_000
    assert rate_limit.token_remaining_total == 240_000


async def test_anthropic_uses_combined_tokens_when_split_limits_are_absent() -> None:
    model = AnthropicModel("claude-sonnet-4-5-20250929", config=_config())
    client = _client_with_headers(
        "messages.with_raw_response.create",
        {
            "date": _DATE_HEADER,
            "anthropic-ratelimit-tokens-limit": "300000",
            "anthropic-ratelimit-tokens-remaining": "250000",
        },
    )

    with patch.object(model, "get_client", return_value=client):
        rate_limit = await model.get_rate_limit()

    assert rate_limit is not None
    assert rate_limit.requests == ()
    assert rate_limit.tokens == TokenRateLimit(
        total=RateLimitCapacity(limit=300_000, remaining=250_000)
    )


@pytest.mark.parametrize("direction", ["input", "output"])
async def test_anthropic_uses_combined_tokens_when_one_split_limit_is_absent(
    direction: str,
) -> None:
    model = AnthropicModel("claude-sonnet-4-5-20250929", config=_config())
    client = _client_with_headers(
        "messages.with_raw_response.create",
        {
            "date": _DATE_HEADER,
            "anthropic-ratelimit-tokens-limit": "300000",
            "anthropic-ratelimit-tokens-remaining": "250000",
            f"anthropic-ratelimit-{direction}-tokens-limit": "200000",
            f"anthropic-ratelimit-{direction}-tokens-remaining": "180000",
        },
    )

    with patch.object(model, "get_client", return_value=client):
        rate_limit = await model.get_rate_limit()

    assert rate_limit is not None
    assert rate_limit.tokens == TokenRateLimit(
        total=RateLimitCapacity(limit=300_000, remaining=250_000)
    )


async def test_openai_keeps_request_limits_when_token_headers_are_absent() -> None:
    model = OpenAIModel("gpt-4o", config=_config())
    client = _client_with_headers(
        "responses.with_raw_response.create",
        {
            "date": _DATE_HEADER,
            "x-ratelimit-limit-requests": "10000",
            "x-ratelimit-remaining-requests": "9999",
        },
    )

    with patch.object(model, "get_client", return_value=client):
        rate_limit = await model.get_rate_limit()

    assert rate_limit is not None
    assert rate_limit.requests == (RequestRateLimit(limit=10_000, remaining=9_999),)
    assert rate_limit.tokens is None


def test_generic_headers_keep_zero_remaining_and_group_capacities() -> None:
    rate_limit = rate_limit_from_headers(
        {
            "date": _DATE_HEADER,
            "x-ratelimit-limit-requests": "10000",
            "x-ratelimit-remaining-requests": "0",
            "x-ratelimit-limit-tokens": "2000000",
            "x-ratelimit-remaining-tokens": "0",
        }
    )

    assert rate_limit == RateLimit(
        requests=(RequestRateLimit(limit=10_000, remaining=0),),
        tokens=TokenRateLimit(total=RateLimitCapacity(limit=2_000_000, remaining=0)),
        unix_timestamp=_DATE_TIMESTAMP,
    )


def test_total_tokens_survive_incomplete_directional_headers() -> None:
    rate_limit = rate_limit_from_headers(
        {
            "date": _DATE_HEADER,
            "x-ratelimit-limit-tokens": "2000000",
            "x-ratelimit-remaining-tokens": "1000000",
            "x-ratelimit-limit-tokens-prompt": "1500000",
            "x-ratelimit-remaining-tokens-prompt": "750000",
        }
    )

    assert rate_limit is not None
    assert rate_limit.tokens == TokenRateLimit(
        total=RateLimitCapacity(limit=2_000_000, remaining=1_000_000)
    )


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
async def test_provider_without_rate_limit_headers_reports_no_data(provider: str) -> None:
    headers = {"date": _DATE_HEADER}
    if provider == "anthropic":
        model = AnthropicModel("claude-sonnet-4-5-20250929", config=_config())
        client = _client_with_headers("messages.with_raw_response.create", headers)
    else:
        model = OpenAIModel("gpt-4o", config=_config())
        client = _client_with_headers("responses.with_raw_response.create", headers)

    with patch.object(model, "get_client", return_value=client):
        assert await model.get_rate_limit() is None


def test_window_suffixed_header_names_are_read() -> None:
    rate_limit = rate_limit_from_headers(
        {
            "date": _DATE_HEADER,
            "x-ratelimit-limit-req-minute": "4000",
            "x-ratelimit-remaining-req-minute": "3999",
            "x-ratelimit-limit-tokens-minute": "4000000",
            "x-ratelimit-remaining-tokens-minute": "3999975",
        }
    )

    assert rate_limit == RateLimit(
        requests=(RequestRateLimit(limit=4_000, remaining=3_999),),
        tokens=TokenRateLimit(
            total=RateLimitCapacity(limit=4_000_000, remaining=3_999_975)
        ),
        unix_timestamp=_DATE_TIMESTAMP,
    )


async def test_fireworks_normalizes_all_three_documented_tps_caps_to_tpm() -> None:
    headers = {
        "date": _DATE_HEADER,
        "x-ratelimit-limit-tokens-prompt": "120000",
        "x-ratelimit-remaining-tokens-prompt": "119900",
        "x-ratelimit-limit-tokens-cache-adjusted-prompt": "60000",
        "x-ratelimit-limit-tokens-generated": "1200",
        "x-ratelimit-remaining-tokens-generated": "1100",
    }
    delegated = rate_limit_from_headers(headers)
    assert delegated is not None
    model = FireworksModel("accounts/test/model", config=_config())
    assert model.delegate is not None
    model.delegate.get_rate_limit = AsyncMock(return_value=delegated)

    rate_limit = await model.get_rate_limit()

    assert rate_limit is not None
    assert rate_limit.tokens == TokenRateLimit(
        input=RateLimitCapacity(limit=7_200_000, remaining=7_194_000),
        uncached_input=RateLimitCapacity(limit=3_600_000),
        output=RateLimitCapacity(limit=72_000, remaining=66_000),
    )


def test_uncached_input_requires_directional_tokens() -> None:
    with pytest.raises(ValidationError, match="directional"):
        TokenRateLimit(uncached_input=RateLimitCapacity(limit=1))


def test_token_total_is_exclusive_with_directional_capacities() -> None:
    with pytest.raises(ValidationError, match="total"):
        TokenRateLimit(
            total=RateLimitCapacity(limit=1),
            input=RateLimitCapacity(limit=1),
            output=RateLimitCapacity(limit=1),
        )


def test_rate_limit_rejects_duplicate_request_modes() -> None:
    with pytest.raises(ValidationError, match="one request limit per mode"):
        RateLimit(
            requests=(RequestRateLimit(limit=1), RequestRateLimit(limit=2)),
            unix_timestamp=_DATE_TIMESTAMP,
        )


async def test_xai_native_probe_uses_default_account() -> None:
    client, post = _probe_client(
        {
            "date": _DATE_HEADER,
            "x-ratelimit-limit-tokens-minute": "4000000",
            "x-ratelimit-remaining-tokens-minute": "3999975",
        }
    )
    with (
        patch.object(XAIModel, "_get_default_api_key", return_value="default-key"),
        patch(
            "model_library.rate_limits.probe.default_httpx_client",
            return_value=client,
        ),
    ):
        model = XAIModel("grok-3-mini", config=LLMConfig())
        assert model.delegate is None
        rate_limit = await model.get_rate_limit()

    assert rate_limit is not None
    assert rate_limit.tokens == TokenRateLimit(
        total=RateLimitCapacity(limit=4_000_000, remaining=3_999_975)
    )
    assert post.call_args.args == ("https://api.x.ai/v1/chat/completions",)
    assert post.call_args.kwargs["headers"] == {
        "Authorization": "Bearer default-key"
    }


async def test_mistral_probe_uses_default_account_for_canonical_model() -> None:
    client, post = _probe_client(
        {"date": _DATE_HEADER, "x-ratelimit-limit-requests": "10"}
    )
    with (
        patch.object(MistralModel, "_get_default_api_key", return_value="default-key"),
        patch(
            "model_library.rate_limits.probe.default_httpx_client",
            return_value=client,
        ),
    ):
        model = MistralModel("mistral-medium-latest", config=LLMConfig())
        await model.get_rate_limit()

    assert post.call_args.args == ("https://api.mistral.ai/v1/chat/completions",)
    assert post.call_args.kwargs["headers"] == {
        "Authorization": "Bearer default-key"
    }
