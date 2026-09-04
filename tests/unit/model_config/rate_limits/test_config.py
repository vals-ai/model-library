"""Tests for configured rate limits and YAML inheritance."""

from typing import Any

from pydantic import ValidationError
import pytest

from model_library.register_models import DefaultRateLimit, parse_yaml_blocks


def test_configured_token_limit_rejects_zero() -> None:
    with pytest.raises(ValidationError):
        DefaultRateLimit.model_validate({"tokens": {"total": {"limit": 0}}})


def test_configured_request_limit_rejects_zero() -> None:
    with pytest.raises(ValidationError):
        DefaultRateLimit.model_validate({"requests": [{"limit": 0}]})


def _model_blocks(model_config: dict[str, object]) -> dict[str, Any]:
    return {
        "base-config": {
            "company": "Test",
            "open_source": False,
            "supports": {},
            "properties": {
                "context_window": 1_000,
                "max_tokens": 100,
                "reasoning_model": False,
            },
            "costs_per_million_token": None,
        },
        "models": {"test/model": {"label": "Test Model", **model_config}},
    }


def test_omitted_rate_limit_loads_without_a_configured_limit() -> None:
    registry = {}

    parse_yaml_blocks(_model_blocks({}), registry)

    assert registry["test/model"].rate_limit is None


def test_model_rate_limit_rejects_explicit_null() -> None:
    with pytest.raises(ValueError):
        parse_yaml_blocks(_model_blocks({"rate_limit": None}), {})


def test_alternative_key_rate_limit_rejects_explicit_null() -> None:
    with pytest.raises(ValueError):
        parse_yaml_blocks(
            _model_blocks(
                {
                    "alternative_keys": [
                        {"test/alias": {"rate_limit": None}},
                    ]
                }
            ),
            {},
        )


def test_policy_only_rate_limit_requires_an_explicit_policy() -> None:
    explicit_false = DefaultRateLimit.model_validate(
        {"supports_live_monitoring": False}
    )
    cache_exclusion = DefaultRateLimit.model_validate(
        {"cache_read_counts_toward_limit": False}
    )

    assert explicit_false.token_retry_defaults == (None, None)
    assert cache_exclusion.token_retry_defaults == (None, None)
    with pytest.raises(ValidationError, match="policy or capacity"):
        DefaultRateLimit.model_validate({})


def test_alternative_key_rate_limit_policy_inherits_only_within_provider() -> None:
    registry = {}
    parse_yaml_blocks(
        _model_blocks(
            {
                "rate_limit": {
                    "supports_live_monitoring": True,
                    "cache_read_counts_toward_limit": False,
                    "tokens": {"total": {"limit": 1_000}},
                },
                "alternative_keys": ["test/alias", "other/alias"],
            }
        ),
        registry,
    )

    assert registry["test/alias"].rate_limit == registry["test/model"].rate_limit
    assert registry["other/alias"].rate_limit is None


def test_rate_limit_algorithm_modes_default_and_validate_by_dimension() -> None:
    limit = DefaultRateLimit.model_validate(
        {"requests": [{"limit": 10}], "tokens": {"total": {"limit": 1_000}}}
    )

    assert limit.requests[0].mode == "sliding_window"
    assert limit.tokens is not None
    assert limit.tokens.mode == "token_bucket"

    with pytest.raises(ValidationError, match="Extra inputs"):
        DefaultRateLimit.model_validate({"request_limit": 10})
    with pytest.raises(ValidationError):
        DefaultRateLimit.model_validate(
            {"requests": [{"limit": 10, "mode": "token_bucket"}]}
        )
    with pytest.raises(ValidationError):
        DefaultRateLimit.model_validate(
            {"tokens": {"total": {"limit": 1_000}, "mode": "concurrency"}}
        )


@pytest.mark.parametrize(
    "tokens",
    [
        {"input": {"limit": 1_000}},
        {"output": {"limit": 1_000}},
    ],
)
def test_directional_default_limits_require_input_and_output(
    tokens: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="directional token limits"):
        DefaultRateLimit.model_validate({"tokens": tokens})

    assert DefaultRateLimit.model_validate(
        {
            "tokens": {
                "input": {"limit": 1_000},
                "output": {"limit": 2_000},
            }
        }
    ).token_retry_defaults == (3_000, None)
