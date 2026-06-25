"""Unit tests for provider-specific configurations.

Tests that provider config fields correctly influence model behavior
(e.g. request body construction, beta headers, extra body params).
"""

from typing import Literal, cast
from unittest.mock import MagicMock

from pydantic import SecretStr, ValidationError

import pytest

from model_library.base import LLMConfig, QueryResult
from model_library.base.input import TextInput
from model_library.providers.anthropic import AnthropicConfig, AnthropicModel
from model_library.providers.google.google import GoogleConfig, GoogleModel
from model_library.providers.openai import OpenAIConfig, OpenAIModel
from model_library.providers.delegates.kimi import KimiConfig, KimiModel
from model_library.providers.delegates.meta import MetaConfig, MetaModel
from model_library.providers.delegates.zai import ZAIConfig, ZAIModel
from model_library.registry_utils import get_registry_model

_INPUT = [TextInput(text="")]


async def _query_anthropic_with_provider_config(
    provider_config: AnthropicConfig,
    *,
    model_name: str = "claude-primary-test",
) -> tuple[dict[str, object], QueryResult]:
    captured: dict[str, object] = {}

    class _DummyIteration:
        type = "fallback_message"
        model = "claude-fallback-test"

    class _DummyUsage:
        input_tokens = 1
        output_tokens = 1
        cache_read_input_tokens = 0
        cache_creation_input_tokens = 0
        iterations = [_DummyIteration()]

    class _DummyText:
        type = "text"
        text = "ok"

    class _DummyMessage:
        id = "msg_test"
        model = "claude-primary-test"
        content = [_DummyText()]
        usage = _DummyUsage()
        stop_reason = "end_turn"

    class _DummyStream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: object) -> bool:
            return False

        async def __aiter__(self):
            if False:
                yield None

        async def get_final_message(self):
            return _DummyMessage()

    class _DummyMessages:
        def stream(self, **kwargs: object) -> _DummyStream:
            captured.update(kwargs)
            return _DummyStream()

    class _DummyBeta:
        messages = _DummyMessages()

    class _DummyClient:
        beta = _DummyBeta()

    model = AnthropicModel(
        model_name,
        config=LLMConfig(
            max_tokens=4096,
            reasoning=False,
            provider_config=provider_config,
        ),
    )
    object.__setattr__(model, "get_client", MagicMock(return_value=_DummyClient()))

    result = await model._query_impl(_INPUT, tools=[], query_logger=MagicMock())
    return captured, result


class TestAnthropicConfig:
    async def test_supports_auto_thinking_uses_adaptive(self):
        model = AnthropicModel(
            "claude-test",
            config=LLMConfig(
                max_tokens=4096,
                reasoning=True,
                provider_config=AnthropicConfig(supports_auto_thinking=True),
            ),
        )

        body = await model.build_body(_INPUT, tools=[])

        assert body["thinking"] == {"type": "adaptive"}

    async def test_no_auto_thinking_uses_enabled_with_budget(self):
        model = AnthropicModel(
            "claude-test",
            config=LLMConfig(
                max_tokens=4096,
                reasoning=True,
                provider_config=AnthropicConfig(supports_auto_thinking=False),
            ),
        )

        body = await model.build_body(_INPUT, tools=[])

        assert body["thinking"]["type"] == "enabled"
        assert "budget_tokens" in body["thinking"]

    async def test_no_thinking_when_reasoning_disabled(self):
        model = AnthropicModel(
            "claude-test",
            config=LLMConfig(
                max_tokens=4096,
                reasoning=False,
                provider_config=AnthropicConfig(supports_auto_thinking=True),
            ),
        )

        body = await model.build_body(_INPUT, tools=[])

        assert body["thinking"] == {"type": "disabled"}

    async def test_supports_compute_effort_adds_output_config(self):
        model = AnthropicModel(
            "claude-test",
            config=LLMConfig(
                max_tokens=4096,
                compute_effort="max",
                provider_config=AnthropicConfig(supports_compute_effort=True),
            ),
        )

        body = await model.build_body(_INPUT, tools=[])

        assert body["output_config"] == {"effort": "max"}

    async def test_compute_effort_not_added_when_unsupported(self):
        model = AnthropicModel(
            "claude-test",
            config=LLMConfig(
                max_tokens=4096,
                compute_effort="max",
                provider_config=AnthropicConfig(supports_compute_effort=False),
            ),
        )

        body = await model.build_body(_INPUT, tools=[])

        assert "output_config" not in body

    async def test_compute_effort_not_added_when_no_effort_value(self):
        model = AnthropicModel(
            "claude-test",
            config=LLMConfig(
                max_tokens=4096,
                provider_config=AnthropicConfig(supports_compute_effort=True),
            ),
        )

        body = await model.build_body(_INPUT, tools=[])

        assert "output_config" not in body

    async def test_server_side_fallback_models_uses_current_request_shape(self):
        captured, result = await _query_anthropic_with_provider_config(
            AnthropicConfig(
                fallback_models=["claude-fallback-test"],
                supports_auto_thinking=True,
            )
        )

        assert captured["betas"] == [
            "files-api-2025-04-14",
            "server-side-fallback-2026-06-01",
        ]
        extra_body = cast(dict[str, object], captured["extra_body"])
        assert extra_body == {"fallbacks": [{"model": "claude-fallback-test"}]}
        assert result.metadata.extra["fallback"] is True

    async def test_server_side_fallback_models_preserve_order(self):
        captured, result = await _query_anthropic_with_provider_config(
            AnthropicConfig(
                fallback_models=["claude-fallback-test", "claude-backup-test"],
                supports_auto_thinking=True,
            )
        )

        assert captured["betas"] == [
            "files-api-2025-04-14",
            "server-side-fallback-2026-06-01",
        ]
        extra_body = cast(dict[str, object], captured["extra_body"])
        assert extra_body == {
            "fallbacks": [
                {"model": "claude-fallback-test"},
                {"model": "claude-backup-test"},
            ]
        }
        assert result.metadata.extra["fallback"] is True

    def test_fallback_model_is_not_a_supported_config_field(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            AnthropicConfig.model_validate({"fallback_model": "claude-fallback-test"})

    def test_fallback_models_rejects_more_than_three_entries(self):
        with pytest.raises(ValueError, match="at most 3"):
            AnthropicConfig(
                fallback_models=[
                    "claude-fallback-test",
                    "claude-backup-test",
                    "claude-third-test",
                    "claude-fourth-test",
                ]
            )

    def test_fallback_models_rejects_duplicates(self):
        with pytest.raises(ValueError, match="duplicate"):
            AnthropicConfig(
                fallback_models=["claude-fallback-test", "claude-fallback-test"]
            )

    async def test_fallback_models_rejects_requested_model(self):
        with pytest.raises(ValueError, match="must not include requested model"):
            await _query_anthropic_with_provider_config(
                AnthropicConfig(
                    fallback_models=["claude-primary-test"],
                    supports_auto_thinking=True,
                ),
                model_name="claude-primary-test",
            )


class TestRegistryProviderConfigs:
    async def test_anthropic_registry_model_has_typed_config(self):
        model = get_registry_model("anthropic/claude-sonnet-4-5-20250929")
        assert isinstance(model, AnthropicModel)
        assert isinstance(model.provider_config, AnthropicConfig)

    async def test_anthropic_registry_config_affects_build_body(self):
        model = get_registry_model("anthropic/claude-opus-4-6-thinking")
        assert isinstance(model, AnthropicModel)
        assert model.provider_config.supports_auto_thinking is True
        body = await model.build_body(_INPUT, tools=[])
        assert body["thinking"] == {"type": "adaptive"}
    async def test_openai_registry_model_has_typed_config(self):
        model = get_registry_model("openai/gpt-4o")
        assert isinstance(model, OpenAIModel)
        assert isinstance(model.provider_config, OpenAIConfig)

    async def test_google_registry_model_has_typed_config(self):
        model = get_registry_model("google/gemini-2.5-flash")
        assert isinstance(model, GoogleModel)
        assert isinstance(model.provider_config, GoogleConfig)

    async def test_zai_registry_model_has_typed_config(self):
        model = get_registry_model("zai/glm-4.7")
        assert isinstance(model, ZAIModel)
        assert isinstance(model.provider_config, ZAIConfig)


class TestMetaConfig:
    async def test_use_responses_configures_openai_delegate_responses_mode(self):
        model = MetaModel(
            "llama-test",
            config=LLMConfig(
                custom_api_key=SecretStr("sk-test"),
                provider_config=MetaConfig(use_responses=True),
            ),
        )

        assert isinstance(model.provider_config, MetaConfig)
        assert model.provider_config.use_responses is True
        assert isinstance(model.delegate, OpenAIModel)
        assert model.delegate.use_completions is False
        body = await model.build_body(_INPUT, tools=[])
        assert "input" in body
        assert "messages" not in body

    async def test_use_responses_reasoning_sets_store_false(self):
        model = MetaModel(
            "llama-test",
            config=LLMConfig(
                custom_api_key=SecretStr("sk-test"),
                reasoning=True,
                provider_config=MetaConfig(use_responses=True),
            ),
        )

        body = await model.build_body(_INPUT, tools=[])

        assert body["include"] == ["reasoning.encrypted_content"]
        assert body["store"] is False
class TestKimiConfig:
    async def test_parallel_tool_calls_configures_openai_delegate(self):
        config = LLMConfig(
            custom_api_key=SecretStr("sk-test"),
            provider_config=KimiConfig(parallel_tool_calls=False),
        )
        model = KimiModel("kimi-k2", config=config)

        assert config.custom_endpoint is None
        assert isinstance(config.provider_config, KimiConfig)
        assert isinstance(model.provider_config, KimiConfig)
        assert model.provider_config.parallel_tool_calls is False
        assert isinstance(model.delegate, OpenAIModel)
        assert isinstance(model.delegate.provider_config, OpenAIConfig)
        assert model.delegate.provider_config.parallel_tool_calls is False

        body = await model.build_body(_INPUT, tools=[])
        assert body["parallel_tool_calls"] is False


class TestOpenAIConfig:
    @pytest.mark.parametrize("verbosity", ["low", "medium", "high"])
    async def test_verbosity_added_to_body(
        self, verbosity: Literal["low", "medium", "high"]
    ):
        model = OpenAIModel(
            "gpt-4o",
            config=LLMConfig(provider_config=OpenAIConfig(verbosity=verbosity)),
        )

        body = await model.build_body(_INPUT, tools=[])

        assert "text" in body
        assert body["text"]["verbosity"] == verbosity

    async def test_verbosity_not_in_body_when_none(self):
        model = OpenAIModel(
            "gpt-4o",
            config=LLMConfig(provider_config=OpenAIConfig(verbosity=None)),
        )

        body = await model.build_body(_INPUT, tools=[])

        assert "text" not in body
