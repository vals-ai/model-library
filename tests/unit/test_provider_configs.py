"""Unit tests for provider-specific configurations.

Tests that provider config fields correctly influence model behavior
(e.g. request body construction, beta headers, extra body params).
"""

import pytest

from model_library.base import LLMConfig
from model_library.providers.anthropic import AnthropicConfig, AnthropicModel
from model_library.providers.google.google import GoogleConfig, GoogleModel
from model_library.providers.openai import OpenAIConfig, OpenAIModel
from model_library.providers.zai import ZAIConfig, ZAIModel
from model_library.registry_utils import get_registry_model


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

        body = await model.build_body([], tools=[])

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

        body = await model.build_body([], tools=[])

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

        body = await model.build_body([], tools=[])

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

        body = await model.build_body([], tools=[])

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

        body = await model.build_body([], tools=[])

        assert "output_config" not in body

    async def test_compute_effort_not_added_when_no_effort_value(self):
        model = AnthropicModel(
            "claude-test",
            config=LLMConfig(
                max_tokens=4096,
                provider_config=AnthropicConfig(supports_compute_effort=True),
            ),
        )

        body = await model.build_body([], tools=[])

        assert "output_config" not in body

    async def test_registry_sonnet_4_5_has_1M_context(self):
        model = get_registry_model("anthropic/claude-sonnet-4-5-20250929")
        assert isinstance(model, AnthropicModel)
        assert model.provider_config.supports_1M_context is True

    async def test_registry_sonnet_4_has_1M_context(self):
        model = get_registry_model("anthropic/claude-sonnet-4-20250514")
        assert isinstance(model, AnthropicModel)
        assert model.provider_config.supports_1M_context is True

    async def test_registry_opus_4_6_no_1M_context(self):
        model = get_registry_model("anthropic/claude-opus-4-6")
        assert isinstance(model, AnthropicModel)
        assert model.provider_config.supports_1M_context is False

    async def test_registry_haiku_no_1M_context(self):
        model = get_registry_model("anthropic/claude-haiku-4-5-20251001")
        assert isinstance(model, AnthropicModel)
        assert model.provider_config.supports_1M_context is False


class TestRegistryProviderConfigs:
    async def test_anthropic_registry_model_has_typed_config(self):
        model = get_registry_model("anthropic/claude-sonnet-4-20250514")
        assert isinstance(model, AnthropicModel)
        assert isinstance(model.provider_config, AnthropicConfig)

    async def test_anthropic_registry_config_affects_build_body(self):
        model = get_registry_model("anthropic/claude-opus-4-6-thinking")
        assert isinstance(model, AnthropicModel)
        assert model.provider_config.supports_auto_thinking is True
        body = await model.build_body([], tools=[])
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


class TestOpenAIConfig:
    @pytest.mark.parametrize("verbosity", ["low", "medium", "high"])
    async def test_verbosity_added_to_body(self, verbosity: str):
        model = OpenAIModel(
            "gpt-4o",
            config=LLMConfig(provider_config=OpenAIConfig(verbosity=verbosity)),
        )

        body = await model.build_body([], tools=[])

        assert "text" in body
        assert body["text"]["verbosity"] == verbosity

    async def test_verbosity_not_in_body_when_none(self):
        model = OpenAIModel(
            "gpt-4o",
            config=LLMConfig(provider_config=OpenAIConfig(verbosity=None)),
        )

        body = await model.build_body([], tools=[])

        assert "text" not in body
