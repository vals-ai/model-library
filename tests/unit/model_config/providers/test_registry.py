"""Tests for small registry-owned provider configurations."""

from pydantic import SecretStr

from model_library.base import LLMConfig
from model_library.base.input import TextInput
from model_library.providers.anthropic import AnthropicConfig, AnthropicModel
from model_library.providers.google.google import GoogleConfig, GoogleModel
from model_library.providers.openai import OpenAIConfig, OpenAIModel
from model_library.providers.delegates.zai import ZAIConfig, ZAIModel
from model_library.registry_utils import get_registry_model

_INPUT = [TextInput(text="")]

class TestRegistryProviderConfigs:
    async def test_anthropic_registry_model_has_typed_config(self):
        model = get_registry_model("anthropic/claude-sonnet-4-5-20250929")
        assert isinstance(model, AnthropicModel)
        assert isinstance(model.provider_config, AnthropicConfig)

    async def test_anthropic_registry_config_affects_build_body(self):
        model = get_registry_model("anthropic/claude-opus-4-6-thinking")
        assert isinstance(model, AnthropicModel)
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

    async def test_zai_model_has_typed_config(self):
        model = ZAIModel(
            "glm-4.7",
            config=LLMConfig(custom_api_key=SecretStr("sk-test")),
        )
        assert isinstance(model, ZAIModel)
        assert isinstance(model.provider_config, ZAIConfig)

    async def test_zai_endpoint_is_international_only_where_configured(self):
        international = get_registry_model("zai/glm-5.3")
        mainland = ZAIModel(
            "glm-4.7",
            config=LLMConfig(custom_api_key=SecretStr("sk-test")),
        )

        assert isinstance(international, ZAIModel)
        assert international.delegate is not None
        assert mainland.delegate is not None
        assert international.delegate.custom_endpoint == "https://api.z.ai/api/paas/v4/"
        assert (
            mainland.delegate.custom_endpoint == "https://open.bigmodel.cn/api/paas/v4/"
        )

    async def test_zai_thinking_stays_enabled_when_disabling_unsupported(self):
        model = get_registry_model("zai/glm-5.3")
        assert isinstance(model, ZAIModel)
        model.reasoning = False

        assert model._get_extra_body() == {  # pyright: ignore[reportPrivateUsage]
            "thinking": {"type": "enabled", "clear_thinking": False}
        }

    async def test_zai_thinking_disabled_without_reasoning(self):
        model = ZAIModel(
            "glm-4.7",
            config=LLMConfig(custom_api_key=SecretStr("sk-test")),
        )
        model.reasoning = False

        thinking = model._get_extra_body()["thinking"]  # pyright: ignore[reportPrivateUsage]
        assert thinking["type"] == "disabled"
