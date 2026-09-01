"""Tests for Meta provider configuration."""

from pydantic import SecretStr

from model_library.base import LLMConfig
from model_library.base.input import RawResponse, SystemInput, TextInput
from model_library.providers.delegates.meta import MetaConfig, MetaModel
from model_library.providers.openai import OpenAIModel
from model_library.registry_utils import get_registry_model

_INPUT = [TextInput(text="")]

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

    async def test_prompt_cache_key_hash_forwarded_to_openai_delegate(self):
        model = MetaModel(
            "llama-test",
            config=LLMConfig(
                provider_config=MetaConfig(prompt_cache_key="hash", use_responses=True),
                custom_api_key=SecretStr("sk-test"),
            ),
        )

        turn_1 = [SystemInput(text="sys"), TextInput(text="first user msg")]
        turn_2 = [
            SystemInput(text="sys"),
            TextInput(text="first user msg"),
            RawResponse(response=[]),
            TextInput(text="later user msg"),
        ]

        body_1 = await model.build_body(turn_1, tools=[])
        body_2 = await model.build_body(turn_2, tools=[])

        assert isinstance(body_1["prompt_cache_key"], str)
        assert body_1["prompt_cache_key"]
        assert body_1["prompt_cache_key"] == body_2["prompt_cache_key"]

    async def test_prompt_cache_key_omitted_when_unconfigured(self):
        model = MetaModel(
            "llama-test",
            config=LLMConfig(
                provider_config=MetaConfig(use_responses=True),
                custom_api_key=SecretStr("sk-test"),
            ),
        )

        body = await model.build_body(_INPUT, tools=[])

        assert "prompt_cache_key" not in body
