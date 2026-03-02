from typing import Any

import pytest
from pydantic import BaseModel

from model_library.base import LLMConfig
from model_library.providers.anthropic import AnthropicModel
from model_library.providers.google import GoogleModel
from model_library.providers.openai import OpenAIModel
from model_library.registry_utils import get_registry_model


class AnswerSchema(BaseModel):
    answer: str
    confidence: float


DICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["answer", "confidence"],
    "additionalProperties": False,
}


class TestOpenAIBuildBodyOutputSchema:
    @pytest.fixture
    def model(self):
        return OpenAIModel("gpt-4o")

    async def test_dict_schema(self, model: OpenAIModel):
        body = await model.build_body([], tools=[], output_schema=DICT_SCHEMA)

        fmt = body["text"]["format"]
        assert fmt["type"] == "json_schema"
        assert fmt["name"] == "structured_output"
        assert fmt["schema"] is DICT_SCHEMA
        assert fmt["strict"] is True

    async def test_pydantic_model(self, model: OpenAIModel):
        body = await model.build_body([], tools=[], output_schema=AnswerSchema)

        fmt = body["text"]["format"]
        assert fmt["type"] == "json_schema"
        assert fmt["name"] == "structured_output"
        assert fmt["schema"]["type"] == "object"
        assert fmt["strict"] is True


class TestGoogleBuildBodyOutputSchema:
    @pytest.fixture
    def model(self):
        return GoogleModel("gemini-2.5-flash")

    async def test_dict_uses_response_json_schema(self, model: GoogleModel):
        body = await model.build_body([], tools=[], output_schema=DICT_SCHEMA)

        config = body["config"]
        assert config.response_json_schema == DICT_SCHEMA
        assert config.response_schema is None
        assert config.response_mime_type == "application/json"

    async def test_pydantic_model_uses_response_schema(self, model: GoogleModel):
        body = await model.build_body([], tools=[], output_schema=AnswerSchema)

        config = body["config"]
        assert config.response_schema is AnswerSchema
        assert config.response_json_schema is None
        assert config.response_mime_type == "application/json"


class TestAnthropicBuildBodyOutputSchema:
    @pytest.fixture
    def model(self):
        return AnthropicModel(
            "claude-sonnet-4-20250514", config=LLMConfig(max_tokens=1024)
        )

    async def test_dict_schema(self, model: AnthropicModel):
        body = await model.build_body([], tools=[], output_schema=DICT_SCHEMA)

        fmt = body["output_config"]["format"]
        assert fmt["type"] == "json_schema"
        assert fmt["schema"]["type"] == "object"

    async def test_pydantic_model(self, model: AnthropicModel):
        body = await model.build_body([], tools=[], output_schema=AnswerSchema)

        fmt = body["output_config"]["format"]
        assert fmt["type"] == "json_schema"
        assert fmt["schema"]["type"] == "object"


UNSUPPORTED_PROVIDER_MODELS = [
    "fireworks/deepseek-v3p2",
    "perplexity/sonar-pro",
    "together/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "cohere/command-a-03-2025",
    "deepseek/deepseek-chat",
    "mistralai/mistral-large-2512",
    "ai21labs/jamba-large-1.7",
    "amazon/amazon.nova-pro-v1:0",
    "grok/grok-3-mini",
]

UNSUPPORTED_BIG_THREE_MODELS = [
    "anthropic/claude-sonnet-4-20250514",
    "anthropic/claude-3-5-sonnet-20241022",
    "openai/gpt-4-turbo",
    "google/gemma-3-27b-it",
]


class TestUnsupportedProviderModelsRejectOutputSchema:
    @pytest.mark.parametrize("model_key", UNSUPPORTED_PROVIDER_MODELS)
    async def test_raises_on_dict_schema(self, model_key: str):
        model = get_registry_model(model_key)
        assert not model.supports_output_schema

        with pytest.raises(Exception, match="does not support structured outputs"):
            await model.query("test", output_schema=DICT_SCHEMA)

    @pytest.mark.parametrize("model_key", UNSUPPORTED_PROVIDER_MODELS)
    async def test_raises_on_pydantic_schema(self, model_key: str):
        model = get_registry_model(model_key)
        assert not model.supports_output_schema

        with pytest.raises(Exception, match="does not support structured outputs"):
            await model.query("test", output_schema=AnswerSchema)


class TestUnsupportedBigThreeModelsRejectOutputSchema:
    @pytest.mark.parametrize("model_key", UNSUPPORTED_BIG_THREE_MODELS)
    async def test_raises_on_dict_schema(self, model_key: str):
        model = get_registry_model(model_key)
        assert not model.supports_output_schema

        with pytest.raises(Exception, match="does not support structured outputs"):
            await model.query("test", output_schema=DICT_SCHEMA)

    @pytest.mark.parametrize("model_key", UNSUPPORTED_BIG_THREE_MODELS)
    async def test_raises_on_pydantic_schema(self, model_key: str):
        model = get_registry_model(model_key)
        assert not model.supports_output_schema

        with pytest.raises(Exception, match="does not support structured outputs"):
            await model.query("test", output_schema=AnswerSchema)


SUPPORTED_MODELS = [
    "openai/gpt-4o",
    "openai/o3-2025-04-16",
    "google/gemini-2.5-flash",
    "google/gemini-3-flash-preview",
    "anthropic/claude-sonnet-4-6",
    "anthropic/claude-haiku-4-5-20251001",
]


class TestSupportedModelsAllowOutputSchema:
    @pytest.mark.parametrize("model_key", SUPPORTED_MODELS)
    def test_supports_output_schema_flag_is_true(self, model_key: str):
        model = get_registry_model(model_key)
        assert model.supports_output_schema
