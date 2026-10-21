"""Tests for OpenAI registry entries."""

import pytest
from pydantic import SecretStr

from model_library.base import LLMConfig
from model_library.base.input import TextInput
from model_library.providers.openai import OpenAIModel
from model_library.registry_utils import get_registry_config, get_registry_model

_INPUT = [TextInput(text="")]
_DAYBREAK_MODELS = [
    "gpt-5.5",
    "gpt-5.5-pro",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-luna",
    "gpt-6-sol",
]
_API_KEY = LLMConfig(custom_api_key=SecretStr("sk-test"))


def _model(key: str) -> OpenAIModel:
    model = get_registry_model(key, _API_KEY)
    assert isinstance(model, OpenAIModel)
    return model


@pytest.mark.parametrize("name", _DAYBREAK_MODELS)
async def test_daybreak_key_sends_the_program_and_keeps_the_base_config(name: str):
    base_entry = get_registry_config(f"openai/{name}")
    daybreak_entry = get_registry_config(f"openai/{name}-daybreak")
    assert base_entry is not None and daybreak_entry is not None
    base, daybreak = _model(f"openai/{name}"), _model(f"openai/{name}-daybreak")

    assert daybreak_entry.provider_endpoint == base_entry.provider_endpoint
    assert daybreak_entry.costs_per_million_token == base_entry.costs_per_million_token
    assert daybreak.reasoning_effort == base.reasoning_effort
    assert daybreak.reasoning_context == base.reasoning_context
    assert daybreak.verbosity == base.verbosity
    assert daybreak.prompt_cache_retention == base.prompt_cache_retention

    body = await daybreak.build_body(_INPUT, tools=[])
    assert body["extra_body"] == {"access_programs": {"cyber": "daybreak_blue"}}
    assert "extra_body" not in await base.build_body(_INPUT, tools=[])
