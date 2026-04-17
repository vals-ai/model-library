from typing import Literal

import pytest

from model_library.base import LLMConfig
from model_library.base.input import TextInput
from model_library.providers.openai import OpenAIConfig, OpenAIModel

_INPUT = [TextInput(text="")]


@pytest.mark.parametrize("verbosity", ["low", "medium", "high"])
async def test_verbosity_added_to_body(verbosity: Literal["low", "medium", "high"]):
    """Test that verbosity is correctly added to request body."""
    model = OpenAIModel(
        "gpt-4o",
        config=LLMConfig(provider_config=OpenAIConfig(verbosity=verbosity)),
    )

    body = await model.build_body(_INPUT, tools=[])

    assert "text" in body
    assert body["text"]["verbosity"] == verbosity


async def test_verbosity_not_in_body_when_none():
    """Test that text field is not added when verbosity is None."""
    model = OpenAIModel(
        "gpt-4o",
        config=LLMConfig(provider_config=OpenAIConfig(verbosity=None)),
    )

    body = await model.build_body(_INPUT, tools=[])

    assert "text" not in body


async def test_deepseek_reasoning_keeps_max_tokens():
    """DeepSeek thinking mode documents max_tokens, not max_completion_tokens."""
    model = OpenAIModel(
        "deepseek-reasoner",
        provider="deepseek",
        config=LLMConfig(reasoning=True, max_tokens=8192),
        use_completions=True,
    )
    body = await model.build_body(_INPUT, tools=[])
    assert body.get("max_tokens") == 8192
    assert "max_completion_tokens" not in body


async def test_google_delegate_thinking_config():
    model = OpenAIModel(
        "gemini-3.1-pro-preview",
        provider="google",
        config=LLMConfig(reasoning=True, reasoning_effort="low"),
        use_completions=True,
    )
    body = await model.build_body(_INPUT, tools=[])
    thinking_config = body["extra_body"]["extra_body"]["google"]["thinking_config"]
    assert thinking_config["include_thoughts"] is True
    assert thinking_config["thinking_level"] == "low"
    assert "reasoning_effort" not in body
