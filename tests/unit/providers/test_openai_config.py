import logging
from typing import Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import Function
from openai.types.completion_usage import (
    CompletionTokensDetails,
    CompletionUsage,
    PromptTokensDetails,
)

from model_library.base import FinishReason, LLMConfig
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


async def test_completions_stream_options_omitted_when_non_streaming():
    model = OpenAIModel(
        "gpt-4o-mini",
        config=LLMConfig(provider_config=OpenAIConfig(stream_completions=False)),
        use_completions=True,
    )

    body = await model.build_body(_INPUT, tools=[])

    assert "stream_options" not in body


async def test_non_streaming_completions_query_parses_response():
    model = OpenAIModel(
        "gpt-4o-mini",
        config=LLMConfig(provider_config=OpenAIConfig(stream_completions=False)),
        use_completions=True,
    )
    response = ChatCompletion(
        id="cmpl_123",
        created=0,
        model="gpt-4o-mini",
        object="chat.completion",
        choices=[
            Choice(
                finish_reason="tool_calls",
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="hello",
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_1",
                            type="function",
                            function=Function(name="lookup", arguments='{"q":"x"}'),
                        )
                    ],
                ),
            )
        ],
        usage=CompletionUsage(
            completion_tokens=5,
            prompt_tokens=10,
            total_tokens=15,
            completion_tokens_details=CompletionTokensDetails(reasoning_tokens=1),
            prompt_tokens_details=PromptTokensDetails(cached_tokens=2),
        ),
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=response)

    with patch.object(model, "get_client", return_value=mock_client):
        with patch.object(model, "build_body", new_callable=AsyncMock, return_value={}):
            result = await model._query_completions(
                _INPUT, tools=[], query_logger=logging.getLogger("test")
            )

    mock_client.chat.completions.create.assert_awaited_once_with(stream=False)
    assert result.output_text == "hello"
    assert result.finish_reason.reason == FinishReason.TOOL_CALLS
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "call_1"
    assert result.tool_calls[0].name == "lookup"
    assert result.tool_calls[0].args == '{"q":"x"}'
    assert result.metadata.in_tokens == 8
    assert result.metadata.out_tokens == 4
    assert result.metadata.reasoning_tokens == 1
    assert result.metadata.cache_read_tokens == 2
