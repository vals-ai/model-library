"""Anthropic turns that ran out of tokens inside a thinking block."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from model_library.base import LLMConfig, RawResponse, TextInput
from model_library.base.output import FinishReason
from model_library.exceptions import MaxOutputTokensExceededError, ModelNoOutputError
from model_library.providers.anthropic import (
    TRUNCATED_THINKING_MARKER,
    AnthropicConfig,
    AnthropicModel,
)


def _block(block_type: str, **fields: Any) -> SimpleNamespace:
    return SimpleNamespace(type=block_type, **fields)


def _message(content: list[SimpleNamespace], stop_reason: str) -> MagicMock:
    message = MagicMock()
    message.id = "msg_test"
    message._request_id = "req_test"
    message.model = "claude-sonnet-4-6"
    message.content = content
    message.stop_reason = stop_reason
    message.usage = SimpleNamespace(
        input_tokens=10,
        output_tokens=512,
        cache_read_input_tokens=None,
        cache_creation_input_tokens=None,
        output_tokens_details=None,
        iterations=None,
    )
    return message


async def _query(message: MagicMock, config: LLMConfig | None = None):
    model = AnthropicModel("claude-sonnet-4-6", config=config)
    stream = AsyncMock()
    stream.__aenter__ = AsyncMock(return_value=stream)
    stream.__aexit__ = AsyncMock(return_value=None)
    stream.get_final_message = AsyncMock(return_value=message)
    client = MagicMock()
    client.beta.messages.stream.return_value = stream
    body: dict[str, Any] = {
        "model": "claude-sonnet-4-6",
        "messages": [],
        "max_tokens": 512,
    }

    with (
        patch.object(model, "get_client", return_value=client),
        patch.object(model, "build_body", AsyncMock(return_value=body)),
    ):
        return await model._query_impl(  # pyright: ignore[reportPrivateUsage]
            [TextInput(text="solve this")],
            tools=[],
            query_logger=MagicMock(),
        )


@pytest.mark.parametrize("thinking_type", ["thinking", "redacted_thinking"])
async def test_turn_truncated_inside_thinking_is_retried(thinking_type: str):
    message = _message(
        [
            _block("text", text="partial answer", citations=None),
            _block(thinking_type, thinking="half a thou"),
        ],
        "max_tokens",
    )

    with pytest.raises(ModelNoOutputError):
        await _query(message)


async def test_thinking_truncated_turn_is_returned_when_the_key_opts_in():
    message = _message(
        [_block("thinking", thinking="half a thou")],
        "max_tokens",
    )

    result = await _query(
        message,
        LLMConfig(
            provider_config=AnthropicConfig(returns_thinking_truncated_turns=True)
        ),
    )

    assert result.reasoning == "half a thou"
    assert result.finish_reason.reason == FinishReason.MAX_TOKENS


async def test_a_turn_cut_off_before_any_thinking_text_is_returned_when_opted_in():
    """Anthropic can return a signed but empty thinking block when the cut lands early."""
    message = _message([_block("thinking", thinking="")], "max_tokens")

    result = await _query(
        message,
        LLMConfig(
            provider_config=AnthropicConfig(returns_thinking_truncated_turns=True)
        ),
    )

    assert result.output_text is None
    assert result.finish_reason.reason == FinishReason.MAX_TOKENS


async def test_a_turn_cut_off_before_any_thinking_text_still_raises_by_default():
    message = _message([_block("thinking", thinking="")], "max_tokens")

    with pytest.raises(MaxOutputTokensExceededError):
        await _query(message)


async def test_replaying_a_thinking_truncated_turn_ends_the_message_with_text():
    truncated = _message([_block("thinking", thinking="half a thou")], "max_tokens")

    parsed = await AnthropicModel(
        "claude-sonnet-4-6",
        config=LLMConfig(
            provider_config=AnthropicConfig(returns_thinking_truncated_turns=True)
        ),
    ).parse_input(
        [
            TextInput(text="solve this"),
            RawResponse(response=truncated),
            TextInput(text="continue"),
        ]
    )

    assistant = next(msg for msg in parsed if msg["role"] == "assistant")
    assert [block.type for block in assistant["content"]] == ["thinking", "text"]
    assert assistant["content"][-1].text == TRUNCATED_THINKING_MARKER


async def test_a_key_that_does_not_opt_in_replays_history_untouched():
    truncated = _message([_block("thinking", thinking="half a thou")], "max_tokens")

    parsed = await AnthropicModel("claude-sonnet-4-6").parse_input(
        [RawResponse(response=truncated), TextInput(text="continue")]
    )

    assistant = next(msg for msg in parsed if msg["role"] == "assistant")
    assert [block.type for block in assistant["content"]] == ["thinking"]


async def test_replaying_an_ordinary_turn_is_unchanged():
    finished = _message(
        [
            _block("thinking", thinking="a complete thought"),
            _block("text", text="the answer", citations=None),
        ],
        "end_turn",
    )

    parsed = await AnthropicModel("claude-sonnet-4-6").parse_input(
        [RawResponse(response=finished), TextInput(text="continue")]
    )

    assistant = next(msg for msg in parsed if msg["role"] == "assistant")
    assert [block.type for block in assistant["content"]] == ["thinking", "text"]
    assert assistant["content"][-1].text == "the answer"


async def test_turn_truncated_after_thinking_is_returned():
    message = _message(
        [
            _block("thinking", thinking="a complete thought"),
            _block("text", text="partial answer", citations=None),
        ],
        "max_tokens",
    )

    result = await _query(message)

    assert result.output_text == "partial answer"
    assert result.reasoning == "a complete thought"


async def test_thinking_before_tool_use_is_returned():
    message = _message(
        [
            _block("thinking", thinking="plan"),
            _block("tool_use", id="toolu_a", name="bash", input={"command": "ls"}),
        ],
        "tool_use",
    )

    result = await _query(message)

    assert [call.name for call in result.tool_calls] == ["bash"]
