"""Replaying assistant turns whose streamed content contains empty text blocks."""

from anthropic.types.beta import BetaMessage

from model_library.base import LLMConfig
from model_library.base.input import RawResponse, TextInput
from model_library.providers.anthropic import (
    EMPTY_TEXT_MARKER,
    AnthropicConfig,
    AnthropicModel,
)

_EMPTY_TEXT_RESPONSE = BetaMessage.model_validate(
    {
        "id": "msg_empty_text",
        "type": "message",
        "role": "assistant",
        "model": "claude-test",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "content": [
            {"type": "thinking", "thinking": "first pass", "signature": "sig-1"},
            {"type": "text", "text": ""},
            {"type": "thinking", "thinking": "second pass", "signature": "sig-2"},
            {"type": "text", "text": "partial answer"},
            {"type": "thinking", "thinking": "third pass", "signature": "sig-3"},
            {"type": "text", "text": ""},
        ],
    }
)


async def test_replayed_assistant_turn_replaces_empty_text_blocks_in_place():
    """Anthropic rejects replayed history containing an empty text block
    ("text content blocks must be non-empty"), and also rejects the latest
    assistant message if its thinking blocks shift position ("`thinking` ...
    blocks in the latest assistant message cannot be modified"), so empty
    blocks must be replaced without changing the index of any other block."""
    parsed = await AnthropicModel("claude-test").parse_input(
        [
            TextInput(text="solve this"),
            RawResponse(response=_EMPTY_TEXT_RESPONSE),
            TextInput(text="continue"),
        ]
    )

    assistant = next(msg for msg in parsed if msg["role"] == "assistant")
    assert [(block.type, getattr(block, "text", None)) for block in assistant["content"]] == [
        ("thinking", None),
        ("text", EMPTY_TEXT_MARKER),
        ("thinking", None),
        ("text", "partial answer"),
        ("thinking", None),
        ("text", EMPTY_TEXT_MARKER),
    ]
    assert [
        getattr(block, "signature", None)
        for block in assistant["content"]
        if block.type == "thinking"
    ] == ["sig-1", "sig-2", "sig-3"]


async def test_configured_replay_only_marks_max_tokens_turns_as_truncated():
    parsed = await AnthropicModel(
        "claude-test",
        config=LLMConfig(
            provider_config=AnthropicConfig(returns_thinking_truncated_turns=True)
        ),
    ).parse_input(
        [
            TextInput(text="solve this"),
            RawResponse(response=_EMPTY_TEXT_RESPONSE),
            TextInput(text="continue"),
        ]
    )

    assistant = next(msg for msg in parsed if msg["role"] == "assistant")
    assert assistant["content"][-1].text == EMPTY_TEXT_MARKER
