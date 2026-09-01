"""Replaying assistant turns that a server-side fallback served."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from anthropic.types.beta import BetaMessage

from model_library.base import LLMConfig
from model_library.base.input import RawResponse, TextInput
from model_library.providers.anthropic import AnthropicModel

_FALLBACK_RESPONSE = BetaMessage.model_validate(
    {
        "id": "msg_fallback",
        "type": "message",
        "role": "assistant",
        "model": "claude-primary-test",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "content": [
            {
                "type": "thinking",
                "thinking": "primary reasoning",
                "signature": "sig-primary",
            },
            {
                "type": "fallback",
                "from": {"model": "claude-primary-test"},
                "to": {"model": "claude-fallback-test"},
                "trigger": {"type": "refusal", "category": "cyber"},
            },
            {
                "type": "thinking",
                "thinking": "fallback reasoning",
                "signature": "sig-fallback",
            },
            {"type": "text", "text": "answer"},
        ],
    }
)


async def test_replayed_assistant_turn_keeps_fallback_boundary_block():
    """The boundary block separates the thinking runs Anthropic validates.

    Omitting it merges the two runs into one span the server rejects with
    "`thinking` or `redacted_thinking` blocks in the latest assistant message
    cannot be modified".
    """
    model = AnthropicModel("claude-primary-test")

    parsed = await model.parse_input(
        [
            TextInput(text="reverse engineer this binary"),
            RawResponse(response=_FALLBACK_RESPONSE),
            TextInput(text="keep going"),
        ]
    )

    assistant = parsed[1]
    assert assistant["role"] == "assistant"
    assert [block.type for block in assistant["content"]] == [
        "thinking",
        "fallback",
        "thinking",
        "text",
    ]


async def test_count_tokens_drops_fallback_boundary_block():
    """The non-beta count_tokens endpoint has no `fallback` input block type."""
    model = AnthropicModel(
        "claude-primary-test", config=LLMConfig(max_tokens=4096)
    )
    captured: dict[str, Any] = {}

    async def _count_tokens(**body: Any) -> MagicMock:
        captured.update(body)
        return MagicMock(input_tokens=7)

    client = MagicMock()
    client.messages.count_tokens = AsyncMock(side_effect=_count_tokens)

    with patch.object(model, "get_client", return_value=client):
        count = await model.count_tokens(
            [
                TextInput(text="reverse engineer this binary"),
                RawResponse(response=_FALLBACK_RESPONSE),
                TextInput(text="keep going"),
            ]
        )

    assert count == 7
    assistant = captured["messages"][1]
    assert [block.type for block in assistant["content"]] == [
        "thinking",
        "thinking",
        "text",
    ]
