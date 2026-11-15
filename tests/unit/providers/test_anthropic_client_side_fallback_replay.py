"""Replaying client-side fallback turns over the wire, through the SDK middleware."""

import json
from typing import Any
from unittest.mock import MagicMock, patch

import httpx2
import pytest
from anthropic import AsyncAnthropic
from anthropic.types.beta import BetaMessage

from model_library.base import LLMConfig
from model_library.base.input import RawResponse, TextInput
from model_library.providers.anthropic import (
    ANTHROPIC_SERVER_SIDE_FALLBACK_BETA,
    AnthropicConfig,
    AnthropicModel,
)

_CHAIN = ["claude-fallback-a", "claude-fallback-b"]


def _served_turn(to_model: str) -> RawResponse:
    return RawResponse(
        response=BetaMessage.model_validate(
            {
                "id": "msg_served",
                "type": "message",
                "role": "assistant",
                "model": "claude-primary-test",
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "content": [
                    {
                        "type": "fallback",
                        "from": {"model": "claude-primary-test"},
                        "to": {"model": to_model},
                        "trigger": {"type": "refusal", "category": "bio"},
                    },
                    {"type": "text", "text": "served"},
                ],
            }
        )
    )


def _sse(model: str) -> bytes:
    events: list[tuple[str, dict[str, Any]]] = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_next",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 0},
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "ok"},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 1},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(
        f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events
    ).encode()


def _anthropic_api(sent: list[dict[str, Any]]) -> httpx2.MockTransport:
    """Stands in for /v1/messages: a `fallback` block is a valid input tag only under the server-side fallback beta."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        sent.append(body)
        betas = request.headers.get("anthropic-beta", "").split(",")
        for i, message in enumerate(body["messages"]):
            for j, block in enumerate(message["content"]):
                if (
                    block.get("type") == "fallback"
                    and ANTHROPIC_SERVER_SIDE_FALLBACK_BETA not in betas
                ):
                    return httpx2.Response(
                        400,
                        json={
                            "type": "error",
                            "error": {
                                "type": "invalid_request_error",
                                "message": f"messages.{i}.content.{j}: Input tag 'fallback' found using 'type' does not match any of the expected tags",
                            },
                        },
                    )
        return httpx2.Response(
            200, content=_sse(body["model"]), headers={"content-type": "text/event-stream"}
        )

    return httpx2.MockTransport(handler)


@pytest.mark.parametrize("pinned", _CHAIN)
async def test_replayed_client_side_fallback_turn_is_accepted(pinned: str):
    """Pinned to the chain's last model, no middleware runs to strip the seam the API rejects."""
    model = AnthropicModel(
        "claude-primary-test",
        config=LLMConfig(
            max_tokens=4096,
            provider_config=AnthropicConfig(
                fallback_models=_CHAIN,
                client_side_fallback=True,
                supports_auto_thinking=True,
            ),
        ),
    )
    sent: list[dict[str, Any]] = []
    client = AsyncAnthropic(
        api_key="test-key",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=_anthropic_api(sent)),
    )

    with patch.object(model, "get_client", return_value=client):
        result = await model._query_impl(  # pyright: ignore[reportPrivateUsage]
            [
                TextInput(text="identify the pathogen"),
                _served_turn(pinned),
                TextInput(text="keep going"),
            ],
            tools=[],
            query_logger=MagicMock(),
        )

    assert result.output_text == "ok"
    assert [request["model"] for request in sent] == [pinned]
    assert [block["type"] for block in sent[0]["messages"][1]["content"]] == ["text"]
