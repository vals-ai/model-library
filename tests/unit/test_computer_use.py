from logging import Logger
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from anthropic.types.beta import BetaMessage
from openai.types.responses.response import Response
from pydantic import SecretStr

from model_library.base import (
    ComputerCallOutput,
    LLM,
    LLMConfig,
    RawResponse,
    TextInput,
    ToolDefinition,
)
from model_gateway.history import sign_history
from model_gateway.model_helpers import require_raw_input_secret
from model_gateway.types import QueryRequest, query_result_response_body
from model_library.base.gateway import GatewayLLM, _parse_query_result
from model_library.exceptions import BadInputError, NoMatchingToolCallError
from model_library.providers.anthropic import AnthropicConfig, AnthropicModel
from model_library.providers.google import GoogleModel
from model_library.providers.openai import OpenAIModel


def computer_tool() -> ToolDefinition:
    return ToolDefinition(
        name="computer",
        body={
            "type": "computer",
            "display_width_px": 1024,
            "display_height_px": 768,
            "display_number": 1,
        },
    )


@pytest.mark.asyncio
async def test_anthropic_computer_call_replays_signed_history() -> None:
    message = BetaMessage.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "computer",
                    "input": {"action": "left_click", "coordinate": [10, 20]},
                }
            ],
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }
    )

    class Stream:
        async def __aenter__(self) -> "Stream":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        def __aiter__(self) -> "Stream":
            return self

        async def __anext__(self) -> object:
            raise StopAsyncIteration

        async def get_final_message(self) -> BetaMessage:
            return message

    class Messages:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] = {}

        def stream(self, **kwargs: object) -> Stream:
            self.kwargs = kwargs
            return Stream()

    messages = Messages()
    model = AnthropicModel(
        "claude-sonnet-4-6",
        config=LLMConfig(
            max_tokens=64,
            custom_api_key=SecretStr("test"),
            provider_config=AnthropicConfig(computer_tool_type="computer_20251124"),
        ),
    )
    model.get_client = lambda: SimpleNamespace(  # type: ignore[method-assign]
        beta=SimpleNamespace(messages=messages)
    )

    result = await model._query_impl(
        [TextInput(text="click")],
        tools=[computer_tool()],
        query_logger=MagicMock(spec=Logger),
    )

    assert messages.kwargs["betas"].count("computer-use-2025-11-24") == 1
    assert messages.kwargs["tools"][0]["type"] == "computer_20251124"
    assert "enable_zoom" not in messages.kwargs["tools"][0]
    assert result.computer_calls[0].call_id == "toolu_1"
    assert result.computer_calls[0].provider == "anthropic"
    assert result.computer_calls[0].actions == [
        {"action": "left_click", "coordinate": [10, 20]}
    ]

    response = query_result_response_body(
        result,
        signed_history=sign_history(result.history, secret=b"test-secret"),
        failed=False,
    )
    client_result = _parse_query_result(response, schema_model=None)
    raw = client_result.history[1]
    assert isinstance(raw, RawResponse)
    assert isinstance(raw.response, dict)
    assert set(raw.response) == {"pickle", "hmac"}
    assert all(isinstance(value, str) for value in raw.response.values())
    output = ComputerCallOutput(
        call_id=client_result.computer_calls[0].call_id,
        output_text="Clicked",
        screenshot_url="data:image/png;base64,AA==",
    )
    gateway = GatewayLLM("claude-sonnet-4-6", "anthropic")
    request = QueryRequest.model_validate(
        await gateway.build_body(
            [*client_result.history, output], tools=[computer_tool()]
        )
    )
    assert request.model == "anthropic/claude-sonnet-4-6"
    assert request.inputs[1] == raw
    require_raw_input_secret(request.inputs, secret=b"test-secret")
    LLM.restore_raw_fields(request.inputs, secret=b"test-secret")
    assert isinstance(request.inputs[1], RawResponse)
    assert isinstance(request.inputs[1].response, BetaMessage)
    await model._query_impl(
        request.inputs,
        tools=request.tools,
        query_logger=MagicMock(spec=Logger),
    )

    assert messages.kwargs["betas"].count("computer-use-2025-11-24") == 1
    assert messages.kwargs["tools"] == [
        {
            "type": "computer_20251124",
            "name": "computer",
            "display_width_px": 1024,
            "display_height_px": 768,
            "display_number": 1,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    replay = messages.kwargs["messages"]
    assert replay[0] == {
        "role": "user",
        "content": [{"type": "text", "text": "click"}],
    }
    assert replay[1]["role"] == "assistant"
    assert [block.model_dump() for block in replay[1]["content"]] == [
        message.content[0].model_dump()
    ]
    assert replay[2] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "cache_control": {"type": "ephemeral"},
                "content": [
                    {"type": "text", "text": "Clicked"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "AA==",
                        },
                    },
                ],
            }
        ],
    }
