from logging import Logger
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

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
    ToolResult,
)
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
        signed_history=LLM.serialize_input(result.history, secret=b"test-secret"),
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


@pytest.mark.asyncio
async def test_google_computer_call_replays_signed_history() -> None:
    from google.genai.types import (
        Candidate,
        Content,
        Environment,
        FinishReason,
        FunctionCall,
        GenerateContentResponse,
        Part,
    )

    from model_library.base import ToolBody

    turn = Content(
        role="model",
        parts=[
            Part(
                function_call=FunctionCall(
                    name="click",
                    args={
                        "x": 500,
                        "y": 250,
                        "intent": "open the menu",
                        "safety_decision": {
                            "decision": "require_confirmation",
                            "explanation": "This sends a form.",
                        },
                    },
                )
            ),
            Part(
                function_call=FunctionCall(
                    id="n", name="notes", args={"text": "menu"}
                )
            ),
        ],
    )

    def stream_of(content: Content):
        async def stream():
            yield GenerateContentResponse(
                candidates=[Candidate(content=content, finish_reason=FinishReason.STOP)]
            )

        return stream()

    client = MagicMock()
    client.aio.models.generate_content_stream = AsyncMock(
        side_effect=[
            stream_of(turn),
            stream_of(Content(role="model", parts=[Part(text="Done")])),
        ]
    )
    notes = ToolDefinition(
        name="notes",
        body=ToolBody(
            name="notes",
            description="Save notes.",
            properties={"text": {"type": "string"}},
            required=["text"],
        ),
    )
    tools = [computer_tool(), notes]
    model = GoogleModel(
        "gemini-test", config=LLMConfig(custom_api_key=SecretStr("test-key"))
    )
    model.get_client = lambda: client  # type: ignore[method-assign]

    result = await model.query([TextInput(text="click")], tools=tools)

    config = client.aio.models.generate_content_stream.call_args.kwargs["config"]
    assert config.tools[0].computer_use.environment == Environment.ENVIRONMENT_DESKTOP
    assert config.tools[1].function_declarations[0].name == "notes"
    call = result.computer_calls[0]
    assert call.call_id.startswith("click_")
    assert call.provider == "google"
    assert call.actions[0]["name"] == "click"
    assert call.actions[0]["args"]["x"] == 500
    assert call.pending_safety_checks[0]["decision"] == "require_confirmation"
    assert [tool_call.name for tool_call in result.tool_calls] == ["notes"]

    response = query_result_response_body(
        result,
        signed_history=LLM.serialize_input(result.history, secret=b"test-secret"),
        failed=False,
    )
    client_result = _parse_query_result(response, schema_model=None)
    replies = [
        ToolResult(tool_call=client_result.tool_calls[0], result="Notes saved."),
        ComputerCallOutput(
            call_id=client_result.computer_calls[0].call_id,
            screenshot_url="data:image/png;base64,AA==",
            acknowledged_safety_checks=client_result.computer_calls[0].pending_safety_checks,
        ),
    ]
    gateway = GatewayLLM("gemini-test", "google")
    request = QueryRequest.model_validate(
        await gateway.build_body([*client_result.history, *replies], tools=tools)
    )
    require_raw_input_secret(request.inputs, secret=b"test-secret")
    LLM.restore_raw_fields(request.inputs, secret=b"test-secret")
    await model.query(request.inputs, tools=request.tools)

    contents = client.aio.models.generate_content_stream.call_args.kwargs["contents"]
    assert [content.role for content in contents] == ["user", "model", "user"]
    assert contents[1] == turn
    click, saved = (part.function_response for part in contents[2].parts)
    assert (click.id, click.name) == (None, "click")
    assert click.response == {"safety_acknowledgement": True}
    assert click.parts[0].inline_data.mime_type == "image/png"
    assert click.parts[0].inline_data.data == b"\x00"
    assert (saved.name, saved.response) == ("notes", {"result": "Notes saved."})
    assert saved.id == "n"


@pytest.mark.asyncio
async def test_google_string_history_passes_through() -> None:
    model = GoogleModel(
        "gemini-test", config=LLMConfig(custom_api_key=SecretStr("test-key"))
    )
    body = await model.build_body(
        [RawResponse(response=["prior text"]), TextInput(text="continue")], tools=[]
    )
    assert body["contents"][0] == "prior text"
    assert body["contents"][1].parts[0].text == "continue"


@pytest.mark.asyncio
@pytest.mark.parametrize("history", [None, ["prior text"], [{"parts": 42}]])
async def test_google_unreadable_reply_history_keeps_fallback(history) -> None:
    model = GoogleModel(
        "gemini-test", config=LLMConfig(custom_api_key=SecretStr("test-key"))
    )
    contents = await model.parse_input(
        [*([RawResponse(response=history)] if history is not None else []),
         ComputerCallOutput(call_id="click_generated", output_text="done")]
    )
    response = contents[-1].parts[0].function_response
    assert (response.id, response.name, response.response) == (
        "click_generated", "click", {"output": "done"}
    )


@pytest.mark.asyncio
async def test_google_reply_matching_uses_only_latest_turn() -> None:
    from google.genai.types import Content, FunctionCall, Part

    from model_library.base import ToolCall

    model = GoogleModel(
        "gemini-test", config=LLMConfig(custom_api_key=SecretStr("test-key"))
    )
    turn = Content(role="model", parts=[
        Part(function_call=FunctionCall(name="notes")),
        Part(function_call=FunctionCall(name="notes", id="n")),
        Part(function_call=FunctionCall(name="notes")),
    ])
    def reply(id: str, name: str, result: str) -> ToolResult:
        return ToolResult(tool_call=ToolCall(id=id, name=name, args={}), result=result)

    contents = await model.parse_input([
        RawResponse(response=["unrelated"]),
        RawResponse(response=[turn]),
        reply("n", "notes", "provider"),
        reply("x", "unmatched", "first"),
        reply("notes_one", "notes", "one"),
        reply("notes_two", "notes", "two"),
        reply("y", "unmatched", "second"),
        TextInput(text="warning"),
    ])
    responses = [part.function_response for part in contents[-2].parts]
    assert [(r.id, r.name, r.response["result"]) for r in responses] == [
        (None, "notes", "one"), ("n", "notes", "provider"),
        (None, "notes", "two"), (None, "unmatched", "first"),
        (None, "unmatched", "second"),
    ]
    assert contents[-1].parts[0].text == "warning"
    assert turn.parts[0].function_call.id is None
    assert await model.parse_input([]) == []


@pytest.mark.asyncio
async def test_google_reply_matching_reads_every_streamed_chunk() -> None:
    from google.genai.types import Content, FunctionCall, Part

    from model_library.base import ToolCall

    model = GoogleModel(
        "gemini-test", config=LLMConfig(custom_api_key=SecretStr("test-key"))
    )
    # One model turn streams its two calls in two chunks.
    chunks = [
        Content(
            role="model",
            parts=[Part(function_call=FunctionCall(id="c", name="click"))],
        ),
        Content(
            role="model",
            parts=[Part(function_call=FunctionCall(id="n", name="notes"))],
        ),
    ]
    contents = await model.parse_input(
        [
            RawResponse(response=chunks),
            ToolResult(
                tool_call=ToolCall(id="n", name="notes", args={}), result="saved"
            ),
            ComputerCallOutput(call_id="c", output_text="done"),
        ]
    )
    responses = [part.function_response for part in contents[-1].parts]
    assert [(r.id, r.name) for r in responses] == [("c", "click"), ("n", "notes")]


@pytest.mark.asyncio
async def test_google_computer_reply_needs_a_data_url() -> None:
    model = GoogleModel(
        "gemini-test", config=LLMConfig(custom_api_key=SecretStr("test-key"))
    )
    with pytest.raises(BadInputError, match="data: screenshot"):
        await model.parse_input(
            [ComputerCallOutput(call_id="click_1", screenshot_url="https://x/y.png")]
        )


@pytest.mark.asyncio
async def test_google_reads_a_data_url_image() -> None:
    from model_library.base import FileWithUrl

    model = GoogleModel(
        "gemini-test", config=LLMConfig(custom_api_key=SecretStr("test-key"))
    )
    part = await model.parse_file(
        FileWithUrl(
            type="image", name="s.jpg", mime="image/jpeg", url="data:image/jpeg;base64,AA=="
        )
    )
    assert part.inline_data is not None
    assert (part.inline_data.mime_type, part.inline_data.data) == ("image/jpeg", b"\x00")


@pytest.mark.asyncio
async def test_google_native_function_stays_a_tool_call_and_dict_history_reads() -> None:
    from google.genai.types import (
        Candidate,
        Content,
        FinishReason,
        FunctionCall,
        FunctionDeclaration,
        GenerateContentResponse,
        Part,
        Tool,
    )

    turn = Content(
        role="model",
        parts=[Part(function_call=FunctionCall(id="call_1", name="lookup", args={}))],
    )

    async def stream():
        yield GenerateContentResponse(
            candidates=[Candidate(content=turn, finish_reason=FinishReason.STOP)]
        )

    client = MagicMock()
    client.aio.models.generate_content_stream = AsyncMock(return_value=stream())
    native = ToolDefinition(
        name="lookup",
        body=Tool(function_declarations=[FunctionDeclaration(name="lookup")]),
    )
    model = GoogleModel(
        "gemini-test", config=LLMConfig(custom_api_key=SecretStr("test-key"))
    )
    model.get_client = lambda: client  # type: ignore[method-assign]
    result = await model._query_impl(
        [TextInput(text="look")],
        tools=[computer_tool(), native],
        query_logger=MagicMock(spec=Logger),
    )
    assert [call.name for call in result.tool_calls] == ["lookup"]
    assert result.computer_calls == []

    contents = await model.parse_input(
        [
            RawResponse(
                response=[
                    {
                        "role": "model",
                        "parts": [
                            {"function_call": {"id": "call_9", "name": "click", "args": {}}}
                        ],
                    }
                ]
            ),
            ComputerCallOutput(call_id="call_9", screenshot_url="data:image/png;base64,AA=="),
        ]
    )
    assert contents[1].parts[0].function_response.name == "click"
