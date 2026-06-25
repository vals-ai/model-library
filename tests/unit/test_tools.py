"""
Provider-agnostic unit tests for tool-calling request shaping.

These assert that models which support tools include the tool specs in their
request bodies without making any network calls.
"""

import logging
from typing import cast

import pytest
from anthropic.types.beta.beta_tool_use_block import BetaToolUseBlock
from anthropic.types.beta.beta_usage import BetaUsage
from anthropic.types.beta.parsed_beta_message import (
    ParsedBetaContentBlock,
    ParsedBetaMessage,
)

from model_library.base import (
    FileWithUrl,
    RawResponse,
    TextInput,
    ToolBody,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from model_library.providers.amazon import AmazonModel
from model_library.providers.anthropic import AnthropicModel
from model_library.providers.google import GoogleModel
from model_library.providers.openai import OpenAIModel


@pytest.mark.parametrize(
    "provider,Model,model_name,expects_body_keys",
    [
        (
            "google",
            GoogleModel,
            "gemini-2.5-flash-lite",
            {"tools", "tool_config", "config", "model"},
        ),
        ("openai", OpenAIModel, "gpt-4o-mini", {"tools", "input", "model"}),
        (
            "amazon",
            AmazonModel,
            "anthropic.claude-3-5-haiku-2024-10-22-v2:0",
            {"toolConfig", "messages", "modelId"},
        ),
    ],
)
async def test_build_body_includes_tools(
    provider, Model, model_name, expects_body_keys
):
    model = Model(model_name)

    tools = [
        ToolDefinition(
            name="get_weather",
            body=ToolBody(
                name="get_weather",
                description="Get weather",
                properties={"location": {"type": "string"}},
                required=["location"],
            ),
        )
    ]

    body = await model.build_body([TextInput(text="hi")], tools=tools)

    if provider == "google":
        # For Google, tools live inside config
        assert set(["model", "config", "contents"]).issubset(set(body.keys()))
        cfg = body["config"]
        assert getattr(cfg, "tools", None)
    else:
        assert expects_body_keys.issubset(set(body.keys()))


async def test_parse_tools_shapes_for_all():
    tools = [
        ToolDefinition(
            name="get_echo",
            body=ToolBody(
                name="get_echo",
                description="Echo",
                properties={"value": {"type": "string"}},
                required=["value"],
            ),
        )
    ]

    # OpenAI
    openai_m = OpenAIModel("gpt-4o-mini")
    openai_tools = await openai_m.parse_tools(tools)
    assert (
        isinstance(openai_tools, list)
        and openai_tools
        and openai_tools[0]["type"] == "function"
    )

    # Anthropic
    anthropic_m = AnthropicModel("claude-3-7-sonnet-latest")
    anthropic_tools = await anthropic_m.parse_tools(tools)
    assert (
        isinstance(anthropic_tools, list)
        and anthropic_tools
        and anthropic_tools[0]["name"] == "get_echo"
    )

    # Amazon (Bedrock)
    amazon_m = AmazonModel("anthropic.claude-3-5-haiku-2024-10-22-v2:0")
    amazon_tools = await amazon_m.parse_tools(tools)
    assert (
        isinstance(amazon_tools, list)
        and amazon_tools
        and "toolSpec" in amazon_tools[0]
        and amazon_tools[0]["toolSpec"]["name"] == "get_echo"
    )

    google_m = GoogleModel("gemini-2.5-flash-lite")
    google_tools = await google_m.parse_tools(tools)
    assert isinstance(google_tools, list) and google_tools


@pytest.mark.parametrize(
    "model_class,model_name",
    [
        (GoogleModel, "gemini-2.5-flash-lite"),
    ],
)
async def test_google_tool_result_roundtrip_no_storage_import(model_class, model_name):
    # Ensure that creating a GoogleModel and building a body with a ToolResult
    # does not require the storage client (batch-only) and shapes content as expected.

    m = model_class(model_name)
    tr = ToolResult(
        tool_call=ToolCall(id="abc123", name="get_weather", args={"location": "SF"}),
        result={"temperature": "21C"},
    )
    body = await m.build_body([tr, TextInput(text="Thanks!")], tools=[])
    contents = body["contents"]
    roles = [getattr(c, "role", None) for c in contents]
    assert "user" in roles

    function_response = contents[0].parts[0].function_response
    assert function_response is not None
    assert function_response.id == "abc123"


async def test_anthropic_rejects_invalid_tool_result():
    """Verify that providing a ToolResult without a matching tool call raises an exception."""

    model = AnthropicModel("claude-3-7-sonnet-latest")
    orphaned_result = ToolResult(
        tool_call=ToolCall(id="nonexistent_id", name="get_weather", args={}),
        result="Sunny",
    )

    with pytest.raises(
        Exception, match="Tool call result provided with no matching tool call"
    ):
        await model.parse_input([TextInput(text="Hello"), orphaned_result])


def _anthropic_response_with_two_tool_uses(tool_calls: list[ToolCall]):
    return ParsedBetaMessage(
        id="msg_test",
        content=[
            cast(
                ParsedBetaContentBlock,
                BetaToolUseBlock(
                    id=tool_calls[0].id,
                    input={"command": "echo a"},
                    name=tool_calls[0].name,
                    type="tool_use",
                ),
            ),
            cast(
                ParsedBetaContentBlock,
                BetaToolUseBlock(
                    id=tool_calls[1].id,
                    input={"command": "echo b"},
                    name=tool_calls[1].name,
                    type="tool_use",
                ),
            ),
        ],
        model="claude-3-7-sonnet-latest",
        role="assistant",
        stop_reason="tool_use",
        stop_sequence=None,
        type="message",
        usage=BetaUsage(input_tokens=1, output_tokens=1),
    )


async def test_anthropic_groups_parallel_tool_results_in_one_user_message():
    model = AnthropicModel("claude-3-7-sonnet-latest")
    tool_calls = [
        ToolCall(id="toolu_a", name="bash", args={"command": "echo a"}),
        ToolCall(id="toolu_b", name="bash", args={"command": "echo b"}),
    ]

    parsed = await model.parse_input(
        [
            RawResponse(response=_anthropic_response_with_two_tool_uses(tool_calls)),
            ToolResult(tool_call=tool_calls[0], result="A"),
            ToolResult(tool_call=tool_calls[1], result="B"),
        ]
    )

    assert [message["role"] for message in parsed] == ["assistant", "user"]
    tool_result_blocks = parsed[1]["content"]
    assert [block["type"] for block in tool_result_blocks] == [
        "tool_result",
        "tool_result",
    ]
    assert [block["tool_use_id"] for block in tool_result_blocks] == [
        "toolu_a",
        "toolu_b",
    ]


@pytest.mark.parametrize(
    "interleaved_item,expected_type",
    [
        (TextInput(text="note"), "text"),
        (
            FileWithUrl(
                type="file",
                name="note.txt",
                mime="text/plain",
                url="https://example.com/note.txt",
            ),
            "document",
        ),
    ],
)
async def test_anthropic_keeps_interleaved_tool_results_in_separate_user_messages(
    interleaved_item,
    expected_type,
):
    model = AnthropicModel("claude-3-7-sonnet-latest")
    tool_calls = [
        ToolCall(id="toolu_a", name="bash", args={"command": "echo a"}),
        ToolCall(id="toolu_b", name="bash", args={"command": "echo b"}),
    ]

    parsed = await model.parse_input(
        [
            RawResponse(response=_anthropic_response_with_two_tool_uses(tool_calls)),
            ToolResult(tool_call=tool_calls[0], result="A"),
            interleaved_item,
            ToolResult(tool_call=tool_calls[1], result="B"),
        ]
    )

    assert [message["role"] for message in parsed] == [
        "assistant",
        "user",
        "user",
        "user",
    ]
    assert [block["type"] for block in parsed[1]["content"]] == ["tool_result"]
    assert [block["type"] for block in parsed[2]["content"]] == [expected_type]
    assert [block["type"] for block in parsed[3]["content"]] == ["tool_result"]


class TestStreamingToolCallAccumulation:
    """
    Tests for streaming tool call chunk accumulation in OpenAI _query_completions.
    """

    @staticmethod
    def _make_chunk(
        tool_call_id: str | None, func_name: str | None, func_args: str | None
    ):
        """Create a mock streaming chunk with tool call data."""
        from openai.types.chat.chat_completion_chunk import (
            ChatCompletionChunk,
            Choice,
            ChoiceDelta,
            ChoiceDeltaToolCall,
            ChoiceDeltaToolCallFunction,
        )

        tool_calls = None
        if tool_call_id is not None or func_name is not None or func_args is not None:
            tool_calls = [
                ChoiceDeltaToolCall(
                    index=0,
                    id=tool_call_id,
                    function=ChoiceDeltaToolCallFunction(
                        name=func_name, arguments=func_args
                    ),
                    type="function" if tool_call_id else None,
                )
            ]

        return ChatCompletionChunk(
            id="chunk",
            created=0,
            model="test",
            object="chat.completion.chunk",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(tool_calls=tool_calls),
                    finish_reason=None,
                )
            ],
        )

    @staticmethod
    async def _run_query(model: OpenAIModel, chunks: list):
        """Run _query_completions with mocked stream."""
        from unittest.mock import AsyncMock, MagicMock, patch

        async def mock_stream():
            for c in chunks:
                yield c

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        with patch.object(model, "get_client", return_value=mock_client):
            with patch.object(
                model, "build_body", new_callable=AsyncMock, return_value={}
            ):
                return await model._query_completions(
                    [], tools=[], query_logger=logging.getLogger("test")
                )

    async def test_openai_multiple_tool_calls_different_ids(self):
        """OpenAI: different IDs should create separate tool calls."""
        model = OpenAIModel("gpt-4o-mini")
        chunks = [
            self._make_chunk("call_1", "get_weather", '{"location": "SF"}'),
            self._make_chunk("call_2", "get_time", '{"tz": "PST"}'),
        ]

        result = await self._run_query(model, chunks)

        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[1].name == "get_time"

    async def test_openai_same_id_accumulates(self):
        """OpenAI: same ID should accumulate (NOT create new call)."""
        model = OpenAIModel("gpt-4o-mini")
        chunks = [
            self._make_chunk("call_1", "get_weather", '{"location":'),
            self._make_chunk("call_1", None, ' "SF"}'),  # same ID, no name = accumulate
        ]

        result = await self._run_query(model, chunks)

        assert len(result.tool_calls) == 1  # should be 1, not 2
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].args == '{"location": "SF"}'

    async def test_poolside_same_index_different_ids_accumulates(self):
        """Poolside may stream one tool call under different IDs but the same index."""
        model = OpenAIModel(
            "poolside/laguna-xs.2", provider="poolside", use_completions=True
        )
        chunks = [
            self._make_chunk("call_name", "get_weather", ""),
            self._make_chunk("call_args", None, '{"location":'),
            self._make_chunk("call_args", None, ' "SF"}'),
        ]

        result = await self._run_query(model, chunks)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_name"
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].args == '{"location": "SF"}'
