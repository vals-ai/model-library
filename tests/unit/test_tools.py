"""
Provider-agnostic unit tests for tool-calling request shaping.

These assert that models which support tools include the tool specs in their
request bodies without making any network calls.
"""

import logging
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCallFunction
from anthropic.types.beta.beta_tool_use_block import BetaToolUseBlock
from anthropic.types.beta.beta_usage import BetaUsage
from anthropic.types.beta.parsed_beta_message import (
    ParsedBetaContentBlock,
    ParsedBetaMessage,
)

from model_library.base import (
    FileWithUrl,
    QueryResultMetadata,
    QueryResultPerformance,
    RawResponse,
    TextInput,
    ToolBody,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from model_library.base.output.builder import QueryResultBuilder
from model_library.providers.amazon import AmazonModel
from model_library.providers.anthropic import AnthropicModel
from model_library.providers.google import GoogleModel

from model_library.providers.openai import OpenAIModel


def _require_performance(metadata: QueryResultMetadata) -> QueryResultPerformance:
    performance = metadata.performance
    assert performance is not None
    return performance


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


async def test_openai_completions_includes_tools_when_support_is_false():
    model = OpenAIModel("gpt-4o-mini", use_completions=True)
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

    assert not model.supports_tools
    body = await model.build_body([TextInput(text="hi")], tools=tools)
    assert body["tools"]


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
        tool_call_id: str | None,
        func_name: str | None,
        func_args: str | None,
        content: str | None = None,
        index: int = 0,
        extra_content: dict[str, Any] | None = None,
    ):
        """Create a mock streaming chunk with tool call data."""
        from openai.types.chat.chat_completion_chunk import (
            ChatCompletionChunk,
            Choice,
            ChoiceDelta,
            ChoiceDeltaToolCall,
        )

        tool_calls = None
        if (
            tool_call_id is not None
            or func_name is not None
            or func_args is not None
            or extra_content is not None
        ):
            function = ChoiceDeltaToolCallFunction(name=func_name, arguments=func_args)
            if extra_content is None:
                tool_calls = [
                    ChoiceDeltaToolCall(
                        index=index,
                        id=tool_call_id,
                        function=function,
                        type="function" if tool_call_id else None,
                    )
                ]
            else:
                tool_calls = [
                    ChoiceDeltaToolCall.model_validate(
                        {
                            "index": index,
                            "id": tool_call_id,
                            "function": function,
                            "type": "function" if tool_call_id else None,
                            "extra_content": extra_content,
                        }
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
                    delta=ChoiceDelta(content=content, tool_calls=tool_calls),
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

        assert result.output_text is None
        assert result.reasoning is None
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[1].name == "get_time"
        assert _require_performance(result.metadata).timeline[0].channel == "tool_call"

    async def test_openai_same_id_accumulates(self):
        """OpenAI: same ID should accumulate (NOT create new call)."""
        model = OpenAIModel("gpt-4o-mini")
        chunks = [
            self._make_chunk("call_1", "get_weather", '{"location":'),
            self._make_chunk("call_1", None, ' "SF"}'),  # same ID, no name = accumulate
        ]

        result = await self._run_query(model, chunks)

        assert result.output_text is None
        assert result.reasoning is None
        assert len(result.tool_calls) == 1  # should be 1, not 2
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].args == '{"location": "SF"}'
        assert _require_performance(result.metadata).timeline[0].channel == "tool_call"

    async def test_openai_interleaved_tool_call_chunks_accumulate_by_index(self):
        model = OpenAIModel("gpt-4o-mini")
        chunks = [
            self._make_chunk("call_1", "get_weather", "", index=0),
            self._make_chunk("call_2", "get_time", "", index=1),
            self._make_chunk(None, None, '{"location": "SF"}', index=0),
            self._make_chunk(None, None, '{"tz": "PST"}', index=1),
        ]

        result = await self._run_query(model, chunks)

        assert [(call.id, call.name, call.args) for call in result.tool_calls] == [
            ("call_1", "get_weather", '{"location": "SF"}'),
            ("call_2", "get_time", '{"tz": "PST"}'),
        ]
        assert [
            entry.channel for entry in _require_performance(result.metadata).timeline
        ] == [
            "tool_call",
            "tool_call",
        ]
        assert [
            [event.type for event in entry.events]
            for entry in _require_performance(result.metadata).timeline
        ] == [
            [
                "tool_call_started",
                "tool_call_ready",
                "tool_call_delta",
                "tool_call_finished",
            ],
            [
                "tool_call_started",
                "tool_call_ready",
                "tool_call_delta",
                "tool_call_finished",
            ],
        ]

    async def test_openai_pre_id_tool_call_arguments_accumulate_by_index(self):
        model = OpenAIModel("gpt-4o-mini")
        chunks = [
            self._make_chunk(None, None, '{"location":', index=0),
            self._make_chunk("call_1", "get_weather", ' "SF"}', index=0),
        ]

        result = await self._run_query(model, chunks)

        assert [(call.id, call.name, call.args) for call in result.tool_calls] == [
            ("call_1", "get_weather", '{"location": "SF"}')
        ]
        response = cast(RawResponse, result.history[-1]).response
        raw_tool_calls = response.tool_calls
        assert raw_tool_calls is not None
        assert [call.id for call in raw_tool_calls] == ["call_1"]
        assert [
            [event.type for event in entry.events]
            for entry in _require_performance(result.metadata).timeline
        ] == [
            [
                "tool_call_started",
                "tool_call_delta",
                "tool_call_ready",
                "tool_call_delta",
                "tool_call_finished",
            ]
        ]

    async def test_openai_pre_id_tool_calls_keep_stream_index_order(self):
        model = OpenAIModel("gpt-4o-mini")
        chunks = [
            self._make_chunk(None, "first", None, index=0),
            self._make_chunk(None, "second", None, index=1),
            self._make_chunk("call_2", None, "{}", index=1),
            self._make_chunk("call_1", None, "{}", index=0),
        ]

        result = await self._run_query(model, chunks)

        assert [(call.id, call.name) for call in result.tool_calls] == [
            ("call_1", "first"),
            ("call_2", "second"),
        ]
        response = cast(RawResponse, result.history[-1]).response
        raw_tool_calls = response.tool_calls
        assert raw_tool_calls is not None
        assert [(call.id, call.function.name) for call in raw_tool_calls] == [
            ("call_1", "first"),
            ("call_2", "second"),
        ]

    async def test_openai_pre_id_tool_call_name_accumulates_by_index(self):
        model = OpenAIModel("gpt-4o-mini")
        chunks = [
            self._make_chunk(None, "get_weather", None, index=0),
            self._make_chunk("call_1", None, '{"location": "SF"}', index=0),
        ]

        result = await self._run_query(model, chunks)

        assert [(call.id, call.name, call.args) for call in result.tool_calls] == [
            ("call_1", "get_weather", '{"location": "SF"}')
        ]
        response = cast(RawResponse, result.history[-1]).response
        raw_tool_calls = response.tool_calls
        assert raw_tool_calls is not None
        assert [(call.id, call.function.name) for call in raw_tool_calls] == [
            ("call_1", "get_weather")
        ]

    async def test_google_pre_id_tool_call_extra_content_is_preserved(self):
        model = OpenAIModel(
            "gemini-2.5-flash-lite", provider="google", use_completions=True
        )
        extra_content = {"google": {"thought_signature": "signature"}}
        chunks = [
            self._make_chunk(None, None, None, index=0, extra_content=extra_content),
            self._make_chunk("call_1", "ping", "{}", index=0),
        ]

        result = await self._run_query(model, chunks)

        response = cast(RawResponse, result.history[-1]).response
        raw_tool_calls = response.tool_calls
        assert raw_tool_calls is not None
        assert raw_tool_calls[0].id == "call_1"
        assert raw_tool_calls[0].model_extra == {"extra_content": extra_content}

    async def test_google_id_bearing_tool_call_extra_content_is_preserved(self):
        model = OpenAIModel(
            "gemini-2.5-flash-lite", provider="google", use_completions=True
        )
        extra_content = {"google": {"thought_signature": "signature"}}
        chunks = [
            self._make_chunk(
                "call_1", "ping", "{}", index=0, extra_content=extra_content
            ),
        ]

        result = await self._run_query(model, chunks)

        response = cast(RawResponse, result.history[-1]).response
        raw_tool_calls = response.tool_calls
        assert raw_tool_calls is not None
        assert raw_tool_calls[0].model_extra == {"extra_content": extra_content}

    async def test_deepseek_same_id_new_name_starts_finished_tool_call_segment(self):
        model = OpenAIModel("deepseek-chat", provider="deepseek", use_completions=True)
        chunks = [
            self._make_chunk("call_1", "first", "{}", index=0),
            self._make_chunk("call_1", "second", "{}", index=0),
        ]

        result = await self._run_query(model, chunks)

        assert [(call.id, call.name, call.args) for call in result.tool_calls] == [
            ("call_1", "first", "{}"),
            ("call_1", "second", "{}"),
        ]
        assert [
            [event.type for event in entry.events]
            for entry in _require_performance(result.metadata).timeline
        ] == [
            [
                "tool_call_started",
                "tool_call_ready",
                "tool_call_delta",
                "tool_call_finished",
            ],
            [
                "tool_call_started",
                "tool_call_ready",
                "tool_call_delta",
                "tool_call_finished",
            ],
        ]

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

        assert result.output_text is None
        assert result.reasoning is None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_name"
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].args == '{"location": "SF"}'
        assert _require_performance(result.metadata).timeline[0].channel == "tool_call"

    async def test_openai_streaming_content_populates_performance_timeline(self):
        model = OpenAIModel("gpt-4o-mini")
        chunks = [self._make_chunk(None, None, None, content="hello")]

        result = await self._run_query(model, chunks)

        assert result.output_text == "hello"
        assert (
            _require_performance(result.metadata).time_to_first_token_ms.content
            is not None
        )
        assert _require_performance(result.metadata).timeline[0].channel == "content"

    async def test_openai_completion_builder_starts_before_stream_open(self):
        model = OpenAIModel("gpt-4o-mini")
        chunks = [self._make_chunk(None, None, None, content="hello")]
        order: list[str] = []

        async def mock_stream():
            for chunk in chunks:
                yield chunk

        async def create_stream(**_kwargs: object):
            order.append("stream-opened")
            return mock_stream()

        def make_builder() -> QueryResultBuilder:
            order.append("builder-created")
            return QueryResultBuilder()

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=create_stream)

        with patch.object(model, "get_client", return_value=mock_client):
            with patch.object(
                model, "build_body", new_callable=AsyncMock, return_value={}
            ):
                with patch(
                    "model_library.providers.openai.QueryResultBuilder",
                    side_effect=make_builder,
                ):
                    result = await model._query_completions(
                        [], tools=[], query_logger=logging.getLogger("test")
                    )

        assert result.output_text == "hello"
        assert order[:2] == ["builder-created", "stream-opened"]
