"""
Integration tests for provider tool-calling.
"""

from model_library.base import (
    RawResponse,
    TextInput,
    ToolBody,
    ToolDefinition,
    ToolResult,
)
from model_library.registry_utils import get_registry_model


async def test_google_tools_roundtrip_optional():
    model = get_registry_model("google/gemini-2.5-flash")
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
    out1 = await model.query(
        [TextInput(text="If helpful, call get_echo with value 'pong'.")], tools=tools
    )
    if out1.tool_calls:
        tr = ToolResult(tool_call=out1.tool_calls[0], result={"value": "pong"})
        _ = await model.query(
            [tr, TextInput(text="Thanks!")], history=out1.history, tools=tools
        )
    else:
        assert len(out1.output_text_str) > 0


async def test_openai_tools_optional():
    model = get_registry_model("openai/gpt-4o-mini")
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
    out = await model.query(
        [TextInput(text="Optionally call get_echo with value 'pong'.")], tools=tools
    )
    # Either tool call or direct answer is acceptable
    assert out is not None


async def test_fireworks_streaming_tools_roundtrip():
    """Test that streaming completions correctly handle tool calls."""
    model = get_registry_model("fireworks/gpt-oss-20b")

    tools = [
        ToolDefinition(
            name="get_weather",
            body=ToolBody(
                name="get_weather",
                description="Get current temperature in a given location",
                properties={
                    "location": {
                        "type": "string",
                        "description": "City and country e.g. San Francisco, USA",
                    },
                },
                required=["location"],
            ),
        ),
    ]

    # Test 1: Model should make a tool call
    result1 = await model.query(
        [TextInput(text="What is the weather in Tokyo right now?")],
        tools=tools,
        system_prompt="You must make exactly 0 or 1 tool calls per answer.",
    )

    # Verify tool call was made
    assert len(result1.tool_calls) == 1, "Expected exactly one tool call"
    assert result1.tool_calls[0].name == "get_weather"
    assert "Tokyo" in result1.tool_calls[0].args

    # Verify history contains the full conversation and assistant message.
    # The system prompt is normalized into a SystemInput and included in history.
    assert len(result1.history) >= 2, (
        "Expected history to contain input and assistant message"
    )
    last_history_item = result1.history[-1]
    assert isinstance(last_history_item, RawResponse)
    assert last_history_item.response.role == "assistant", (
        "Last message should be from assistant"
    )

    # Test 2: Provide tool result and get final answer
    result2 = await model.query(
        [
            ToolResult(tool_call=result1.tool_calls[0], result="18°C, cloudy"),
        ],
        history=result1.history,
        tools=tools,
    )

    # Verify final response
    assert result2.output_text is not None
    assert len(result2.output_text) > 0
    assert len(result2.tool_calls) == 0, "Should not make additional tool calls"

    # Verify token counts
    assert result1.metadata.total_input_tokens > 0
    assert result1.metadata.total_output_tokens > 0
    assert result2.metadata.total_input_tokens > 0
    assert result2.metadata.total_output_tokens > 0
