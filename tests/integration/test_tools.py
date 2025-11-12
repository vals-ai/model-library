"""
Integration tests for provider tool-calling.
"""

import os

import pytest

from model_library.base import (
    TextInput,
    ToolBody,
    ToolDefinition,
    ToolResult,
)
from model_library.registry_utils import get_registry_model


def has(var: str) -> bool:
    return bool(os.getenv(var))


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(not has("GOOGLE_API_KEY"), reason="No Google key")
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


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(not has("OPENAI_API_KEY"), reason="No OpenAI key")
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


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(not has("FIREWORKS_API_KEY"), reason="No Fireworks key")
async def test_fireworks_streaming_tools_roundtrip():
    """Test that streaming completions correctly handle tool calls."""
    model = get_registry_model("fireworks/glm-4p6")

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

    # Verify history contains the full conversation (input + assistant message)
    assert len(result1.history) == 2, (
        "Expected history to contain input and assistant message"
    )
    assert result1.history[-1].role == "assistant", (
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
    assert result1.metadata.in_tokens > 0
    assert result1.metadata.out_tokens > 0
    assert result2.metadata.in_tokens > 0
    assert result2.metadata.out_tokens > 0
