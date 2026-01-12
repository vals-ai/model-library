"""
Unit tests for count_tokens functionality.
"""

import pytest

from model_library.base import TextInput, ToolBody, ToolDefinition
from model_library.registry_utils import get_registry_model


@pytest.mark.asyncio
async def test_empty_input_returns_zero():
    """Test that count_tokens returns 0 for empty input."""
    model = get_registry_model("anthropic/claude-opus-4-5-20251101")

    token_count = await model.count_tokens([])

    assert token_count == 0


@pytest.mark.asyncio
async def test_count_tokens_with_text_and_tools():
    """Test that count_tokens returns positive counts for non-empty input."""
    model = get_registry_model("openai/gpt-4o")

    tools = [
        ToolDefinition(
            name="get_weather",
            body=ToolBody(
                name="get_weather",
                description="Get current temperature in a given location",
                properties={
                    "location": {
                        "type": "string",
                        "description": "City and country e.g. Bogotá, Colombia",
                    },
                },
                required=["location"],
            ),
        ),
        ToolDefinition(
            name="get_danger",
            body=ToolBody(
                name="get_danger",
                description="Get current danger in a given location",
                properties={
                    "location": {
                        "type": "string",
                        "description": "City and country e.g. Bogotá, Colombia",
                    },
                },
                required=["location"],
            ),
        ),
    ]

    system_prompt = "You must make exactly 0 or 1 tool calls per answer."
    user_prompt = "What is the weather in San Francisco right now?"

    token_count = await model.count_tokens(
        [TextInput(text=user_prompt)],
        tools=tools,
        system_prompt=system_prompt,
    )

    assert isinstance(token_count, int)
    assert token_count > 0
