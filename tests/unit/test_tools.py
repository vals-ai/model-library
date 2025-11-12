"""
Provider-agnostic unit tests for tool-calling request shaping.

These assert that models which support tools include the tool specs in their
request bodies without making any network calls.
"""

import pytest

from model_library.base import TextInput, ToolBody, ToolDefinition
from model_library.providers.openai import OpenAIModel
from model_library.providers.amazon import AmazonModel
from model_library.providers.anthropic import AnthropicModel
from model_library.providers.google import GoogleModel


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
async def test_create_body_includes_tools(
    provider, Model, model_name, expects_body_keys, mock_model_settings
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

    if provider == "openai":
        body = await model.build_body([TextInput(text="hi")], tools=tools)
    elif provider == "amazon":
        body = await model.build_body([TextInput(text="hi")], tools=tools)
    else:  # google
        body = await model.create_body([TextInput(text="hi")], tools=tools)
    if provider == "google":
        # For Google, tools live inside config
        assert set(["model", "config", "contents"]).issubset(set(body.keys()))
        cfg = body["config"]
        assert getattr(cfg, "tools", None)
    else:
        assert expects_body_keys.issubset(set(body.keys()))


async def test_parse_tools_shapes_for_all(mock_model_settings):
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
async def test_google_tool_result_roundtrip_no_storage_import(
    model_class, model_name, mock_model_settings
):
    # Ensure that creating a GoogleModel and building a body with a ToolResult
    # does not require the storage client (batch-only) and shapes content as expected.
    from model_library.base import ToolCall, ToolResult

    m = model_class(model_name)
    tr = ToolResult(
        tool_call=ToolCall(id="abc123", name="get_weather", args={"location": "SF"}),
        result={"temperature": "21C"},
    )
    body = await m.create_body([tr, TextInput(text="Thanks!")], tools=[])
    contents = body["contents"]
    roles = [getattr(c, "role", None) for c in contents]
    assert "function" in roles and "user" in roles


async def test_anthropic_rejects_invalid_tool_result(mock_model_settings):
    """Verify that providing a ToolResult without a matching tool call raises an exception."""
    from model_library.base import ToolCall, ToolResult

    model = AnthropicModel("claude-3-7-sonnet-latest")
    orphaned_result = ToolResult(
        tool_call=ToolCall(id="nonexistent_id", name="get_weather", args={}),
        result="Sunny",
    )

    with pytest.raises(
        Exception, match="Tool call result provided with no matching tool call"
    ):
        await model.parse_input([TextInput(text="Hello"), orphaned_result])
