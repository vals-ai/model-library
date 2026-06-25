"""Tests for OpenAI built-in web search (web_search_call) handling."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai.types.responses.response_function_web_search import (
    ActionSearch,
    ActionSearchSource,
    ResponseFunctionWebSearch,
)

from model_library.agent.tool import ProviderTool
from model_library.base.input import TextInput
from model_library.base.output import FinishReason, QueryResult
from model_library.base.output.result import ProviderToolEvent
from model_library.providers.openai import OpenAIModel

from tests.unit.agent.helpers import DoneTool, make_agent, make_metadata, mock_llm


def _make_web_search_response(query: str = "test query", status: str = "completed"):
    """Minimal Responses API response with a web_search_call output item."""
    action = ActionSearch(
        type="search",
        query=query,
        queries=None,
        sources=[ActionSearchSource(type="url", url="https://example.com")],
    )
    web_search_item = ResponseFunctionWebSearch(
        id="ws_1",
        type="web_search_call",
        status=status,  # type: ignore[arg-type]
        action=action,
    )
    return SimpleNamespace(
        id="resp_1",
        status="completed",
        output_text="The answer is 42.",
        tools=[],
        reasoning=None,
        incomplete_details=None,
        output=[
            web_search_item,
            SimpleNamespace(type="message", content=[]),
        ],
        usage=None,
    )


async def _parse_response(response):
    model = OpenAIModel("gpt-5.5")
    mock_client = MagicMock()
    mock_client.responses.create = AsyncMock(return_value=response)
    with patch.object(model, "get_client", return_value=mock_client):
        return await model._query_impl(  # pyright: ignore[reportPrivateUsage]
            [TextInput(text="search for something")],
            tools=[],
            stream=False,
            query_logger=MagicMock(),
        )


# --- Parser tests ---


async def test_web_search_call_maps_to_provider_tool_events():
    """web_search_call output items produce ProviderToolEvents, not ToolCalls."""
    result = await _parse_response(_make_web_search_response())

    assert result.tool_calls == []
    assert len(result.provider_tool_events) == 1
    event = result.provider_tool_events[0]
    assert isinstance(event, ProviderToolEvent)
    assert event.name == "web_search"
    assert event.type == "web_search_call"
    assert event.provider == "openai"
    assert event.input == "test query"
    assert event.status == "completed"


async def test_web_search_call_finish_reason_is_stop():
    """When only web_search_call items are present, finish reason is STOP not TOOL_CALLS."""
    result = await _parse_response(_make_web_search_response())

    assert result.finish_reason.reason == FinishReason.STOP


# --- ProviderTool definition tests ---


async def test_provider_tool_definition_is_sent_to_model():
    """ProviderTool definitions appear in tool_definitions (sent to the model)."""
    agent = make_agent(mock_llm(), tools=[ProviderTool(name="web_search", body={"type": "web_search"}), DoneTool()])

    names = [d.name for d in agent.tool_definitions]
    assert "web_search" in names
    assert "submit" in names


async def test_provider_tool_excluded_from_local_execution():
    """ProviderTool is absent from the agent's local dispatch table."""
    agent = make_agent(mock_llm(), tools=[ProviderTool(name="web_search", body={"type": "web_search"}), DoneTool()])

    assert "web_search" not in agent._tools  # pyright: ignore[reportPrivateUsage]
    assert "submit" in agent._tools  # pyright: ignore[reportPrivateUsage]


# --- Agent loop tests ---


async def test_agent_turn_with_only_provider_events_does_not_execute_tools():
    """A turn with only provider_tool_events produces no tool call records."""
    provider_event_response = QueryResult(
        output_text=None,
        metadata=make_metadata(),
        tool_calls=[],
        provider_tool_events=[
            ProviderToolEvent(
                id="ws_1",
                provider="openai",
                type="web_search_call",
                name="web_search",
                status="completed",
                input="test query",
                output=["https://example.com"],
            )
        ],
        history=[TextInput(text="prompt")],
    )
    done_response = QueryResult(
        output_text="done",
        metadata=make_metadata(),
        tool_calls=[],
        history=[TextInput(text="prompt")],
    )

    llm = mock_llm(provider_event_response, done_response)
    agent = make_agent(llm, tools=[ProviderTool(name="web_search", body={"type": "web_search"}), DoneTool()])

    result = await agent.run([TextInput(text="search for something")], question_id="q1")

    assert result.turns[0].tool_calls == []
