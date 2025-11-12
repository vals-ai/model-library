from unittest.mock import AsyncMock, Mock

import pytest

from model_library.base import LLMConfig
from model_library.providers.openai import OpenAIConfig, OpenAIModel


async def test_check_deep_research_args_raises_without_web_search():
    """
    Test that check for web search tool raises an exception if not present
    """

    model = OpenAIModel(
        "gpt-4o-deep-research",
        config=LLMConfig(provider_config=OpenAIConfig(deep_research=True)),
    )
    tools = [Mock(body={"type": "non_search_tool"})]

    with pytest.raises(Exception):
        await model._check_deep_research_args(tools)  # pyright: ignore[reportPrivateUsage]


async def test_check_deep_research_args_warns_low_tokens_and_missing_background():
    """
    Test that checks for low tokens and background flag are logged as warnings
    """

    model = OpenAIModel(
        "gpt-4o-deep-research",
        config=LLMConfig(provider_config=OpenAIConfig(deep_research=True)),
    )
    model.max_tokens = 10000
    model.logger = Mock()

    tools = [Mock(body={"type": "web_search"})]

    await model._check_deep_research_args(tools)  # pyright: ignore[reportPrivateUsage]

    logged_messages = [call.args[0] for call in model.logger.warning.call_args_list]
    assert any("max_tokens >=" in msg for msg in logged_messages)
    assert any("background=True" in msg for msg in logged_messages)


async def test_query_impl_parses_deep_research_response():
    """
    Test parsing of annotations from deep research response
    """

    model = OpenAIModel(
        "o3-deep-research",
        config=LLMConfig(provider_config=OpenAIConfig(deep_research=True)),
    )
    model.get_client = Mock()

    from openai.types.responses import Response, ResponseCompletedEvent

    response = Response.model_validate(
        {
            "id": "resp_123",
            "created_at": 1_726_000_000,
            "model": "o3-deep-research",
            "object": "response",
            "status": "completed",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "output": [
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Report [1]",
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "title": "Example",
                                    "url": "https://example.com",
                                    "start_index": 7,
                                    "end_index": 10,
                                }
                            ],
                        }
                    ],
                },
                {
                    "id": "reason_1",
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "Reasoning"}],
                    "status": "completed",
                },
            ],
        }
    )

    stream_event = ResponseCompletedEvent(
        type="response.completed",
        response=response,
        sequence_number=1,
    )

    async def fake_stream():
        yield stream_event

    model.get_client().responses.create = AsyncMock(return_value=fake_stream())

    web_search_tool = Mock()
    web_search_tool.body = {"type": "web_search"}
    result = await model._query_impl([], tools=[web_search_tool], stream=False)  # pyright: ignore[reportPrivateUsage]

    assert result.output_text == "Report [1]"
    assert result.reasoning == "Reasoning"
    assert len(result.extras.citations) == 1
    citation = result.extras.citations[0]
    assert citation.url == "https://example.com"
    assert citation.title == "Example"
