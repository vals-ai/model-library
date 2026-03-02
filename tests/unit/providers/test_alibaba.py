"""Unit tests for Alibaba (Qwen) provider."""

import pytest

from model_library.base import RawResponse
from model_library.providers.alibaba import AlibabaModel
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function


class TestAlibabaBuildBodyContentFix:
    """Test that assistant messages with content=None are normalized to content=\"\" in the request body for Qwen API."""

    @pytest.mark.asyncio
    async def test_raw_response_with_content_none_normalized_to_empty_string_in_body(self):
        """Tool-call-only assistant messages (content=None) are sent as content=\"\" to satisfy Qwen API."""
        model = AlibabaModel("qwen3.5-flash")
        msg = ChatCompletionMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id="call_abc",
                    type="function",
                    function=Function(name="get_weather", arguments='{"location": "SF"}'),
                )
            ],
        )
        input_items = [RawResponse(response=msg)]
        body = await model.build_body(input_items, tools=[])
        messages = body["messages"]
        assert len(messages) == 1
        m = messages[0]
        content = m.content if hasattr(m, "content") else m.get("content")
        assert content == ""

    @pytest.mark.asyncio
    async def test_raw_response_with_content_not_none_unchanged_in_body(self):
        """Assistant messages that already have content are left unchanged."""
        model = AlibabaModel("qwen3.5-flash")
        msg = ChatCompletionMessage(
            role="assistant",
            content="Here is the weather.",
            tool_calls=None,
        )
        input_items = [RawResponse(response=msg)]
        body = await model.build_body(input_items, tools=[])
        messages = body["messages"]
        assert len(messages) == 1
        m = messages[0]
        content = m.content if hasattr(m, "content") else m.get("content")
        assert content == "Here is the weather."
