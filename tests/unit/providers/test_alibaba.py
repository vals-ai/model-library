"""Unit tests for Alibaba (Qwen) provider."""

import pytest

from model_library.base import LLMConfig, RawResponse
from model_library.providers.delegates.alibaba import AlibabaConfig, AlibabaModel
from openai.types.chat import ChatCompletionMessage


class TestAlibabaBuildBodyContentFix:
    """Test that assistant messages with content=None are normalized to content=\"\" in the request body for Qwen API."""

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


class TestAlibabaPreserveThinking:
    """Test that configured thinking preservation reaches the request body."""

    def test_preserve_thinking_sent_for_codenamed_endpoint(self):
        """Endpoints whose name does not embed a version still send preserve_thinking."""
        model = AlibabaModel(
            "qwen-codename-endpoint",
            config=LLMConfig(
                reasoning=True,
                provider_config=AlibabaConfig(preserve_thinking=True),
            ),
        )
        assert model._get_extra_body() == {
            "enable_thinking": True,
            "preserve_thinking": True,
        }

    def test_preserve_thinking_omitted_when_not_configured(self):
        model = AlibabaModel("qwen-codename-endpoint", config=LLMConfig(reasoning=True))
        assert model._get_extra_body() == {"enable_thinking": True}
