"""Unit tests for Alibaba (Qwen) provider."""

import pytest

from model_library.base import LLMConfig, RawResponse, TextInput
from model_library.providers.delegates.alibaba import AlibabaConfig, AlibabaModel
from model_library.registry_utils import get_registry_model
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


class TestAlibabaReasoningEffort:
    """Qwen 3.8 Max effort keys send reasoning_effort beside enable_thinking.

    https://docs.qwencloud.com/developer-guides/text-generation/thinking lists
    low, medium and xhigh for qwen3.8-max, with xhigh as the default.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("model_key", "expected_effort"),
        [
            ("alibaba/qwen3.8-max", None),
            ("alibaba/qwen3.8-max-low", "low"),
            ("alibaba/qwen3.8-max-medium", "medium"),
        ],
    )
    async def test_registry_key_request_body(
        self, model_key: str, expected_effort: str | None
    ):
        model = get_registry_model(model_key)
        assert isinstance(model, AlibabaModel)
        body = await model.build_body([TextInput(text="hi")], tools=[])

        assert model._get_extra_body() == {
            "enable_thinking": True,
            "preserve_thinking": True,
        }
        assert body.get("reasoning_effort") == expected_effort
