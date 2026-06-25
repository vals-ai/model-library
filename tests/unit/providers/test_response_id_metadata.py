import logging
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from ai21.models.chat import AssistantMessage
from ai21.models.chat.chat_completion_response import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
)
from ai21.models.usage_info import UsageInfo
from google.genai.types import (
    Candidate,
    Content,
    FinishReason,
    GenerateContentResponse,
    GenerateContentResponseUsageMetadata,
    Part,
)

from model_library.base import TextInput
from model_library.providers.ai21labs import AI21LabsModel
from model_library.providers.amazon import AmazonModel
from model_library.providers.anthropic import AnthropicModel
from model_library.providers.google.batch import parse_predictions_jsonl
from model_library.providers.google.google import GoogleModel
from model_library.providers.mistral import MistralModel
from model_library.providers.vals import DummyAIModel
from model_library.providers.xai import XAIModel

_INPUT = [TextInput(text="hello")]
_LOGGER = logging.getLogger("test")


async def test_ai21_query_captures_response_id_metadata():
    response = ChatCompletionResponse(
        id="ai21-response-1",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=AssistantMessage(content="hello"),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )
    model = AI21LabsModel("jamba-mini")
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)

    with (
        patch.object(model, "get_client", return_value=client),
        patch.object(model, "build_body", new_callable=AsyncMock, return_value={}),
    ):
        result = await model._query_impl(_INPUT, tools=[], query_logger=_LOGGER)

    assert result.extras.response_id == "ai21-response-1"


async def test_amazon_query_captures_request_id_metadata():
    response = {
        "ResponseMetadata": {"RequestId": "amazon-request-1"},
        "stream": [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"delta": {"text": "hello"}}},
            {"contentBlockStop": {}},
            {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 2}}},
            {"messageStop": {"stopReason": "end_turn"}},
        ],
    }
    model = AmazonModel("anthropic.claude")
    client = MagicMock()
    client.converse_stream.return_value = response

    with (
        patch.object(model, "get_client", return_value=client),
        patch.object(model, "build_body", new_callable=AsyncMock, return_value={}),
    ):
        result = await model._query_impl(_INPUT, tools=[], query_logger=_LOGGER)

    assert result.extras.response_id == "amazon-request-1"


async def test_anthropic_query_captures_message_id_metadata():
    message = SimpleNamespace(
        id="anthropic-message-1",
        model="claude-test",
        content=[SimpleNamespace(type="text", text="hello")],
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=1,
            output_tokens=2,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
            iterations=None,
        ),
    )

    class MockStream:
        async def __aenter__(self) -> "MockStream":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def get_final_message(self) -> SimpleNamespace:
            return message

    client = SimpleNamespace(
        beta=SimpleNamespace(messages=SimpleNamespace(stream=lambda **_: MockStream()))
    )
    model = AnthropicModel("claude-test")

    with (
        patch.object(model, "get_client", return_value=client),
        patch.object(model, "build_body", new_callable=AsyncMock, return_value={}),
    ):
        result = await model._query_impl(_INPUT, tools=[], query_logger=_LOGGER)

    assert result.extras.response_id == "anthropic-message-1"


async def test_google_query_captures_response_id_metadata():
    async def stream() -> AsyncIterator[GenerateContentResponse]:
        yield GenerateContentResponse(
            response_id="google-response-1",
            candidates=[
                Candidate(
                    content=Content(parts=[Part(text="hello")]),
                    finish_reason=FinishReason.STOP,
                )
            ],
            usage_metadata=GenerateContentResponseUsageMetadata(
                prompt_token_count=1,
                candidates_token_count=2,
            ),
        )

    client = MagicMock()
    client.aio.models.generate_content_stream = AsyncMock(return_value=stream())
    model = GoogleModel("gemini-test")

    with (
        patch.object(model, "get_client", return_value=client),
        patch.object(model, "build_body", new_callable=AsyncMock, return_value={}),
    ):
        result = await model._query_impl(_INPUT, tools=[], query_logger=_LOGGER)

    assert result.extras.response_id == "google-response-1"


def test_google_batch_parse_captures_response_id_metadata():
    results = parse_predictions_jsonl(
        '{"key":"request-1","response":{"responseId":"google-batch-1",'
        '"candidates":[{"content":{"parts":[{"text":"hello"}]}}],'
        '"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":2}}}'
    )

    assert results[0].output.extras.response_id == "google-batch-1"


async def test_mistral_query_captures_response_id_metadata():
    async def stream() -> AsyncIterator[Any]:
        yield SimpleNamespace(
            data=SimpleNamespace(
                id="mistral-response-1",
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="hello", tool_calls=None),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2),
            )
        )

    client = MagicMock()
    client.chat.stream_async = AsyncMock(return_value=stream())
    model = MistralModel("mistral-test")

    with (
        patch.object(model, "get_client", return_value=client),
        patch.object(model, "build_body", new_callable=AsyncMock, return_value={}),
    ):
        result = await model._query_impl(_INPUT, tools=[], query_logger=_LOGGER)

    assert result.extras.response_id == "mistral-response-1"


async def test_vals_query_captures_response_id_metadata():
    model = DummyAIModel("response-id-evaluator")

    result = await model._query_impl(_INPUT, tools=[], query_logger=_LOGGER)

    assert result.extras.response_id == "mock-id"


async def test_xai_query_captures_response_id_metadata():
    latest_response = SimpleNamespace(
        id="xai-response-1",
        tool_calls=[],
        content="hello",
        reasoning_content="",
        finish_reason="stop",
        usage=SimpleNamespace(
            prompt_tokens=3,
            cached_prompt_text_tokens=1,
            completion_tokens=2,
            reasoning_tokens=0,
        ),
    )

    async def stream() -> AsyncIterator[tuple[SimpleNamespace, None]]:
        yield latest_response, None

    chat = SimpleNamespace(stream=stream)
    client = SimpleNamespace(chat=SimpleNamespace(create=lambda **_: chat))
    model = XAIModel("grok-test")

    with (
        patch.object(model, "get_client", return_value=client),
        patch.object(model, "build_body", new_callable=AsyncMock, return_value={}),
    ):
        result = await model._query_impl(_INPUT, tools=[], query_logger=_LOGGER)

    assert result.extras.response_id == "xai-response-1"
