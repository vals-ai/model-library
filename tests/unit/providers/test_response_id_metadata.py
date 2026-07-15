from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from anthropic.types.beta import BetaTextBlock
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

from model_library.providers.ai21labs import AI21LabsModel
from model_library.providers.anthropic import AnthropicBatchMixin, AnthropicModel
from model_library.providers.google.batch import parse_predictions_jsonl
from model_library.providers.google.google import GoogleModel
from model_library.providers.mistral import MistralModel
from model_library.providers.vals import DummyAIModel
from tests.unit.provider_response_helpers import (
    _AMAZON_BLOCK_STOP,
    _INPUT,
    _LOGGER,
    _amazon_response,
    _amazon_text,
    _query_amazon,
    _query_anthropic,
    _query_xai,
    _xai_response,
)


class _AsyncItems:
    def __init__(self, *items: object):
        self._items = items

    async def __aiter__(self):
        for item in self._items:
            yield item


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
    assert result.extras.provider_response_id == "ai21-response-1"


async def test_amazon_query_captures_request_id_metadata():
    result = await _query_amazon(
        _amazon_response(_amazon_text("hello"), _AMAZON_BLOCK_STOP, output_tokens=2)
    )

    assert result.extras.response_id == "amazon-request-1"
    assert result.extras.provider_response_id == "amazon-request-1"
    assert result.extras.provider_request_id == "amazon-request-1"


async def test_anthropic_query_captures_message_id_metadata():
    message = SimpleNamespace(
        id="anthropic-message-1",
        model="claude-test",
        content=[BetaTextBlock(type="text", text="hello")],
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=1,
            output_tokens=2,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
            iterations=None,
        ),
        _request_id="anthropic-request-1",
    )

    result = await _query_anthropic(
        message,
        [
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="text_delta", text="hello"),
            )
        ],
    )

    assert result.extras.response_id == "anthropic-message-1"
    assert result.extras.provider_response_id == "anthropic-message-1"
    assert result.extras.provider_request_id == "anthropic-request-1"


async def test_anthropic_batch_captures_message_id_metadata():
    model = AnthropicModel("claude-test")
    client = MagicMock()
    client.messages.batches.retrieve = AsyncMock(
        return_value=SimpleNamespace(processing_status="ended")
    )
    client.messages.batches.results = AsyncMock(
        return_value=_AsyncItems(
            SimpleNamespace(
                model_dump=lambda: {
                    "custom_id": "request-1",
                    "result": {
                        "type": "succeeded",
                        "message": {
                            "id": "anthropic-batch-message-1",
                            "content": [{"type": "text", "text": "hello"}],
                            "usage": {"input_tokens": 1, "output_tokens": 2},
                        },
                    },
                }
            )
        )
    )

    with patch.object(model, "get_client", return_value=client):
        results = await AnthropicBatchMixin(model).get_batch_results("batch-1")

    assert results[0].custom_id == "request-1"
    assert results[0].output.extras.response_id == "anthropic-batch-message-1"
    assert results[0].output.extras.provider_response_id == "anthropic-batch-message-1"


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
    assert result.extras.provider_response_id == "google-response-1"


async def test_google_query_allows_missing_response_id_metadata():
    async def stream() -> AsyncIterator[GenerateContentResponse]:
        yield GenerateContentResponse(
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

    assert result.output_text == "hello"
    assert result.extras.response_id is None


def test_google_batch_parse_captures_response_id_metadata():
    results = parse_predictions_jsonl(
        '{"key":"request-1","response":{"responseId":"google-batch-1",'
        '"candidates":[{"content":{"parts":[{"text":"hello"}]}}],'
        '"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":2}}}'
    )

    assert results[0].output.extras.response_id == "google-batch-1"
    assert results[0].output.extras.provider_response_id == "google-batch-1"


async def test_mistral_query_captures_response_id_metadata():
    async def stream() -> AsyncIterator[Any]:
        yield SimpleNamespace(
            data=SimpleNamespace(
                id="mistral-response-1",
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="hello", tool_calls=None),
                        finish_reason=None,
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2),
            )
        )
        yield SimpleNamespace(
            data=SimpleNamespace(
                id=None,
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=" world", tool_calls=None),
                        finish_reason="stop",
                    )
                ],
                usage=None,
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
    assert result.extras.provider_response_id == "mistral-response-1"


async def test_vals_query_captures_response_id_metadata():
    model = DummyAIModel("response-id-evaluator")

    with patch("model_library.providers.vals.random.random", return_value=1.0):
        result = await model._query_impl(_INPUT, tools=[], query_logger=_LOGGER)

    assert result.extras.response_id == "mock-id"
    assert result.extras.provider_response_id == "mock-id"


async def test_xai_query_captures_response_id_metadata():
    result = await _query_xai(
        _xai_response(
            response_id="xai-response-1",
            content="hello",
            completion_tokens=2,
        ),
        chunk_content="hello",
    )

    assert result.extras.response_id == "xai-response-1"
    assert result.extras.provider_response_id == "xai-response-1"


async def test_xai_query_allows_missing_response_id_metadata():
    result = await _query_xai(
        _xai_response(content="hello", completion_tokens=2),
        chunk_content="hello",
    )

    assert result.output_text == "hello"
    assert result.extras.response_id is None
