from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai.types import (
    Candidate,
    Content,
    FinishReason,
    FunctionCall,
    GenerateContentResponse,
    GenerateContentResponseUsageMetadata,
    Part,
)

from model_library.exceptions import MaxOutputTokensExceededError, ModelNoOutputError
from model_library.providers.google.batch import parse_predictions_jsonl
from model_library.providers.google.google import GoogleModel
from model_library.providers.mistral import MistralModel
from tests.unit.provider_response_helpers import (
    _AMAZON_BLOCK_STOP,
    _AMAZON_TOOL_DELTA,
    _AMAZON_TOOL_START,
    _INPUT,
    _LOGGER,
    _amazon_reasoning,
    _amazon_response,
    _amazon_text,
    _query_amazon,
    _query_anthropic,
    _query_xai,
    _xai_response,
)


async def test_amazon_tool_only_query_keeps_unobserved_output_text_none():
    result = await _query_amazon(
        _amazon_response(
            _AMAZON_TOOL_START,
            _AMAZON_TOOL_DELTA,
            _AMAZON_BLOCK_STOP,
            stop_reason="tool_use",
            output_tokens=2,
        )
    )

    assert result.output_text is None
    assert result.reasoning is None
    assert result.tool_calls[0].name == "ping"


async def test_amazon_completed_empty_text_raises_no_output():
    with pytest.raises(ModelNoOutputError):
        await _query_amazon(_amazon_response(_amazon_text(""), _AMAZON_BLOCK_STOP))


async def test_amazon_completed_empty_stop_sequence_raises_no_output():
    with pytest.raises(ModelNoOutputError):
        await _query_amazon(
            _amazon_response(
                _amazon_text(""), _AMAZON_BLOCK_STOP, stop_reason="stop_sequence"
            )
        )


async def test_amazon_completed_absent_stop_sequence_raises_no_output():
    with pytest.raises(ModelNoOutputError):
        await _query_amazon(_amazon_response(stop_reason="stop_sequence"))


async def test_amazon_empty_text_with_reasoning_keeps_reasoning_only():
    result = await _query_amazon(
        _amazon_response(
            _amazon_text(""),
            _AMAZON_BLOCK_STOP,
            _amazon_reasoning("think"),
            _AMAZON_BLOCK_STOP,
            output_tokens=1,
        )
    )

    assert result.output_text is None
    assert result.reasoning == "think"
    assert not result.tool_calls


async def test_amazon_empty_max_tokens_raises_max_output_tokens():
    with pytest.raises(MaxOutputTokensExceededError):
        await _query_amazon(
            _amazon_response(
                _amazon_text(""), _AMAZON_BLOCK_STOP, stop_reason="max_tokens"
            )
        )


async def test_amazon_completed_absent_content_raises_no_output():
    with pytest.raises(ModelNoOutputError):
        await _query_amazon(_amazon_response())


async def test_amazon_empty_text_before_tool_keeps_tool_only_output_text_none():
    result = await _query_amazon(
        _amazon_response(
            _amazon_text(""),
            _AMAZON_BLOCK_STOP,
            _AMAZON_TOOL_START,
            _AMAZON_TOOL_DELTA,
            _AMAZON_BLOCK_STOP,
            stop_reason="tool_use",
        )
    )

    assert result.output_text is None
    assert result.tool_calls[0].name == "ping"


async def test_amazon_empty_text_before_real_text_does_not_prefix_space():
    result = await _query_amazon(
        _amazon_response(
            _amazon_text(""),
            _AMAZON_BLOCK_STOP,
            _amazon_text("hello"),
            _AMAZON_BLOCK_STOP,
            output_tokens=1,
        )
    )

    assert result.output_text == "hello"
    assert not result.tool_calls


async def test_amazon_multi_block_text_preserves_space_separator():
    result = await _query_amazon(
        _amazon_response(
            _amazon_text("hello"),
            _AMAZON_BLOCK_STOP,
            _AMAZON_TOOL_START,
            _AMAZON_TOOL_DELTA,
            _AMAZON_BLOCK_STOP,
            _amazon_text("world"),
            _AMAZON_BLOCK_STOP,
            stop_reason="tool_use",
            output_tokens=2,
        )
    )

    assert result.output_text == "hello world"
    assert result.tool_calls[0].name == "ping"


async def test_amazon_multi_block_reasoning_preserves_space_separator():
    result = await _query_amazon(
        _amazon_response(
            _amazon_reasoning("think"),
            _AMAZON_BLOCK_STOP,
            _amazon_reasoning("more"),
            _AMAZON_BLOCK_STOP,
            output_tokens=2,
        )
    )

    assert result.output_text is None
    assert result.reasoning == "think more"


async def test_anthropic_query_empty_output_text_raises_no_output():
    message = SimpleNamespace(
        id="anthropic-message-empty",
        model="claude-test",
        content=[SimpleNamespace(type="text", text="")],
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=1,
            output_tokens=0,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
            iterations=None,
        ),
    )

    with pytest.raises(ModelNoOutputError):
        await _query_anthropic(message)


async def test_anthropic_empty_output_text_with_reasoning_keeps_reasoning_only():
    message = SimpleNamespace(
        id="anthropic-message-empty-with-thinking",
        model="claude-test",
        content=[
            SimpleNamespace(type="thinking", thinking="considered"),
            SimpleNamespace(type="text", text=""),
        ],
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=1,
            output_tokens=1,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
            iterations=None,
        ),
    )

    result = await _query_anthropic(
        message,
        [
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="thinking_delta", thinking="considered"),
            )
        ],
    )

    assert result.output_text is None
    assert result.reasoning == "considered"
    assert result.tool_calls == []


async def test_anthropic_empty_text_with_reasoning_and_tool_keeps_useful_outputs():
    message = SimpleNamespace(
        id="anthropic-message-empty-with-thinking-tool",
        model="claude-test",
        content=[
            SimpleNamespace(type="thinking", thinking="considered"),
            SimpleNamespace(type="text", text=""),
            SimpleNamespace(type="tool_use", id="tool-1", name="ping", input={}),
        ],
        stop_reason="tool_use",
        usage=SimpleNamespace(
            input_tokens=1,
            output_tokens=1,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
            iterations=None,
        ),
    )

    result = await _query_anthropic(
        message,
        [
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="thinking_delta", thinking="considered"),
            ),
            SimpleNamespace(
                type="content_block_start",
                content_block=SimpleNamespace(type="tool_use"),
            ),
        ],
    )

    assert result.output_text is None
    assert result.reasoning == "considered"
    assert result.tool_calls[0].name == "ping"


@pytest.mark.parametrize("stop_reason", ["end_turn", "stop_sequence"])
async def test_anthropic_query_absent_content_raises_no_output(
    stop_reason: str,
):
    message = SimpleNamespace(
        id="anthropic-message-empty-content",
        model="claude-test",
        content=[],
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=1,
            output_tokens=0,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
            iterations=None,
        ),
    )

    with pytest.raises(ModelNoOutputError):
        await _query_anthropic(message)


@pytest.mark.parametrize(
    ("stop_reason", "expected_error"),
    [
        ("max_tokens", MaxOutputTokensExceededError),
        ("pause_turn", ModelNoOutputError),
    ],
)
async def test_anthropic_empty_non_success_raises_mapped_error(
    stop_reason: str,
    expected_error: type[Exception],
):
    message = SimpleNamespace(
        id="anthropic-message-empty-non-success",
        model="claude-test",
        content=[SimpleNamespace(type="text", text="")],
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=1,
            output_tokens=0,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
            iterations=None,
        ),
    )

    with pytest.raises(expected_error):
        await _query_anthropic(message)


async def test_anthropic_tool_only_query_keeps_unobserved_output_text_none():
    message = SimpleNamespace(
        id="anthropic-message-1",
        model="claude-test",
        content=[
            SimpleNamespace(
                type="tool_use",
                id="tool-1",
                name="ping",
                input={},
            )
        ],
        stop_reason="tool_use",
        usage=SimpleNamespace(
            input_tokens=1,
            output_tokens=2,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
            iterations=None,
        ),
    )

    result = await _query_anthropic(
        message,
        [
            SimpleNamespace(
                type="content_block_start",
                content_block=SimpleNamespace(type="tool_use"),
            )
        ],
    )

    assert result.output_text is None
    assert result.reasoning is None
    assert result.tool_calls[0].name == "ping"


async def test_google_tool_only_query_keeps_unobserved_output_text_none():
    async def stream() -> AsyncIterator[GenerateContentResponse]:
        yield GenerateContentResponse(
            response_id="google-response-1",
            candidates=[
                Candidate(
                    content=Content(
                        parts=[
                            Part(
                                function_call=FunctionCall(
                                    id="tool-1",
                                    name="ping",
                                    args={},
                                )
                            )
                        ]
                    ),
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

    assert result.output_text is None
    assert result.reasoning is None
    assert result.tool_calls[0].name == "ping"


def test_google_batch_empty_text_normalizes_to_none_with_reasoning():
    results = parse_predictions_jsonl(
        '{"key":"request-1","response":{"responseId":"google-batch-1",'
        '"candidates":[{"content":{"parts":['
        '{"thought":true,"text":"thinking"},{"text":""}]}}]}}'
    )

    assert results[0].output.output_text is None
    assert results[0].output.reasoning == "thinking"


async def test_mistral_tool_only_query_keeps_unobserved_output_text_none():
    from mistralai.client.models import FunctionCall as MistralFunctionCall
    from mistralai.client.models import ToolCall as MistralToolCall

    raw_tool_call = MistralToolCall(
        id="tool-1",
        function=MistralFunctionCall(name="ping", arguments="{}"),
    )

    async def stream() -> AsyncIterator[Any]:
        yield SimpleNamespace(
            data=SimpleNamespace(
                id="mistral-response-1",
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=None, tool_calls=[raw_tool_call]),
                        finish_reason="tool_calls",
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

    assert result.output_text is None
    assert result.reasoning is None
    assert result.tool_calls[0].name == "ping"


async def test_xai_query_empty_output_text_raises_no_output():
    with pytest.raises(ModelNoOutputError):
        await _query_xai(
            _xai_response(response_id="xai-response-empty", content=""),
            chunk_content="",
        )


async def test_xai_query_observed_empty_stream_chunk_with_absent_final_content_raises_no_output():
    with pytest.raises(ModelNoOutputError):
        await _query_xai(
            _xai_response(response_id="xai-response-empty-chunk", content=None),
            chunk_content="",
        )


async def test_xai_empty_text_with_reasoning_keeps_reasoning_only():
    result = await _query_xai(
        _xai_response(
            response_id="xai-response-empty-reasoning",
            content="",
            reasoning_content="considered",
            reasoning_tokens=2,
        ),
        chunk_content="",
    )

    assert result.output_text is None
    assert result.reasoning == "considered"
    assert result.tool_calls == []


async def test_xai_completed_absent_content_raises_no_output():
    with pytest.raises(ModelNoOutputError):
        await _query_xai(
            _xai_response(response_id="xai-response-absent-content", content=None),
            chunk_content=None,
        )


async def test_xai_empty_max_tokens_raises_max_output_tokens():
    with pytest.raises(MaxOutputTokensExceededError):
        await _query_xai(
            _xai_response(
                response_id="xai-response-empty-max-tokens",
                content="",
                finish_reason="REASON_MAX_LEN",
            ),
            chunk_content="",
        )


async def test_xai_mixed_content_and_tool_call_preserves_output_text():
    tool_calls = [
        SimpleNamespace(
            id="tool-1",
            type=1,
            function=SimpleNamespace(name="ping", arguments="{}"),
        )
    ]
    result = await _query_xai(
        _xai_response(
            response_id="xai-response-mixed-content-tool",
            content="hello",
            finish_reason="REASON_TOOL_CALLS",
            tool_calls=tool_calls,
            completion_tokens=2,
        ),
        chunk_content="",
    )

    assert result.output_text == "hello"
    assert result.tool_calls[0].name == "ping"


@pytest.mark.parametrize("content", [None, ""])
async def test_xai_tool_only_query_uses_none_for_absent_or_empty_output_text(
    content: str | None,
):
    tool_calls = [
        SimpleNamespace(
            id="tool-1",
            type=1,
            function=SimpleNamespace(name="ping", arguments="{}"),
        )
    ]
    result = await _query_xai(
        _xai_response(
            response_id="xai-response-1",
            content=content,
            finish_reason="REASON_TOOL_CALLS",
            tool_calls=tool_calls,
            completion_tokens=2,
        ),
        chunk_content=content,
    )

    assert result.output_text is None
    assert result.reasoning is None
    assert result.tool_calls[0].name == "ping"

