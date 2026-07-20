import logging
from collections.abc import AsyncIterator, Sequence
from types import SimpleNamespace
from typing import Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, SecretStr
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import Function
from openai.types.completion_usage import (
    CompletionTokensDetails,
    CompletionUsage,
    PromptTokensDetails,
)
from openai.types.responses import Response, ResponseOutputMessage, ResponseOutputText
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall

from model_library.base import FinishReason, LLMConfig
from model_library.base.input import (
    FileWithBase64,
    RawResponse,
    SystemInput,
    TextInput,
    ToolBody,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from model_library.exceptions import (
    ContentFilterError,
    MaxOutputTokensExceededError,
    ModelNoOutputError,
)
from model_library.providers.delegates.kimi import KimiModel
from model_library.providers.openai import (
    OpenAIConfig,
    OpenAIModel,
    _safe_search_results,
)

_INPUT = [TextInput(text="")]


@pytest.mark.parametrize("mime", ["png", "image/png"])
@pytest.mark.parametrize("use_completions", [False, True])
async def test_base64_image_mime_is_normalized(mime: str, use_completions: bool):
    model = OpenAIModel(
        "test-model",
        config=LLMConfig(custom_api_key=SecretStr("test-key")),
        use_completions=use_completions,
    )

    parsed = await model.parse_image(
        FileWithBase64(
            type="image",
            name="test.png",
            mime=mime,
            base64="dGVzdA==",
        )
    )

    image_url = parsed["image_url"]["url"] if use_completions else parsed["image_url"]
    assert image_url == "data:image/png;base64,dGVzdA=="


def _response_output_text_message(text: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id="msg_1",
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(annotations=[], text=text, type="output_text")],
    )


def _responses_response(
    *,
    response_id: str,
    status: Literal[
        "completed", "failed", "in_progress", "cancelled", "queued", "incomplete"
    ] = "completed",
    text_block_text: str | None = None,
    output: Sequence[object] = (),
    incomplete_details: object | None = None,
    usage: object | None = None,
    request_id: str | None = None,
) -> Response:
    output_items: list[object] = []
    if text_block_text is not None:
        output_items.append(_response_output_text_message(text_block_text))
    output_items.extend(output)
    return Response.model_construct(
        id=response_id,
        created_at=0.0,
        model="gpt-4o-mini",
        object="response",
        output=output_items,
        parallel_tool_calls=True,
        status=status,
        tool_choice="auto",
        tools=[],
        incomplete_details=incomplete_details,
        usage=usage,
        _request_id=request_id,
    )


async def test_non_streaming_responses_uses_sdk_output_text_aggregation():
    response = _responses_response(
        response_id="resp_multi_text",
        output=[
            _response_output_text_message("A"),
            _response_output_text_message("C"),
        ],
    )
    model = OpenAIModel("gpt-4o-mini")
    mock_client = MagicMock()
    mock_client.responses.create = AsyncMock(return_value=response)

    with patch.object(model, "get_client", return_value=mock_client):
        result = await model._query_impl(  # pyright: ignore[reportPrivateUsage]
            [TextInput(text="say letters")],
            tools=[],
            stream=False,
            query_logger=MagicMock(),
        )

    assert response.output_text == "AC"
    assert result.output_text == "AC"


@pytest.mark.parametrize("verbosity", ["low", "medium", "high"])
async def test_verbosity_added_to_body(verbosity: Literal["low", "medium", "high"]):
    """Test that verbosity is correctly added to request body."""
    model = OpenAIModel(
        "gpt-4o",
        config=LLMConfig(provider_config=OpenAIConfig(verbosity=verbosity)),
    )

    body = await model.build_body(_INPUT, tools=[])

    assert "text" in body
    assert body["text"]["verbosity"] == verbosity


async def test_verbosity_not_in_body_when_none():
    """Test that text field is not added when verbosity is None."""
    model = OpenAIModel(
        "gpt-4o",
        config=LLMConfig(provider_config=OpenAIConfig(verbosity=None)),
    )

    body = await model.build_body(_INPUT, tools=[])

    assert "text" not in body


async def test_reasoning_context_added_to_body():
    """reasoning.context is set on the request body when configured (persisted reasoning)."""
    model = OpenAIModel(
        "public-test-responses-model",
        config=LLMConfig(
            reasoning=True,
            provider_config=OpenAIConfig(reasoning_context="all_turns"),
        ),
    )

    body = await model.build_body(_INPUT, tools=[])

    assert body["reasoning"]["context"] == "all_turns"
    assert body["include"] == ["reasoning.encrypted_content"]


async def test_reasoning_context_omitted_when_none():
    """reasoning.context is not added when unconfigured, leaving default behavior."""
    model = OpenAIModel(
        "gpt-4o",
        config=LLMConfig(
            reasoning=True,
            provider_config=OpenAIConfig(reasoning_context=None),
        ),
    )

    body = await model.build_body(_INPUT, tools=[])

    assert "context" not in body["reasoning"]


async def test_responses_parallel_tool_calls_added_to_body_when_configured():
    model = OpenAIModel(
        "gpt-4o",
        config=LLMConfig(
            custom_api_key=SecretStr("sk-test"),
            provider_config=OpenAIConfig(parallel_tool_calls=False),
        ),
    )

    body = await model.build_body(_INPUT, tools=[])

    assert body["parallel_tool_calls"] is False


async def test_completions_parallel_tool_calls_added_to_body_when_configured():
    model = OpenAIModel(
        "gpt-4o-mini",
        config=LLMConfig(
            custom_api_key=SecretStr("sk-test"),
            provider_config=OpenAIConfig(parallel_tool_calls=False),
        ),
        use_completions=True,
    )

    body = await model.build_body(_INPUT, tools=[])

    assert body["parallel_tool_calls"] is False


async def test_parallel_tool_calls_omitted_when_unconfigured():
    model = OpenAIModel(
        "gpt-4o",
        config=LLMConfig(custom_api_key=SecretStr("sk-test")),
    )

    body = await model.build_body(_INPUT, tools=[])

    assert "parallel_tool_calls" not in body


async def test_responses_prompt_cache_retention_added_to_body():
    model = OpenAIModel(
        "gpt-4o",
        config=LLMConfig(provider_config=OpenAIConfig(prompt_cache_retention="24h")),
    )

    body = await model.build_body(_INPUT, tools=[])

    assert body["prompt_cache_retention"] == "24h"


async def test_responses_prompt_cache_key_from_query_ids_added_to_body():
    model = OpenAIModel(
        "gpt-4o",
        config=LLMConfig(provider_config=OpenAIConfig(prompt_cache_key="id")),
    )

    body = await model.build_body(_INPUT, tools=[], run_id="run-a", question_id="q-a")

    assert isinstance(body["prompt_cache_key"], str)
    assert len(body["prompt_cache_key"]) == 32


async def test_prompt_cache_key_hash_is_stable_across_turns():
    model = OpenAIModel(
        "gpt-4o",
        config=LLMConfig(provider_config=OpenAIConfig(prompt_cache_key="hash")),
    )

    turn_1 = [SystemInput(text="sys"), TextInput(text="first user msg")]
    turn_2 = [
        SystemInput(text="sys"),
        TextInput(text="first user msg"),
        RawResponse(response=[]),
        TextInput(text="later user msg"),
    ]

    body_1 = await model.build_body(turn_1, tools=[])
    body_2 = await model.build_body(turn_2, tools=[])

    assert isinstance(body_1["prompt_cache_key"], str)
    assert body_1["prompt_cache_key"]
    assert body_1["prompt_cache_key"] == body_2["prompt_cache_key"]


async def test_prompt_cache_key_hash_differs_for_different_initial_inputs():
    model = OpenAIModel(
        "gpt-4o",
        config=LLMConfig(provider_config=OpenAIConfig(prompt_cache_key="hash")),
    )

    body_1 = await model.build_body(
        [SystemInput(text="sys"), TextInput(text="first user msg")],
        tools=[],
    )
    body_2 = await model.build_body(
        [SystemInput(text="sys"), TextInput(text="different first user msg")],
        tools=[],
    )

    assert body_1["prompt_cache_key"] != body_2["prompt_cache_key"]


async def test_prompt_cache_key_id_uses_query_ids():
    model = OpenAIModel(
        "gpt-4o",
        config=LLMConfig(provider_config=OpenAIConfig(prompt_cache_key="id")),
    )

    body_1 = await model.build_body(
        [SystemInput(text="sys"), TextInput(text="first user msg")],
        tools=[],
        run_id="run-a",
        question_id="q-a",
    )
    body_2 = await model.build_body(
        [SystemInput(text="sys"), TextInput(text="different first user msg")],
        tools=[],
        run_id="run-a",
        question_id="q-a",
    )
    body_3 = await model.build_body(
        [SystemInput(text="sys"), TextInput(text="first user msg")],
        tools=[],
        run_id="run-a",
        question_id="q-b",
    )

    assert body_1["prompt_cache_key"] == body_2["prompt_cache_key"]
    assert body_1["prompt_cache_key"] != body_3["prompt_cache_key"]


async def test_completions_prompt_cache_retention_added_to_body():
    model = OpenAIModel(
        "gpt-4o",
        config=LLMConfig(
            provider_config=OpenAIConfig(prompt_cache_retention="in_memory")
        ),
        use_completions=True,
    )

    body = await model.build_body(_INPUT, tools=[])

    assert body["prompt_cache_retention"] == "in_memory"


async def test_prompt_cache_key_hash_is_stable_across_turns_completions():
    model = OpenAIModel(
        "gpt-4o",
        config=LLMConfig(provider_config=OpenAIConfig(prompt_cache_key="hash")),
        use_completions=True,
    )

    turn_1 = [SystemInput(text="sys"), TextInput(text="first user msg")]
    turn_2 = [
        SystemInput(text="sys"),
        TextInput(text="first user msg"),
        RawResponse(response=[]),
        TextInput(text="later user msg"),
    ]

    body_1 = await model.build_body(turn_1, tools=[])
    body_2 = await model.build_body(turn_2, tools=[])

    assert body_1["prompt_cache_key"] == body_2["prompt_cache_key"]


async def test_prompt_cache_key_omitted_when_unconfigured():
    model = OpenAIModel(
        "gpt-4o",
        config=LLMConfig(provider_config=OpenAIConfig()),
    )

    body = await model.build_body(_INPUT, tools=[])

    assert "prompt_cache_key" not in body


async def test_deepseek_reasoning_keeps_max_tokens():
    """DeepSeek thinking mode documents max_tokens, not max_completion_tokens."""
    model = OpenAIModel(
        "deepseek-reasoner",
        provider="deepseek",
        config=LLMConfig(reasoning=True, max_tokens=8192),
        use_completions=True,
    )
    body = await model.build_body(_INPUT, tools=[])
    assert body.get("max_tokens") == 8192
    assert "max_completion_tokens" not in body
async def test_kimi_k3_public_reasoning_uses_max_completion_tokens():
    model = OpenAIModel(
        "kimi-k3",
        provider="kimi",
        config=LLMConfig(reasoning=True, reasoning_effort="max", max_tokens=256 * 1024),
        use_completions=True,
    )
    body = await model.build_body(_INPUT, tools=[])
    assert body.get("max_completion_tokens") == 256 * 1024
    assert "max_tokens" not in body
    assert body["reasoning_effort"] == "max"


def test_kimi_k3_public_route_uses_standard_key():
    model = KimiModel("kimi-k3")
    expected_key = "mock_ENV_KIMI_API_KEY"
    assert model.delegate is not None
    assert model.delegate.custom_endpoint == "https://api.moonshot.ai/v1/"
    assert model._default_api_key() == expected_key  # pyright: ignore[reportPrivateUsage]


async def test_plain_text_string_serialization_is_kimi_only():
    kimi = OpenAIModel(
        "kimi-k3",
        provider="kimi",
        config=LLMConfig(),
        use_completions=True,
    )
    openai = OpenAIModel(
        "gpt-4o",
        provider="openai",
        config=LLMConfig(),
        use_completions=True,
    )

    kimi_body = await kimi.build_body(_INPUT, tools=[])
    openai_body = await openai.build_body(_INPUT, tools=[])

    assert kimi_body["messages"] == [{"role": "user", "content": ""}]
    assert openai_body["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": ""}]}
    ]


async def test_google_delegate_thinking_config():
    model = OpenAIModel(
        "gemini-3.1-pro-preview",
        provider="google",
        config=LLMConfig(reasoning=True, reasoning_effort="low"),
        use_completions=True,
    )
    body = await model.build_body(_INPUT, tools=[])
    thinking_config = body["extra_body"]["extra_body"]["google"]["thinking_config"]
    assert thinking_config["include_thoughts"] is True
    assert thinking_config["thinking_level"] == "low"
    assert "reasoning_effort" not in body


async def test_completions_stream_options_omitted_when_non_streaming():
    model = OpenAIModel(
        "gpt-4o-mini",
        config=LLMConfig(provider_config=OpenAIConfig(stream_completions=False)),
        use_completions=True,
    )

    body = await model.build_body(_INPUT, tools=[])

    assert "stream_options" not in body


async def _async_iter(items: Sequence[object]) -> AsyncIterator[object]:
    for item in items:
        yield item


async def _query_completions(model: OpenAIModel, completion: object):
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=completion)

    with (
        patch.object(model, "get_client", return_value=mock_client),
        patch.object(model, "build_body", new_callable=AsyncMock, return_value={}),
    ):
        return await model._query_completions(
            _INPUT, tools=[], query_logger=logging.getLogger("test")
        )


def _tool_call_delta(
    *,
    index: int,
    call_id: str | None,
    name: str | None = None,
    arguments: str | None = None,
    extra_content: object | None = None,
) -> SimpleNamespace:
    model_extra = {}
    if extra_content is not None:
        model_extra["extra_content"] = extra_content
    return SimpleNamespace(
        index=index,
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
        model_extra=model_extra,
    )


def _completion_chunk(
    *,
    chunk_id: str,
    tool_calls: list[SimpleNamespace],
    finish_reason: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=chunk_id,
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                delta=SimpleNamespace(content=None, tool_calls=tool_calls),
            )
        ],
        usage=None,
    )


async def test_streaming_completions_google_tool_call_preserves_extra_content():
    model = OpenAIModel("gemini-test", provider="google", use_completions=True)
    extra_content = {"google": {"thought_signature": "sig-1"}}
    chunks = [
        _completion_chunk(
            chunk_id="cmpl_google_tool",
            tool_calls=[
                _tool_call_delta(
                    index=0,
                    call_id="call_1",
                    name="lookup",
                    arguments='{"q"',
                    extra_content=extra_content,
                )
            ],
        ),
        _completion_chunk(
            chunk_id="cmpl_google_tool",
            tool_calls=[_tool_call_delta(index=0, call_id=None, arguments=':"x"}')],
            finish_reason="tool_calls",
        ),
    ]

    result = await _query_completions(model, _async_iter(chunks))

    assert result.tool_calls[0].id == "call_1"
    assert result.tool_calls[0].name == "lookup"
    assert result.tool_calls[0].args == '{"q":"x"}'
    final_message = result.history[-1]
    assert isinstance(final_message, RawResponse)
    assert isinstance(final_message.response, ChatCompletionMessage)
    assert final_message.response.tool_calls is not None
    assert final_message.response.tool_calls[0].model_extra == {
        "extra_content": extra_content
    }


async def test_streaming_completions_poolside_reuses_existing_indexed_tool_call():
    model = OpenAIModel("poolside-test", provider="poolside", use_completions=True)
    chunks = [
        _completion_chunk(
            chunk_id="cmpl_poolside_tool",
            tool_calls=[
                _tool_call_delta(
                    index=0,
                    call_id="call_1",
                    name="lookup",
                    arguments='{"q"',
                )
            ],
        ),
        _completion_chunk(
            chunk_id="cmpl_poolside_tool",
            tool_calls=[
                _tool_call_delta(
                    index=0,
                    call_id="call_2",
                    name="lookup",
                    arguments=':"x"}',
                )
            ],
            finish_reason="tool_calls",
        ),
    ]

    result = await _query_completions(model, _async_iter(chunks))

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "call_1"
    assert result.tool_calls[0].args == '{"q":"x"}'


async def test_streaming_completions_deepseek_same_id_named_chunk_starts_new_tool_call():
    model = OpenAIModel("deepseek-test", provider="deepseek", use_completions=True)
    chunks = [
        _completion_chunk(
            chunk_id="cmpl_deepseek_tool",
            tool_calls=[
                _tool_call_delta(
                    index=0,
                    call_id="call_1",
                    name="first",
                    arguments="{}",
                )
            ],
        ),
        _completion_chunk(
            chunk_id="cmpl_deepseek_tool",
            tool_calls=[
                _tool_call_delta(
                    index=0,
                    call_id="call_1",
                    name="second",
                    arguments="{}",
                )
            ],
            finish_reason="tool_calls",
        ),
    ]

    result = await _query_completions(model, _async_iter(chunks))

    assert [(tool.id, tool.name, tool.args) for tool in result.tool_calls] == [
        ("call_1", "first", "{}"),
        ("call_1", "second", "{}"),
    ]


async def test_non_streaming_completions_query_parses_response():
    model = OpenAIModel(
        "gpt-4o-mini",
        config=LLMConfig(provider_config=OpenAIConfig(stream_completions=False)),
        use_completions=True,
    )
    response = ChatCompletion(
        id="cmpl_123",
        created=0,
        model="gpt-4o-mini",
        object="chat.completion",
        choices=[
            Choice(
                finish_reason="tool_calls",
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="hello",
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_1",
                            type="function",
                            function=Function(name="lookup", arguments='{"q":"x"}'),
                        )
                    ],
                ),
            )
        ],
        usage=CompletionUsage(
            completion_tokens=5,
            prompt_tokens=10,
            total_tokens=15,
            completion_tokens_details=CompletionTokensDetails(reasoning_tokens=1),
            prompt_tokens_details=PromptTokensDetails(cached_tokens=2),
        ),
    )
    object.__setattr__(response, "_request_id", "openai-request-1")
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=response)

    with patch.object(model, "get_client", return_value=mock_client):
        with patch.object(model, "build_body", new_callable=AsyncMock, return_value={}):
            result = await model._query_completions(
                _INPUT, tools=[], query_logger=logging.getLogger("test")
            )

    mock_client.chat.completions.create.assert_awaited_once_with(stream=False)
    assert result.output_text == "hello"
    assert result.finish_reason.reason == FinishReason.TOOL_CALLS
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "call_1"
    assert result.tool_calls[0].name == "lookup"
    assert result.tool_calls[0].args == '{"q":"x"}'
    assert result.metadata.in_tokens == 8
    assert result.metadata.out_tokens == 4
    assert result.metadata.reasoning_tokens == 1
    assert result.metadata.cache_read_tokens == 2
    assert result.extras.response_id == "cmpl_123"
    assert result.extras.provider_response_id == "cmpl_123"
    assert result.extras.provider_request_id == "openai-request-1"
    assert result.metadata.performance is None


async def test_non_streaming_kimi_parses_top_level_cached_tokens():
    model = OpenAIModel(
        "kimi-k3",
        provider="kimi",
        config=LLMConfig(provider_config=OpenAIConfig(stream_completions=False)),
        use_completions=True,
    )
    usage = CompletionUsage(
        completion_tokens=5,
        prompt_tokens=10,
        total_tokens=15,
    )
    object.__setattr__(usage, "cached_tokens", 4)
    response = ChatCompletion(
        id="cmpl_kimi_cached",
        created=0,
        model="kimi-k3",
        object="chat.completion",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content="hello"),
            )
        ],
        usage=usage,
    )

    result = await _query_completions(model, response)

    assert result.metadata.in_tokens == 6
    assert result.metadata.out_tokens == 5
    assert result.metadata.cache_read_tokens == 4


async def test_non_streaming_completions_empty_output_text_raises_no_output():
    model = OpenAIModel(
        "gpt-4o-mini",
        config=LLMConfig(provider_config=OpenAIConfig(stream_completions=False)),
        use_completions=True,
    )
    response = ChatCompletion(
        id="cmpl_empty",
        created=0,
        model="gpt-4o-mini",
        object="chat.completion",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content=""),
            )
        ],
        usage=None,
    )
    with pytest.raises(ModelNoOutputError):
        await _query_completions(model, response)


@pytest.mark.parametrize(
    ("finish_reason", "expected_error"),
    [
        ("length", MaxOutputTokensExceededError),
        ("content_filter", ContentFilterError),
    ],
)
async def test_non_streaming_completions_empty_non_success_raises_mapped_error(
    finish_reason: Literal["length", "content_filter"],
    expected_error: type[Exception],
):
    model = OpenAIModel(
        "gpt-4o-mini",
        config=LLMConfig(provider_config=OpenAIConfig(stream_completions=False)),
        use_completions=True,
    )
    response = ChatCompletion(
        id="cmpl_empty_non_success",
        created=0,
        model="gpt-4o-mini",
        object="chat.completion",
        choices=[
            Choice(
                finish_reason=finish_reason,
                index=0,
                message=ChatCompletionMessage(role="assistant", content=""),
            )
        ],
        usage=None,
    )
    with pytest.raises(expected_error):
        await _query_completions(model, response)


async def test_non_streaming_completions_empty_text_with_reasoning_keeps_reasoning_only():
    model = OpenAIModel(
        "gpt-4o-mini",
        config=LLMConfig(provider_config=OpenAIConfig(stream_completions=False)),
        use_completions=True,
    )
    message = ChatCompletionMessage(role="assistant", content="")
    object.__setattr__(message, "reasoning_content", "considered")
    response = ChatCompletion(
        id="cmpl_empty_non_success_reasoning",
        created=0,
        model="gpt-4o-mini",
        object="chat.completion",
        choices=[Choice(finish_reason="length", index=0, message=message)],
        usage=None,
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=response)

    with patch.object(model, "get_client", return_value=mock_client):
        with patch.object(model, "build_body", new_callable=AsyncMock, return_value={}):
            result = await model._query_completions(
                _INPUT, tools=[], query_logger=logging.getLogger("test")
            )

    assert result.output_text is None
    assert result.reasoning == "considered"
    assert result.finish_reason.reason == FinishReason.MAX_TOKENS


@pytest.mark.parametrize("content", [None, ""])
async def test_non_streaming_completions_tool_only_uses_none_for_absent_or_empty_text(
    content: str | None,
):
    model = OpenAIModel(
        "gpt-4o-mini",
        config=LLMConfig(provider_config=OpenAIConfig(stream_completions=False)),
        use_completions=True,
    )
    response = ChatCompletion(
        id="cmpl_tool_only",
        created=0,
        model="gpt-4o-mini",
        object="chat.completion",
        choices=[
            Choice(
                finish_reason="tool_calls",
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=content,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_1",
                            type="function",
                            function=Function(name="lookup", arguments='{"q":"x"}'),
                        )
                    ],
                ),
            )
        ],
        usage=None,
    )
    result = await _query_completions(model, response)

    assert result.output_text is None
    assert result.reasoning is None
    assert result.finish_reason.reason == FinishReason.TOOL_CALLS
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "call_1"
    assert result.tool_calls[0].name == "lookup"
    assert result.tool_calls[0].args == '{"q":"x"}'
    assert result.extras.response_id == "cmpl_tool_only"
    final_message = result.history[-1]
    assert isinstance(final_message, RawResponse)
    assert isinstance(final_message.response, ChatCompletionMessage)
    assert final_message.response.content is None


@pytest.mark.parametrize(
    ("model_name", "expects_reasoning_content"),
    [
        ("kimi-k3", True),
        ("kimi-k2.6", False),
    ],
)
async def test_kimi_k3_only_preserves_empty_reasoning_content(
    model_name: str,
    expects_reasoning_content: bool,
):
    model = OpenAIModel(
        model_name,
        provider="kimi",
        config=LLMConfig(
            reasoning=True,
            provider_config=OpenAIConfig(stream_completions=False),
        ),
        use_completions=True,
    )
    response = ChatCompletion(
        id="cmpl_kimi_tool_only",
        created=0,
        model=model_name,
        object="chat.completion",
        choices=[
            Choice(
                finish_reason="tool_calls",
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_1",
                            type="function",
                            function=Function(name="lookup", arguments='{"q":"x"}'),
                        )
                    ],
                ),
            )
        ],
        usage=None,
    )

    result = await _query_completions(model, response)

    final_message = result.history[-1]
    assert isinstance(final_message, RawResponse)
    assert isinstance(final_message.response, ChatCompletionMessage)
    if expects_reasoning_content:
        assert getattr(final_message.response, "reasoning_content") == ""
    else:
        assert not hasattr(final_message.response, "reasoning_content")


async def test_streaming_completions_empty_output_text_raises_no_output():
    model = OpenAIModel("gpt-4o-mini", use_completions=True)
    chunks = [
        SimpleNamespace(
            id="cmpl_stream_empty",
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    delta=SimpleNamespace(content="", tool_calls=None),
                )
            ],
            usage=None,
        )
    ]

    with pytest.raises(ModelNoOutputError):
        await _query_completions(model, _async_iter(chunks))


@pytest.mark.parametrize(
    ("finish_reason", "expected_error"),
    [
        ("length", MaxOutputTokensExceededError),
        ("content_filter", ContentFilterError),
    ],
)
async def test_streaming_completions_empty_non_success_raises_mapped_error(
    finish_reason: Literal["length", "content_filter"],
    expected_error: type[Exception],
):
    model = OpenAIModel("gpt-4o-mini", use_completions=True)
    chunks = [
        SimpleNamespace(
            id="cmpl_stream_empty_non_success",
            choices=[
                SimpleNamespace(
                    finish_reason=finish_reason,
                    delta=SimpleNamespace(content="", tool_calls=None),
                )
            ],
            usage=None,
        )
    ]
    with pytest.raises(expected_error):
        await _query_completions(model, _async_iter(chunks))


async def test_reasoning_delegate_tag_only_output_raises_no_output():
    model = OpenAIModel(
        "sonar-reasoning-pro",
        provider="perplexity",
        config=LLMConfig(reasoning=True),
        use_completions=True,
    )
    chunk = SimpleNamespace(
        id="cmpl_stream_tag_only",
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                delta=SimpleNamespace(
                    content="<think></think>",
                    reasoning_content=None,
                    tool_calls=None,
                ),
            )
        ],
        usage=CompletionUsage(completion_tokens=1, prompt_tokens=6, total_tokens=7),
    )
    with pytest.raises(ModelNoOutputError):
        await _query_completions(model, _async_iter([chunk]))


async def test_non_streaming_reasoning_delegate_tag_only_non_success_raises_mapped_error():
    model = OpenAIModel(
        "sonar-reasoning-pro",
        provider="perplexity",
        config=LLMConfig(
            reasoning=True,
            provider_config=OpenAIConfig(stream_completions=False),
        ),
        use_completions=True,
    )
    response = ChatCompletion(
        id="cmpl_tag_only_non_success",
        created=0,
        model="sonar-reasoning-pro",
        object="chat.completion",
        choices=[
            Choice(
                finish_reason="length",
                index=0,
                message=ChatCompletionMessage(
                    role="assistant", content="<think></think>"
                ),
            )
        ],
        usage=None,
    )
    with pytest.raises(MaxOutputTokensExceededError):
        await _query_completions(model, response)


@pytest.mark.parametrize(
    ("finish_reason", "expected_error"),
    [
        ("length", MaxOutputTokensExceededError),
        ("content_filter", ContentFilterError),
    ],
)
async def test_streaming_reasoning_delegate_tag_only_non_success_raises_mapped_error(
    finish_reason: Literal["length", "content_filter"],
    expected_error: type[Exception],
):
    model = OpenAIModel(
        "sonar-reasoning-pro",
        provider="perplexity",
        config=LLMConfig(reasoning=True),
        use_completions=True,
    )
    chunk = SimpleNamespace(
        id="cmpl_stream_tag_only_non_success",
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                delta=SimpleNamespace(
                    content="<think></think>",
                    reasoning_content=None,
                    tool_calls=None,
                ),
            )
        ],
        usage=None,
    )

    with pytest.raises(expected_error):
        await _query_completions(model, _async_iter([chunk]))


async def test_completions_stream_reasoning_without_usage_split_preserves_timeline():
    model = OpenAIModel(
        "sonar-reasoning-pro",
        provider="perplexity",
        config=LLMConfig(reasoning=True),
        use_completions=True,
    )
    usage = CompletionUsage(completion_tokens=6, prompt_tokens=10, total_tokens=16)
    chunks = [
        SimpleNamespace(
            id="cmpl_stream",
            choices=[
                SimpleNamespace(
                    finish_reason=None,
                    delta=SimpleNamespace(
                        content=None,
                        reasoning_content="why",
                        tool_calls=None,
                    ),
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            id="cmpl_stream",
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    delta=SimpleNamespace(
                        content="OK",
                        reasoning_content=None,
                        tool_calls=None,
                    ),
                )
            ],
            usage=usage,
        ),
    ]
    result = await _query_completions(model, _async_iter(chunks))

    assert result.reasoning == "why"
    assert result.output_text == "OK"
    assert result.metadata.out_tokens == 6
    assert result.metadata.reasoning_tokens is None
    performance = result.metadata.performance
    assert performance is not None
    assert [entry.channel for entry in performance.timeline] == ["reasoning", "content"]
    assert [
        [event.type for event in entry.events] for entry in performance.timeline
    ] == [
        ["reasoning_started", "reasoning_delta", "reasoning_finished"],
        ["content_started", "content_delta", "content_finished"],
    ]


async def test_reasoning_delegate_with_tags_keeps_streamed_content_timing():
    model = OpenAIModel(
        "sonar-reasoning-pro",
        provider="perplexity",
        config=LLMConfig(reasoning=True),
        use_completions=True,
    )
    usage = CompletionUsage(completion_tokens=3, prompt_tokens=6, total_tokens=9)
    chunk = SimpleNamespace(
        id="cmpl_stream",
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                delta=SimpleNamespace(
                    content="<think>why</think>OK",
                    reasoning_content=None,
                    tool_calls=None,
                ),
            )
        ],
        usage=usage,
    )
    result = await _query_completions(model, _async_iter([chunk]))

    assert result.output_text == "OK"
    assert result.reasoning == "why"
    performance = result.metadata.performance
    assert performance is not None
    assert performance.time_to_first_token_ms.content is not None
    assert [entry.channel for entry in performance.timeline] == ["content"]


async def test_reasoning_delegate_with_tags_splits_reasoning_after_query_finalization():
    model = OpenAIModel(
        "sonar-reasoning-pro",
        provider="perplexity",
        config=LLMConfig(reasoning=True),
        use_completions=True,
    )
    usage = CompletionUsage(completion_tokens=3, prompt_tokens=6, total_tokens=9)
    chunk = SimpleNamespace(
        id="cmpl_stream",
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                delta=SimpleNamespace(
                    content="<think>why</think>OK",
                    reasoning_content=None,
                    tool_calls=None,
                ),
            )
        ],
        usage=usage,
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=_async_iter([chunk]))

    with (
        patch.object(model, "get_client", return_value=mock_client),
        patch.object(model, "build_body", new_callable=AsyncMock, return_value={}),
    ):
        result = await model.query(_INPUT)

    assert result.output_text == "OK"
    assert result.reasoning == "why"


async def test_reasoning_delegate_split_thought_tags_replay_visible_history():
    model = OpenAIModel(
        "sonar-reasoning-pro",
        provider="perplexity",
        config=LLMConfig(reasoning=True),
        use_completions=True,
    )
    chunks = [
        SimpleNamespace(
            id="cmpl_split_thought",
            choices=[
                SimpleNamespace(
                    finish_reason=None,
                    delta=SimpleNamespace(
                        content="<thought>wh",
                        reasoning_content=None,
                        tool_calls=None,
                    ),
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            id="cmpl_split_thought",
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    delta=SimpleNamespace(
                        content="y</thought>OK",
                        reasoning_content=None,
                        tool_calls=None,
                    ),
                )
            ],
            usage=CompletionUsage(completion_tokens=3, prompt_tokens=6, total_tokens=9),
        ),
    ]
    result = await _query_completions(model, _async_iter(chunks))

    parsed_followup = await model.parse_input(
        [*result.history, TextInput(text="follow up")]
    )

    assert result.output_text == "OK"
    assert result.reasoning == "why"
    final_message = result.history[-1]
    assert isinstance(final_message, RawResponse)
    assert isinstance(final_message.response, ChatCompletionMessage)
    assert final_message.response.content == "OK"
    assert getattr(final_message.response, "reasoning_content") == "why"
    assert parsed_followup[-2] == final_message.response


@pytest.mark.parametrize("stream", [False, True])
async def test_reasoning_delegate_tag_only_tool_call_keeps_tool_only_output_text_none(
    stream: bool,
):
    model = OpenAIModel(
        "sonar-reasoning-pro",
        provider="perplexity",
        config=LLMConfig(
            reasoning=True,
            provider_config=OpenAIConfig(stream_completions=stream),
        ),
        use_completions=True,
    )
    raw_tool_call = ChatCompletionMessageToolCall(
        id="call_1",
        type="function",
        function=Function(name="lookup", arguments='{"q":"x"}'),
    )
    if stream:
        chunks: Sequence[object] | ChatCompletion = [
            SimpleNamespace(
                id="cmpl_tag_tool",
                choices=[
                    SimpleNamespace(
                        finish_reason=None,
                        delta=SimpleNamespace(
                            content="<think></think>",
                            reasoning_content=None,
                            tool_calls=None,
                        ),
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                id="cmpl_tag_tool",
                choices=[
                    SimpleNamespace(
                        finish_reason="tool_calls",
                        delta=SimpleNamespace(
                            content=None,
                            reasoning_content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id="call_1",
                                    function=SimpleNamespace(
                                        name="lookup", arguments='{"q":"x"}'
                                    ),
                                    model_extra={},
                                )
                            ],
                        ),
                    )
                ],
                usage=None,
            ),
        ]
        response = _async_iter(chunks)
    else:
        response = ChatCompletion(
            id="cmpl_tag_tool",
            created=0,
            model="sonar-reasoning-pro",
            object="chat.completion",
            choices=[
                Choice(
                    finish_reason="tool_calls",
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content="<think></think>",
                        tool_calls=[raw_tool_call],
                    ),
                )
            ],
            usage=None,
        )

    result = await _query_completions(model, response)

    assert result.output_text is None
    assert result.reasoning is None
    assert result.finish_reason.reason == FinishReason.TOOL_CALLS
    assert len(result.tool_calls) == 1
    final_message = result.history[-1]
    assert isinstance(final_message, RawResponse)
    assert isinstance(final_message.response, ChatCompletionMessage)
    assert final_message.response.content is None


async def test_reasoning_delegate_without_tags_keeps_streamed_content_timing():
    model = OpenAIModel(
        "sonar-reasoning-pro",
        provider="perplexity",
        config=LLMConfig(reasoning=True),
        use_completions=True,
    )
    usage = CompletionUsage(completion_tokens=1, prompt_tokens=6, total_tokens=7)
    chunk = SimpleNamespace(
        id="cmpl_stream",
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                delta=SimpleNamespace(
                    content="OK",
                    reasoning_content=None,
                    tool_calls=None,
                ),
            )
        ],
        usage=usage,
    )
    result = await _query_completions(model, _async_iter([chunk]))

    assert result.output_text == "OK"
    assert result.reasoning is None
    performance = result.metadata.performance
    assert performance is not None
    assert performance.time_to_first_token_ms.content is not None
    assert [entry.channel for entry in performance.timeline] == ["content"]


async def test_default_tool_call_mode_leaves_allowed_callers_omitted():
    model = OpenAIModel("public-test-responses-model")
    tools = [
        ToolDefinition(
            name="list_files",
            body=ToolBody(
                name="list_files",
                description="List files",
                properties={"path": {"type": "string"}},
                required=["path"],
            ),
        ),
        ToolDefinition(
            name="lookup",
            body=ToolBody(
                name="lookup",
                description="Lookup",
                properties={"query": {"type": "string"}},
                required=["query"],
                allowed_callers=["direct"],
            ),
        ),
    ]

    body = await model.build_body([TextInput(text="inspect")], tools=tools)

    assert "allowed_callers" not in body["tools"][0]
    assert body["tools"][1]["allowed_callers"] == ["direct"]
    assert len(body["tools"]) == 2


@pytest.mark.parametrize(
    ("tool_call_mode", "allowed_callers", "has_code_mode_tool"),
    [
        ("auto", ["code_mode", "direct"], True),
        ("code_mode", ["code_mode"], True),
    ],
)
async def test_tool_call_mode_sets_missing_allowed_callers(
    tool_call_mode: Literal["auto", "code_mode"],
    allowed_callers: list[str],
    has_code_mode_tool: bool,
):
    model = OpenAIModel(
        "public-test-responses-model",
        config=LLMConfig(provider_config=OpenAIConfig(tool_call_mode=tool_call_mode)),
    )
    tools = [
        ToolDefinition(
            name="list_files",
            body=ToolBody(
                name="list_files",
                description="List files",
                properties={"path": {"type": "string"}},
                required=["path"],
            ),
        ),
        ToolDefinition(
            name="lookup",
            body=ToolBody(
                name="lookup",
                description="Lookup",
                properties={"query": {"type": "string"}},
                required=["query"],
                allowed_callers=["direct"],
            ),
        ),
        ToolDefinition(
            name="code_mode",
            body={"type": "code_mode", "language": "javascript"},
        ),
    ]

    body = await model.build_body([TextInput(text="inspect")], tools=tools)

    assert body["tools"][0]["allowed_callers"] == allowed_callers
    assert body["tools"][1]["allowed_callers"] == ["direct"]
    assert (body["tools"][-1] == {"type": "code_mode", "language": "javascript"}) is (
        has_code_mode_tool
    )


async def test_code_mode_tool_shape_and_roundtrip():
    model = OpenAIModel("public-test-responses-model")
    tools = [
        ToolDefinition(
            name="list_files",
            body=ToolBody(
                name="list_files",
                description="List files",
                properties={"path": {"type": "string"}},
                required=["path"],
                allowed_callers=["code_mode"],
            ),
        ),
        ToolDefinition(
            name="code_mode",
            body={"type": "code_mode", "language": "javascript"},
        ),
    ]

    body = await model.build_body([TextInput(text="inspect")], tools=tools, store=False)

    assert body["store"] is False
    assert body["tools"][0]["allowed_callers"] == ["code_mode"]
    assert body["tools"][1] == {"type": "code_mode", "language": "javascript"}

    raw_tool_call = ResponseFunctionToolCall.model_construct(
        id="fc_1",
        call_id="call_1",
        name="list_files",
        arguments='{"path":"/repo"}',
        type="function_call",
        code_mode_id="cm_1",
    )
    tool_result = ToolResult(
        tool_call=ToolCall(
            id="fc_1",
            call_id="call_1",
            name="list_files",
            args='{"path":"/repo"}',
            code_mode_id="cm_1",
        ),
        result='{"files":["README.md"]}',
    )

    parsed_input = await model.parse_input(
        [RawResponse(response=[raw_tool_call]), tool_result]
    )

    assert parsed_input[-1] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": '{"files":["README.md"]}',
        "code_mode_id": "cm_1",
    }


async def test_code_mode_output_appends_to_output_text():
    response = _responses_response(
        response_id="resp_1",
        text_block_text="assistant text",
        output=[
            SimpleNamespace(
                type="code_mode_output",
                id="cmo_1",
                code_mode_id="cm_1",
                result=8,
                status="completed",
            )
        ],
        request_id="openai-request-2",
    )
    model = OpenAIModel("public-test-responses-model")
    mock_client = MagicMock()
    mock_client.responses.create = AsyncMock(return_value=response)

    with patch.object(model, "get_client", return_value=mock_client):
        result = await model._query_impl(  # pyright: ignore[reportPrivateUsage]
            [TextInput(text="compute")],
            tools=[],
            stream=False,
            query_logger=MagicMock(),
        )

    assert result.output_text == "assistant text\n8"
    assert result.extras.response_id == "resp_1"
    assert result.extras.provider_response_id == "resp_1"
    assert result.extras.provider_request_id == "openai-request-2"


async def test_code_mode_output_without_text_block_does_not_prefix_separator():
    response = _responses_response(
        response_id="resp_1",
        output=[
            SimpleNamespace(
                type="code_mode_output",
                id="cmo_1",
                code_mode_id="cm_1",
                result=8,
                status="completed",
            )
        ],
    )
    model = OpenAIModel("public-test-responses-model")
    mock_client = MagicMock()
    mock_client.responses.create = AsyncMock(return_value=response)

    with patch.object(model, "get_client", return_value=mock_client):
        result = await model._query_impl(  # pyright: ignore[reportPrivateUsage]
            [TextInput(text="compute")],
            tools=[],
            stream=False,
            query_logger=MagicMock(),
        )

    assert result.output_text == "8"


async def test_empty_code_mode_output_raises_no_output():
    response = _responses_response(
        response_id="resp_empty_code_mode",
        output=[
            SimpleNamespace(
                type="code_mode_output",
                id="cmo_1",
                code_mode_id="cm_1",
                result="",
                status="completed",
            )
        ],
    )
    model = OpenAIModel("public-test-responses-model")
    mock_client = MagicMock()
    mock_client.responses.create = AsyncMock(return_value=response)

    with patch.object(model, "get_client", return_value=mock_client):
        with pytest.raises(ModelNoOutputError):
            await model._query_impl(  # pyright: ignore[reportPrivateUsage]
                [TextInput(text="compute")],
                tools=[],
                stream=False,
                query_logger=MagicMock(),
            )


async def test_non_streaming_responses_usage_populates_normalized_metadata():
    response = _responses_response(
        response_id="resp_usage",
        text_block_text="assistant text",
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            input_tokens_details=SimpleNamespace(cached_tokens=2),
            output_tokens_details=SimpleNamespace(reasoning_tokens=1),
        ),
        request_id="openai-request-usage",
    )
    model = OpenAIModel("public-test-responses-model")
    mock_client = MagicMock()
    mock_client.responses.create = AsyncMock(return_value=response)

    with patch.object(model, "get_client", return_value=mock_client):
        result = await model._query_impl(  # pyright: ignore[reportPrivateUsage]
            [TextInput(text="compute")],
            tools=[],
            stream=False,
            query_logger=MagicMock(),
        )

    assert result.metadata.in_tokens == 8
    assert result.metadata.out_tokens == 4
    assert result.metadata.reasoning_tokens == 1
    assert result.metadata.cache_read_tokens == 2
    assert result.extras.response_id == "resp_usage"
    assert result.extras.provider_response_id == "resp_usage"
    assert result.extras.provider_request_id == "openai-request-usage"


async def test_non_streaming_responses_empty_output_text_raises_no_output():
    response = _responses_response(
        response_id="resp_empty",
        text_block_text="",
    )
    model = OpenAIModel("gpt-4o-mini")
    mock_client = MagicMock()
    mock_client.responses.create = AsyncMock(return_value=response)

    with patch.object(model, "get_client", return_value=mock_client):
        with pytest.raises(ModelNoOutputError):
            await model._query_impl(  # pyright: ignore[reportPrivateUsage]
                [TextInput(text="lookup")],
                tools=[],
                stream=False,
                query_logger=MagicMock(),
            )


@pytest.mark.parametrize(
    ("incomplete_reason", "expected_error"),
    [
        ("max_output_tokens", MaxOutputTokensExceededError),
        ("content_filter", ContentFilterError),
    ],
)
async def test_non_streaming_responses_empty_non_success_raises_mapped_error(
    incomplete_reason: Literal["max_output_tokens", "content_filter"],
    expected_error: type[Exception],
):
    response = _responses_response(
        response_id="resp_empty_non_success",
        status="incomplete",
        text_block_text="",
        incomplete_details=SimpleNamespace(reason=incomplete_reason),
    )
    model = OpenAIModel("gpt-4o-mini")
    mock_client = MagicMock()
    mock_client.responses.create = AsyncMock(return_value=response)

    with patch.object(model, "get_client", return_value=mock_client):
        with pytest.raises(expected_error):
            await model._query_impl(  # pyright: ignore[reportPrivateUsage]
                [TextInput(text="lookup")],
                tools=[],
                stream=False,
                query_logger=MagicMock(),
            )


async def test_non_streaming_responses_empty_text_with_reasoning_keeps_reasoning_only():
    response = _responses_response(
        response_id="resp_empty_non_success_reasoning",
        status="incomplete",
        text_block_text="",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        output=[
            SimpleNamespace(
                type="reasoning",
                summary=[SimpleNamespace(text="considered")],
            )
        ],
    )
    model = OpenAIModel("gpt-4o-mini")
    mock_client = MagicMock()
    mock_client.responses.create = AsyncMock(return_value=response)

    with patch.object(model, "get_client", return_value=mock_client):
        result = await model._query_impl(  # pyright: ignore[reportPrivateUsage]
            [TextInput(text="lookup")],
            tools=[],
            stream=False,
            query_logger=MagicMock(),
        )

    assert result.output_text is None
    assert result.reasoning == "considered"
    assert result.finish_reason.reason == FinishReason.MAX_TOKENS


async def test_non_streaming_responses_empty_reasoning_summary_raises_no_output():
    response = _responses_response(
        response_id="resp_empty_reasoning_summary",
        output=[SimpleNamespace(type="reasoning", summary=[])],
    )
    model = OpenAIModel("gpt-4o-mini")
    mock_client = MagicMock()
    mock_client.responses.create = AsyncMock(return_value=response)

    with patch.object(model, "get_client", return_value=mock_client):
        with pytest.raises(ModelNoOutputError):
            await model._query_impl(  # pyright: ignore[reportPrivateUsage]
                [TextInput(text="lookup")],
                tools=[],
                stream=False,
                query_logger=MagicMock(),
            )


@pytest.mark.parametrize("text_block_text", [None, ""])
async def test_non_streaming_responses_tool_only_uses_none_for_absent_or_empty_text(
    text_block_text: str | None,
):
    raw_tool_call = ResponseFunctionToolCall.model_construct(
        id="fc_1",
        call_id="call_1",
        name="lookup",
        arguments='{"q":"x"}',
        type="function_call",
    )
    output: list[object] = [raw_tool_call]
    if text_block_text is not None:
        output.append(
            ResponseOutputMessage(
                id="msg_1",
                type="message",
                role="assistant",
                status="completed",
                content=[
                    ResponseOutputText(
                        annotations=[],
                        text=text_block_text,
                        type="output_text",
                    )
                ],
            )
        )
    response = Response.model_construct(
        id="resp_tool_only",
        created_at=0.0,
        model="gpt-4o-mini",
        object="response",
        output=output,
        parallel_tool_calls=True,
        status="completed",
        tool_choice="auto",
        tools=[],
        incomplete_details=None,
        usage=None,
    )
    model = OpenAIModel("gpt-4o-mini")
    mock_client = MagicMock()
    mock_client.responses.create = AsyncMock(return_value=response)

    with patch.object(model, "get_client", return_value=mock_client):
        result = await model._query_impl(  # pyright: ignore[reportPrivateUsage]
            [TextInput(text="lookup")],
            tools=[],
            stream=False,
            query_logger=MagicMock(),
        )

    assert result.output_text is None
    assert result.reasoning is None
    assert result.finish_reason.reason == FinishReason.TOOL_CALLS
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "fc_1"
    assert result.tool_calls[0].call_id == "call_1"
    assert result.tool_calls[0].name == "lookup"
    assert result.tool_calls[0].args == '{"q":"x"}'
    assert result.extras.response_id == "resp_tool_only"
    assert result.metadata.performance is None


class ProviderSearchResult(BaseModel):
    title: str
    score: float


def test_safe_search_results_converts_provider_models_to_json_values():
    search_results = [ProviderSearchResult(title="doc", score=0.5)]

    result = _safe_search_results(search_results, logging.getLogger("test"))

    assert result == [{"title": "doc", "score": 0.5}]


def test_safe_search_results_drops_non_json_values(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.WARNING)

    result = _safe_search_results(object(), logging.getLogger("test"))

    assert result is None
    assert "Dropping non-JSON-serializable search results" in caplog.text
