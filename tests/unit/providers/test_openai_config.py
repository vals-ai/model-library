import logging
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
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall

from model_library.base import FinishReason, LLMConfig
from model_library.base.input import (
    RawResponse,
    TextInput,
    ToolBody,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from model_library.providers.openai import (
    OpenAIConfig,
    OpenAIModel,
    _safe_search_results,
)

_INPUT = [TextInput(text="")]


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
    response = SimpleNamespace(
        id="resp_1",
        status="completed",
        output_text="assistant text",
        tools=[],
        reasoning=None,
        incomplete_details=None,
        output=[
            SimpleNamespace(
                type="code_mode_output",
                id="cmo_1",
                code_mode_id="cm_1",
                result=8,
                status="completed",
            )
        ],
        usage=None,
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
