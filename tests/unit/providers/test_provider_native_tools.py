"""Tests for built-in provider web search (OpenAI web_search_call, Anthropic web_search_tool_result, Google grounding, xAI web_search_tool) handling."""

from typing import Any, Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai.types import (
    Candidate,
    FinishReason as GoogleFinishReason,
    GenerateContentResponse,
    GroundingChunk,
    GroundingChunkWeb,
    GroundingMetadata,
)
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputMessage,
    ResponseOutputText,
)
from openai.types.responses.response_function_web_search import (
    ActionSearch,
    ActionSearchSource,
    ResponseFunctionWebSearch,
)

from model_library.agent.metadata import TurnSummary
from model_library.agent.tool import ProviderTool
from model_library.base.input import TextInput
from model_library.base.output import FinishReason, FinishReasonInfo, QueryResult
from model_library.base.output.result import ProviderToolEvent
from model_library.providers.google.google import GoogleModel
from model_library.providers.openai import OpenAIModel

from tests.unit.agent.helpers import DoneTool, make_agent, make_metadata, mock_llm


def _make_web_search_response(
    query: str = "test query",
    status: Literal["in_progress", "searching", "completed", "failed"] = "completed",
    text_block_text: str | None = "The answer is 42.",
    action: ActionSearch | None = None,
) -> Response:
    """Minimal Responses API response with a web_search_call output item."""
    action = action or ActionSearch(
        type="search",
        query=query,
        queries=None,
        sources=[ActionSearchSource(type="url", url="https://example.com")],
    )
    web_search_item = ResponseFunctionWebSearch(
        id="ws_1",
        type="web_search_call",
        status=status,
        action=action,
    )
    output: list[object] = [web_search_item]
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
    return Response.model_construct(
        id="resp_1",
        created_at=0.0,
        model="gpt-5.5",
        object="response",
        output=output,
        parallel_tool_calls=True,
        status="completed",
        tool_choice="auto",
        tools=[],
        incomplete_details=None,
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


class _OpenAIResponsesStream:
    def __init__(
        self, events: list[ResponseOutputItemAddedEvent | ResponseCompletedEvent]
    ):
        self._events = events

    async def __aiter__(self):
        for event in self._events:
            yield event


class _OpenAIRawResponsesStream:
    request_id = "openai-request-1"

    def __init__(self, stream: _OpenAIResponsesStream):
        self._stream = stream

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc: object):
        return None

    async def parse(self):
        return self._stream


async def _parse_streaming_response(response):
    model = OpenAIModel("gpt-5.5")
    mock_client = MagicMock()
    mock_client.responses.with_streaming_response.create = MagicMock(
        return_value=_OpenAIRawResponsesStream(
            _OpenAIResponsesStream(
                [
                    ResponseOutputItemAddedEvent(
                        item=response.output[0],
                        output_index=0,
                        sequence_number=0,
                        type="response.output_item.added",
                    ),
                    ResponseCompletedEvent(
                        response=response,
                        sequence_number=1,
                        type="response.completed",
                    ),
                ]
            )
        )
    )
    with patch.object(model, "get_client", return_value=mock_client):
        return await model._query_impl(  # pyright: ignore[reportPrivateUsage]
            [TextInput(text="search for something")],
            tools=[],
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


@pytest.mark.parametrize("text_block_text", [None, ""])
async def test_web_search_only_response_is_not_empty_response(
    text_block_text: str | None,
):
    result = await _parse_response(
        _make_web_search_response(text_block_text=text_block_text)
    )

    assert result.output_text is None
    assert result.tool_calls == []
    assert len(result.provider_tool_events) == 1
    assert result.provider_tool_events[0].name == "web_search"


@pytest.mark.parametrize("text_block_text", [None, ""])
async def test_web_search_streaming_response_is_not_empty_response(
    text_block_text: str | None,
):
    result = await _parse_streaming_response(
        _make_web_search_response(text_block_text=text_block_text)
    )

    assert result.output_text is None
    assert result.finish_reason.reason == FinishReason.STOP
    assert result.tool_calls == []
    assert len(result.provider_tool_events) == 1
    assert result.provider_tool_events[0].name == "web_search"


# --- ProviderTool definition tests ---


async def test_provider_tool_definition_is_sent_to_model():
    """ProviderTool definitions appear in tool_definitions (sent to the model)."""
    agent = make_agent(
        mock_llm(),
        tools=[
            ProviderTool(name="web_search", body={"type": "web_search"}),
            DoneTool(),
        ],
    )

    names = [d.name for d in agent.tool_definitions]
    assert "web_search" in names
    assert "submit" in names


async def test_provider_tool_excluded_from_local_execution():
    """ProviderTool is absent from the agent's local dispatch table."""
    agent = make_agent(
        mock_llm(),
        tools=[
            ProviderTool(name="web_search", body={"type": "web_search"}),
            DoneTool(),
        ],
    )

    assert "web_search" not in agent._tools  # pyright: ignore[reportPrivateUsage]
    assert "submit" in agent._tools  # pyright: ignore[reportPrivateUsage]


# --- Agent loop tests ---


async def test_paused_provider_turn_continues_loop():
    """A paused provider turn (finish_reason=PAUSED) must not stop the loop.

    This is the Anthropic pause_turn case: the provider ran a server-side search
    and paused before emitting the answer, so the loop re-queries to resume.
    Also verifies that provider events are not dispatched as local tool calls.
    """
    paused_response = QueryResult(
        output_text=None,
        metadata=make_metadata(),
        finish_reason=FinishReasonInfo(reason=FinishReason.PAUSED, raw="pause_turn"),
        tool_calls=[],
        provider_tool_events=[
            ProviderToolEvent(
                id="ws_1",
                provider="anthropic",
                type="web_search_tool_result",
                name="web_search",
                input="test query",
                output=["https://example.com"],
            )
        ],
        history=[TextInput(text="prompt")],
    )
    done_response = QueryResult(
        output_text="the answer",
        metadata=make_metadata(),
        tool_calls=[],
        history=[TextInput(text="prompt")],
    )

    llm = mock_llm(paused_response, done_response)
    agent = make_agent(
        llm,
        tools=[
            ProviderTool(name="web_search", body={"type": "web_search"}),
            DoneTool(),
        ],
    )

    result = await agent.run([TextInput(text="search for something")], question_id="q1")

    first_turn = result.turns[0]
    assert isinstance(first_turn, TurnSummary)
    assert llm.query.call_count == 2
    assert first_turn.tool_calls == []
    assert result.final_answer == "the answer"


async def test_completed_provider_event_turn_with_no_text_stops_loop():
    """A completed provider-event turn with NO text (not paused) is terminal.

    This pins the grounding-only shape that motivated the finish-reason gate:
    STOP + provider event + no answer text. Only a PAUSED turn continues; a
    finished (STOP) provider search must not trigger a redundant re-query, even
    when it produced no text — otherwise the loop would run to the time limit.
    """
    completed_response = QueryResult(
        output_text=None,
        metadata=make_metadata(),
        finish_reason=FinishReasonInfo(reason=FinishReason.STOP, raw="stop"),
        tool_calls=[],
        provider_tool_events=[
            ProviderToolEvent(
                id="ws_1",
                provider="google",
                type="google_search_call",
                name="web_search",
                status="completed",
                input="test query",
                output=["https://example.com"],
            )
        ],
        history=[TextInput(text="prompt")],
    )

    llm = mock_llm(completed_response)
    agent = make_agent(
        llm,
        tools=[
            ProviderTool(name="web_search", body={"type": "web_search"}),
            DoneTool(),
        ],
    )

    result = await agent.run([TextInput(text="search for something")], question_id="q1")

    # Stopped after one turn (no re-query), and no text → empty final answer
    assert llm.query.call_count == 1
    assert result.final_answer == ""


async def test_text_with_provider_event_stops_loop():
    """A turn with output_text alongside provider events stops the loop immediately.

    Text means the model has answered. A grounded synthesis turn (text + events)
    is treated as a terminal response — no extra query is issued.
    """
    text_and_events_response = QueryResult(
        output_text="the answer",
        metadata=make_metadata(),
        tool_calls=[],
        provider_tool_events=[
            ProviderToolEvent(
                id="ws_1",
                provider="google",
                type="google_search_call",
                name="web_search",
                status="completed",
            )
        ],
        history=[TextInput(text="prompt")],
    )

    llm = mock_llm(text_and_events_response)
    agent = make_agent(
        llm,
        tools=[
            ProviderTool(name="web_search", body={"type": "web_search"}),
            DoneTool(),
        ],
    )

    result = await agent.run([TextInput(text="search for something")], question_id="q1")

    assert llm.query.call_count == 1
    assert result.final_answer == "the answer"


# ---------------------------------------------------------------------------
# Google grounding tests
# ---------------------------------------------------------------------------


def _make_grounding_chunk(uri: str) -> GenerateContentResponse:
    """Minimal streaming chunk containing grounding metadata."""
    chunk = MagicMock(spec=GenerateContentResponse)
    chunk.response_id = None
    chunk.usage_metadata = None

    candidate = MagicMock(spec=Candidate)
    candidate.content = None
    candidate.finish_reason = GoogleFinishReason.STOP
    candidate.grounding_metadata = GroundingMetadata(
        web_search_queries=["test query"],
        grounding_chunks=[
            GroundingChunk(web=GroundingChunkWeb(uri=uri, title="Example")),
        ],
    )
    chunk.candidates = [candidate]
    return chunk


def _make_text_chunk(text: str) -> GenerateContentResponse:
    """Minimal streaming chunk with text content."""
    from google.genai.types import Content, Part

    chunk = MagicMock(spec=GenerateContentResponse)
    chunk.response_id = "resp_1"
    chunk.usage_metadata = None

    candidate = MagicMock(spec=Candidate)
    candidate.finish_reason = None
    candidate.grounding_metadata = None
    part = MagicMock(spec=Part)
    part.function_call = None
    part.text = text
    part.thought = False
    candidate.content = MagicMock(spec=Content)
    candidate.content.parts = [part]
    chunk.candidates = [candidate]
    return chunk


async def _parse_google_response(chunks: list[GenerateContentResponse]) -> QueryResult:
    model = GoogleModel("gemini-2.5-flash")
    mock_client = MagicMock()

    async def fake_stream():
        for c in chunks:
            yield c

    mock_client.aio.models.generate_content_stream = AsyncMock(
        return_value=fake_stream()
    )
    with patch.object(model, "get_client", return_value=mock_client):
        return await model._query_impl(  # pyright: ignore[reportPrivateUsage]
            [TextInput(text="search for something")],
            tools=[],
            query_logger=MagicMock(),
        )


async def test_google_grounding_maps_to_provider_tool_events():
    """Grounding metadata produces ProviderToolEvents, not ToolCalls."""
    result = await _parse_google_response(
        [_make_text_chunk("The answer."), _make_grounding_chunk("https://example.com")]
    )

    assert result.tool_calls == []
    assert len(result.provider_tool_events) == 1
    event = result.provider_tool_events[0]
    assert isinstance(event, ProviderToolEvent)
    assert event.name == "web_search"
    assert event.type == "google_search_call"
    assert event.provider == "google"
    assert event.input == "test query"
    assert event.output == ["https://example.com"]


async def test_google_grounding_finish_reason_is_stop():
    """Finish reason is STOP when only grounding events are present."""
    result = await _parse_google_response(
        [_make_text_chunk("The answer."), _make_grounding_chunk("https://example.com")]
    )

    assert result.finish_reason.reason == FinishReason.STOP


async def test_google_grounding_only_response_does_not_raise():
    """A response with only grounding metadata (no text) must not trigger handle_empty_response."""
    result = await _parse_google_response(
        [_make_grounding_chunk("https://example.com")]
    )

    assert len(result.provider_tool_events) == 1
    assert result.tool_calls == []


async def test_google_truly_empty_response_raises():
    """A response with no text, no tool calls, and no grounding must raise."""
    from model_library.exceptions import ModelNoOutputError

    empty_chunk = MagicMock(spec=GenerateContentResponse)
    empty_chunk.response_id = "resp_1"
    empty_chunk.usage_metadata = None
    candidate = MagicMock(spec=Candidate)
    candidate.finish_reason = GoogleFinishReason.STOP
    candidate.grounding_metadata = None
    candidate.content = None
    empty_chunk.candidates = [candidate]

    with pytest.raises(ModelNoOutputError):
        await _parse_google_response([empty_chunk])


# ---------------------------------------------------------------------------
# Anthropic web search tests
# ---------------------------------------------------------------------------


def _make_anthropic_web_search_message(
    query: str = "test query", url: str = "https://example.com"
) -> MagicMock:
    """Minimal Anthropic beta message with a web_search_tool_result content block."""
    from anthropic.types.beta import (
        BetaTextBlock,
        BetaWebSearchResultBlock,
        BetaWebSearchToolResultBlock,
    )
    from anthropic.types.beta.beta_server_tool_use_block import BetaServerToolUseBlock

    tool_use_id = "tu_1"

    tool_use_block = BetaServerToolUseBlock(
        type="server_tool_use",
        id=tool_use_id,
        name="web_search",
        input={"query": query},
    )

    result_block = BetaWebSearchToolResultBlock(
        tool_use_id=tool_use_id,
        type="web_search_tool_result",
        content=[
            BetaWebSearchResultBlock(
                type="web_search_result",
                title="Example",
                url=url,
                encrypted_content="enc",
            )
        ],
    )

    text_block = BetaTextBlock(type="text", text="The answer.")

    message = MagicMock()
    message.id = "msg_1"
    message._request_id = None
    message.content = [tool_use_block, result_block, text_block]
    message.stop_reason = "end_turn"
    message.model = "claude-sonnet-4-6"
    usage = MagicMock()
    usage.input_tokens = 10
    usage.output_tokens = 20
    usage.cache_read_input_tokens = None
    usage.cache_creation_input_tokens = None
    usage.iterations = None
    message.usage = usage
    return message


async def _parse_anthropic_response(message: MagicMock) -> QueryResult:
    from model_library.providers.anthropic import AnthropicModel

    model = AnthropicModel("claude-sonnet-4-6")
    mock_stream = AsyncMock()
    mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_stream.__aexit__ = AsyncMock(return_value=None)
    mock_stream.get_final_message = AsyncMock(return_value=message)
    mock_client = MagicMock()
    mock_client.beta.messages.stream.return_value = mock_stream
    minimal_body: dict[str, Any] = {
        "model": "claude-sonnet-4-6",
        "messages": [],
        "max_tokens": 1024,
    }
    with (
        patch.object(model, "get_client", return_value=mock_client),
        patch.object(model, "build_body", AsyncMock(return_value=minimal_body)),
    ):
        return await model._query_impl(  # pyright: ignore[reportPrivateUsage]
            [TextInput(text="search for something")],
            tools=[],
            query_logger=MagicMock(),
        )


async def test_anthropic_web_search_maps_to_provider_tool_events():
    """web_search_tool_result blocks produce ProviderToolEvents, not ToolCalls."""
    result = await _parse_anthropic_response(_make_anthropic_web_search_message())

    assert result.tool_calls == []
    assert len(result.provider_tool_events) == 1
    event = result.provider_tool_events[0]
    assert isinstance(event, ProviderToolEvent)
    assert event.name == "web_search"
    assert event.type == "web_search_tool_result"
    assert event.provider == "anthropic"
    assert event.input == "test query"
    assert event.output == ["https://example.com"]


async def test_anthropic_web_search_finish_reason_is_stop():
    """Finish reason is STOP when only web search events are present."""
    result = await _parse_anthropic_response(_make_anthropic_web_search_message())

    assert result.finish_reason.reason == FinishReason.STOP


def test_anthropic_pause_turn_maps_to_paused():
    """Anthropic pause_turn maps to FinishReason.PAUSED so the agent loop resumes it."""
    from model_library.providers.anthropic import map_anthropic_finish_reason

    info = map_anthropic_finish_reason("pause_turn")
    assert info.reason == FinishReason.PAUSED
    assert info.raw == "pause_turn"


async def test_anthropic_replays_container_id_from_paused_turn():
    """A stored code-execution response's container id is replayed on the next request."""
    from model_library.base.input import RawResponse
    from model_library.providers.anthropic import AnthropicModel

    message = MagicMock()
    message.content = []
    message.container = MagicMock(id="cont_abc123")

    model = AnthropicModel("claude-sonnet-4-6")
    model.max_tokens = 1024

    body = await model.build_body(
        [TextInput(text="continue"), RawResponse(response=message)],
        tools=[],
    )

    assert body["container"] == "cont_abc123"


async def test_anthropic_no_container_key_without_prior_container():
    """No container key is sent when history never had a code-execution response."""
    from model_library.providers.anthropic import AnthropicModel

    model = AnthropicModel("claude-sonnet-4-6")
    model.max_tokens = 1024

    body = await model.build_body([TextInput(text="hello")], tools=[])

    assert "container" not in body


async def test_anthropic_replays_latest_container_id_across_multiple_turns():
    """When history has several code-execution turns, the most recent container id wins."""
    from model_library.base.input import RawResponse
    from model_library.providers.anthropic import AnthropicModel

    older_message = MagicMock()
    older_message.content = []
    older_message.container = MagicMock(id="cont_old")

    newer_message = MagicMock()
    newer_message.content = []
    newer_message.container = MagicMock(id="cont_new")

    model = AnthropicModel("claude-sonnet-4-6")
    model.max_tokens = 1024

    body = await model.build_body(
        [
            TextInput(text="first"),
            RawResponse(response=older_message),
            TextInput(text="second"),
            RawResponse(response=newer_message),
        ],
        tools=[],
    )

    assert body["container"] == "cont_new"


async def test_anthropic_container_id_survives_a_later_containerless_turn():
    """A later containerless turn doesn't erase an earlier turn's container id."""
    from model_library.base.input import RawResponse
    from model_library.providers.anthropic import AnthropicModel

    code_exec_message = MagicMock()
    code_exec_message.content = []
    code_exec_message.container = MagicMock(id="cont_from_code_exec")

    plain_message = MagicMock()
    plain_message.content = []
    plain_message.container = None

    model = AnthropicModel("claude-sonnet-4-6")
    model.max_tokens = 1024

    body = await model.build_body(
        [
            TextInput(text="first"),
            RawResponse(response=code_exec_message),
            TextInput(text="second"),
            RawResponse(response=plain_message),
        ],
        tools=[],
    )

    assert body["container"] == "cont_from_code_exec"


async def test_anthropic_web_search_only_response_does_not_raise():
    """A response with only web-search blocks (no text) must not trigger handle_empty_response."""
    from anthropic.types.beta import (
        BetaWebSearchResultBlock,
        BetaWebSearchToolResultBlock,
    )
    from anthropic.types.beta.beta_server_tool_use_block import BetaServerToolUseBlock

    tool_use_id = "tu_1"
    tool_use_block = BetaServerToolUseBlock(
        type="server_tool_use",
        id=tool_use_id,
        name="web_search",
        input={"query": "test"},
    )

    result_block = BetaWebSearchToolResultBlock(
        tool_use_id=tool_use_id,
        type="web_search_tool_result",
        content=[
            BetaWebSearchResultBlock(
                type="web_search_result",
                title="Example",
                url="https://example.com",
                encrypted_content="enc",
            )
        ],
    )

    message = MagicMock()
    message.id = "msg_1"
    message._request_id = None
    message.content = [tool_use_block, result_block]  # no text block
    message.stop_reason = "end_turn"
    message.model = "claude-sonnet-4-6"
    usage = MagicMock()
    usage.input_tokens = 10
    usage.output_tokens = 20
    usage.cache_read_input_tokens = None
    usage.cache_creation_input_tokens = None
    usage.iterations = None
    message.usage = usage

    result = await _parse_anthropic_response(message)

    assert len(result.provider_tool_events) == 1
    assert result.tool_calls == []


async def test_anthropic_truly_empty_response_raises():
    """A response with no text, no tool calls, and no web-search blocks must raise."""
    from model_library.exceptions import ModelNoOutputError

    message = MagicMock()
    message.id = "msg_1"
    message._request_id = None
    message.content = []
    message.stop_reason = "end_turn"
    message.model = "claude-sonnet-4-6"
    usage = MagicMock()
    usage.input_tokens = 10
    usage.output_tokens = 0
    usage.cache_read_input_tokens = None
    usage.cache_creation_input_tokens = None
    usage.iterations = None
    message.usage = usage

    with pytest.raises(ModelNoOutputError):
        await _parse_anthropic_response(message)


async def test_anthropic_web_search_error_maps_to_provider_tool_event():
    """An error-shaped web_search_tool_result becomes an observable provider event.

    When the web search fails, `content` is a BetaWebSearchToolResultError rather
    than a list of results. The error must be recorded as a provider event
    (status="error", output=error_code) so an error-only turn is not misread as
    empty output and raised as ModelNoOutputError.
    """
    from anthropic.types.beta import BetaWebSearchToolResultBlock
    from anthropic.types.beta.beta_server_tool_use_block import BetaServerToolUseBlock
    from anthropic.types.beta.beta_web_search_tool_result_error import (
        BetaWebSearchToolResultError,
    )

    tool_use_id = "tu_err"
    tool_use_block = BetaServerToolUseBlock(
        type="server_tool_use",
        id=tool_use_id,
        name="web_search",
        input={"query": "test"},
    )
    result_block = BetaWebSearchToolResultBlock(
        tool_use_id=tool_use_id,
        type="web_search_tool_result",
        content=BetaWebSearchToolResultError(
            type="web_search_tool_result_error",
            error_code="unavailable",
        ),
    )

    message = MagicMock()
    message.id = "msg_1"
    message._request_id = None
    message.content = [tool_use_block, result_block]  # error only, no text
    message.stop_reason = "end_turn"
    message.model = "claude-sonnet-4-6"
    usage = MagicMock()
    usage.input_tokens = 10
    usage.output_tokens = 20
    usage.cache_read_input_tokens = None
    usage.cache_creation_input_tokens = None
    usage.iterations = None
    message.usage = usage

    # Must not raise ModelNoOutputError despite having no text/tool_calls
    result = await _parse_anthropic_response(message)

    assert result.tool_calls == []
    assert len(result.provider_tool_events) == 1
    event = result.provider_tool_events[0]
    assert event.name == "web_search"
    assert event.type == "web_search_tool_result"
    assert event.status == "error"
    assert event.output == "unavailable"
    assert event.input == "test"


# ---------------------------------------------------------------------------
# xAI web search tests
# ---------------------------------------------------------------------------


def _make_xai_web_search_tool_call(query: str = "test query", id: str = "tc_1"):
    """Minimal xAI proto ToolCall with type WEB_SEARCH_TOOL."""
    from xai_sdk.proto.v6.chat_pb2 import FunctionCall, ToolCall, ToolCallType
    import json

    return ToolCall(
        id=id,
        type=ToolCallType.TOOL_CALL_TYPE_WEB_SEARCH_TOOL,
        function=FunctionCall(
            name="web_search",
            arguments=json.dumps({"query": query, "num_results": "5"}),
        ),
    )


def _make_xai_mock_response(
    content: str = "The answer.",
    tool_calls=None,
):
    """Minimal mock xAI Response object."""
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.cached_prompt_text_tokens = 0
    usage.completion_tokens = 20
    usage.reasoning_tokens = 0

    response = MagicMock()
    response.id = "resp_1"
    response.content = content
    response.reasoning_content = ""
    response.finish_reason = "REASON_STOP"
    response.tool_calls = tool_calls or []
    response.usage = usage
    return response


async def _parse_xai_response(mock_response) -> QueryResult:
    from model_library.providers.xai import XAIModel

    model = XAIModel("grok-3-latest")

    mock_chunk = MagicMock()
    mock_chunk.tool_calls = []
    mock_chunk.reasoning_content = None
    mock_chunk.content = None

    async def fake_stream():
        yield mock_response, mock_chunk

    mock_chat = MagicMock()
    mock_chat.stream = MagicMock(return_value=fake_stream())
    mock_client = MagicMock()
    mock_client.chat.create.return_value = mock_chat

    with patch.object(model, "get_client", return_value=mock_client):
        return await model._query_impl(
            [TextInput(text="search for something")],
            tools=[],
            query_logger=MagicMock(),
        )


async def test_xai_web_search_maps_to_provider_tool_events():
    """web_search_tool calls produce ProviderToolEvents, not ToolCalls."""
    import json

    tc = _make_xai_web_search_tool_call("test query")
    result = await _parse_xai_response(_make_xai_mock_response(tool_calls=[tc]))

    assert result.tool_calls == []
    assert len(result.provider_tool_events) == 1
    event = result.provider_tool_events[0]
    assert isinstance(event, ProviderToolEvent)
    assert event.name == "web_search"
    assert event.type == "web_search_tool"
    assert event.provider == "xai"
    # Raw args string is stored as-is (not json-parsed); the query is nested inside
    assert event.input == json.dumps({"query": "test query", "num_results": "5"})


async def test_xai_web_search_finish_reason_is_stop():
    """Finish reason is STOP when only web search events are present."""
    tc = _make_xai_web_search_tool_call()
    result = await _parse_xai_response(_make_xai_mock_response(tool_calls=[tc]))

    assert result.finish_reason.reason == FinishReason.STOP


async def test_xai_web_search_only_response_does_not_raise():
    """A response with only web_search_tool calls (no text) must not raise."""
    tc = _make_xai_web_search_tool_call()
    result = await _parse_xai_response(
        _make_xai_mock_response(content="", tool_calls=[tc])
    )

    assert len(result.provider_tool_events) == 1
    assert result.tool_calls == []


async def test_xai_truly_empty_response_raises():
    """A response with no text, no tool calls, and no web search must raise."""
    from model_library.exceptions import ModelNoOutputError

    with pytest.raises(ModelNoOutputError):
        await _parse_xai_response(_make_xai_mock_response(content="", tool_calls=[]))


async def test_xai_mixed_local_and_provider_tool_sequences():
    """Parsing a mixed turn assigns each list its content-index `sequence`.

    A web_search -> local_tool -> web_search response splits into separate
    tool_calls / provider_tool_events lists, but each item keeps `sequence` set
    to its index in the provider's tool-call list. Downstream (the Agent) uses
    these to merge the two lists back into the original order; here we only
    assert the parser-level sequence assignment that contract depends on.
    """
    import json

    from xai_sdk.proto.v6.chat_pb2 import FunctionCall, ToolCall, ToolCallType

    local_tool_call = ToolCall(
        id="tc_local",
        type=ToolCallType.TOOL_CALL_TYPE_CLIENT_SIDE_TOOL,
        function=FunctionCall(name="get_weather", arguments=json.dumps({"city": "SF"})),
    )
    # Interleaved order: web search, local tool, web search
    tool_calls = [
        _make_xai_web_search_tool_call("first query", id="tc_ws_1"),
        local_tool_call,
        _make_xai_web_search_tool_call("second query", id="tc_ws_2"),
    ]

    result = await _parse_xai_response(_make_xai_mock_response(tool_calls=tool_calls))

    # Local tool call keeps its interleaved position (index 1)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "get_weather"
    assert result.tool_calls[0].sequence == 1

    # Both web searches keep their positions (index 0 and 2)
    assert len(result.provider_tool_events) == 2
    assert [e.sequence for e in result.provider_tool_events] == [0, 2]
    # Raw args stored as-is (not json-parsed); the query is nested inside
    assert [e.input for e in result.provider_tool_events] == [
        json.dumps({"query": "first query", "num_results": "5"}),
        json.dumps({"query": "second query", "num_results": "5"}),
    ]


# ---------------------------------------------------------------------------
# NativeWebSearch parse_tools mapping tests
# ---------------------------------------------------------------------------


async def test_native_web_search_definition_uses_self_as_sentinel():
    """NativeWebSearch.definition.body is the JSON sentinel dict (gateway-safe)."""
    from model_library.agent.tool import NativeWebSearch, is_native_web_search

    tool = NativeWebSearch()
    defn = tool.definition
    assert defn.name == "web_search"
    assert is_native_web_search(defn.body)
    assert is_native_web_search(defn.body.model_dump())  # survives model_dump()


async def test_native_web_search_anthropic_parse_tools():
    """Anthropic parse_tools maps NativeWebSearch to the web_search_20260209 dict."""
    from model_library.agent.tool import NativeWebSearch
    from model_library.providers.anthropic import AnthropicModel

    model = AnthropicModel("claude-sonnet-4-6")
    tool = NativeWebSearch()
    result = await model.parse_tools([tool.definition])

    assert result == [{"type": "web_search_20260209", "name": "web_search"}]


async def test_native_web_search_google_parse_tools():
    """Google parse_tools maps NativeWebSearch to Tool(google_search=GoogleSearch())."""
    from google.genai.types import GoogleSearch, Tool as GoogleTool

    from model_library.agent.tool import NativeWebSearch
    from model_library.providers.google.google import GoogleModel

    model = GoogleModel("gemini-3-flash-preview")
    tool = NativeWebSearch()
    result = await model.parse_tools([tool.definition])

    assert len(result) == 1
    assert isinstance(result[0], GoogleTool)
    assert isinstance(result[0].google_search, GoogleSearch)


async def test_native_web_search_google_build_body_injects_tool_config():
    """Google build_body auto-injects tool_config when NativeWebSearch is present."""
    from google.genai.types import ToolConfig

    from model_library.agent.tool import NativeWebSearch
    from model_library.base.input import TextInput, SystemInput
    from model_library.providers.google.google import GoogleModel

    model = GoogleModel("gemini-3-flash-preview")
    tool = NativeWebSearch()
    body = await model.build_body(
        [SystemInput(text="sys"), TextInput(text="hi")],
        tools=[tool.definition],
    )

    config = body["config"]
    assert config.tool_config is not None
    assert isinstance(config.tool_config, ToolConfig)
    assert config.tool_config.include_server_side_tool_invocations is True


async def test_native_web_search_google_build_body_no_tool_config_without_native_search():
    """Google build_body does NOT inject tool_config when NativeWebSearch is absent."""
    from model_library.base.input import TextInput, SystemInput
    from model_library.providers.google.google import GoogleModel

    model = GoogleModel("gemini-3-flash-preview")
    body = await model.build_body(
        [SystemInput(text="sys"), TextInput(text="hi")],
        tools=[],
    )

    config = body["config"]
    assert config.tool_config is None


async def test_native_web_search_xai_parse_tools():
    """xAI parse_tools maps NativeWebSearch to the web_search proto (native path)."""
    from xai_sdk.proto.v6.chat_pb2 import Tool as XAITool

    from model_library.agent.tool import NativeWebSearch
    from model_library.base.base import LLMConfig
    from model_library.providers.xai import XAIModel

    model = XAIModel("grok-3-latest", config=LLMConfig(native=True))
    tool = NativeWebSearch()
    result = await model.parse_tools([tool.definition])

    assert len(result) == 1
    assert isinstance(result[0], XAITool)


async def test_native_web_search_openai_parse_tools():
    """OpenAI parse_tools maps NativeWebSearch to web_search_preview (Responses API)."""
    from model_library.agent.tool import NativeWebSearch
    from model_library.providers.openai import OpenAIModel

    model = OpenAIModel("gpt-5.5")  # use_completions=False by default
    tool = NativeWebSearch()
    result = await model.parse_tools([tool.definition])

    assert result == [{"type": "web_search_preview"}]


async def test_native_web_search_openai_raises_on_completions_path():
    """OpenAI parse_tools raises NotImplementedError for NativeWebSearch on completions path."""
    from model_library.agent.tool import NativeWebSearch
    from model_library.providers.openai import OpenAIModel

    model = OpenAIModel("gpt-5.5", use_completions=True)
    tool = NativeWebSearch()

    with pytest.raises(NotImplementedError, match="Chat Completions"):
        await model.parse_tools([tool.definition])


# ---------------------------------------------------------------------------
# Unsupported provider tests
# ---------------------------------------------------------------------------


async def test_native_web_search_ai21labs_raises():
    """AI21 Labs parse_tools raises NotImplementedError for NativeWebSearch."""
    from model_library.agent.tool import NativeWebSearch
    from model_library.providers.ai21labs import AI21LabsModel

    model = AI21LabsModel("jamba-1.5-large")
    tool = NativeWebSearch()

    with pytest.raises(NotImplementedError, match="AI21"):
        await model.parse_tools([tool.definition])


async def test_native_web_search_amazon_raises():
    """Amazon parse_tools raises NotImplementedError for NativeWebSearch."""
    from model_library.agent.tool import NativeWebSearch
    from model_library.providers.amazon import AmazonModel

    model = AmazonModel("amazon.nova-pro-v1:0")
    tool = NativeWebSearch()

    with pytest.raises(NotImplementedError, match="Amazon"):
        await model.parse_tools([tool.definition])


async def test_native_web_search_mistral_raises():
    """Mistral parse_tools raises NotImplementedError for NativeWebSearch."""
    from model_library.agent.tool import NativeWebSearch
    from model_library.providers.mistral import MistralModel

    model = MistralModel("mistral-large-latest")
    tool = NativeWebSearch()

    with pytest.raises(NotImplementedError, match="Mistral"):
        await model.parse_tools([tool.definition])


async def test_mistral_preserves_raw_provider_tool_passthrough():
    """Mistral still passes through raw ProviderTool bodies unchanged.

    Only the NativeWebSearch sentinel raises NotImplementedError; an arbitrary
    raw dict body must still be forwarded.
    """
    from model_library.base import ToolDefinition
    from model_library.providers.mistral import MistralModel

    model = MistralModel("mistral-large-latest")
    raw_tool = ToolDefinition(name="custom_search", body={"type": "web_search_preview"})

    result = await model.parse_tools([raw_tool])
    assert result == [{"type": "web_search_preview"}]


async def test_openai_deep_research_accepts_native_web_search_sentinel():
    """_check_deep_research_args does not raise when NativeWebSearch sentinel is present."""
    from model_library.agent.tool import NativeWebSearch

    model = OpenAIModel("openai/o3-pro")
    model.deep_research = True
    tool = NativeWebSearch()

    # Should not raise — sentinel matches is_native_web_search check
    await model._check_deep_research_args([tool.definition])  # pyright: ignore[reportPrivateUsage]


async def test_unsupported_provider_preserves_raw_provider_tool_passthrough():
    """Unsupported providers still pass through raw ProviderTool bodies unchanged.

    Only the NativeWebSearch sentinel raises NotImplementedError; an arbitrary
    raw dict body (e.g. a hand-crafted ProviderTool) must still be forwarded.
    """
    from model_library.base import ToolDefinition
    from model_library.providers.ai21labs import AI21LabsModel

    model = AI21LabsModel("jamba-1.5-large")
    raw_tool = ToolDefinition(name="custom_search", body={"type": "web_search_preview"})

    # Should not raise — raw dict body is not the NativeWebSearch sentinel
    result = await model.parse_tools([raw_tool])
    assert result == [{"type": "web_search_preview"}]
