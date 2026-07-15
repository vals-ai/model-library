import logging
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningSummaryTextDoneEvent,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)
from openai.types.responses.response_reasoning_item import Summary

from model_library.base import QueryResultMetadata, QueryResultPerformance
from model_library.base.output.builder import QueryResultBuilder
from model_library.base.output.result import FinishReason as ValsFinishReason
from model_library.exceptions import ModelNoOutputError
from model_library.providers.openai import OpenAIModel


def _require_performance(metadata: QueryResultMetadata) -> QueryResultPerformance:
    performance = metadata.performance
    assert performance is not None
    return performance


def _response_completed(
    response: object, sequence_number: int = 0
) -> ResponseCompletedEvent:
    return ResponseCompletedEvent.model_construct(
        response=response,
        sequence_number=sequence_number,
        type="response.completed",
    )


def _response_text_delta(
    delta: str, sequence_number: int = 0
) -> ResponseTextDeltaEvent:
    return ResponseTextDeltaEvent(
        content_index=0,
        delta=delta,
        item_id="msg_1",
        logprobs=[],
        output_index=0,
        sequence_number=sequence_number,
        type="response.output_text.delta",
    )


def _response_text_done(
    text: str = "", sequence_number: int = 0
) -> ResponseTextDoneEvent:
    return ResponseTextDoneEvent(
        content_index=0,
        item_id="msg_1",
        logprobs=[],
        output_index=0,
        sequence_number=sequence_number,
        text=text,
        type="response.output_text.done",
    )


def _response_reasoning_delta(
    delta: str, sequence_number: int = 0
) -> ResponseReasoningSummaryTextDeltaEvent:
    return ResponseReasoningSummaryTextDeltaEvent(
        delta=delta,
        item_id="rs_1",
        output_index=0,
        sequence_number=sequence_number,
        summary_index=0,
        type="response.reasoning_summary_text.delta",
    )


def _response_reasoning_done(
    text: str = "", sequence_number: int = 0
) -> ResponseReasoningSummaryTextDoneEvent:
    return ResponseReasoningSummaryTextDoneEvent(
        item_id="rs_1",
        output_index=0,
        sequence_number=sequence_number,
        summary_index=0,
        text=text,
        type="response.reasoning_summary_text.done",
    )


def _response_function_call(
    *,
    item_id: str | None = "fc_1",
    call_id: str = "call_1",
    name: str = "ping",
    arguments: str = "{}",
) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall.model_construct(
        id=item_id,
        call_id=call_id,
        name=name,
        arguments=arguments,
        type="function_call",
    )


def _response_output_text_message(text: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id="msg_1",
        content=[ResponseOutputText(annotations=[], text=text, type="output_text")],
        role="assistant",
        status="completed",
        type="message",
    )


def _response_reasoning_item(text: str):
    from openai.types.responses import ResponseReasoningItem

    return ResponseReasoningItem(
        id="rs_1",
        summary=[Summary(text=text, type="summary_text")],
        type="reasoning",
    )


def _openai_response(
    *,
    response_id: str = "resp_1",
    text_block_text: str | None = None,
    output: list[object] | None = None,
    status: str = "completed",
    usage: object | None = None,
    incomplete_details: object | None = None,
) -> Response:
    output_items: list[object] = []
    if text_block_text is not None:
        output_items.append(_response_output_text_message(text_block_text))
    if output is not None:
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
    )


def _response_output_item_added(
    item: object, output_index: int = 0, sequence_number: int = 0
) -> ResponseOutputItemAddedEvent:
    return ResponseOutputItemAddedEvent.model_construct(
        item=item,
        output_index=output_index,
        sequence_number=sequence_number,
        type="response.output_item.added",
    )


def _response_function_arguments_delta(
    *,
    item_id: str = "fc_1",
    output_index: int = 0,
    delta: str = "{}",
    sequence_number: int = 0,
) -> ResponseFunctionCallArgumentsDeltaEvent:
    return ResponseFunctionCallArgumentsDeltaEvent(
        delta=delta,
        item_id=item_id,
        output_index=output_index,
        sequence_number=sequence_number,
        type="response.function_call_arguments.delta",
    )


def _response_function_arguments_done(
    *,
    item_id: str = "fc_1",
    output_index: int = 0,
    arguments: str = "{}",
    sequence_number: int = 0,
) -> ResponseFunctionCallArgumentsDoneEvent:
    return ResponseFunctionCallArgumentsDoneEvent(
        arguments=arguments,
        item_id=item_id,
        name="ping",
        output_index=output_index,
        sequence_number=sequence_number,
        type="response.function_call_arguments.done",
    )


def _response_output_item_done(
    item: object, output_index: int = 0, sequence_number: int = 0
) -> ResponseOutputItemDoneEvent:
    return ResponseOutputItemDoneEvent.model_construct(
        item=item,
        output_index=output_index,
        sequence_number=sequence_number,
        type="response.output_item.done",
    )


class _OpenAIResponsesStream:
    def __init__(self, events: list[ResponseStreamEvent]):
        self._events = events

    async def __aiter__(self):
        for event in self._events:
            yield event


class _TimedOpenAIResponsesStream:
    def __init__(
        self,
        events: list[tuple[float, ResponseStreamEvent]],
        clock_time: dict[str, float],
    ):
        self._events = events
        self._clock_time = clock_time

    async def __aiter__(self):
        for timestamp, event in self._events:
            self._clock_time["now"] = timestamp
            yield event


class _OpenAIRawResponsesStream:
    request_id = "openai-request-1"

    def __init__(self, stream: object):
        self._stream = stream

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc: object):
        return None

    async def parse(self):
        return self._stream


class TestOpenAIResponsesStreamingPerformance:
    @staticmethod
    async def _run_query(model: OpenAIModel, events: list[ResponseStreamEvent]):
        mock_client = MagicMock()
        mock_client.responses.with_streaming_response.create = MagicMock(
            return_value=_OpenAIRawResponsesStream(_OpenAIResponsesStream(events))
        )

        with patch.object(model, "get_client", return_value=mock_client):
            with patch.object(
                model, "build_body", new_callable=AsyncMock, return_value={}
            ):
                return await model._query_impl(
                    [], tools=[], query_logger=logging.getLogger("test")
                )

    async def test_openai_responses_streaming_empty_completed_response_raises_no_output(
        self,
    ):
        model = OpenAIModel("gpt-4o-mini")
        response = _openai_response()
        events = [cast(ResponseStreamEvent, _response_completed(response))]

        with pytest.raises(ModelNoOutputError):
            await self._run_query(model, events)

    async def test_openai_responses_streaming_content_populates_performance_timeline(
        self,
    ):
        model = OpenAIModel("gpt-4o-mini")
        response = _openai_response()
        events = [
            _response_text_delta("hello"),
            _response_completed(response),
        ]

        result = await self._run_query(model, events)

        assert result.output_text == "hello"
        assert _require_performance(result.metadata).timeline[0].channel == "content"
        assert (
            _require_performance(result.metadata).time_to_first_token_ms.content
            is not None
        )

    async def test_openai_responses_streaming_uses_completed_text_without_delta(
        self,
    ):
        model = OpenAIModel("gpt-4o-mini")
        response = _openai_response(output=[_response_output_text_message("completed")])
        events = [cast(ResponseStreamEvent, _response_completed(response))]

        result = await self._run_query(model, events)

        assert result.output_text == "completed"

    async def test_openai_responses_streaming_uses_streamed_text_over_completed_response(
        self,
    ):
        model = OpenAIModel("gpt-4o-mini")
        response = _openai_response(output=[_response_output_text_message("completed")])
        events = [
            _response_text_delta("streamed"),
            _response_completed(response),
        ]

        result = await self._run_query(model, events)

        assert result.output_text == "streamed"

    async def test_openai_responses_streaming_uses_completed_reasoning_without_delta(
        self,
    ):
        model = OpenAIModel("gpt-4o-mini")
        response = _openai_response(output=[_response_reasoning_item("completed")])
        events = [cast(ResponseStreamEvent, _response_completed(response))]

        result = await self._run_query(model, events)

        assert result.reasoning == "completed"

    async def test_openai_responses_streaming_uses_streamed_reasoning_over_completed_response(
        self,
    ):
        model = OpenAIModel("gpt-4o-mini")
        response = _openai_response(output=[_response_reasoning_item("completed")])
        events = [
            _response_reasoning_delta("streamed"),
            _response_completed(response),
        ]

        result = await self._run_query(model, events)

        assert result.reasoning == "streamed"

    async def test_openai_responses_streaming_keeps_terminal_artifacts(self):
        model = OpenAIModel("gpt-4o-mini")
        response = _openai_response(
            output=[
                _response_output_text_message("completed"),
                _response_reasoning_item("completed reasoning"),
                _response_function_call(item_id="fc_1"),
                SimpleNamespace(
                    type="code_mode_output",
                    id="cmo_1",
                    code_mode_id="cm_1",
                    result=8,
                    status="completed",
                ),
            ],
            status="incomplete",
            incomplete_details=SimpleNamespace(reason="max_output_tokens"),
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=5,
                input_tokens_details=SimpleNamespace(cached_tokens=2),
                output_tokens_details=SimpleNamespace(reasoning_tokens=1),
            ),
        )
        events = [
            _response_text_delta("streamed"),
            _response_reasoning_delta("streamed reasoning"),
            _response_completed(response),
        ]

        result = await self._run_query(model, events)

        assert result.output_text == "streamed\n8"
        assert result.reasoning == "streamed reasoning"
        assert result.tool_calls[0].id == "fc_1"
        assert result.finish_reason.reason == ValsFinishReason.MAX_TOKENS
        assert result.metadata.in_tokens == 8
        assert result.metadata.out_tokens == 4
        assert result.metadata.cache_read_tokens == 2
        assert result.metadata.reasoning_tokens == 1

    async def test_openai_responses_streaming_appends_code_mode_output(self):
        model = OpenAIModel("gpt-4o-mini")
        clock_time = {"now": 0.0}
        response = _openai_response(
            output=[
                _response_output_text_message("completed"),
                SimpleNamespace(
                    type="code_mode_output",
                    id="cmo_1",
                    code_mode_id="cm_1",
                    result=8,
                    status="completed",
                ),
            ],
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=5,
                input_tokens_details=SimpleNamespace(cached_tokens=0),
                output_tokens_details=SimpleNamespace(reasoning_tokens=0),
            ),
        )
        events = [
            (0.100, _response_text_delta("streamed")),
            (0.200, _response_text_done("streamed")),
            (1.000, _response_completed(response)),
        ]
        mock_client = MagicMock()
        mock_client.responses.with_streaming_response.create = MagicMock(
            return_value=_OpenAIRawResponsesStream(
                _TimedOpenAIResponsesStream(events, clock_time)
            )
        )

        with patch.object(model, "get_client", return_value=mock_client):
            with patch.object(
                model, "build_body", new_callable=AsyncMock, return_value={}
            ):
                with patch(
                    "model_library.providers.openai.QueryResultBuilder",
                    return_value=QueryResultBuilder(clock=lambda: clock_time["now"]),
                ):
                    result = await model._query_impl(
                        [], tools=[], query_logger=logging.getLogger("test")
                    )

        assert result.output_text == "streamed\n8"
        assert result.metadata.out_tokens == 5

    async def test_openai_responses_builder_starts_before_stream_open(self):
        model = OpenAIModel("gpt-4o-mini")
        order: list[str] = []
        response = _openai_response()
        events = [
            _response_text_delta("hello"),
            _response_completed(response),
        ]

        def create_response(**_kwargs: object):
            order.append("stream-opened")
            return _OpenAIRawResponsesStream(_OpenAIResponsesStream(events))

        def make_builder() -> QueryResultBuilder:
            order.append("builder-created")
            return QueryResultBuilder()

        mock_client = MagicMock()
        mock_client.responses.with_streaming_response.create = MagicMock(
            side_effect=create_response
        )

        with patch.object(model, "get_client", return_value=mock_client):
            with patch.object(
                model, "build_body", new_callable=AsyncMock, return_value={}
            ):
                with patch(
                    "model_library.providers.openai.QueryResultBuilder",
                    side_effect=make_builder,
                ):
                    result = await model._query_impl(
                        [], tools=[], query_logger=logging.getLogger("test")
                    )

        assert result.output_text == "hello"
        assert order[:2] == ["builder-created", "stream-opened"]

    async def test_openai_responses_streaming_content_done_closes_timeline_segment(
        self,
    ):
        model = OpenAIModel("gpt-4o-mini")
        clock_time = {"now": 0.0}
        response = _openai_response()
        events = [
            (0.100, _response_text_delta("hello")),
            (0.200, _response_text_done()),
            (1.000, _response_completed(response)),
        ]
        mock_client = MagicMock()
        mock_client.responses.with_streaming_response.create = MagicMock(
            return_value=_OpenAIRawResponsesStream(
                _TimedOpenAIResponsesStream(events, clock_time)
            )
        )

        with patch.object(model, "get_client", return_value=mock_client):
            with patch.object(
                model, "build_body", new_callable=AsyncMock, return_value={}
            ):
                with patch(
                    "model_library.providers.openai.QueryResultBuilder",
                    return_value=QueryResultBuilder(clock=lambda: clock_time["now"]),
                ):
                    result = await model._query_impl(
                        [], tools=[], query_logger=logging.getLogger("test")
                    )

        assert _require_performance(result.metadata).timeline[0].end_ms == 200

    async def test_openai_responses_streaming_tool_done_closes_keyed_segment(
        self,
    ):
        model = OpenAIModel("gpt-4o-mini")
        clock_time = {"now": 0.0}
        response = _openai_response(output=[_response_function_call(item_id="fc_1")])
        events = [
            (
                0.100,
                _response_output_item_added(
                    _response_function_call(item_id="fc_1"), output_index=0
                ),
            ),
            (
                0.150,
                _response_function_arguments_delta(item_id="fc_1", output_index=0),
            ),
            (
                0.250,
                _response_function_arguments_done(item_id="fc_1", output_index=0),
            ),
            (1.000, _response_completed(response)),
        ]
        mock_client = MagicMock()
        mock_client.responses.with_streaming_response.create = MagicMock(
            return_value=_OpenAIRawResponsesStream(
                _TimedOpenAIResponsesStream(events, clock_time)
            )
        )

        with patch.object(model, "get_client", return_value=mock_client):
            with patch.object(
                model, "build_body", new_callable=AsyncMock, return_value={}
            ):
                with patch(
                    "model_library.providers.openai.QueryResultBuilder",
                    return_value=QueryResultBuilder(clock=lambda: clock_time["now"]),
                ):
                    result = await model._query_impl(
                        [], tools=[], query_logger=logging.getLogger("test")
                    )

        assert _require_performance(result.metadata).timeline[0].end_ms == 250

    async def test_openai_responses_streaming_reasoning_done_closes_timeline_segment(
        self,
    ):
        model = OpenAIModel("gpt-4o-mini")
        clock_time = {"now": 0.0}
        response = _openai_response(output=[_response_reasoning_item("summary")])
        events = [
            (0.100, _response_reasoning_delta("summary")),
            (0.300, _response_reasoning_done()),
            (1.000, _response_completed(response)),
        ]
        mock_client = MagicMock()
        mock_client.responses.with_streaming_response.create = MagicMock(
            return_value=_OpenAIRawResponsesStream(
                _TimedOpenAIResponsesStream(events, clock_time)
            )
        )

        with patch.object(model, "get_client", return_value=mock_client):
            with patch.object(
                model, "build_body", new_callable=AsyncMock, return_value={}
            ):
                with patch(
                    "model_library.providers.openai.QueryResultBuilder",
                    return_value=QueryResultBuilder(clock=lambda: clock_time["now"]),
                ):
                    result = await model._query_impl(
                        [], tools=[], query_logger=logging.getLogger("test")
                    )

        assert _require_performance(result.metadata).timeline[0].end_ms == 300

    async def test_openai_responses_streaming_output_item_done_closes_keyed_segment(
        self,
    ):
        model = OpenAIModel("gpt-4o-mini")
        clock_time = {"now": 0.0}
        response = _openai_response(output=[_response_function_call(item_id="fc_1")])
        events = [
            (
                0.100,
                _response_output_item_added(
                    _response_function_call(item_id="fc_1"), output_index=0
                ),
            ),
            (
                0.150,
                _response_function_arguments_delta(item_id="fc_1", output_index=0),
            ),
            (
                0.350,
                _response_output_item_done(
                    _response_function_call(item_id="fc_1"), output_index=0
                ),
            ),
            (1.000, _response_completed(response)),
        ]
        mock_client = MagicMock()
        mock_client.responses.with_streaming_response.create = MagicMock(
            return_value=_OpenAIRawResponsesStream(
                _TimedOpenAIResponsesStream(events, clock_time)
            )
        )

        with patch.object(model, "get_client", return_value=mock_client):
            with patch.object(
                model, "build_body", new_callable=AsyncMock, return_value={}
            ):
                with patch(
                    "model_library.providers.openai.QueryResultBuilder",
                    return_value=QueryResultBuilder(clock=lambda: clock_time["now"]),
                ):
                    result = await model._query_impl(
                        [], tools=[], query_logger=logging.getLogger("test")
                    )

        assert _require_performance(result.metadata).timeline[0].end_ms == 350

    async def test_openai_responses_streaming_reasoning_summary_populates_performance_timeline(
        self,
    ):
        model = OpenAIModel("gpt-4o-mini")
        response = _openai_response(output=[_response_reasoning_item("summary")])
        events = [
            _response_reasoning_delta("summary"),
            _response_completed(response),
        ]

        result = await self._run_query(model, events)

        assert result.reasoning == "summary"
        assert _require_performance(result.metadata).timeline[0].channel == "reasoning"
        assert (
            _require_performance(result.metadata).time_to_first_token_ms.reasoning
            is not None
        )

    async def test_openai_responses_streaming_tool_call_populates_performance_timeline(
        self,
    ):
        model = OpenAIModel("gpt-4o-mini")
        response = _openai_response(output=[_response_function_call(item_id="fc_1")])
        events = [
            _response_output_item_added(_response_function_call(item_id="fc_1")),
            _response_function_arguments_delta(item_id="fc_1"),
            _response_completed(response),
        ]

        result = await self._run_query(model, events)

        assert result.output_text is None
        assert result.reasoning is None
        assert result.tool_calls[0].name == "ping"
        assert _require_performance(result.metadata).timeline[0].channel == "tool_call"
        assert (
            _require_performance(result.metadata).time_to_first_token_ms.tool_call
            is not None
        )
        assert (
            _require_performance(result.metadata).time_to_first_token_ms.answer
            is not None
        )

    async def test_openai_responses_interleaved_tool_call_deltas_use_item_segments(
        self,
    ):
        model = OpenAIModel("gpt-4o-mini")
        response = _openai_response(
            output=[
                _response_function_call(
                    item_id="fc_1",
                    call_id="call_1",
                    name="weather",
                    arguments='{"location": "SF"}',
                ),
                _response_function_call(
                    item_id="fc_2",
                    call_id="call_2",
                    name="time",
                    arguments='{"tz": "PST"}',
                ),
            ]
        )
        events = [
            _response_output_item_added(
                _response_function_call(item_id="fc_1"), output_index=0
            ),
            _response_output_item_added(
                _response_function_call(item_id="fc_2"), output_index=1
            ),
            _response_function_arguments_delta(
                item_id="fc_1", output_index=0, delta='{"location": "SF"}'
            ),
            _response_function_arguments_delta(
                item_id="fc_2", output_index=1, delta='{"tz": "PST"}'
            ),
            _response_completed(response),
        ]

        result = await self._run_query(model, events)

        assert [
            (call.id, call.call_id, call.name, call.args) for call in result.tool_calls
        ] == [
            ("fc_1", "call_1", "weather", '{"location": "SF"}'),
            ("fc_2", "call_2", "time", '{"tz": "PST"}'),
        ]
        assert [
            entry.channel for entry in _require_performance(result.metadata).timeline
        ] == [
            "tool_call",
            "tool_call",
        ]
        assert [
            [event.type for event in entry.events]
            for entry in _require_performance(result.metadata).timeline
        ] == [
            [
                "tool_call_started",
                "tool_call_ready",
                "tool_call_delta",
                "tool_call_finished",
            ],
            [
                "tool_call_started",
                "tool_call_ready",
                "tool_call_delta",
                "tool_call_finished",
            ],
        ]

    async def test_openai_responses_non_streaming_falls_back_to_message_text(self):
        model = OpenAIModel("gpt-4o-mini")
        response = _openai_response(text_block_text="fallback text")
        mock_client = MagicMock()
        mock_client.responses.create = AsyncMock(return_value=response)

        with patch.object(model, "get_client", return_value=mock_client):
            with patch.object(
                model, "build_body", new_callable=AsyncMock, return_value={}
            ):
                result = await model._query_impl(
                    [], tools=[], query_logger=logging.getLogger("test"), stream=False
                )

        assert result.output_text == "fallback text"
