import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch


from model_library.base import QueryResultMetadata, QueryResultPerformance
from model_library.base.output.builder import QueryResultBuilder
from model_library.providers.anthropic import AnthropicModel


def _require_performance(metadata: QueryResultMetadata) -> QueryResultPerformance:
    performance = metadata.performance
    assert performance is not None
    return performance


class _AnthropicStream:
    def __init__(self, events: list[SimpleNamespace], message: SimpleNamespace):
        self._events = events
        self._message = message

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return None

    async def __aiter__(self):
        for event in self._events:
            yield event

    async def get_final_message(self):
        return self._message


class _TimedAnthropicStream:
    def __init__(
        self,
        events: list[tuple[float, SimpleNamespace]],
        message: SimpleNamespace,
        clock_time: dict[str, float],
        final_message_time: float,
    ):
        self._events = events
        self._message = message
        self._clock_time = clock_time
        self._final_message_time = final_message_time

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return None

    async def __aiter__(self):
        for timestamp, event in self._events:
            self._clock_time["now"] = timestamp
            yield event

    async def get_final_message(self):
        self._clock_time["now"] = self._final_message_time
        return self._message


class TestAnthropicStreamingPerformance:
    @staticmethod
    async def _run_query(
        model: AnthropicModel,
        events: list[SimpleNamespace],
        message: SimpleNamespace,
    ):
        mock_client = MagicMock()
        mock_client.beta.messages.stream = MagicMock(
            return_value=_AnthropicStream(events, message)
        )

        with patch.object(model, "get_client", return_value=mock_client):
            with patch.object(
                model, "build_body", new_callable=AsyncMock, return_value={}
            ):
                return await model._query_impl(
                    [], tools=[], query_logger=logging.getLogger("test")
                )

    async def test_anthropic_raw_stream_events_populate_performance_timeline(self):
        model = AnthropicModel("claude-haiku-4-5-20251001")
        events = [
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="thinking_delta", thinking="thinking"),
            ),
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="text_delta", text="answer"),
            ),
            SimpleNamespace(
                type="content_block_start",
                content_block=SimpleNamespace(type="tool_use", name="lookup"),
            ),
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="input_json_delta", partial_json='{"q"'),
            ),
        ]
        message = SimpleNamespace(
            id="msg_1",
            model="claude-haiku-4-5-20251001",
            stop_reason="tool_use",
            usage=SimpleNamespace(
                input_tokens=3,
                output_tokens=2,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
                iterations=None,
            ),
            content=[
                SimpleNamespace(type="thinking", thinking="thinking"),
                SimpleNamespace(type="text", text="answer"),
                SimpleNamespace(
                    type="tool_use",
                    id="toolu_1",
                    name="lookup",
                    input={"q": "x"},
                ),
            ],
        )

        result = await self._run_query(model, events, message)

        assert result.reasoning == "thinking"
        assert result.output_text == "answer"
        assert result.tool_calls[0].name == "lookup"
        assert [
            entry.channel for entry in _require_performance(result.metadata).timeline
        ] == [
            "reasoning",
            "content",
            "tool_call",
        ]
        assert (
            _require_performance(result.metadata).time_to_first_token_ms.reasoning
            is not None
        )
        assert (
            _require_performance(result.metadata).time_to_first_token_ms.content
            is not None
        )
        assert (
            _require_performance(result.metadata).time_to_first_token_ms.tool_call
            is not None
        )

    async def test_anthropic_content_block_stop_closes_timeline_segment(self):
        model = AnthropicModel("claude-haiku-4-5-20251001")
        clock_time = {"now": 0.0}
        events = [
            (
                0.100,
                SimpleNamespace(
                    type="content_block_delta",
                    delta=SimpleNamespace(type="text_delta", text="answer"),
                ),
            ),
            (0.250, SimpleNamespace(type="content_block_stop")),
        ]
        message = SimpleNamespace(
            id="msg_1",
            model="claude-haiku-4-5-20251001",
            stop_reason="end_turn",
            usage=SimpleNamespace(
                input_tokens=3,
                output_tokens=2,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
                iterations=None,
            ),
            content=[SimpleNamespace(type="text", text="answer")],
        )
        mock_client = MagicMock()
        mock_client.beta.messages.stream = MagicMock(
            return_value=_TimedAnthropicStream(events, message, clock_time, 1.000)
        )

        with patch.object(model, "get_client", return_value=mock_client):
            with patch.object(
                model, "build_body", new_callable=AsyncMock, return_value={}
            ):
                with patch(
                    "model_library.providers.anthropic.QueryResultBuilder",
                    return_value=QueryResultBuilder(clock=lambda: clock_time["now"]),
                ):
                    result = await model._query_impl(
                        [], tools=[], query_logger=logging.getLogger("test")
                    )

        assert _require_performance(result.metadata).timeline[0].end_ms == 250

    async def test_anthropic_tool_block_stop_closes_timeline_segment(self):
        model = AnthropicModel("claude-haiku-4-5-20251001")
        clock_time = {"now": 0.0}
        events = [
            (
                0.100,
                SimpleNamespace(
                    type="content_block_start",
                    content_block=SimpleNamespace(type="tool_use", name="lookup"),
                ),
            ),
            (
                0.150,
                SimpleNamespace(
                    type="content_block_delta",
                    delta=SimpleNamespace(type="input_json_delta", partial_json="{}"),
                ),
            ),
            (0.250, SimpleNamespace(type="content_block_stop")),
        ]
        message = SimpleNamespace(
            id="msg_1",
            model="claude-haiku-4-5-20251001",
            stop_reason="tool_use",
            usage=SimpleNamespace(
                input_tokens=3,
                output_tokens=2,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
                iterations=None,
            ),
            content=[
                SimpleNamespace(
                    type="tool_use",
                    id="toolu_1",
                    name="lookup",
                    input={},
                )
            ],
        )
        mock_client = MagicMock()
        mock_client.beta.messages.stream = MagicMock(
            return_value=_TimedAnthropicStream(events, message, clock_time, 1.000)
        )

        with patch.object(model, "get_client", return_value=mock_client):
            with patch.object(
                model, "build_body", new_callable=AsyncMock, return_value={}
            ):
                with patch(
                    "model_library.providers.anthropic.QueryResultBuilder",
                    return_value=QueryResultBuilder(clock=lambda: clock_time["now"]),
                ):
                    result = await model._query_impl(
                        [], tools=[], query_logger=logging.getLogger("test")
                    )

        assert result.tool_calls[0].name == "lookup"
        assert _require_performance(result.metadata).timeline[0].channel == "tool_call"
        assert _require_performance(result.metadata).timeline[0].end_ms == 250

    async def test_anthropic_zero_arg_tool_call_records_ready_without_first_token(self):
        model = AnthropicModel("claude-haiku-4-5-20251001")
        events = [
            SimpleNamespace(
                type="content_block_start",
                content_block=SimpleNamespace(type="tool_use", name="ping"),
            ),
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="input_json_delta", partial_json=""),
            ),
        ]
        message = SimpleNamespace(
            id="msg_1",
            model="claude-haiku-4-5-20251001",
            stop_reason="tool_use",
            usage=SimpleNamespace(
                input_tokens=3,
                output_tokens=2,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
                iterations=None,
            ),
            content=[
                SimpleNamespace(
                    type="tool_use",
                    id="toolu_1",
                    name="ping",
                    input={},
                ),
            ],
        )

        result = await self._run_query(model, events, message)

        assert result.tool_calls[0].name == "ping"
        assert result.tool_calls[0].args == {}
        performance = _require_performance(result.metadata)
        assert performance.timeline[0].channel == "tool_call"
        assert performance.timeline[0].ready_ms is not None
        assert performance.time_to_first_token_ms.tool_call is None
        assert performance.time_to_first_token_ms.answer is None


