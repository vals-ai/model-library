import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch


from model_library.base import QueryResultMetadata, QueryResultPerformance
from model_library.base.output.builder import QueryResultBuilder
from model_library.providers.mistral import MistralModel


def _require_performance(metadata: QueryResultMetadata) -> QueryResultPerformance:
    performance = metadata.performance
    assert performance is not None
    return performance


class TestMistralStreamingPerformance:
    @staticmethod
    async def _run_query(model: MistralModel, chunks: list[SimpleNamespace]):
        async def mock_stream():
            for chunk in chunks:
                yield chunk

        mock_client = MagicMock()
        mock_client.chat.stream_async = AsyncMock(return_value=mock_stream())

        with patch.object(model, "get_client", return_value=mock_client):
            with patch.object(
                model, "build_body", new_callable=AsyncMock, return_value={}
            ):
                return await model._query_impl(
                    [], tools=[], query_logger=logging.getLogger("test")
                )

    async def test_mistral_streaming_content_populates_performance_timeline(self):
        from mistralai.client.models import TextChunk, ThinkChunk

        model = MistralModel("mistral-small-latest")
        chunks = [
            SimpleNamespace(
                data=SimpleNamespace(
                    id="mistral-performance-response",
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                content=[
                                    ThinkChunk(
                                        thinking=[
                                            TextChunk(text="thinking", type="text")
                                        ],
                                        closed=None,
                                        type="thinking",
                                    ),
                                    TextChunk(text="answer", type="text"),
                                ],
                                tool_calls=None,
                            ),
                            finish_reason="stop",
                        )
                    ],
                    usage=None,
                )
            )
        ]

        result = await self._run_query(model, chunks)

        assert result.reasoning == "thinking"
        assert result.output_text == "answer"
        assert [
            entry.channel for entry in _require_performance(result.metadata).timeline
        ] == [
            "reasoning",
            "content",
        ]
        assert (
            _require_performance(result.metadata).time_to_first_token_ms.reasoning
            is not None
        )
        assert (
            _require_performance(result.metadata).time_to_first_token_ms.content
            is not None
        )

    async def test_mistral_streaming_tool_call_populates_performance_timeline(self):
        from mistralai.client.models import FunctionCall, ToolCall as MistralToolCall

        raw_tool_call = MistralToolCall(
            id="tool-1",
            function=FunctionCall(name="ping", arguments="{}"),
        )
        model = MistralModel("mistral-small-latest")
        chunks = [
            SimpleNamespace(
                data=SimpleNamespace(
                    id="mistral-performance-response",
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                content=None, tool_calls=[raw_tool_call]
                            ),
                            finish_reason="tool_calls",
                        )
                    ],
                    usage=None,
                )
            )
        ]

        result = await self._run_query(model, chunks)

        assert result.output_text is None
        assert result.tool_calls[0].name == "ping"
        assert [
            entry.channel for entry in _require_performance(result.metadata).timeline
        ] == ["tool_call"]
        assert (
            _require_performance(result.metadata).time_to_first_token_ms.tool_call
            is not None
        )

    async def test_mistral_streaming_chunks_for_same_index_share_timeline(self):
        from mistralai.client.models import FunctionCall, ToolCall as MistralToolCall

        model = MistralModel("mistral-small-latest")
        chunks = [
            SimpleNamespace(
                data=SimpleNamespace(
                    id="mistral-performance-response",
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                content=None,
                                tool_calls=[
                                    MistralToolCall(
                                        index=0,
                                        function=FunctionCall(
                                            name="ping", arguments='{"a"'
                                        ),
                                    )
                                ],
                            ),
                            finish_reason=None,
                        )
                    ],
                    usage=None,
                )
            ),
            SimpleNamespace(
                data=SimpleNamespace(
                    id="mistral-performance-response",
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                content=None,
                                tool_calls=[
                                    MistralToolCall(
                                        index=0,
                                        function=FunctionCall(
                                            name="ping", arguments=": 1}"
                                        ),
                                    )
                                ],
                            ),
                            finish_reason="tool_calls",
                        )
                    ],
                    usage=None,
                )
            ),
        ]

        result = await self._run_query(model, chunks)

        timeline = _require_performance(result.metadata).timeline
        assert [entry.channel for entry in timeline] == ["tool_call"]
        assert [event.type for event in timeline[0].events] == [
            "tool_call_started",
            "tool_call_delta",
            "tool_call_delta",
            "tool_call_finished",
        ]

    async def test_mistral_streaming_late_id_with_same_index_shares_timeline(self):
        from mistralai.client.models import FunctionCall, ToolCall as MistralToolCall

        model = MistralModel("mistral-small-latest")
        chunks = [
            SimpleNamespace(
                data=SimpleNamespace(
                    id="mistral-performance-response",
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                content=None,
                                tool_calls=[
                                    MistralToolCall(
                                        index=0,
                                        function=FunctionCall(
                                            name="ping", arguments='{"a"'
                                        ),
                                    )
                                ],
                            ),
                            finish_reason=None,
                        )
                    ],
                    usage=None,
                )
            ),
            SimpleNamespace(
                data=SimpleNamespace(
                    id="mistral-performance-response",
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                content=None,
                                tool_calls=[
                                    MistralToolCall(
                                        id="tool-1",
                                        index=0,
                                        function=FunctionCall(
                                            name="ping", arguments=": 1}"
                                        ),
                                    )
                                ],
                            ),
                            finish_reason="tool_calls",
                        )
                    ],
                    usage=None,
                )
            ),
        ]

        result = await self._run_query(model, chunks)

        timeline = _require_performance(result.metadata).timeline
        assert [entry.channel for entry in timeline] == ["tool_call"]
        assert [event.type for event in timeline[0].events] == [
            "tool_call_started",
            "tool_call_delta",
            "tool_call_delta",
            "tool_call_finished",
        ]

    async def test_mistral_streaming_tool_calls_without_id_or_index_do_not_merge_timeline(
        self,
    ):
        from mistralai.client.models import FunctionCall, ToolCall as MistralToolCall

        raw_tool_calls = [
            MistralToolCall(function=FunctionCall(name="ping", arguments="{}")),
            MistralToolCall(
                function=FunctionCall(name="pong", arguments='{"ok": true}')
            ),
        ]
        model = MistralModel("mistral-small-latest")
        chunks = [
            SimpleNamespace(
                data=SimpleNamespace(
                    id="mistral-performance-response",
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                content=None, tool_calls=[raw_tool_calls[0]]
                            ),
                            finish_reason=None,
                        )
                    ],
                    usage=None,
                )
            ),
            SimpleNamespace(
                data=SimpleNamespace(
                    id="mistral-performance-response",
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                content=None, tool_calls=[raw_tool_calls[1]]
                            ),
                            finish_reason="tool_calls",
                        )
                    ],
                    usage=None,
                )
            ),
        ]

        result = await self._run_query(model, chunks)

        assert [tool_call.name for tool_call in result.tool_calls] == ["ping", "pong"]
        assert [
            entry.channel for entry in _require_performance(result.metadata).timeline
        ] == [
            "tool_call",
            "tool_call",
        ]

    async def test_mistral_streaming_multiple_complete_tool_calls_split_timeline(self):
        from mistralai.client.models import FunctionCall, ToolCall as MistralToolCall

        raw_tool_calls = [
            MistralToolCall(
                id="tool-1",
                function=FunctionCall(name="ping", arguments="{}"),
            ),
            MistralToolCall(
                id="tool-2",
                function=FunctionCall(name="pong", arguments='{"ok": true}'),
            ),
        ]
        model = MistralModel("mistral-small-latest")
        chunks = [
            SimpleNamespace(
                data=SimpleNamespace(
                    id="mistral-performance-response",
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                content=None, tool_calls=raw_tool_calls
                            ),
                            finish_reason="tool_calls",
                        )
                    ],
                    usage=None,
                )
            )
        ]

        result = await self._run_query(model, chunks)

        assert [tool_call.name for tool_call in result.tool_calls] == ["ping", "pong"]
        assert [
            entry.channel for entry in _require_performance(result.metadata).timeline
        ] == [
            "tool_call",
            "tool_call",
        ]

    async def test_mistral_builder_starts_before_stream_open(self):
        model = MistralModel("mistral-small-latest")
        order: list[str] = []
        chunks = [
            SimpleNamespace(
                data=SimpleNamespace(
                    id="mistral-performance-response",
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content="answer", tool_calls=None),
                            finish_reason="stop",
                        )
                    ],
                    usage=None,
                )
            )
        ]

        async def mock_stream():
            for chunk in chunks:
                yield chunk

        async def stream_async(**_kwargs: object):
            order.append("stream-opened")
            return mock_stream()

        def make_builder() -> QueryResultBuilder:
            order.append("builder-created")
            return QueryResultBuilder()

        mock_client = MagicMock()
        mock_client.chat.stream_async = AsyncMock(side_effect=stream_async)

        with patch.object(model, "get_client", return_value=mock_client):
            with patch.object(
                model, "build_body", new_callable=AsyncMock, return_value={}
            ):
                with patch(
                    "model_library.providers.mistral.QueryResultBuilder",
                    side_effect=make_builder,
                ):
                    result = await model._query_impl(
                        [], tools=[], query_logger=logging.getLogger("test")
                    )

        assert result.output_text == "answer"
        assert order[:2] == ["builder-created", "stream-opened"]
