import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from google.genai.types import FinishReason

from model_library.base import QueryResultMetadata, QueryResultPerformance
from model_library.providers.google import GoogleModel


def _require_performance(metadata: QueryResultMetadata) -> QueryResultPerformance:
    performance = metadata.performance
    assert performance is not None
    return performance


class TestGoogleStreamingPerformance:
    @staticmethod
    def _make_chunk(*, text: str, thought: bool = False):
        part = SimpleNamespace(function_call=None, text=text, thought=thought)
        content = SimpleNamespace(parts=[part])
        candidate = SimpleNamespace(content=content, finish_reason=FinishReason.STOP, grounding_metadata=None)
        return SimpleNamespace(
            response_id=None, candidates=[candidate], usage_metadata=None
        )

    @staticmethod
    async def _run_query(model: GoogleModel, chunks: list[SimpleNamespace]):
        async def mock_stream():
            for chunk in chunks:
                yield chunk

        mock_client = MagicMock()
        mock_client.aio.models.generate_content_stream = AsyncMock(
            return_value=mock_stream()
        )

        with patch.object(model, "get_client", return_value=mock_client):
            with patch.object(
                model, "build_body", new_callable=AsyncMock, return_value={}
            ):
                return await model._query_impl(
                    [], tools=[], query_logger=logging.getLogger("test")
                )

    async def test_google_streaming_parts_populate_performance_timeline(self):
        model = GoogleModel("gemini-2.5-flash-lite")
        chunks = [
            self._make_chunk(text="thinking", thought=True),
            self._make_chunk(text="answer"),
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

    async def test_google_streaming_tool_call_populates_performance_timeline(self):
        model = GoogleModel("gemini-2.5-flash-lite")
        part = SimpleNamespace(
            function_call=SimpleNamespace(id="tool-1", name="ping", args={}),
            text=None,
            thought=False,
        )
        content = SimpleNamespace(parts=[part])
        candidate = SimpleNamespace(content=content, finish_reason=FinishReason.STOP, grounding_metadata=None)
        chunks = [
            SimpleNamespace(
                response_id=None, candidates=[candidate], usage_metadata=None
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

    async def test_google_streaming_multiple_complete_tool_calls_split_timeline(self):
        model = GoogleModel("gemini-2.5-flash-lite")
        parts = [
            SimpleNamespace(
                function_call=SimpleNamespace(id="tool-1", name="ping", args={}),
                text=None,
                thought=False,
            ),
            SimpleNamespace(
                function_call=SimpleNamespace(
                    id="tool-2", name="pong", args={"ok": True}
                ),
                text=None,
                thought=False,
            ),
        ]
        content = SimpleNamespace(parts=parts)
        candidate = SimpleNamespace(content=content, finish_reason=FinishReason.STOP, grounding_metadata=None)
        chunks = [
            SimpleNamespace(
                response_id=None, candidates=[candidate], usage_metadata=None
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
