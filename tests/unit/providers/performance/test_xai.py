import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from model_library.base import QueryResultMetadata, QueryResultPerformance
from model_library.providers.xai import XAIModel


def _require_performance(metadata: QueryResultMetadata) -> QueryResultPerformance:
    performance = metadata.performance
    assert performance is not None
    return performance


class _XAIChat:
    def __init__(self, pairs: list[tuple[SimpleNamespace, SimpleNamespace]]):
        self._pairs = pairs

    async def stream(self):
        for pair in self._pairs:
            yield pair


class TestXAIStreamingPerformance:
    @staticmethod
    async def _run_query(
        model: XAIModel, pairs: list[tuple[SimpleNamespace, SimpleNamespace]]
    ):
        mock_client = MagicMock()
        mock_client.chat.create = MagicMock(return_value=_XAIChat(pairs))

        with patch.object(model, "get_client", return_value=mock_client):
            with patch.object(
                model, "build_body", new_callable=AsyncMock, return_value={}
            ):
                return await model._query_impl(
                    [], tools=[], query_logger=logging.getLogger("test")
                )

    async def test_xai_streaming_chunks_populate_performance_timeline(self):
        model = XAIModel("grok-3-mini")
        usage = SimpleNamespace(
            prompt_tokens=3,
            cached_prompt_text_tokens=0,
            completion_tokens=2,
            reasoning_tokens=1,
        )
        final_response = SimpleNamespace(
            content="answer",
            reasoning_content="thinking",
            finish_reason="stop",
            tool_calls=[],
            usage=usage,
        )
        pairs = [
            (
                SimpleNamespace(
                    content="",
                    reasoning_content="thinking",
                    finish_reason=None,
                    tool_calls=[],
                    usage=usage,
                ),
                SimpleNamespace(
                    content="", reasoning_content="thinking", tool_calls=[]
                ),
            ),
            (
                final_response,
                SimpleNamespace(content="answer", reasoning_content="", tool_calls=[]),
            ),
        ]

        result = await self._run_query(model, pairs)

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

    async def test_xai_streaming_falls_back_to_final_response_content(self):
        model = XAIModel("grok-3-mini")
        usage = SimpleNamespace(
            prompt_tokens=3,
            cached_prompt_text_tokens=0,
            completion_tokens=2,
            reasoning_tokens=1,
        )
        final_response = SimpleNamespace(
            content="answer",
            reasoning_content="thinking",
            finish_reason="stop",
            tool_calls=[],
            usage=usage,
        )
        pairs = [
            (
                final_response,
                SimpleNamespace(content="", reasoning_content="", tool_calls=[]),
            ),
        ]

        result = await self._run_query(model, pairs)

        assert result.output_text == "answer"
        assert result.reasoning == "thinking"
        assert result.metadata.performance is None

    async def test_xai_streaming_keeps_content_delta_when_final_content_is_empty(self):
        model = XAIModel("grok-3-mini")
        usage = SimpleNamespace(
            prompt_tokens=3,
            cached_prompt_text_tokens=0,
            completion_tokens=2,
            reasoning_tokens=0,
        )
        final_response = SimpleNamespace(
            content="",
            reasoning_content="",
            finish_reason="REASON_STOP",
            tool_calls=[],
            usage=usage,
        )
        pairs = [
            (
                final_response,
                SimpleNamespace(content="hello", reasoning_content="", tool_calls=[]),
            ),
        ]

        result = await self._run_query(model, pairs)

        assert result.output_text == "hello"

    async def test_xai_streaming_tool_call_populates_performance_timeline(self):
        model = XAIModel("grok-3-mini")
        usage = SimpleNamespace(
            prompt_tokens=3,
            cached_prompt_text_tokens=0,
            completion_tokens=2,
            reasoning_tokens=0,
        )
        raw_tool_call = SimpleNamespace(
            id="tool-1",
            type=1,
            function=SimpleNamespace(name="ping", arguments="{}"),
        )
        final_response = SimpleNamespace(
            content=None,
            reasoning_content="",
            finish_reason="REASON_TOOL_CALLS",
            tool_calls=[raw_tool_call],
            usage=usage,
        )
        pairs = [
            (
                final_response,
                SimpleNamespace(
                    content=None, reasoning_content=None, tool_calls=[raw_tool_call]
                ),
            )
        ]

        result = await self._run_query(model, pairs)

        assert result.output_text is None
        assert result.tool_calls[0].name == "ping"
        assert [
            entry.channel for entry in _require_performance(result.metadata).timeline
        ] == ["tool_call"]
        assert (
            _require_performance(result.metadata).time_to_first_token_ms.tool_call
            is not None
        )

    @pytest.mark.parametrize("first_chunk_id", ["tool-1", ""])
    async def test_xai_streaming_same_tool_call_chunks_share_one_timeline_segment(
        self, first_chunk_id: str
    ):
        model = XAIModel("grok-3-mini")
        usage = SimpleNamespace(
            prompt_tokens=3,
            cached_prompt_text_tokens=0,
            completion_tokens=2,
            reasoning_tokens=0,
        )
        final_tool_call = SimpleNamespace(
            id="tool-1",
            type=1,
            function=SimpleNamespace(name="ping", arguments='{"a":1}'),
        )
        chunk_tool_calls = [
            SimpleNamespace(
                id=first_chunk_id,
                type=1,
                function=SimpleNamespace(name="ping", arguments='{"a"'),
            ),
            SimpleNamespace(
                id="tool-1",
                type=1,
                function=SimpleNamespace(name="ping", arguments=":1}"),
            ),
        ]
        pairs = [
            (
                SimpleNamespace(
                    content=None,
                    reasoning_content="",
                    finish_reason=None,
                    tool_calls=[],
                    usage=usage,
                ),
                SimpleNamespace(
                    content=None,
                    reasoning_content=None,
                    tool_calls=[chunk_tool_calls[0]],
                ),
            ),
            (
                SimpleNamespace(
                    content=None,
                    reasoning_content="",
                    finish_reason="REASON_TOOL_CALLS",
                    tool_calls=[final_tool_call],
                    usage=usage,
                ),
                SimpleNamespace(
                    content=None,
                    reasoning_content=None,
                    tool_calls=[chunk_tool_calls[1]],
                ),
            ),
        ]

        result = await self._run_query(model, pairs)

        assert [(tool_call.id, tool_call.args) for tool_call in result.tool_calls] == [
            ("tool-1", '{"a":1}')
        ]
        assert [
            entry.channel for entry in _require_performance(result.metadata).timeline
        ] == ["tool_call"]

    async def test_xai_streaming_multiple_complete_tool_calls_split_timeline(self):
        model = XAIModel("grok-3-mini")
        usage = SimpleNamespace(
            prompt_tokens=3,
            cached_prompt_text_tokens=0,
            completion_tokens=2,
            reasoning_tokens=0,
        )
        raw_tool_calls = [
            SimpleNamespace(
                id="tool-1",
                type=1,
                function=SimpleNamespace(name="ping", arguments="{}"),
            ),
            SimpleNamespace(
                id="tool-2",
                type=1,
                function=SimpleNamespace(name="pong", arguments='{"ok": true}'),
            ),
        ]
        final_response = SimpleNamespace(
            content="",
            reasoning_content="",
            finish_reason="REASON_TOOL_CALLS",
            tool_calls=raw_tool_calls,
            usage=usage,
        )
        pairs = [
            (
                final_response,
                SimpleNamespace(
                    content=None, reasoning_content=None, tool_calls=raw_tool_calls
                ),
            )
        ]

        result = await self._run_query(model, pairs)

        assert [tool_call.name for tool_call in result.tool_calls] == ["ping", "pong"]
        assert [
            entry.channel for entry in _require_performance(result.metadata).timeline
        ] == [
            "tool_call",
            "tool_call",
        ]


