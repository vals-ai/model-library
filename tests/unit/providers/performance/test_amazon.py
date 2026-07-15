import logging
from unittest.mock import AsyncMock, MagicMock, patch


from model_library.base import QueryResultMetadata, QueryResultPerformance
from model_library.base.output.builder import QueryResultBuilder
from model_library.providers.amazon import AmazonModel


def _require_performance(metadata: QueryResultMetadata) -> QueryResultPerformance:
    performance = metadata.performance
    assert performance is not None
    return performance


class TestAmazonStreamingPerformance:
    async def test_amazon_builder_starts_before_stream_open(self):
        model = AmazonModel("anthropic.claude-3-5-haiku-2024-10-22-v2:0")
        order: list[str] = []
        response = {
            "stream": [
                {"messageStart": {"role": "assistant"}},
                {"contentBlockDelta": {"delta": {"text": "answer"}}},
                {"contentBlockStop": {}},
                {
                    "metadata": {
                        "usage": {
                            "inputTokens": 1,
                            "outputTokens": 1,
                        }
                    }
                },
                {"messageStop": {"stopReason": "end_turn"}},
            ]
        }

        def converse_stream(**_kwargs: object):
            order.append("stream-opened")
            return response

        def make_builder() -> QueryResultBuilder:
            order.append("builder-created")
            return QueryResultBuilder()

        mock_client = MagicMock()
        mock_client.converse_stream = MagicMock(side_effect=converse_stream)

        with patch.object(model, "get_client", return_value=mock_client):
            with patch.object(
                model, "build_body", new_callable=AsyncMock, return_value={}
            ):
                with patch(
                    "model_library.providers.amazon.QueryResultBuilder",
                    side_effect=make_builder,
                ):
                    result = await model._query_impl(
                        [], tools=[], query_logger=logging.getLogger("test")
                    )

        assert result.output_text == "answer"
        assert order[:2] == ["builder-created", "stream-opened"]

    async def test_amazon_content_block_stop_closes_timeline_segment(self):
        model = AmazonModel("anthropic.claude-3-5-haiku-2024-10-22-v2:0")
        clock_time = {"now": 0.0}

        class TimedAmazonStream:
            def __iter__(self):
                events = [
                    (0.100, {"contentBlockDelta": {"delta": {"text": "answer"}}}),
                    (0.250, {"contentBlockStop": {}}),
                    (
                        1.000,
                        {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1}}},
                    ),
                    (1.000, {"messageStop": {"stopReason": "end_turn"}}),
                ]
                for timestamp, event in events:
                    clock_time["now"] = timestamp
                    yield event

        response = {"stream": TimedAmazonStream()}
        messages, _stop_reason, metadata, builder = await model.stream_response(
            response,
            result_builder=QueryResultBuilder(clock=lambda: clock_time["now"]),
        )
        result = builder.set_output_text("answer").build(metadata=metadata)

        assert messages["content"][0]["text"] == "answer"
        assert _require_performance(result.metadata).timeline[0].end_ms == 250

    async def test_amazon_tool_block_stop_closes_timeline_segment(self):
        model = AmazonModel("anthropic.claude-3-5-haiku-2024-10-22-v2:0")
        clock_time = {"now": 0.0}

        class TimedAmazonStream:
            def __iter__(self):
                events = [
                    (
                        0.100,
                        {
                            "contentBlockStart": {
                                "start": {
                                    "toolUse": {
                                        "toolUseId": "toolu_1",
                                        "name": "lookup",
                                    }
                                }
                            }
                        },
                    ),
                    (
                        0.150,
                        {"contentBlockDelta": {"delta": {"toolUse": {"input": "{}"}}}},
                    ),
                    (0.250, {"contentBlockStop": {}}),
                    (
                        1.000,
                        {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1}}},
                    ),
                    (1.000, {"messageStop": {"stopReason": "tool_use"}}),
                ]
                for timestamp, event in events:
                    clock_time["now"] = timestamp
                    yield event

        response = {"stream": TimedAmazonStream()}
        messages, _stop_reason, metadata, builder = await model.stream_response(
            response,
            result_builder=QueryResultBuilder(clock=lambda: clock_time["now"]),
        )
        result = builder.build(metadata=metadata)

        assert messages["content"][0]["toolUse"]["name"] == "lookup"
        assert _require_performance(result.metadata).timeline[0].channel == "tool_call"
        assert _require_performance(result.metadata).timeline[0].end_ms == 250

    async def test_amazon_stream_response_populates_performance_timeline(self):
        model = AmazonModel("anthropic.claude-3-5-haiku-2024-10-22-v2:0")
        response = {
            "stream": [
                {"messageStart": {"role": "assistant"}},
                {
                    "contentBlockDelta": {
                        "delta": {"reasoningContent": {"text": "thinking"}}
                    }
                },
                {"contentBlockDelta": {"delta": {"text": "answer"}}},
                {
                    "contentBlockStart": {
                        "start": {"toolUse": {"toolUseId": "toolu_1", "name": "lookup"}}
                    }
                },
                {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"q": 1}'}}}},
                {"contentBlockStop": {}},
                {
                    "metadata": {
                        "usage": {
                            "inputTokens": 3,
                            "outputTokens": 2,
                        }
                    }
                },
                {"messageStop": {"stopReason": "tool_use"}},
            ]
        }

        messages, stop_reason, metadata, builder = await model.stream_response(response)
        result = (
            builder.set_output_text("answer")
            .set_reasoning("thinking")
            .build(
                finish_reason=None,
                metadata=metadata,
            )
        )

        assert stop_reason == "tool_use"
        assert messages["content"][0]["toolUse"]["name"] == "lookup"
        assert result.metadata.in_tokens == 3
        assert result.metadata.out_tokens == 2
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

    async def test_amazon_zero_arg_tool_call_defaults_empty_input_to_dict(self):
        model = AmazonModel("anthropic.claude-3-5-haiku-2024-10-22-v2:0")
        response = {
            "stream": [
                {
                    "contentBlockStart": {
                        "start": {"toolUse": {"toolUseId": "toolu_1", "name": "ping"}}
                    }
                },
                {"contentBlockDelta": {"delta": {"toolUse": {"input": ""}}}},
                {"contentBlockStop": {}},
                {"messageStop": {"stopReason": "tool_use"}},
            ]
        }

        messages, stop_reason, metadata, builder = await model.stream_response(response)
        result = builder.build(metadata=metadata)

        assert stop_reason == "tool_use"
        assert messages["content"][0]["toolUse"]["input"] == {}
        performance = _require_performance(result.metadata)
        assert performance.timeline[0].channel == "tool_call"
        assert performance.timeline[0].ready_ms is not None
        assert performance.time_to_first_token_ms.tool_call is None
        assert performance.time_to_first_token_ms.answer is None


