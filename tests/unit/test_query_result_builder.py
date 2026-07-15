import pytest
from pydantic import ValidationError

from model_library.base import (
    QueryPerformanceEvent,
    QueryResultCost,
    QueryResultMetadata,
    QueryResultPerformance,
    QueryPerformanceTimelineEntry,
)
from model_library.base.output.builder import QueryResultBuilder
from model_library.base.output.result import (
    ProviderToolEvent,
    QueryResult,
)


def _require_performance(metadata: QueryResultMetadata) -> QueryResultPerformance:
    performance = metadata.performance
    assert performance is not None
    return performance


def _content_performance() -> QueryResultPerformance:
    return QueryResultPerformance(
        timeline=[
            QueryPerformanceTimelineEntry(
                channel="content",
                index=0,
                events=[
                    QueryPerformanceEvent(type="content_started", timestamp_ms=0),
                    QueryPerformanceEvent(type="content_delta", timestamp_ms=100),
                    QueryPerformanceEvent(type="content_finished", timestamp_ms=600),
                ],
            )
        ]
    )


class TestQueryResultBuilder:
    async def test_builder_records_channel_timeline_and_first_token_rollups(self):
        times = iter([0.0, 0.100, 0.250, 0.500])
        builder = QueryResultBuilder(clock=lambda: next(times))

        builder.append_reasoning_delta("thinking")
        builder.append_content_delta("answer")
        result = builder.build(metadata=QueryResultMetadata(in_tokens=3, out_tokens=2))

        assert result.output_text == "answer"
        assert result.reasoning == "thinking"
        assert result.metadata.in_tokens == 3
        assert result.metadata.out_tokens == 2
        assert _require_performance(result.metadata).time_to_first_token_ms.any == 100
        assert (
            _require_performance(result.metadata).time_to_first_token_ms.reasoning
            == 100
        )
        assert (
            _require_performance(result.metadata).time_to_first_token_ms.answer == 250
        )
        assert (
            _require_performance(result.metadata).time_to_first_token_ms.content == 250
        )
        assert [
            entry.channel for entry in _require_performance(result.metadata).timeline
        ] == [
            "reasoning",
            "content",
        ]
        assert (
            _require_performance(result.metadata).timeline[0].events[-1].type
            == "reasoning_finished"
        )
        assert (
            _require_performance(result.metadata).timeline[1].events[1].type
            == "content_delta"
        )

    async def test_builder_preserves_text_delta_offsets_when_final_text_matches_stream(
        self,
    ):
        times = iter([0.0, 0.100, 0.200, 0.300, 0.400, 0.500])
        builder = QueryResultBuilder(clock=lambda: next(times))

        builder.append_content_delta("hello ")
        builder.append_content_delta("world")
        builder.append_reasoning_delta("think")
        builder.append_reasoning_delta("ing")
        result = (
            builder.set_output_text("hello world").set_reasoning("thinking").build()
        )

        performance = _require_performance(result.metadata)
        content_delta_events = [
            event
            for entry in performance.timeline
            if entry.channel == "content"
            for event in entry.events
            if event.type == "content_delta"
        ]
        reasoning_delta_events = [
            event
            for entry in performance.timeline
            if entry.channel == "reasoning"
            for event in entry.events
            if event.type == "reasoning_delta"
        ]

        assert result.output_text == "hello world"
        assert result.reasoning == "thinking"
        assert [
            (event.channel_text_start_char, event.channel_text_end_char)
            for event in content_delta_events
        ] == [(0, 6), (6, 11)]
        assert [
            result.output_text[
                event.channel_text_start_char : event.channel_text_end_char
            ]
            for event in content_delta_events
        ] == ["hello ", "world"]
        assert [
            (event.channel_text_start_char, event.channel_text_end_char)
            for event in reasoning_delta_events
        ] == [(0, 5), (5, 8)]
        assert [
            result.reasoning[
                event.channel_text_start_char : event.channel_text_end_char
            ]
            for event in reasoning_delta_events
        ] == ["think", "ing"]

    async def test_builder_omits_text_delta_offsets_after_final_text_is_rewritten(
        self,
    ):
        times = iter([0.0, 0.100, 0.200, 0.300, 0.400, 0.500])
        result = (
            QueryResultBuilder(clock=lambda: next(times))
            .append_content_delta("raw chunk")
            .set_output_text("normalized")
            .append_content_delta("!")
            .append_reasoning_delta("raw thought")
            .set_reasoning("normalized thought")
            .append_reasoning_delta("!")
            .build()
        )

        performance = _require_performance(result.metadata)
        text_delta_events = [
            event
            for entry in performance.timeline
            for event in entry.events
            if event.type in {"content_delta", "reasoning_delta"}
        ]

        assert result.output_text == "normalized!"
        assert result.reasoning == "normalized thought!"
        assert [event.model_dump(mode="json") for event in text_delta_events] == [
            {"type": "content_delta", "timestamp_ms": 100},
            {"type": "content_delta", "timestamp_ms": 200},
            {"type": "reasoning_delta", "timestamp_ms": 300},
            {"type": "reasoning_delta", "timestamp_ms": 400},
        ]

    async def test_builder_closes_remaining_keyed_tool_call_segments_on_channel_switch(
        self,
    ):
        current_time = 0.0
        builder = QueryResultBuilder(clock=lambda: current_time)

        current_time = 0.100
        builder.start_tool_call_segment("call_1")
        current_time = 0.110
        builder.record_tool_call_ready("call_1")
        current_time = 0.120
        builder.start_tool_call_segment("call_2")
        current_time = 0.130
        builder.record_tool_call_ready("call_2")
        current_time = 0.150
        builder.record_tool_call_delta("call_1")
        current_time = 0.160
        builder.record_tool_call_delta("call_2")
        current_time = 0.200
        builder.finish_tool_call_segment("call_2")
        current_time = 1.000
        builder.append_content_delta("answer")
        current_time = 1.100

        performance = _require_performance(builder.build().metadata)

        assert [entry.channel for entry in performance.timeline] == [
            "tool_call",
            "tool_call",
            "content",
        ]
        assert [entry.end_ms for entry in performance.timeline] == [1000, 200, 1100]
        assert [event.type for event in performance.timeline[0].events] == [
            "tool_call_started",
            "tool_call_ready",
            "tool_call_delta",
            "tool_call_finished",
        ]
        assert [event.type for event in performance.timeline[1].events] == [
            "tool_call_started",
            "tool_call_ready",
            "tool_call_delta",
            "tool_call_finished",
        ]

    async def test_builder_timeline_overrides_supplied_performance(self):
        supplied_performance = QueryResultPerformance(
            timeline=[
                QueryPerformanceTimelineEntry(
                    channel="tool_call",
                    index=0,
                    events=[
                        QueryPerformanceEvent(
                            type="tool_call_started", timestamp_ms=10
                        ),
                        QueryPerformanceEvent(type="tool_call_delta", timestamp_ms=20),
                        QueryPerformanceEvent(
                            type="tool_call_finished", timestamp_ms=40
                        ),
                    ],
                )
            ]
        )
        metadata = QueryResultMetadata(
            in_tokens=3,
            out_tokens=2,
            performance=supplied_performance,
        )
        original_performance = _require_performance(metadata).model_copy(deep=True)
        times = iter([0.0, 0.100, 0.300])

        result = (
            QueryResultBuilder(clock=lambda: next(times))
            .append_content_delta("answer")
            .build(metadata=metadata)
        )

        result_performance = _require_performance(result.metadata)
        assert [entry.channel for entry in result_performance.timeline] == ["content"]
        assert result_performance.time_to_first_token_ms.content == 100
        assert metadata.performance == original_performance

    async def test_builder_non_streaming_setters_do_not_invent_timeline_events(self):
        builder = QueryResultBuilder(clock=lambda: 0.0)

        result = builder.set_output_text("answer").set_reasoning("thinking").build()

        assert result.output_text == "answer"
        assert result.reasoning == "thinking"
        assert result.metadata.performance is None

    async def test_builder_uses_none_for_unobserved_output_text_and_reasoning(self):
        result = QueryResultBuilder(clock=lambda: 0.0).record_tool_call_ready().build()

        assert result.output_text is None
        assert result.output_text_str == ""
        assert result.reasoning is None

    async def test_builder_normalizes_empty_output_text_and_reasoning_to_none(self):
        result = (
            QueryResultBuilder(clock=lambda: 0.0)
            .set_output_text("")
            .set_reasoning("")
            .build()
        )

        assert result.output_text is None
        assert result.reasoning is None

    async def test_query_result_only_normalizes_empty_strings(self):
        assert QueryResult(output_text="", reasoning="").output_text is None

        with pytest.raises(ValidationError):
            QueryResult(output_text=0)  # pyright: ignore[reportArgumentType]
        with pytest.raises(ValidationError):
            QueryResult(reasoning=False)  # pyright: ignore[reportArgumentType]

    async def test_builder_ignores_empty_content_delta(self):
        result = QueryResultBuilder(clock=lambda: 0.0).append_content_delta("").build()

        assert result.output_text is None
        assert result.metadata.performance is None

    async def test_builder_ignores_empty_reasoning_delta(self):
        result = (
            QueryResultBuilder(clock=lambda: 0.0).append_reasoning_delta("").build()
        )

        assert result.reasoning is None
        assert result.metadata.performance is None

    async def test_builder_reports_observed_text_and_reasoning(self):
        builder = QueryResultBuilder(clock=lambda: 0.0)

        assert not builder.has_output_text
        assert not builder.has_reasoning

        builder.set_output_text("")
        assert builder.output_text is None
        assert not builder.has_output_text
        builder.set_output_text("answer")
        assert builder.output_text == "answer"
        assert builder.has_output_text

        builder = QueryResultBuilder(clock=lambda: 0.0)
        builder.append_content_delta("answer")
        assert builder.output_text == "answer"
        assert builder.has_output_text

        builder = QueryResultBuilder(clock=lambda: 0.0)
        builder.append_reasoning_delta("thinking")
        assert builder.reasoning == "thinking"
        assert builder.has_reasoning

    async def test_builder_does_not_mutate_supplied_metadata(self):
        times = iter([0.0, 0.100, 0.200])
        metadata = QueryResultMetadata(in_tokens=3, out_tokens=2)
        original_performance = metadata.performance

        result = (
            QueryResultBuilder(clock=lambda: next(times))
            .append_content_delta("answer")
            .build(metadata=metadata)
        )

        assert metadata.performance == original_performance
        assert (
            _require_performance(result.metadata).time_to_first_token_ms.content == 100
        )

    async def test_builder_normalizes_supplied_empty_performance_without_builder_timeline(
        self,
    ):
        result = (
            QueryResultBuilder(clock=lambda: 0.0)
            .set_output_text("ok")
            .build(
                metadata=QueryResultMetadata(
                    in_tokens=3,
                    out_tokens=2,
                    performance=QueryResultPerformance(),
                )
            )
        )

        assert result.metadata.performance is None

    async def test_builder_preserves_supplied_performance_without_builder_timeline(
        self,
    ):
        supplied_performance = QueryResultPerformance(
            timeline=[
                QueryPerformanceTimelineEntry(
                    channel="content",
                    index=0,
                    events=[
                        QueryPerformanceEvent(type="content_started", timestamp_ms=10),
                        QueryPerformanceEvent(type="content_delta", timestamp_ms=20),
                        QueryPerformanceEvent(type="content_finished", timestamp_ms=40),
                    ],
                )
            ]
        )
        metadata = QueryResultMetadata(
            in_tokens=3,
            out_tokens=2,
            performance=supplied_performance,
        )

        original_performance = _require_performance(metadata).model_copy(deep=True)

        result = (
            QueryResultBuilder(clock=lambda: 0.0)
            .set_output_text("ok")
            .build(metadata=metadata)
        )

        result_performance = _require_performance(result.metadata)
        assert result_performance == original_performance
        assert result_performance is not metadata.performance
        assert metadata.performance == original_performance

    async def test_builder_does_not_alias_supplied_metadata_nested_fields(self):
        metadata = QueryResultMetadata(
            cost=QueryResultCost(input=1.0, output=2.0),
            extra={"provider": {"request_id": "original"}},
        )

        result = QueryResultBuilder(clock=lambda: 0.0).build(metadata=metadata)

        assert result.metadata.cost is not metadata.cost
        assert result.metadata.extra is not metadata.extra
        assert result.metadata.extra["provider"] is not metadata.extra["provider"]
        assert result.metadata.cost is not None
        result.metadata.cost.input = 99.0
        result.metadata.extra["provider"]["request_id"] = "changed"
        assert metadata.cost is not None
        assert metadata.cost.input == 1.0
        assert metadata.extra["provider"]["request_id"] == "original"

    async def test_builder_preserves_provider_tool_events(self):
        event = ProviderToolEvent(
            id="ws_1",
            provider="openai",
            type="web_search_call",
            name="web_search",
            status="completed",
            input="query",
            output=["https://example.com"],
            sequence=2,
        )

        result = QueryResultBuilder(clock=lambda: 0.0).build(
            provider_tool_events=[event]
        )

        assert result.provider_tool_events == [event]

    async def test_builder_is_single_use_after_build(self):
        builder = QueryResultBuilder(clock=lambda: 0.0)
        builder.append_content_delta("answer")

        builder.build()

        with pytest.raises(RuntimeError, match="QueryResultBuilder cannot be reused"):
            builder.append_content_delta(" again")
        with pytest.raises(RuntimeError, match="QueryResultBuilder cannot be reused"):
            builder.build()

    async def test_builder_splits_consecutive_tool_call_segments(self):
        builder = QueryResultBuilder(clock=lambda: 0.0)

        result = (
            builder.record_tool_call_ready()
            .record_tool_call_delta()
            .start_tool_call_segment()
            .record_tool_call_ready()
            .record_tool_call_delta()
            .build()
        )

        timeline = _require_performance(result.metadata).timeline
        assert [(entry.channel, entry.index) for entry in timeline] == [
            ("tool_call", 0),
            ("tool_call", 1),
        ]
        assert [[event.type for event in entry.events] for entry in timeline] == [
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

    async def test_builder_tracks_interleaved_keyed_tool_call_segments(self):
        builder = QueryResultBuilder(clock=lambda: 0.0)

        result = (
            builder.start_tool_call_segment("call_1")
            .record_tool_call_ready("call_1")
            .start_tool_call_segment("call_2")
            .record_tool_call_ready("call_2")
            .record_tool_call_delta("call_1")
            .record_tool_call_delta("call_2")
            .build()
        )

        timeline = _require_performance(result.metadata).timeline
        assert [(entry.channel, entry.index) for entry in timeline] == [
            ("tool_call", 0),
            ("tool_call", 1),
        ]
        assert [[event.type for event in entry.events] for entry in timeline] == [
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

    async def test_builder_records_tool_call_progress_at_same_timestamp(self):
        times = iter([0.0, 0.100, 0.200])
        builder = QueryResultBuilder(clock=lambda: next(times))

        assert builder.record_tool_call_progress("call_1") is False
        assert (
            builder.record_tool_call_progress("call_1", ready=True, delta=True) is True
        )
        result = builder.build()

        timeline_entry = _require_performance(result.metadata).timeline[0]
        assert [
            (event.type, event.timestamp_ms) for event in timeline_entry.events
        ] == [
            ("tool_call_started", 100),
            ("tool_call_ready", 100),
            ("tool_call_delta", 100),
            ("tool_call_finished", 200),
        ]
        assert timeline_entry.ready_ms == 100

    async def test_builder_splits_keyed_then_unkeyed_tool_call_segments(self):
        builder = QueryResultBuilder(clock=lambda: 0.0)

        result = (
            builder.start_tool_call_segment("call_1")
            .record_tool_call_ready("call_1")
            .record_tool_call_delta("call_1")
            .start_tool_call_segment()
            .record_tool_call_ready()
            .record_tool_call_delta()
            .build()
        )

        timeline = _require_performance(result.metadata).timeline
        assert [(entry.channel, entry.index) for entry in timeline] == [
            ("tool_call", 0),
            ("tool_call", 1),
        ]
        assert [[event.type for event in entry.events] for entry in timeline] == [
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

    async def test_builder_finish_current_tool_call_segment_closes_all_keyed_segments(
        self,
    ):
        current_time = 0.0
        builder = QueryResultBuilder(clock=lambda: current_time)

        current_time = 0.100
        builder.record_tool_call_delta("call_1")
        current_time = 0.200
        builder.record_tool_call_delta("call_2")
        current_time = 0.300
        builder.finish_current_segment("tool_call")
        current_time = 1.000
        result = builder.build()

        timeline = _require_performance(result.metadata).timeline
        assert [entry.end_ms for entry in timeline] == [300, 300]

    async def test_builder_finish_current_tool_call_closes_remaining_keyed_segments_after_latest_key_closed(
        self,
    ):
        current_time = 0.0
        builder = QueryResultBuilder(clock=lambda: current_time)

        current_time = 0.100
        builder.record_tool_call_delta("call_1")
        current_time = 0.200
        builder.record_tool_call_delta("call_2")
        current_time = 0.250
        builder.finish_tool_call_segment("call_2")
        current_time = 0.300
        builder.finish_current_segment("tool_call")
        current_time = 1.000
        result = builder.build()

        timeline = _require_performance(result.metadata).timeline
        assert [entry.end_ms for entry in timeline] == [300, 250]

    async def test_builder_splits_unkeyed_then_keyed_tool_call_segments(self):
        builder = QueryResultBuilder(clock=lambda: 0.0)

        result = (
            builder.start_tool_call_segment()
            .record_tool_call_ready()
            .record_tool_call_delta()
            .start_tool_call_segment("call_1")
            .record_tool_call_ready("call_1")
            .record_tool_call_delta("call_1")
            .build()
        )

        timeline = _require_performance(result.metadata).timeline
        assert [(entry.channel, entry.index) for entry in timeline] == [
            ("tool_call", 0),
            ("tool_call", 1),
        ]
        assert [[event.type for event in entry.events] for entry in timeline] == [
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

    async def test_builder_finishes_restarted_keyed_tool_call_segment(self):
        builder = QueryResultBuilder(clock=lambda: 0.0)

        result = (
            builder.start_tool_call_segment("call_1")
            .record_tool_call_delta("call_1")
            .start_tool_call_segment("call_1")
            .record_tool_call_delta("call_1")
            .build()
        )

        assert [
            [event.type for event in entry.events]
            for entry in _require_performance(result.metadata).timeline
        ] == [
            ["tool_call_started", "tool_call_delta", "tool_call_finished"],
            ["tool_call_started", "tool_call_delta", "tool_call_finished"],
        ]

    async def test_builder_closes_all_keyed_tool_call_segments_before_unkeyed_segment(
        self,
    ):
        times = iter([0.0, 0.100, 0.200, 0.300, 0.400])
        builder = QueryResultBuilder(clock=lambda: next(times))

        result = (
            builder.start_tool_call_segment("call_1")
            .start_tool_call_segment("call_2")
            .start_tool_call_segment()
            .build()
        )

        timeline = _require_performance(result.metadata).timeline
        assert [entry.channel for entry in timeline] == [
            "tool_call",
            "tool_call",
            "tool_call",
        ]
        assert [event.type for event in timeline[0].events] == [
            "tool_call_started",
            "tool_call_finished",
        ]
        assert [event.type for event in timeline[1].events] == [
            "tool_call_started",
            "tool_call_finished",
        ]
        assert [event.type for event in timeline[2].events] == [
            "tool_call_started",
            "tool_call_finished",
        ]
        first_keyed_end_ms = timeline[0].end_ms
        second_keyed_end_ms = timeline[1].end_ms
        unkeyed_end_ms = timeline[2].end_ms
        assert first_keyed_end_ms is not None
        assert second_keyed_end_ms == first_keyed_end_ms
        assert unkeyed_end_ms is not None
        assert unkeyed_end_ms > first_keyed_end_ms

    async def test_metadata_has_no_performance_without_timeline(self):
        metadata = QueryResultMetadata(
            duration_seconds=2.0,
            out_tokens=10,
            reasoning_tokens=4,
        )

        assert metadata.performance is None
