import pytest
from pydantic import ValidationError

from model_library.base import (
    QueryPerformanceEvent,
    QueryResultCost,
    QueryResultMetadata,
    QueryResultPerformance,
    QueryPerformanceTimelineEntry,
    QueryTokensPerSecond,
)


class TestQueryResultCostAddition:
    async def test_add_full_costs(self):
        cost1 = QueryResultCost(
            input=0.01,
            output=0.02,
            reasoning=0.005,
            cache_read=0.001,
            cache_write=0.002,
        )
        cost2 = QueryResultCost(
            input=0.02,
            output=0.03,
            reasoning=0.01,
            cache_read=0.002,
            cache_write=0.003,
        )

        result = cost1 + cost2

        assert result.input == 0.03
        assert result.output == 0.05
        assert result.reasoning == 0.015
        assert result.cache_read == 0.003
        assert result.cache_write == 0.005

    async def test_add_costs_with_none_fields(self):
        cost1 = QueryResultCost(input=0.01, output=0.02, reasoning=0.005)
        cost2 = QueryResultCost(input=0.02, output=0.03, cache_read=0.002)

        result = cost1 + cost2

        assert result.input == 0.03
        assert result.output == 0.05
        assert result.reasoning == 0.005
        assert result.cache_read == 0.002
        assert result.cache_write is None

    async def test_add_costs_both_none_fields(self):
        cost1 = QueryResultCost(input=0.01, output=0.02)
        cost2 = QueryResultCost(input=0.02, output=0.03)

        result = cost1 + cost2

        assert result.input == 0.03
        assert result.output == 0.05
        assert result.reasoning is None
        assert result.cache_read is None
        assert result.cache_write is None

    async def test_add_costs_one_none(self):
        cost1 = QueryResultCost(
            input=0.01, output=0.02, reasoning=0.005, cache_write=0.001
        )
        cost2 = QueryResultCost(input=0.01, output=0.02, cache_read=0.001)

        result = cost1 + cost2

        assert result.input == 0.02
        assert result.output == 0.04
        assert result.reasoning == 0.005
        assert result.cache_read == 0.001
        assert result.cache_write == 0.001

    async def test_cost_total_computed(self):
        cost1 = QueryResultCost(input=0.01, output=0.02, reasoning=0.005)
        cost2 = QueryResultCost(input=0.02, output=0.03, cache_read=0.001)

        result = cost1 + cost2

        expected_total = 0.03 + 0.05 + 0.005 + 0.001
        assert abs(result.total - expected_total) < 1e-10

    async def test_cost_fields_round_float_artifacts_to_12_decimals(self):
        cost = QueryResultCost(
            input=2.6250000000000003e-07,
            output=2.1000000000000002e-06,
            reasoning=7.000000000000001e-07,
            cache_read=4.0000000000000003e-07,
            cache_write=1.4000000000000001e-06,
            total_override=3.7500000000000005e-06,
        )

        assert repr(cost.input) == "2.625e-07"
        assert repr(cost.output) == "2.1e-06"
        assert repr(cost.reasoning) == "7e-07"
        assert repr(cost.cache_read) == "4e-07"
        assert repr(cost.cache_write) == "1.4e-06"
        assert repr(cost.total_override) == "3.75e-06"

    async def test_cost_addition_rounds_accumulated_artifacts(self):
        cost = QueryResultCost(input=2.25e-07, output=1.5e-07)
        total = QueryResultCost(input=0.0, output=0.0)

        for _ in range(10):
            total += cost

        assert repr(total.total) == "3.75e-06"


class TestQueryResultPerformance:
    async def test_metadata_accepts_nested_timeline_performance_dict(self):
        metadata = QueryResultMetadata.model_validate(
            {
                "performance": {
                    "tokens_per_second": {
                        "reasoning": 7.8912,
                        "content": 12.3456,
                        "tool_call": None,
                    },
                    "timeline": [
                        {
                            "channel": "content",
                            "index": 0,
                            "start_ms": 700,
                            "first_token_ms": 760,
                            "end_ms": 1300,
                            "events": [
                                {"type": "content_started", "timestamp_ms": 700},
                                {"type": "content_delta", "timestamp_ms": 760},
                                {"type": "content_finished", "timestamp_ms": 1300},
                            ],
                        },
                        {
                            "channel": "tool_call",
                            "index": 0,
                            "start_ms": 1500,
                            "first_token_ms": 1580,
                            "ready_ms": 2100,
                            "end_ms": 2100,
                            "events": [
                                {"type": "tool_call_started", "timestamp_ms": 1500},
                                {"type": "tool_call_delta", "timestamp_ms": 1580},
                                {"type": "tool_call_ready", "timestamp_ms": 2100},
                                {"type": "tool_call_finished", "timestamp_ms": 2100},
                            ],
                        },
                        {
                            "channel": "content",
                            "index": 1,
                            "start_ms": 2600,
                            "first_token_ms": 2660,
                            "end_ms": 4000,
                            "events": [
                                {"type": "content_started", "timestamp_ms": 2600},
                                {"type": "content_delta", "timestamp_ms": 2660},
                                {"type": "content_finished", "timestamp_ms": 4000},
                            ],
                        },
                    ],
                }
            }
        )

        assert metadata.performance.time_to_first_token_ms.any == 760
        assert metadata.performance.time_to_first_token_ms.answer == 760
        assert metadata.performance.time_to_first_token_ms.reasoning is None
        assert metadata.performance.time_to_first_token_ms.content == 760
        assert metadata.performance.time_to_first_token_ms.tool_call == 1580
        assert metadata.performance.tokens_per_second == QueryTokensPerSecond(
            reasoning=7.891,
            content=12.346,
            tool_call=None,
        )
        assert metadata.model_dump()["performance"] == {
            "tokens_per_second": {
                "reasoning": 7.891,
                "content": 12.346,
                "tool_call": None,
            },
            "timeline": [
                {
                    "channel": "content",
                    "index": 0,
                    "start_ms": 700,
                    "first_token_ms": 760,
                    "ready_ms": None,
                    "end_ms": 1300,
                    "events": [
                        {"type": "content_started", "timestamp_ms": 700},
                        {"type": "content_delta", "timestamp_ms": 760},
                        {"type": "content_finished", "timestamp_ms": 1300},
                    ],
                    "duration_ms": 600,
                },
                {
                    "channel": "tool_call",
                    "index": 0,
                    "start_ms": 1500,
                    "first_token_ms": 1580,
                    "ready_ms": 2100,
                    "end_ms": 2100,
                    "events": [
                        {"type": "tool_call_started", "timestamp_ms": 1500},
                        {"type": "tool_call_delta", "timestamp_ms": 1580},
                        {"type": "tool_call_ready", "timestamp_ms": 2100},
                        {"type": "tool_call_finished", "timestamp_ms": 2100},
                    ],
                    "duration_ms": 600,
                },
                {
                    "channel": "content",
                    "index": 1,
                    "start_ms": 2600,
                    "first_token_ms": 2660,
                    "ready_ms": None,
                    "end_ms": 4000,
                    "events": [
                        {"type": "content_started", "timestamp_ms": 2600},
                        {"type": "content_delta", "timestamp_ms": 2660},
                        {"type": "content_finished", "timestamp_ms": 4000},
                    ],
                    "duration_ms": 1400,
                },
            ],
            "time_to_first_token_ms": {
                "any": 760,
                "answer": 760,
                "reasoning": None,
                "content": 760,
                "tool_call": 1580,
            },
        }

    async def test_metadata_addition_does_not_aggregate_query_performance(self):
        meta1 = QueryResultMetadata(
            duration_seconds=1.0,
            performance=QueryResultPerformance(
                tokens_per_second=QueryTokensPerSecond(content=10.0),
                timeline=[
                    QueryPerformanceTimelineEntry(
                        channel="content",
                        index=0,
                        start_ms=100,
                        first_token_ms=120,
                        end_ms=200,
                        events=[
                            QueryPerformanceEvent(
                                type="content_delta",
                                timestamp_ms=120,
                            )
                        ],
                    )
                ],
            ),
        )
        meta2 = QueryResultMetadata(duration_seconds=2.0)

        result = meta1 + meta2

        assert result.duration_seconds == 3.0
        assert result.performance == QueryResultPerformance()

    async def test_output_package_exports_public_result_and_performance_types(self):
        from model_library.base.output import (
            QueryPerformanceEvent as OutputQueryPerformanceEvent,
            QueryResult as OutputQueryResult,
            QueryResultMetadata as OutputQueryResultMetadata,
        )

        assert OutputQueryPerformanceEvent is QueryPerformanceEvent
        assert OutputQueryResultMetadata is QueryResultMetadata
        assert OutputQueryResult.__name__ == "QueryResult"

    async def test_performance_models_reject_event_payload_extra_keys(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            QueryPerformanceEvent.model_validate(
                {"type": "content_delta", "timestamp_ms": 1, "raw_text": "x"}
            )

    async def test_performance_models_reject_legacy_answer_started_event(self):
        with pytest.raises(ValidationError, match="Input should be"):
            QueryPerformanceEvent.model_validate(
                {"type": "answer_started", "timestamp_ms": 0}
            )

    async def test_timeline_accepts_zero_timestamp_started_event(self):
        entry = QueryPerformanceTimelineEntry(
            channel="content",
            index=0,
            events=[
                QueryPerformanceEvent(type="content_started", timestamp_ms=0),
                QueryPerformanceEvent(type="content_delta", timestamp_ms=5),
                QueryPerformanceEvent(type="content_finished", timestamp_ms=10),
            ],
        )

        assert entry.start_ms == 0
        assert entry.first_token_ms == 5
        assert entry.end_ms == 10
        assert entry.duration_ms == 10

    async def test_timeline_accepts_zero_timestamp_delta_without_started_event(self):
        entry = QueryPerformanceTimelineEntry(
            channel="content",
            index=0,
            events=[
                QueryPerformanceEvent(type="content_delta", timestamp_ms=0),
                QueryPerformanceEvent(type="content_finished", timestamp_ms=10),
            ],
        )

        assert entry.start_ms == 0
        assert entry.first_token_ms == 0
        assert entry.end_ms == 10
        assert entry.duration_ms == 10

    async def test_tool_call_ready_counts_as_first_token_without_args_delta(self):
        performance = QueryResultPerformance(
            timeline=[
                QueryPerformanceTimelineEntry(
                    channel="tool_call",
                    index=0,
                    events=[
                        QueryPerformanceEvent(
                            type="tool_call_started", timestamp_ms=100
                        ),
                        QueryPerformanceEvent(type="tool_call_ready", timestamp_ms=120),
                        QueryPerformanceEvent(
                            type="tool_call_finished", timestamp_ms=125
                        ),
                    ],
                )
            ]
        )

        entry = performance.timeline[0]
        assert entry.ready_ms == 120
        assert entry.first_token_ms == 120
        assert performance.time_to_first_token_ms.tool_call == 120
        assert performance.time_to_first_token_ms.answer == 120
        assert performance.time_to_first_token_ms.any == 120

    async def test_performance_metadata_rejects_flat_performance_payload(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            QueryResultMetadata.model_validate(
                {
                    "performance": {
                        "time_to_first_token_seconds": 0.1234,
                        "time_to_first_content_token_seconds": 0.4567,
                        "output_tokens_per_second": 12.3456,
                        "reasoning_tokens_per_second": 7.89,
                    }
                }
            )

    async def test_performance_rollups_reject_negative_values(self):
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            QueryResultPerformance.model_validate(
                {"time_to_first_token_ms": {"any": -1}}
            )

        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            QueryResultPerformance.model_validate(
                {"tokens_per_second": {"content": -3.5}}
            )

    async def test_performance_timeline_rejects_unordered_entries(self):
        with pytest.raises(
            ValidationError,
            match="timeline entries must be ordered by start_ms",
        ):
            QueryResultPerformance.model_validate(
                {
                    "timeline": [
                        {
                            "channel": "content",
                            "index": 0,
                            "events": [{"type": "content_delta", "timestamp_ms": 200}],
                        },
                        {
                            "channel": "reasoning",
                            "index": 0,
                            "events": [
                                {"type": "reasoning_delta", "timestamp_ms": 100}
                            ],
                        },
                    ]
                }
            )

        with pytest.raises(
            ValidationError,
            match="timeline content indexes must be contiguous from 0",
        ):
            QueryResultPerformance.model_validate(
                {
                    "timeline": [
                        {
                            "channel": "content",
                            "index": 1,
                            "events": [{"type": "content_delta", "timestamp_ms": 100}],
                        }
                    ]
                }
            )

    async def test_performance_metadata_round_trips_through_model_dump(self):
        metadata = QueryResultMetadata(
            performance=QueryResultPerformance(
                timeline=[
                    QueryPerformanceTimelineEntry(
                        channel="content",
                        index=0,
                        events=[
                            QueryPerformanceEvent(
                                type="content_started",
                                timestamp_ms=100,
                            ),
                            QueryPerformanceEvent(
                                type="content_delta",
                                timestamp_ms=120,
                            ),
                            QueryPerformanceEvent(
                                type="content_finished",
                                timestamp_ms=200,
                            ),
                        ],
                    )
                ]
            )
        )

        reparsed = QueryResultMetadata.model_validate(metadata.model_dump())

        assert reparsed == metadata
        assert reparsed.performance.timeline[0].duration_ms == 100
        assert reparsed.performance.time_to_first_token_ms.content == 120

    async def test_timeline_timing_fields_are_derived_from_events(self):
        entry = QueryPerformanceTimelineEntry(
            channel="content",
            index=0,
            start_ms=900,
            first_token_ms=900,
            end_ms=900,
            duration_ms=0,
            events=[
                QueryPerformanceEvent(type="content_started", timestamp_ms=100),
                QueryPerformanceEvent(type="content_delta", timestamp_ms=120),
                QueryPerformanceEvent(type="content_finished", timestamp_ms=200),
            ],
        )

        assert entry.start_ms == 100
        assert entry.first_token_ms == 120
        assert entry.end_ms == 200
        assert entry.duration_ms == 100
        assert (
            QueryResultPerformance(timeline=[entry]).time_to_first_token_ms.content
            == 120
        )

    async def test_timeline_validates_segment_ordering_and_event_channel(self):
        with pytest.raises(ValidationError, match="timeline entry must include events"):
            QueryPerformanceTimelineEntry(
                channel="content",
                index=0,
                start_ms=100,
            )

        with pytest.raises(ValidationError, match="timeline events must be ordered"):
            QueryPerformanceTimelineEntry(
                channel="content",
                index=0,
                events=[
                    QueryPerformanceEvent(
                        type="content_delta",
                        timestamp_ms=200,
                    ),
                    QueryPerformanceEvent(
                        type="content_finished",
                        timestamp_ms=100,
                    ),
                ],
            )

        with pytest.raises(ValidationError, match="does not belong to content segment"):
            QueryPerformanceTimelineEntry(
                channel="content",
                index=0,
                events=[
                    QueryPerformanceEvent(
                        type="tool_call_delta",
                        timestamp_ms=150,
                    )
                ],
            )


class TestQueryResultMetadataAddition:
    async def test_add_full_metadata(self):
        meta1 = QueryResultMetadata(
            in_tokens=100,
            out_tokens=50,
            reasoning_tokens=20,
            cache_read_tokens=10,
            cache_write_tokens=5,
            duration_seconds=1.5,
            cost=QueryResultCost(input=0.01, output=0.02),
        )
        meta2 = QueryResultMetadata(
            in_tokens=200,
            out_tokens=100,
            reasoning_tokens=30,
            cache_read_tokens=15,
            cache_write_tokens=10,
            duration_seconds=2.0,
            cost=QueryResultCost(input=0.02, output=0.03),
        )

        result = meta1 + meta2

        assert result.in_tokens == 300
        assert result.out_tokens == 150
        assert result.reasoning_tokens == 50
        assert result.cache_read_tokens == 25
        assert result.cache_write_tokens == 15
        assert result.duration_seconds == 3.5
        assert result.cost is not None
        assert result.cost.input == 0.03
        assert result.cost.output == 0.05

    async def test_add_metadata_missing_optional_tokens(self):
        meta1 = QueryResultMetadata(
            in_tokens=100,
            out_tokens=50,
            reasoning_tokens=20,
            duration_seconds=1.0,
        )
        meta2 = QueryResultMetadata(
            in_tokens=200,
            out_tokens=100,
            cache_read_tokens=15,
            duration_seconds=2.0,
        )

        result = meta1 + meta2

        assert result.in_tokens == 300
        assert result.out_tokens == 150
        assert result.reasoning_tokens == 20
        assert result.cache_read_tokens == 15
        assert result.cache_write_tokens is None

    async def test_add_metadata_both_missing_optional_tokens(self):
        meta1 = QueryResultMetadata(in_tokens=100, out_tokens=50)
        meta2 = QueryResultMetadata(in_tokens=200, out_tokens=100)

        result = meta1 + meta2

        assert result.in_tokens == 300
        assert result.out_tokens == 150
        assert result.reasoning_tokens is None
        assert result.cache_read_tokens is None
        assert result.cache_write_tokens is None

    async def test_add_metadata_one_has_cost(self):
        meta1 = QueryResultMetadata(
            in_tokens=100,
            out_tokens=50,
            cost=QueryResultCost(input=0.01, output=0.02),
        )
        meta2 = QueryResultMetadata(in_tokens=200, out_tokens=100, cost=None)

        result = meta1 + meta2

        assert result.cost is not None
        assert result.cost.input == 0.01
        assert result.cost.output == 0.02

    async def test_add_metadata_other_has_cost(self):
        meta1 = QueryResultMetadata(in_tokens=100, out_tokens=50, cost=None)
        meta2 = QueryResultMetadata(
            in_tokens=200,
            out_tokens=100,
            cost=QueryResultCost(input=0.02, output=0.03),
        )

        result = meta1 + meta2

        assert result.cost is not None
        assert result.cost.input == 0.02
        assert result.cost.output == 0.03

    async def test_add_metadata_neither_has_cost(self):
        meta1 = QueryResultMetadata(in_tokens=100, out_tokens=50)
        meta2 = QueryResultMetadata(in_tokens=200, out_tokens=100)

        result = meta1 + meta2

        assert result.cost is None

    async def test_add_metadata_default_duration(self):
        meta1 = QueryResultMetadata(in_tokens=100, out_tokens=50, duration_seconds=None)
        meta2 = QueryResultMetadata(in_tokens=200, out_tokens=100, duration_seconds=2.0)

        result = meta1 + meta2

        assert result.duration_seconds == 2.0

    async def test_add_metadata_both_none_duration(self):
        meta1 = QueryResultMetadata(in_tokens=100, out_tokens=50, duration_seconds=None)
        meta2 = QueryResultMetadata(
            in_tokens=200, out_tokens=100, duration_seconds=None
        )

        result = meta1 + meta2

        assert result.duration_seconds == 0.0

    async def test_computed_total_tokens(self):
        meta1 = QueryResultMetadata(
            in_tokens=100,
            out_tokens=50,
            reasoning_tokens=20,
            cache_read_tokens=10,
            cache_write_tokens=5,
        )
        meta2 = QueryResultMetadata(
            in_tokens=200,
            out_tokens=100,
            reasoning_tokens=30,
            cache_read_tokens=15,
        )

        result = meta1 + meta2

        assert result.total_input_tokens == 300 + 25 + 5
        assert result.total_output_tokens == 150 + 50
