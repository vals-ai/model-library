from model_gateway.types import query_result_response_body
from model_library.base.gateway import _parse_query_result
from model_library.base.output import (
    QueryPerformanceEvent,
    QueryPerformanceTimelineEntry,
    QueryResult,
    QueryResultMetadata,
    QueryResultPerformance,
)


def test_query_result_response_body_sends_first_token_fields_without_timeline():
    performance = QueryResultPerformance(
        timeline=[
            QueryPerformanceTimelineEntry(
                channel="content",
                index=0,
                events=[QueryPerformanceEvent(type="content_delta", timestamp_ms=10)],
            )
        ]
    )
    result = QueryResult(metadata=QueryResultMetadata(performance=performance))

    body = query_result_response_body(result, signed_history="[]", failed=False)

    assert body["metadata"]["performance"] == {
        "time_to_first_token_ms": {"any": 10, "answer": 10, "content": 10}
    }

    reparsed = _parse_query_result(body, schema_model=None)

    assert reparsed.metadata.performance == QueryResultPerformance(
        time_to_first_token_ms=performance.time_to_first_token_ms
    )


def test_query_result_response_body_omits_missing_performance():
    body = query_result_response_body(QueryResult(), signed_history="[]", failed=False)

    assert "performance" not in body["metadata"]
