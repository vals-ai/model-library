from model_gateway.types import query_result_response_body
from model_library.base.gateway import _parse_query_result
from model_library.base.output import (
    CompressedQueryResultPerformance,
    QueryPerformanceEvent,
    QueryPerformanceTimelineEntry,
    QueryResult,
    QueryResultMetadata,
    QueryResultPerformance,
    decompress_query_result_performance,
)
from model_library.base.output.performance import compress_query_result_performance


def test_query_result_response_body_sends_compressed_performance():
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
        "time_to_first_token_ms": {"any": 10, "answer": 10, "content": 10},
        "timeline": {
            "encoding": "gzip+base64",
            "data": compress_query_result_performance(performance).timeline.data,
        },
    }

    reparsed = _parse_query_result(body, schema_model=None)

    assert isinstance(reparsed.metadata.performance, CompressedQueryResultPerformance)
    assert (
        decompress_query_result_performance(reparsed.metadata.performance)
        == performance
    )


def test_query_result_response_body_omits_missing_performance():
    body = query_result_response_body(QueryResult(), signed_history="[]", failed=False)

    assert "performance" not in body["metadata"]
