from model_gateway.types import query_result_response_body
from model_library.base.gateway import _parse_query_result
from model_library.base.output import (
    QueryPerformanceEvent,
    QueryPerformanceTimelineEntry,
    decompress_query_result_performance,
    QueryResult,
    QueryResultMetadata,
    QueryResultPerformance,
    CompressedQueryResultPerformance,
)


def test_query_result_response_body_compresses_performance_for_client_round_trip():
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

    body = query_result_response_body(result, signed_history="[]")

    envelope = body["metadata"]["performance"]
    assert envelope["encoding"] == "gzip+base64"

    reparsed = _parse_query_result(body, schema_model=None)

    assert isinstance(reparsed.metadata.performance, CompressedQueryResultPerformance)
    assert reparsed.metadata.performance.model_dump(mode="json") == envelope

    assert (
        decompress_query_result_performance(reparsed.metadata.performance)
        == performance
    )
