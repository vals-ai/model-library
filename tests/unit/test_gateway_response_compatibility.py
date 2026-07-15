from model_library.base import FinishReason, QueryResult


def test_current_gateway_query_result_is_parseable() -> None:
    result = QueryResult.model_validate(
        {
            "output_text": "ok",
            "finish_reason": {"reason": "paused", "raw": "pause_turn"},
            "metadata": {
                "performance": {
                    "timeline": [
                        {
                            "channel": "content",
                            "index": 0,
                            "events": [
                                {
                                    "type": "content_delta",
                                    "timestamp_ms": 10,
                                    "channel_text_start_char": 0,
                                    "channel_text_end_char": 2,
                                }
                            ],
                        }
                    ]
                }
            },
            "extras": {
                "provider_response_id": "response-1",
                "provider_request_id": "request-1",
            },
        }
    )

    event = result.metadata.performance.timeline[0].events[0]
    assert result.finish_reason.reason is FinishReason.PAUSED
    assert event.channel_text_start_char == 0
    assert event.channel_text_end_char == 2
    assert result.extras.response_id == "response-1"
    assert result.extras.provider_response_id == "response-1"
    assert result.extras.provider_request_id == "request-1"
