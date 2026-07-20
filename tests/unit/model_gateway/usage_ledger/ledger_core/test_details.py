from __future__ import annotations

import json
from typing import Any, cast

from pydantic import create_model
import pytest

from model_gateway.types import QueryRequest
from model_gateway.usage_ledger.details import (
    MAX_LEDGER_VALUE_JSON_CHARS,
    build_usage_event_details,
    snapshot_usage_request,
)
from model_library.base.input import (
    RawResponse,
    TextInput,
    ToolCall,
    ToolDefinition,
)
from model_library.base.output import (
    ProviderToolEvent,
    QueryResult,
    QueryResultExtras,
    QueryResultMetadata,
    QueryResultPerformance,
    CompressedQueryResultPerformance,
    decompress_query_result_performance,
)


def _request(**overrides: object) -> QueryRequest:
    values: dict[str, object] = {
        "model": "openai/gpt-test",
        "inputs": [TextInput(text="forbidden user prompt")],
        "config": {"temperature": 0.2},
        "run_id": "run-1",
        "question_id": "question-1",
        "query_id": "query-1",
        "identity": {
            "email": "user@example.com",
            "benchmark_name": "benchmark-1",
            "agent_name": "agent-1",
            "organization": "org-1",
        },
    }
    values.update(overrides)
    return QueryRequest.model_validate(values)


def _result(**overrides: object) -> QueryResult:
    values: dict[str, object] = {
        "output_text": "forbidden model output",
        "output_parsed": {"answer": "forbidden parsed output"},
        "reasoning": "forbidden reasoning",
        "metadata": QueryResultMetadata(
            in_tokens=10,
            out_tokens=20,
            extra={"token_metadata": {"estimated": 12, "actual": 10}},
        ),
        "tool_calls": [
            ToolCall(
                id="tool-call-1",
                name="search",
                args={"query": "forbidden tool argument", "limit": 3},
            ),
            ToolCall(
                id="tool-call-2",
                name="notify",
                args='{"recipient":"user@example.com"}',
            ),
        ],
        "provider_tool_events": [
            ProviderToolEvent(
                id="provider-tool-1",
                provider="openai",
                type="web_search_call",
                name="web_search",
                input="forbidden provider search query",
                output=["https://example.com/source"],
                sequence=0,
            )
        ],
        "history": [TextInput(text="omitted history")],
        "extras": QueryResultExtras(
            provider_request_id="provider-request-1",
            provider_response_id="provider-response-1",
            search_results={"answer": "forbidden search result"},
        ),
    }
    values.update(overrides)
    return QueryResult.model_validate(values)


def _details(
    *,
    request: QueryRequest | None = None,
    result: QueryResult | None = None,
) -> dict[str, dict[str, Any]]:
    request = request or _request()
    return cast(
        dict[str, dict[str, Any]],
        build_usage_event_details(
            request=snapshot_usage_request(request),
            result=result or _result(),
        ),
    )


def test_details_preserve_request_and_result_except_explicit_reductions() -> None:
    request = _request(
        config={
            "temperature": 0.2,
            "custom_api_key": "provider-secret",
            "provider_config": {"prompt_cache_retention": "24h"},
        }
    )

    details = _details(request=request)

    assert set(details) == {"request", "result"}
    request_data = details["request"]
    result_data = details["result"]
    assert request_data["inputs"][0] == {
        "kind": "text",
        "text_length": len("forbidden user prompt"),
    }
    assert request_data["identity"] == {
        "email": "user@example.com",
        "benchmark_name": "benchmark-1",
        "agent_name": "agent-1",
        "organization": "org-1",
    }
    assert request_data["config"]["custom_api_key"] == "**********"
    assert request_data["config"]["provider_config"] == {
        "prompt_cache_retention": "24h"
    }
    assert request_data["tools"] == []
    assert "tool_names" not in request_data
    assert request_data["output_schema_length"] == 0
    assert "output_schema" not in request_data
    assert "output_text" not in result_data
    assert "output_parsed" not in result_data
    assert "reasoning" not in result_data
    assert "history" not in result_data
    assert result_data["output_text_length"] == len("forbidden model output")
    assert result_data["output_parsed_length"] == len(
        '{"answer":"forbidden parsed output"}'
    )
    assert result_data["reasoning_length"] == len("forbidden reasoning")
    assert result_data["metadata"]["extra"]["token_metadata"] == {
        "estimated": 12,
        "actual": 10,
    }
    tool_call = result_data["tool_calls"][0]
    assert "args" not in tool_call
    assert "parsed_args" not in tool_call
    assert tool_call["args_length"] == len(
        '{"query":"forbidden tool argument","limit":3}'
    )
    string_args_tool_call = result_data["tool_calls"][1]
    assert "args" not in string_args_tool_call
    assert "parsed_args" not in string_args_tool_call
    assert string_args_tool_call["args_length"] == len(
        '{"recipient":"user@example.com"}'
    )
    provider_tool_event = result_data["provider_tool_events"][0]
    assert "input" not in provider_tool_event
    assert provider_tool_event["input_length"] == len("forbidden provider search query")
    assert "output" not in provider_tool_event
    assert provider_tool_event["output_length"] == _compact_length(
        ["https://example.com/source"]
    )
    assert "search_results" not in result_data["extras"]
    assert result_data["extras"]["search_results_length"] == _compact_length(
        {"answer": "forbidden search result"}
    )
    assert result_data["extras"]["provider_request_id"] == "provider-request-1"
    assert result_data["extras"]["provider_response_id"] == "provider-response-1"


def test_request_details_delegate_content_sanitization() -> None:
    tool_body = {"private": "definition"}
    output_schema = {"type": "object"}
    request = _request(
        inputs=[TextInput(text="private user prompt")],
        tools=[ToolDefinition(name="search", body=tool_body)],
        output_schema=output_schema,
    )

    request_data = snapshot_usage_request(request)

    assert request_data["inputs"] == [
        {
            "kind": "text",
            "text_length": len("private user prompt"),
        }
    ]
    assert request_data["tools"] == [
        {
            "name": "search",
            "body_length": _compact_length(tool_body),
        }
    ]
    assert request_data["output_schema_length"] == _compact_length(output_schema)
    assert "output_schema" not in request_data


def _compact_length(value: object) -> int:
    return len(json.dumps(value, sort_keys=True, separators=(",", ":")))


@pytest.mark.parametrize(
    ("output_parsed", "expected_length"),
    [
        ("text", 4),
        ({"answer": "text"}, 17),
        (create_model("ParsedOutput", answer=(str, ...))(answer="text"), 17),
        (
            create_model("UnicodeParsedOutput", answer=(str, ...))(answer="é"),
            len('{"answer":"\\u00e9"}'),
        ),
        (["text", 7], 10),
        (7, 1),
        (True, 4),
        (None, 0),
    ],
)
def test_details_replace_every_output_parsed_shape_with_length(
    output_parsed: object,
    expected_length: int,
) -> None:
    result = _result(output_parsed=None)
    cast(Any, result).output_parsed = output_parsed

    result_data = _details(result=result)["result"]

    assert "output_parsed" not in result_data
    assert result_data["output_parsed_length"] == expected_length


def test_details_omit_non_json_provider_history_before_serialization() -> None:
    result = _result(history=[RawResponse(response=object())])

    result_data = _details(result=result)["result"]

    assert "history" not in result_data


def test_details_preserve_absent_performance_as_none() -> None:
    details = _details(result=_result(metadata=QueryResultMetadata()))

    assert details["result"]["metadata"]["performance"] is None


def test_details_preserve_compressed_performance_envelope() -> None:
    performance = QueryResultPerformance.model_validate(
        {
            "timeline": [
                {
                    "channel": "content",
                    "index": 0,
                    "events": [
                        {"type": "content_started", "timestamp_ms": 10},
                        {"type": "content_delta", "timestamp_ms": 15},
                        {"type": "content_finished", "timestamp_ms": 25},
                    ],
                }
            ]
        }
    )

    details = _details(
        result=_result(metadata=QueryResultMetadata(performance=performance))
    )

    stored = details["result"]["metadata"]["performance"]
    serialized = CompressedQueryResultPerformance.model_validate(stored)
    assert decompress_query_result_performance(serialized) == performance


def test_details_preserve_compressed_performance_over_generic_string_limit() -> None:
    performance = CompressedQueryResultPerformance(
        data="x" * (MAX_LEDGER_VALUE_JSON_CHARS + 1)
    )

    details = _details(
        result=_result(metadata=QueryResultMetadata(performance=performance))
    )

    assert details["result"]["metadata"]["performance"] == performance.model_dump()


def test_details_replace_only_oversized_strings_and_lists_with_length() -> None:
    ordinary = "x" * 1_000
    boundary = "x" * (MAX_LEDGER_VALUE_JSON_CHARS - 2)
    oversized = "x" * (MAX_LEDGER_VALUE_JSON_CHARS - 1)
    oversized_list = list(range(20_000))
    request = _request(
        config={
            "provider_config": {
                "ordinary": ordinary,
                "boundary": boundary,
                "oversized": oversized,
            }
        }
    )
    result = _result(
        metadata=QueryResultMetadata(
            extra={
                "ordinary": ["keep", 7],
                "oversized": oversized_list,
            }
        )
    )

    details = _details(request=request, result=result)

    provider_config = details["request"]["config"]["provider_config"]
    assert provider_config["ordinary"] == ordinary
    assert provider_config["boundary"] == boundary
    assert provider_config["oversized"] == len(oversized)
    extra = details["result"]["metadata"]["extra"]
    assert extra["ordinary"] == ["keep", 7]
    assert extra["oversized"] == len(oversized_list)
