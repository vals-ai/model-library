from __future__ import annotations

from collections.abc import Mapping
import json
import math
from typing import Any, cast

from pydantic import BaseModel

from model_gateway.types import QueryRequest
from model_library.base.output import QueryResult
from model_library.utils import content_length

MAX_LEDGER_VALUE_JSON_CHARS = 64_000


def snapshot_usage_request(request: QueryRequest) -> dict[str, object]:
    data = _model_json(request)
    data["inputs"] = [
        item.sanitize_content(show_content=False) for item in request.inputs
    ]
    data["tools"] = [
        tool.sanitize_content(show_content=False) for tool in request.tools
    ]
    _replace_with_length(data, "output_schema")
    return _bounded_value(data)


def build_usage_event_details(
    *,
    request: Mapping[str, object],
    result: QueryResult,
) -> dict[str, object]:
    result_data = _model_json(
        result, exclude={"history", "output_parsed", "tool_calls"}
    )
    _replace_with_length(result_data, "output_text")
    result_data["output_parsed_length"] = content_length(result.output_parsed)
    _replace_with_length(result_data, "reasoning")
    result_data["tool_calls"] = [
        tool_call.sanitize_content(show_content=False)
        for tool_call in result.tool_calls
    ]
    for provider_event in result_data["provider_tool_events"]:
        _replace_with_length(provider_event, "input")
        _replace_with_length(provider_event, "output")
    extras = result_data["extras"]
    _replace_with_length(extras, "search_results")

    metadata = cast(dict[str, object], result_data["metadata"])
    performance = metadata.pop("performance")
    bounded_result = cast(dict[str, object], _bounded_value(result_data))
    bounded_metadata = cast(dict[str, object], bounded_result["metadata"])
    bounded_metadata["performance"] = performance
    return {
        "request": dict(request),
        "result": bounded_result,
    }


def _model_json(model: BaseModel, *, exclude: set[str] | None = None) -> dict[str, Any]:
    return model.model_dump(mode="json", exclude=exclude)


def _replace_with_length(values: dict[str, object], field: str) -> None:
    value = values.pop(field)
    values[f"{field}_length"] = content_length(value)


def _bounded_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str):
        return (
            len(value)
            if _compact_json_length(value) > MAX_LEDGER_VALUE_JSON_CHARS
            else value
        )
    if isinstance(value, list):
        items = cast(list[Any], value)
        if _compact_json_length(items) > MAX_LEDGER_VALUE_JSON_CHARS:
            return len(items)
        return [_bounded_value(item) for item in items]
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, Any], value)
        return {key: _bounded_value(item) for key, item in mapping.items()}
    return value


def _compact_json_length(value: object) -> int:
    return len(json.dumps(value, sort_keys=True, separators=(",", ":")))
