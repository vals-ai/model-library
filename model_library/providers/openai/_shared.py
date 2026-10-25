from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

from openai.lib._pydantic import to_strict_json_schema
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.responses.response import Response
from pydantic import BaseModel

from model_library.base import (
    FileBase,
    FileInput,
    InputItem,
    RawInput,
    RawResponse,
    SystemInput,
    TextInput,
    ToolBody,
    ToolDefinition,
    ToolResult,
)
from model_library.base.query_ids import prompt_cache_key_from_query_ids
from model_library.exceptions import NoMatchingToolCallError

if TYPE_CHECKING:
    from .openai import OpenAIModel


def iter_input_groups(
    input: Sequence[InputItem],
) -> Iterator[
    list[TextInput | FileInput] | SystemInput | ToolResult | RawInput | RawResponse
]:
    content: list[TextInput | FileInput] = []
    for item in input:
        if isinstance(item, (TextInput, FileBase)):
            content.append(item)
            continue
        if content:
            yield content
            content = []
        yield item
    if content:
        yield content


async def user_content(
    model: OpenAIModel, items: list[TextInput | FileInput], text_type: str
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, TextInput):
            content.append({"type": text_type, "text": item.text})
        else:
            match item.type:
                case "image":
                    content.append(await model.parse_image(item))
                case "file":
                    content.append(await model.parse_file(item))
    return content


def validate_tool_result(item: ToolResult, tool_call_ids: list[str]) -> None:
    if item.tool_call.id not in tool_call_ids:
        raise NoMatchingToolCallError()


def function_tool_payload(body: ToolBody) -> dict[str, Any]:
    parameters = {
        "type": "object",
        "properties": body.properties,
        "required": body.required,
        "additionalProperties": body.kwargs.get("additional_properties", False),
    }
    payload = {
        "name": body.name,
        "description": body.description,
        "parameters": parameters,
        "strict": body.kwargs.get("strict", False),
    }
    if body.allowed_callers is not None:
        payload["allowed_callers"] = body.allowed_callers
    return payload


def apply_common_body(
    model: OpenAIModel,
    body: dict[str, Any],
    input: Sequence[InputItem],
    tools: Sequence[ToolDefinition],
    sampling: dict[str, object],
) -> None:
    if model.prompt_cache_retention is not None:
        body["prompt_cache_retention"] = model.prompt_cache_retention
    # GPT-5.6+ caches implicitly and bills writes; explicit mode with no breakpoints writes nothing.
    if model.prompt_caching == "off" and model.supports_prompt_cache_options:
        body["prompt_cache_options"] = {"mode": "explicit"}
    if model.service_tier is not None:
        body["service_tier"] = model.service_tier
    if model.prompt_cache_key_mode is not None:
        body["prompt_cache_key"] = prompt_cache_key_from_query_ids(
            model_name=model.model_name, input=input, tools=tools
        )
    if model.parallel_tool_calls is not None:
        body["parallel_tool_calls"] = model.parallel_tool_calls
    if "temperature" in sampling:
        body["temperature"] = sampling["temperature"]
    if "top_p" in sampling:
        body["top_p"] = sampling["top_p"]


def output_schema_value(
    output_schema: dict[str, Any] | type[BaseModel],
) -> dict[str, Any]:
    return (
        output_schema
        if isinstance(output_schema, dict)
        else to_strict_json_schema(output_schema)
    )


def capture_routing(
    routing: dict[str, Any],
    response: ChatCompletion | ChatCompletionChunk | Response,
) -> None:
    if response.model:
        routing["served_model"] = response.model
    if provider := (response.model_extra or {}).get("provider"):
        routing["upstream_provider"] = provider
    if response.usage is not None:
        cost = (response.usage.model_extra or {}).get("cost")
        if cost is not None:
            routing["upstream_cost"] = float(cost)
