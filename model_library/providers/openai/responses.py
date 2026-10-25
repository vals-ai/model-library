from __future__ import annotations

import json
import logging
from collections import deque
from collections.abc import AsyncIterator, Hashable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from openai import APIConnectionError
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputRefusal
from openai.types.responses.response_function_web_search import ActionSearch
from openai.types.responses.response import Response
from openai.types.responses.response_stream_event import ResponseStreamEvent
from openai.types.responses.tool_param import ToolParam as ResponsesToolParam
from pydantic import BaseModel, JsonValue

from model_library.base import (
    Citation,
    FileInput,
    FileWithBase64,
    FileWithId,
    FileWithUrl,
    FinishReason,
    FinishReasonInfo,
    InputItem,
    ProviderToolEvent,
    PydanticT,
    QueryResult,
    QueryResultExtras,
    QueryResultMetadata,
    RawInput,
    RawResponse,
    SystemInput,
    ToolBody,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from model_library.base.output.builder import QueryResultBuilder
from model_library.exceptions import (
    BadInputError,
    ContentFilterError,
    ImmediateRetryException,
    ModelNoOutputError,
    UnexpectedSystemInputError,
    handle_empty_response,
)
from model_library.rate_limits import RateLimit, rate_limit_from_headers
from model_library.agent.tool import is_native_web_search
from model_library.retriers.base import BaseRetrier

from ._shared import (
    apply_common_body,
    capture_routing,
    function_tool_payload,
    iter_input_groups,
    output_schema_value,
    user_content,
    validate_tool_result,
)

if TYPE_CHECKING:
    from .openai import OpenAIModel


def _first_hashable_value(*values: object) -> Hashable | None:
    for value in values:
        if value is not None and isinstance(value, Hashable):
            return value
    return None


def _response_tool_call_event_key(event: object) -> Hashable | None:
    item = getattr(event, "item", None)
    return _first_hashable_value(
        getattr(item, "id", None),
        getattr(event, "item_id", None),
        getattr(item, "output_index", None),
        getattr(event, "output_index", None),
    )


def _to_json_value(value: object) -> JsonValue:
    if isinstance(value, BaseModel):
        return _to_json_value(value.model_dump(mode="json"))

    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {str(key): _to_json_value(item) for key, item in mapping.items()}

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        sequence = cast(Sequence[object], value)
        return [_to_json_value(item) for item in sequence]

    return cast(JsonValue, json.loads(json.dumps(value, allow_nan=False)))


def _safe_search_results(
    value: object, query_logger: logging.Logger
) -> JsonValue | None:
    if value is None:
        return None

    try:
        return _to_json_value(value)
    except (TypeError, ValueError):
        query_logger.warning("Dropping non-JSON-serializable search results")
        return None


def map_openai_responses_finish_reason(
    status: str | None,
    incomplete_reason: str | None,
    has_tool_calls: bool = False,
) -> FinishReasonInfo:
    match status:
        case "completed":
            reason = FinishReason.TOOL_CALLS if has_tool_calls else FinishReason.STOP
        case "incomplete":
            match incomplete_reason:
                case "max_output_tokens":
                    reason = FinishReason.MAX_TOKENS
                case "content_filter":
                    reason = FinishReason.CONTENT_FILTER
                case _:
                    reason = FinishReason.UNKNOWN
        case "failed":
            reason = FinishReason.ERROR
        case _:
            reason = FinishReason.UNKNOWN

    raw = status
    if status == "incomplete" and incomplete_reason:
        raw = f"incomplete:{incomplete_reason}"

    return FinishReasonInfo(reason=reason, raw=raw)


async def get_tool_call_ids(
    model: OpenAIModel, input: Sequence[InputItem]
) -> list[str]:
    calls = [
        call
        for item in input
        if isinstance(item, RawResponse)
        for call in item.response
        if isinstance(call, ResponseFunctionToolCall)
    ]
    return [call.id for call in calls if call.id]


async def parse_input(
    model: OpenAIModel, input: Sequence[InputItem], **kwargs: Any
) -> list[dict[str, Any] | Any]:
    new_input: list[dict[str, Any] | Any] = []
    if isinstance(input[0], SystemInput):
        new_input.append(
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": input[0].text}],
            }
        )
        input = input[1:]
    tool_call_ids = await model.get_tool_call_ids(input)
    for item in iter_input_groups(input):
        if isinstance(item, list):
            new_input.append(
                {
                    "role": "user",
                    "content": await user_content(model, item, "input_text"),
                }
            )
            continue
        match item:
            case ToolResult():
                validate_tool_result(item, tool_call_ids)
                new_input.append(
                    {
                        "type": "function_call_output",
                        "call_id": item.tool_call.call_id,
                        "output": item.result,
                        **(
                            {"code_mode_id": item.tool_call.code_mode_id}
                            if item.tool_call.code_mode_id is not None
                            else {}
                        ),
                    }
                )
            case RawResponse():
                new_input.extend(item.response)
            case RawInput():
                new_input.append(item.input)
            case SystemInput():
                raise UnexpectedSystemInputError()
    return new_input


async def parse_image(model: OpenAIModel, image: FileInput) -> dict[str, Any]:
    base_dict: dict[str, Any] = {"type": "input_image", "detail": "auto"}
    match image:
        case FileWithBase64():
            base_dict["image_url"] = f"data:{image.mime};base64,{image.base64}"
        case FileWithUrl():
            base_dict["image_url"] = image.url
        case FileWithId():
            if model.provider in ("grok", "xai"):
                raise BadInputError(
                    "xAI Responses API does not support image file_id; use base64 or URL"
                )
            base_dict["file_id"] = image.file_id
        case _:
            raise BadInputError("OpenAI does not support byte-backed files")
    return base_dict


async def parse_file(model: OpenAIModel, file: FileInput) -> dict[str, Any]:
    base_dict: dict[str, Any] = {"type": "input_file"}
    match file:
        case FileWithBase64():
            base_dict["filename"] = file.name
            base_dict["file_data"] = f"data:{file.mime};base64,{file.base64}"
        case FileWithUrl():
            base_dict["file_url"] = file.url
        case FileWithId():
            base_dict["file_id"] = file.file_id
        case _:
            raise BadInputError("OpenAI does not support byte-backed files")
    return base_dict


async def parse_tools(
    model: OpenAIModel, tools: Sequence[ToolDefinition]
) -> list[ResponsesToolParam | Any]:
    parsed_tools: list[ResponsesToolParam | Any] = []
    for tool in tools:
        body = tool.body
        if is_native_web_search(body):
            parsed_tools.append(model.search_tool)
            continue
        if not isinstance(body, ToolBody):
            parsed_tools.append(body)
            continue
        parsed_tools.append(
            cast(
                ResponsesToolParam,
                {"type": "function", **function_tool_payload(body)},
            )
        )
    return parsed_tools


def search_tool(model: OpenAIModel) -> dict[str, str]:
    if model.provider in ("grok", "xai"):
        return {"type": "web_search"}
    return {"type": "web_search_preview"}


async def _check_deep_research_args(
    model: OpenAIModel, tools: Sequence[ToolDefinition], **kwargs: object
) -> None:
    min_tokens = 30_000
    if not model.max_tokens or model.max_tokens < min_tokens:
        model.instance_logger.warning(
            f"Recommended to set max_tokens >= {min_tokens} for deep research models"
        )

    if "background" not in kwargs:
        model.instance_logger.warning(
            "Recommended to set background=True for deep research models"
        )

    valid = False
    for tool in tools:
        tool_body = tool.body
        if is_native_web_search(tool_body):
            valid = True
            continue
        if not isinstance(tool_body, dict):
            continue
        tool_body = cast(dict[str, Any], tool_body)
        tool_type = tool_body.get("type", None)
        if tool_type in {"web_search", "web_search_preview"}:
            valid = True
    if not valid:
        raise Exception("Deep research models require web search tools")


async def build_body(
    model: OpenAIModel,
    input: Sequence[InputItem],
    *,
    tools: list[ToolDefinition],
    output_schema: dict[str, Any] | type[BaseModel] | None = None,
    sampling: dict[str, object],
    **kwargs: object,
) -> dict[str, Any]:
    if model.deep_research:
        await _check_deep_research_args(model, tools, **kwargs)

    parsed_tools: list[Any] = await model.parse_tools(tools)
    if model.tool_call_mode != "default":
        if model.tool_call_mode == "auto":
            allowed_callers = ["code_mode", "direct"]
        elif model.tool_call_mode == "code_mode":
            allowed_callers = ["code_mode"]
        else:
            raise ValueError(f"Unknown tool_call_mode: {model.tool_call_mode}")

        has_code_mode_tool = any(
            isinstance(tool, dict)
            and cast(dict[str, Any], tool).get("type") == "code_mode"
            for tool in parsed_tools
        )
        for tool in parsed_tools:
            if not isinstance(tool, dict):
                continue
            tool_body = cast(dict[str, Any], tool)
            if (
                tool_body.get("type") == "function"
                and "allowed_callers" not in tool_body
            ):
                tool_body["allowed_callers"] = allowed_callers
        if model.tool_call_mode in {"auto", "code_mode"} and not has_code_mode_tool:
            parsed_tools.append({"type": "code_mode", "language": "javascript"})

    body: dict[str, Any] = {
        "model": model.model_name,
        "input": await model.parse_input(input),
    }

    if model.max_tokens:
        body["max_output_tokens"] = model.max_tokens

    if parsed_tools:
        body["tools"] = parsed_tools

    if model.reasoning:
        body["include"] = ["reasoning.encrypted_content"]
        body["reasoning"] = {"summary": "auto"}
        body["store"] = False
        if model.reasoning_effort is not None:
            body["reasoning"]["effort"] = model.reasoning_effort  # type: ignore[reportArgumentType]
        if model.reasoning_context is not None:
            body["reasoning"]["context"] = model.reasoning_context

    if model.verbosity is not None:
        body["text"] = {"format": {"type": "text"}, "verbosity": model.verbosity}

    apply_common_body(model, body, input, tools, sampling)

    if model.access_programs:
        # Write into kwargs["extra_body"] so the final body.update(kwargs) keeps
        # access_programs when the caller also passes extra_body.
        extra_body = cast(dict[str, Any], kwargs.setdefault("extra_body", {}))
        extra_body["access_programs"] = dict(model.access_programs)

    if output_schema is not None:
        schema = output_schema_value(output_schema)
        text_block: dict[str, Any] = {
            "format": {
                "type": "json_schema",
                "name": "structured_output",
                "schema": schema,
                "strict": True,
            }
        }
        if model.verbosity is not None:
            text_block["verbosity"] = model.verbosity
        body["text"] = text_block

    _ = kwargs.pop("stream", None)

    body.update(kwargs)
    return body


async def query(
    model: OpenAIModel,
    input: Sequence[InputItem],
    *,
    tools: list[ToolDefinition],
    query_logger: logging.Logger,
    output_schema: dict[str, Any] | type[BaseModel] | None = None,
    **kwargs: object,
) -> QueryResult:
    stream_responses = bool(kwargs.pop("stream", True))
    body = await model.build_body(
        input, tools=tools, output_schema=output_schema, **kwargs
    )

    result_builder = QueryResultBuilder()
    response: Response | None = None
    provider_request_id: str | None = None
    recent_stream_events: deque[dict[str, str | None]] = deque(maxlen=5)
    try:
        if stream_responses:
            async with model.get_client().responses.with_streaming_response.create(
                **body,  # pyright: ignore[reportAny]
                stream=True,
            ) as raw_response:
                provider_request_id = raw_response.request_id
                stream = cast(
                    AsyncIterator[ResponseStreamEvent], await raw_response.parse()
                )
                async for event in stream:
                    event_response = getattr(event, "response", None)
                    recent_stream_events.append(
                        {
                            "type": event.type,
                            "response_id": getattr(event_response, "id", None),
                        }
                    )
                    match event.type:
                        case "response.created":
                            query_logger.debug(f"Response created: {event.response.id}")
                            response = event.response
                        case "response.completed":
                            query_logger.debug(
                                f"Response completed: {event.response.id}"
                            )
                            response = event.response
                        case "response.incomplete":
                            query_logger.error(
                                f"Response incomplete: {event.response.id}"
                            )
                            response = event.response
                        case "response.in_progress":
                            query_logger.debug(
                                f"Response in progress: {event.response.id}"
                            )
                            response = event.response
                        case "response.failed":
                            query_logger.error(
                                f"Response failed: {event.response.id} | {event.response.error}"
                            )
                            response = event.response
                        case "response.output_text.delta":
                            result_builder.append_content_delta(
                                getattr(event, "delta", None)
                            )
                        case "response.output_text.done":
                            result_builder.finish_current_segment("content")
                        case "response.reasoning_summary_text.delta":
                            result_builder.append_reasoning_delta(
                                getattr(event, "delta", None)
                            )
                        case "response.reasoning_summary_text.done":
                            result_builder.finish_current_segment("reasoning")
                        case "response.output_item.added":
                            output_item = getattr(event, "item", None)
                            if getattr(output_item, "type", None) == "function_call":
                                tool_call_key = _response_tool_call_event_key(event)
                                result_builder.start_tool_call_segment(
                                    tool_call_key
                                ).record_tool_call_ready(tool_call_key)
                        case "response.function_call_arguments.delta":
                            if getattr(event, "delta", None) is not None:
                                result_builder.record_tool_call_delta(
                                    _response_tool_call_event_key(event)
                                )
                        case "response.function_call_arguments.done":
                            result_builder.finish_tool_call_segment(
                                _response_tool_call_event_key(event)
                            )
                        case "response.output_item.done":
                            output_item = getattr(event, "item", None)
                            if getattr(output_item, "type", None) == "function_call":
                                result_builder.finish_tool_call_segment(
                                    _response_tool_call_event_key(event)
                                )
                        case _:
                            continue
        else:
            openai_response = await model.get_client().responses.create(
                **body,  # pyright: ignore[reportAny]
                stream=False,
            )
            response = openai_response
            provider_request_id = getattr(response, "_request_id", None)
    except APIConnectionError:
        raise ImmediateRetryException("Failed to connect to OpenAI")

    if not response:
        query_logger.error(
            f"Model returned no response. Recent events: {list(recent_stream_events)}"
        )
        raise ImmediateRetryException("Model returned no response")
    if stream_responses and response.status in {"queued", "in_progress"}:
        raise ImmediateRetryException(
            f"Response stream ended before completion: {response.id}"
        )
    query_logger.debug(f"Response finished: {response.id}")

    finish_reason = (
        None if not response.incomplete_details else response.incomplete_details.reason
    )

    tool_calls: list[ToolCall] = []
    provider_tool_events: list[ProviderToolEvent] = []
    citations: list[Citation] = []
    parsed_reasoning = None
    code_mode_outputs: list[str] = []
    for i, output in enumerate(response.output):
        if output.type == "message":
            for content in output.content:
                if isinstance(content, ResponseOutputRefusal):
                    raise ContentFilterError(content.refusal)
                for citation in content.annotations:
                    citations.append(Citation(**citation.model_dump()))

        if output.type == "reasoning":
            parsed_reasoning = " ".join(s.text for s in output.summary) or None
            continue
        if output.type == "code_mode_output":  # pyright: ignore[reportUnnecessaryComparison]
            code_mode_result = getattr(output, "result", None)
            if code_mode_result is not None:
                code_mode_outputs.append(
                    code_mode_result
                    if isinstance(code_mode_result, str)
                    else json.dumps(code_mode_result)
                )
            continue
        if output.type == "web_search_call":
            if isinstance(output.action, ActionSearch):
                query = output.action.query or next(
                    iter(output.action.queries or []), ""
                )
                sources: list[JsonValue] = [
                    s.url for s in (output.action.sources or [])
                ]
                provider_tool_events.append(
                    ProviderToolEvent.web_search(
                        provider=model.provider,
                        kind="web_search_call",
                        query=query,
                        sources=sources,
                        sequence=i,
                        id=output.id,
                        status=output.status,
                    )
                )
            continue
        if output.type != "function_call":
            continue
        query_logger.debug(f"Found tool call for response: {response.id}")
        if not output.id:
            raise Exception(f"Tool call is missing id for response: {response.id}")
        tool_calls.append(
            ToolCall(
                id=output.id,
                call_id=output.call_id,
                name=output.name,
                args=output.arguments,
                code_mode_id=getattr(output, "code_mode_id", None),
                sequence=i,
            )
        )

    mapped_finish_reason = map_openai_responses_finish_reason(
        response.status, finish_reason, has_tool_calls=bool(tool_calls)
    )

    parsed_response_text = response.output_text or None
    if stream_responses:
        base_output_text = result_builder.output_text or parsed_response_text
        if not result_builder.has_reasoning:
            result_builder.set_reasoning(parsed_reasoning)
    else:
        base_output_text = parsed_response_text
        result_builder.set_reasoning(parsed_reasoning)
    output_parts = [part for part in [base_output_text, *code_mode_outputs] if part]
    result_builder.set_output_text("\n".join(output_parts) if output_parts else None)

    no_useful_content = (
        not result_builder.has_output_text
        and not result_builder.has_reasoning
        and not tool_calls
        and not provider_tool_events
    )
    if no_useful_content:
        handle_empty_response(
            mapped_finish_reason,
            {
                "status": response.status,
                "response_id": response.id,
                "incomplete_details": response.incomplete_details,
            },
        )

    result_metadata = QueryResultMetadata()
    if usage := response.usage:
        # see _calculate_cost
        input_details = usage.input_tokens_details
        cache_read_tokens = (input_details.cached_tokens or 0) if input_details else 0
        cache_write_tokens = (
            (input_details.cache_write_tokens or 0) if input_details else 0
        )
        output_details = usage.output_tokens_details
        reasoning_tokens = output_details.reasoning_tokens if output_details else None
        result_metadata = QueryResultMetadata(
            in_tokens=(
                usage.input_tokens - cache_read_tokens - cache_write_tokens
                if usage.input_tokens
                else 0
            ),
            out_tokens=usage.output_tokens - (reasoning_tokens or 0),
            reasoning_tokens=reasoning_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        )
    if model.router_mode:
        routing: dict[str, Any] = {}
        capture_routing(routing, response)
        result_metadata.served_model = routing.pop("served_model", None)
        result_metadata.extra.update(routing)
    elif model.access_programs and response.model:
        result_metadata.served_model = response.model
    elif model.provider == "openai":
        result_metadata.extra["openai_response_model"] = response.model

    result = result_builder.build(
        finish_reason=mapped_finish_reason,
        tool_calls=tool_calls,
        provider_tool_events=provider_tool_events,
        history=[*input, RawResponse(response=response.output)],
        extras=QueryResultExtras(
            provider_response_id=response.id,
            provider_request_id=provider_request_id,
            citations=citations,
            search_results=_safe_search_results(
                getattr(response, "search_results", None), query_logger
            ),
        ),
        metadata=result_metadata,
    )
    return result


async def get_rate_limit(model: OpenAIModel) -> RateLimit | None:
    # NOTE: with_streaming_response doesn't seem to always work
    response = await model.get_client().responses.with_raw_response.create(
        max_output_tokens=16,
        input="Do not think. Say 'ok'",
        model=model.model_name,
    )
    # NOTE: for openai, max_tokens is used to reject requests if the amount of tokens left is less than the max_tokens
    # we calculate estimated_tokens as (character_count / 4) + max_tokens. Note that OpenAI's rate limiter doesn't tokenize the request using the model's specific tokenizer but relies on a character count-based heuristic.
    return rate_limit_from_headers(
        response.headers,
        default_scope="shared" if model.provider == "azure" else None,
    )


async def query_json(
    model: OpenAIModel,
    input: Sequence[InputItem],
    pydantic_model: type[PydanticT],
    output_schema: dict[str, Any] | type[BaseModel] | None = None,
    **kwargs: object,
) -> PydanticT:
    # re-use existing body
    body = await model.build_body(
        input,
        tools=[],
        output_schema=output_schema,
        **kwargs,
    )

    async def _query():
        try:
            return await model.get_client().responses.parse(
                text_format=pydantic_model,
                **body,  # pyright: ignore[reportAny]
            )
        except APIConnectionError:
            raise ImmediateRetryException("Failed to connect to OpenAI")

    response = await BaseRetrier.immediate_retry_wrapper(
        func=_query, logger=model.instance_logger
    )

    parsed: PydanticT | None = response.output_parsed
    if parsed is None:
        raise ModelNoOutputError("Model returned empty response")

    return parsed
