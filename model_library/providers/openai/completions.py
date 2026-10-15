from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any, cast

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallUnion,
)
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from openai.types.chat.chat_completion_message_tool_call import Function
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel

from model_library.base import (
    FileInput,
    FileWithBase64,
    FileWithId,
    FileWithUrl,
    FinishReason,
    FinishReasonInfo,
    InputItem,
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
    UnexpectedSystemInputError,
    handle_empty_response,
)
from model_library.model_utils import get_reasoning_in_tag
from model_library.agent.tool import is_native_web_search
from model_library.rate_limits import RateLimit, rate_limit_from_headers

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


def map_openai_completions_finish_reason(
    finish_reason: str | None,
) -> FinishReasonInfo:
    match finish_reason:
        case "stop":
            reason = FinishReason.STOP
        case "length":
            reason = FinishReason.MAX_TOKENS
        case "tool_calls" | "function_call":
            reason = FinishReason.TOOL_CALLS
        case "content_filter" | "sensitive":
            reason = FinishReason.CONTENT_FILTER
        case "model_context_window_exceeded":
            reason = FinishReason.CONTEXT_WINDOW_EXCEEDED
        case _:
            reason = FinishReason.UNKNOWN

    return FinishReasonInfo(reason=reason, raw=finish_reason)


async def get_tool_call_ids(
    model: OpenAIModel, input: Sequence[InputItem]
) -> list[str]:
    calls = [
        call
        for item in input
        if isinstance(item, RawResponse)
        and isinstance(item.response, ChatCompletionMessage)
        and item.response.tool_calls
        for call in item.response.tool_calls
    ]
    return [call.id for call in calls if call.id]


async def parse_input(
    model: OpenAIModel, input: Sequence[InputItem], **kwargs: Any
) -> list[dict[str, Any] | Any]:
    new_input: list[dict[str, Any] | Any] = []
    if isinstance(input[0], SystemInput):
        new_input.append({"role": "system", "content": input[0].text})
        input = input[1:]
    tool_call_ids = await model.get_tool_call_ids(input)
    for item in iter_input_groups(input):
        if isinstance(item, list):
            content = await user_content(model, item, "text")
            if model.provider == "kimi" and all(
                item.get("type") == "text" and isinstance(item.get("text"), str)
                for item in content
            ):
                new_input.append(
                    {
                        "role": "user",
                        "content": "".join(cast(str, item["text"]) for item in content),
                    }
                )
            else:
                new_input.append({"role": "user", "content": content})
            continue
        match item:
            case ToolResult():
                validate_tool_result(item, tool_call_ids)
                new_input.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.tool_call.id,
                        "content": item.result,
                    }
                )
            case RawResponse():
                new_input.append(item.response)
            case RawInput():
                new_input.append(item.input)
            case SystemInput():
                raise UnexpectedSystemInputError()
    return new_input


async def parse_image(model: OpenAIModel, image: FileInput) -> dict[str, Any]:
    base_dict: dict[str, Any] = {
        "type": "image_url",
        "image_url": {"detail": "auto"},
    }
    match image:
        case FileWithBase64():
            base_dict["image_url"]["url"] = f"data:{image.mime};base64,{image.base64}"
        case FileWithUrl():
            base_dict["image_url"]["url"] = image.url
        case FileWithId():
            raise Exception("Completions endpoint does not support file_id")
        case _:
            raise BadInputError("OpenAI does not support byte-backed files")
    return base_dict


async def parse_file(model: OpenAIModel, file: FileInput) -> dict[str, Any]:
    base_dict: dict[str, Any] = {"type": "file", "file": {}}
    match file:
        case FileWithBase64():
            base_dict["file"]["file_data"] = f"data:{file.mime};base64,{file.base64}"
        case FileWithUrl():
            if model.provider != "openrouter":
                raise BadInputError("Completions endpoint does not support url")
            # OpenRouter's PDF parser accepts a direct URL in file_data.
            base_dict["file"]["filename"] = file.name
            base_dict["file"]["file_data"] = file.url
        case FileWithId():
            base_dict["file"]["file_id"] = file.file_id
        case _:
            raise BadInputError("OpenAI does not support byte-backed files")
    return base_dict


async def parse_tools(
    model: OpenAIModel, tools: Sequence[ToolDefinition]
) -> list[ChatCompletionToolParam | Any]:
    if any(is_native_web_search(tool.body) for tool in tools):
        raise NotImplementedError(
            "Native web search is not supported on the Chat Completions path. "
            "Use the Responses API (use_completions=False)."
        )
    parsed_tools: list[ChatCompletionToolParam | Any] = []
    for tool in tools:
        body = tool.body
        if not isinstance(body, ToolBody):
            parsed_tools.append(body)
            continue
        payload = {"type": "function", "function": function_tool_payload(body)}
        parsed_tools.append(cast(ChatCompletionToolParam, payload))
    return parsed_tools


async def build_body(
    model: OpenAIModel,
    input: Sequence[InputItem],
    *,
    tools: list[ToolDefinition],
    output_schema: dict[str, Any] | type[BaseModel] | None = None,
    **kwargs: object,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model.model_name,
        "messages": await model.parse_input(input),
    }
    if model.stream_completions:
        # enable usage data in streaming responses
        body["stream_options"] = {"include_usage": True}

    if model.max_tokens:
        body["max_tokens"] = model.max_tokens

    parsed_tools = await model.parse_tools(tools)
    if parsed_tools:
        body["tools"] = parsed_tools

    # DeepSeek, Ant, Baseten, Together and older Kimi modes use max_tokens. Ant ignores
    # max_completion_tokens outright, which would leave generation uncapped.
    # Kimi K3 follows the current API contract and uses max_completion_tokens.
    if (
        model.reasoning
        and model.max_tokens
        and model.provider not in ("deepseek", "ant", "baseten", "together")
        and (model.provider != "kimi" or model.model_name == "kimi-k3")
    ):
        del body["max_tokens"]
        body["max_completion_tokens"] = model.max_tokens

    if model.provider == "google":
        if model.reasoning:
            # Google's OpenAI-compat endpoint uses a nested extra_body.google.thinking_config
            # to return visible thinking tokens. reasoning_effort conflicts with thinking_config
            # so only one can be used; map reasoning_effort → thinking_level inside thinking_config.
            thinking_config: dict[str, Any] = {"include_thoughts": True}
            if isinstance(model.reasoning_effort, str):
                thinking_config["thinking_level"] = model.reasoning_effort
            google_extra = cast(dict[str, Any], body.setdefault("extra_body", {}))
            nested = cast(dict[str, Any], google_extra.setdefault("extra_body", {}))
            google_section = cast(dict[str, Any], nested.setdefault("google", {}))
            google_section["thinking_config"] = thinking_config
    elif model.reasoning_effort is not None:
        # some model endpoints (like `fireworks/deepseek-v3p2`)
        # require explicitly setting reasoning effort to disable thinking
        body["reasoning_effort"] = model.reasoning_effort

    await apply_common_body(model, body, input, kwargs)

    # top_k isn't a standard chat-completions field; forward it via
    # extra_body so non-OpenAI backends (e.g. Gemini OpenAI-compat)
    # receive it while the OpenAI SDK doesn't reject it client-side.
    if model.supports_temperature and model.top_k is not None:
        extra_body = cast(dict[str, Any], body.setdefault("extra_body", {}))
        extra_body.setdefault("top_k", model.top_k)

    if output_schema is not None:
        schema = output_schema_value(output_schema)
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_output",
                "schema": schema,
                "strict": True,
            },
        }

    # Merge rather than replace extra_body: delegates pass their own via
    # kwargs, which would otherwise discard entries set above (notably top_k).
    extra_body_kwarg = kwargs.pop("extra_body", None)
    body.update(kwargs)
    if extra_body_kwarg:
        merged = cast(dict[str, Any], body.setdefault("extra_body", {}))
        merged.update(cast(dict[str, Any], extra_body_kwarg))

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
    """
    Completions endpoint
    Generally not used for openai models
    Used by providers using openai as a delegate
    """
    if model.deep_research:
        raise Exception("Use responses endpoint for deep research models")
    body = await model.build_body(
        input, tools=tools, output_schema=output_schema, **kwargs
    )

    raw_tool_calls: list[ChatCompletionMessageToolCall] = []
    tool_calls_by_stream_index: dict[int, ChatCompletionMessageToolCall] = {}

    finish_reason: str | None = None
    refusal: str | None = None
    metadata: QueryResultMetadata = QueryResultMetadata()
    routing: dict[str, Any] | None = {} if model.router_mode else None

    def metadata_from_usage(usage: CompletionUsage) -> QueryResultMetadata:
        reasoning_tokens = (
            usage.completion_tokens_details.reasoning_tokens
            if usage.completion_tokens_details
            else None
        )
        if reasoning_tokens is None:
            reasoning_tokens = getattr(usage, "reasoning_tokens", None)  # SGLang
        details = usage.prompt_tokens_details
        if details:
            cache_read_tokens = details.cached_tokens
        else:
            cache_read_tokens = getattr(usage, "cached_tokens", None)  # for kimi
        if cache_read_tokens is None:
            cache_read_tokens = getattr(usage, "cached_input_tokens", 0)
        cache_write_tokens = getattr(details, "cache_write_tokens", None)
        in_tokens = usage.prompt_tokens
        if model.provider == "digitalocean":
            usage_extra = usage.model_extra or {}
            cache_read_tokens = (
                cache_read_tokens or usage_extra.get("cache_read_input_tokens") or 0
            )
            cache_write_tokens = (
                cache_write_tokens or usage_extra.get("cache_created_input_tokens") or 0
            )
        if in_tokens:
            in_tokens -= cache_read_tokens + (cache_write_tokens or 0)
        out_tokens = usage.completion_tokens - (reasoning_tokens or 0)
        if model.provider == "sakana":
            # Orchestration tokens are billed like normal ones; the cached ones
            # are a subset of the orchestration input.
            orchestration_in = getattr(details, "orchestration_input_tokens", None) or 0
            orchestration_cached = (
                getattr(details, "orchestration_input_cached_tokens", None) or 0
            )
            cache_read_tokens += orchestration_cached
            in_tokens += orchestration_in - orchestration_cached
            out_tokens += (
                getattr(
                    usage.completion_tokens_details, "orchestration_output_tokens", None
                )
                or 0
            )
        return QueryResultMetadata(
            in_tokens=in_tokens,
            out_tokens=out_tokens,
            reasoning_tokens=reasoning_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        )

    def should_start_new_tool_call_segment(
        existing: ChatCompletionMessageToolCall | None,
        chunk: ChoiceDeltaToolCall,
    ) -> bool:
        if not chunk.id:
            return False
        if existing is None:
            return True
        if not existing.id:
            return False
        func = chunk.function
        is_deepseek_new_same_id_call = (
            existing.id == chunk.id
            and model.provider == "deepseek"
            and func is not None
            and bool(func.name)
        )
        return (
            model.provider != "poolside" and existing.id != chunk.id
        ) or is_deepseek_new_same_id_call

    def create_empty_tool_call() -> ChatCompletionMessageToolCall:
        return ChatCompletionMessageToolCall(
            id="",
            type="function",
            function=Function(name="", arguments=""),
        )

    def apply_tool_call_chunk(
        tool_call: ChatCompletionMessageToolCall,
        chunk: ChoiceDeltaToolCall,
    ) -> None:
        if chunk.id and (not tool_call.id or model.provider != "poolside"):
            tool_call.id = chunk.id
        func = chunk.function
        if func is not None:
            if func.name:
                tool_call.function.name = func.name
            if func.arguments:
                tool_call.function.arguments += func.arguments
        if model.provider == "google":
            extra_content = (chunk.model_extra or {}).get("extra_content")
            if extra_content is not None:
                setattr(tool_call, "extra_content", extra_content)

    result_builder = QueryResultBuilder()
    completion = await model.get_client().chat.completions.create(
        **body,  # pyright: ignore[reportAny]
        stream=model.stream_completions,
    )

    completion_id: str | None = None
    provider_request_id: str | None = None
    if not model.stream_completions:
        completion = cast(ChatCompletion, completion)
        completion_id = completion.id
        provider_request_id = getattr(completion, "_request_id", None)
        query_logger.debug(f"Completion created: {completion.id}")
        if completion.choices:
            choice = completion.choices[0]
            finish_reason = choice.finish_reason
            message = choice.message
            result_builder.set_output_text(message.content)
            refusal = message.refusal
            if hasattr(message, "reasoning_content"):
                raw_reasoning_content = getattr(message, "reasoning_content")
                if raw_reasoning_content is not None:
                    result_builder.set_reasoning(cast(str, raw_reasoning_content))
            elif hasattr(message, "reasoning"):
                raw_reasoning = getattr(message, "reasoning")
                if raw_reasoning is not None:
                    result_builder.set_reasoning(cast(str, raw_reasoning))
            raw_tool_calls = cast(
                list[ChatCompletionMessageToolCall], message.tool_calls or []
            )
        if completion.usage:
            metadata = metadata_from_usage(completion.usage)
        if routing is not None:
            capture_routing(routing, completion)
    else:
        if routing is not None:
            headers = getattr(getattr(completion, "response", None), "headers", {})
            if route := headers.get("x-model-router-selected-route"):
                routing["router_route"] = route
        stream = cast(AsyncIterator[ChatCompletionChunk], completion)
        async for chunk in stream:
            if not completion_id:
                completion_id = chunk.id
                provider_request_id = getattr(chunk, "_request_id", None)
                query_logger.debug(f"Completion created: {completion_id}")
            if routing is not None:
                capture_routing(routing, chunk)

            if chunk.choices:
                choice = chunk.choices[0]

                if choice.finish_reason:
                    finish_reason = choice.finish_reason

                if choice.delta and choice.delta.content is not None:
                    result_builder.append_content_delta(choice.delta.content)
                if choice.delta and choice.delta.refusal:
                    refusal = (refusal or "") + choice.delta.refusal

                if choice.delta and hasattr(choice.delta, "reasoning_content"):
                    raw_reasoning_delta = getattr(choice.delta, "reasoning_content")
                    if raw_reasoning_delta is not None:
                        result_builder.append_reasoning_delta(
                            cast(str, raw_reasoning_delta)
                        )
                elif choice.delta and hasattr(choice.delta, "reasoning"):
                    raw_reasoning_delta = getattr(choice.delta, "reasoning")
                    if raw_reasoning_delta is not None:
                        result_builder.append_reasoning_delta(
                            cast(str, raw_reasoning_delta)
                        )

                if choice.delta and choice.delta.tool_calls:
                    for tool_call_chunk in choice.delta.tool_calls:
                        existing_tool_call = tool_calls_by_stream_index.get(
                            tool_call_chunk.index
                        )
                        if should_start_new_tool_call_segment(
                            existing_tool_call, tool_call_chunk
                        ):
                            result_builder.start_tool_call_segment(
                                tool_call_chunk.index
                            )
                            existing_tool_call = create_empty_tool_call()
                            raw_tool_calls.append(existing_tool_call)
                            tool_calls_by_stream_index[tool_call_chunk.index] = (
                                existing_tool_call
                            )
                        elif existing_tool_call is None:
                            existing_tool_call = create_empty_tool_call()
                            raw_tool_calls.append(existing_tool_call)
                            tool_calls_by_stream_index[tool_call_chunk.index] = (
                                existing_tool_call
                            )

                        apply_tool_call_chunk(existing_tool_call, tool_call_chunk)

                        func = tool_call_chunk.function
                        has_tool_ready = bool(
                            tool_call_chunk.id or (func is not None and func.name)
                        )
                        extra_content = (tool_call_chunk.model_extra or {}).get(
                            "extra_content"
                        )
                        has_tool_delta = (
                            func is not None and bool(func.arguments)
                        ) or extra_content is not None
                        result_builder.record_tool_call_progress(
                            tool_call_chunk.index,
                            ready=has_tool_ready,
                            delta=has_tool_delta,
                        )
            if chunk.usage:
                # NOTE: see _calculate_cost
                metadata = metadata_from_usage(chunk.usage)

    if model.stream_completions:
        raw_tool_calls = [tool_call for tool_call in raw_tool_calls if tool_call.id]

    if routing is not None:
        metadata.served_model = routing.pop("served_model", None)
        metadata.extra.update(routing)

    mapped_finish_reason = map_openai_completions_finish_reason(finish_reason)

    output_text = result_builder.output_text
    reasoning_text = result_builder.reasoning
    if (
        model.reasoning
        and model.provider not in {"openai", "azure", "deepseek", "openrouter"}
        and output_text is not None
        and reasoning_text is None
    ):
        stripped_output_text, extracted_reasoning_text = get_reasoning_in_tag(
            output_text
        )
        reasoning_text = extracted_reasoning_text or None
        result_builder.set_output_text(stripped_output_text)
        result_builder.set_reasoning(reasoning_text)
        output_text = result_builder.output_text
    no_useful_content = (
        not result_builder.has_output_text
        and not result_builder.has_reasoning
        and not raw_tool_calls
    )

    if refusal:
        raise ContentFilterError(refusal)

    if no_useful_content:
        handle_empty_response(mapped_finish_reason, {"metadata": metadata})

    tool_calls: list[ToolCall] = []
    for raw_tool_call in raw_tool_calls:
        tool_calls.append(
            ToolCall(
                id=raw_tool_call.id,
                name=raw_tool_call.function.name,
                args=raw_tool_call.function.arguments,
            )
        )

    # build final message for history
    omit_null_message_fields = model.normalize_null_assistant_history_fields
    if result_builder.has_output_text:
        history_content = output_text
    elif omit_null_message_fields and reasoning_text and not raw_tool_calls:
        history_content = "[reasoning-only response omitted]"
    elif omit_null_message_fields:
        history_content = ""
    else:
        history_content = None

    final_message = ChatCompletionMessage(
        role="assistant",
        content=history_content,
    )
    if raw_tool_calls:
        final_message.tool_calls = cast(
            list[ChatCompletionMessageToolCallUnion], raw_tool_calls
        )
    elif not omit_null_message_fields:
        final_message.tool_calls = None
    preserve_empty_reasoning = (
        model.provider == "kimi" and model.model_name == "kimi-k3"
    )
    if reasoning_text or preserve_empty_reasoning:
        setattr(final_message, "reasoning_content", reasoning_text or "")

    result = result_builder.build(
        finish_reason=mapped_finish_reason,
        tool_calls=tool_calls,
        history=[*input, RawResponse(response=final_message)],
        metadata=metadata,
        extras=QueryResultExtras(
            provider_response_id=completion_id,
            provider_request_id=provider_request_id,
        ),
    )
    return result


async def get_rate_limit(model: OpenAIModel) -> RateLimit | None:
    # NOTE: with_streaming_response doesn't seem to always work
    response = await model.get_client().chat.completions.with_raw_response.create(
        max_completion_tokens=16,
        model=model.model_name,
        messages=[{"role": "user", "content": "Do not think. Say 'ok'"}],
        stream=True,
    )
    # NOTE: for openai, max_tokens is used to reject requests if the amount of tokens left is less than the max_tokens
    # we calculate estimated_tokens as (character_count / 4) + max_tokens. Note that OpenAI's rate limiter doesn't tokenize the request using the model's specific tokenizer but relies on a character count-based heuristic.
    return rate_limit_from_headers(
        response.headers,
        default_scope="shared" if model.provider == "azure" else None,
    )
