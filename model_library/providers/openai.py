from __future__ import annotations

import datetime
import io
import json
import logging
import time
from collections import deque
from collections.abc import Hashable, Mapping, Sequence
from typing import Any, AsyncIterator, Literal, cast

from openai import APIConnectionError, AsyncOpenAI
from openai.lib._pydantic import to_strict_json_schema
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
from openai.types.create_embedding_response import CreateEmbeddingResponse
from openai.types.moderation_create_response import ModerationCreateResponse
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputText,
)
from openai.types.responses.response_function_web_search import ActionSearch
from openai.types.responses.response import Response
from openai.types.responses.response_stream_event import ResponseStreamEvent
from openai.types.responses.tool_param import ToolParam as ResponsesToolParam
from pydantic import BaseModel, JsonValue
from typing_extensions import deprecated, override

from model_library import model_library_settings
from model_library.base import (
    LLM,
    BatchResult,
    Citation,
    FileBase,
    FileInput,
    FileWithBase64,
    FileWithId,
    FileWithUrl,
    FinishReason,
    FinishReasonInfo,
    InputItem,
    LLMBatchMixin,
    LLMConfig,
    ProviderConfig,
    PydanticT,
    ProviderToolEvent,
    QueryResult,
    QueryResultCost,
    QueryResultExtras,
    QueryResultMetadata,
    RateLimit,
    RawInput,
    RawResponse,
    SystemInput,
    TextInput,
    ToolBody,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from model_library.base.output.builder import QueryResultBuilder
from model_library.base.query_ids import PromptCacheKeyMode, resolve_prompt_cache_key
from model_library.exceptions import (
    ImmediateRetryException,
    ModelNoOutputError,
    NoMatchingToolCallError,
    UnexpectedSystemInputError,
    handle_empty_response,
)
from model_library.model_utils import get_reasoning_in_tag
from model_library.agent.tool import is_native_web_search
from model_library.register_models import register_provider
from model_library.retriers.base import BaseRetrier
from model_library.utils import create_openai_client_with_defaults


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
        case "content_filter":
            reason = FinishReason.CONTENT_FILTER
        case "model_context_window_exceeded":
            reason = FinishReason.CONTEXT_WINDOW_EXCEEDED
        case _:
            reason = FinishReason.UNKNOWN

    return FinishReasonInfo(reason=reason, raw=finish_reason)


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


class OpenAIBatchMixin(LLMBatchMixin):
    COMPLETED_BATCH_STATUSES: list[str] = [
        "failed",
        "completed",
        "expired",
        "cancelled",
    ]

    def __init__(self, openai: OpenAIModel):
        self._root: OpenAIModel = openai
        self._client: AsyncOpenAI = self._root.get_client()

    @override
    async def create_batch_query_request(
        self,
        custom_id: str,
        input: Sequence[InputItem],
        output_schema: dict[str, Any] | type[BaseModel] | None = None,
        **kwargs: object,
    ) -> dict[str, Any]:
        tools_override = kwargs.pop("tools", None)
        normalized_tools: list[ToolDefinition]
        if tools_override is None:
            normalized_tools = []
        elif isinstance(tools_override, ToolDefinition):
            normalized_tools = [tools_override]
        elif isinstance(tools_override, Sequence) and not isinstance(
            tools_override, (str, bytes)
        ):
            normalized_tools = []
            tools_override_seq = cast(Sequence[object], tools_override)
            for tool in tools_override_seq:
                if not isinstance(tool, ToolDefinition):
                    raise TypeError(
                        "tools must contain ToolDefinition instances when batching"
                    )
                normalized_tools.append(tool)
        else:
            raise TypeError(
                "tools must be a ToolDefinition or a sequence of ToolDefinition instances"
            )
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": await self._root.build_body(
                input,
                tools=normalized_tools,
                output_schema=output_schema,
                **kwargs,  # pyright: ignore[reportArgumentType]
            ),
        }

    @override
    async def batch_query(
        self,
        batch_name: str,
        requests: list[dict[str, Any]],
    ) -> str:
        """Sends a batch api query and returns batch id."""
        input_jsonl_str = "\n".join(json.dumps(req) for req in requests)
        input_jsonl_bytes = io.BytesIO(input_jsonl_str.encode("utf-8"))
        input_jsonl_bytes.name = batch_name

        batch_input_file = await self._client.files.create(
            file=input_jsonl_bytes, purpose="batch"
        )

        # TODO: Parameterize completion window
        completion_window = "24h"

        batch = await self._client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/responses",
            completion_window=completion_window,
            metadata={"description": batch_name},
        )
        return batch.id

    @override
    async def get_batch_results(self, batch_id: str) -> list[BatchResult]:
        batch = await self._client.batches.retrieve(batch_id)

        if not batch:
            raise Exception(f"Couldn't retrieve batch results for batch {batch_id}.")

        batch_results: list[BatchResult] = []

        if batch.output_file_id:
            successful_responses = await self._client.files.content(
                batch.output_file_id
            )
            successful_results: list[dict[str, Any]] = [
                json.loads(line) for line in successful_responses.iter_lines() if line
            ]
            for result in successful_results:
                id = cast(str, result["response"]["body"]["id"])
                response: Response = await self._client.responses.retrieve(id)

                output = QueryResult(
                    output_text=response.output_text,
                )
                if response.usage:
                    output.metadata.in_tokens = response.usage.input_tokens
                    output.metadata.out_tokens = response.usage.output_tokens
                output.extras.response_id = response.id
                output.extras.provider_response_id = response.id
                output.extras.provider_request_id = getattr(
                    response, "_request_id", None
                )

                batch_results.append(
                    BatchResult(
                        custom_id=cast(str, result["custom_id"]),
                        output=output,
                    )
                )

        if batch.error_file_id:
            failed_responses = await self._client.files.content(batch.error_file_id)
            failed_results: list[dict[str, Any]] = [
                json.loads(line) for line in failed_responses.iter_lines() if line
            ]
            for result in failed_results:
                error_message = cast(
                    str, result["response"]["body"]["error"]["message"]
                )
                output = QueryResult(
                    output_text=error_message,
                )
                batch_results.append(
                    BatchResult(
                        custom_id=cast(str, result["custom_id"]),
                        output=output,
                        error_message=error_message,
                    )
                )

        return batch_results

    @override
    async def get_batch_progress(self, batch_id: str) -> int:
        batch = await self._client.batches.retrieve(batch_id)
        if batch and batch.request_counts:
            completed = batch.request_counts.completed
        else:
            self._root.instance_logger.error(f"Couldn't retrieve {batch_id}")
            completed = 0
        return completed

    @override
    async def cancel_batch_request(self, batch_id: str):
        self._root.instance_logger.debug(f"Cancelling {batch_id}")
        _ = await self._client.batches.cancel(batch_id)

    @override
    async def get_batch_status(self, batch_id: str) -> str:
        batch = await self._client.batches.retrieve(batch_id)
        return batch.status

    @override
    @classmethod
    def is_batch_status_completed(cls, batch_status: str) -> bool:
        return batch_status in OpenAIBatchMixin.COMPLETED_BATCH_STATUSES

    @override
    @classmethod
    def is_batch_status_failed(cls, batch_status: str) -> bool:
        return batch_status == "failed"

    @override
    @classmethod
    def is_batch_status_cancelled(cls, batch_status: str) -> bool:
        return batch_status == "cancelled"


OpenAIToolCallMode = Literal["default", "auto", "code_mode"]


class OpenAIConfig(ProviderConfig):
    deep_research: bool = False
    tool_call_mode: OpenAIToolCallMode = "default"
    verbosity: Literal["low", "medium", "high"] | None = None
    prompt_cache_retention: Literal["24h", "in_memory"] | None = None
    prompt_cache_key: PromptCacheKeyMode | None = None
    reasoning_context: Literal["current_turn", "all_turns"] | None = None
    parallel_tool_calls: bool | None = None
    # TODO: move to LLMConfig so OpenAI-compatible delegate providers can configure it.
    stream_completions: bool = True


@register_provider("openai")
class OpenAIModel(LLM):
    provider_config = OpenAIConfig()

    @override
    def _get_default_api_key(self) -> str:
        return model_library_settings.OPENAI_API_KEY

    @override
    def get_client(
        self, api_key: str | None = None, base_url: str | None = None
    ) -> AsyncOpenAI:
        if not self.has_client():
            assert api_key
            dns_resolve: dict[str, str] | None = None
            client = create_openai_client_with_defaults(
                base_url=base_url,
                api_key=api_key,
                dns_resolve=dns_resolve,
            )
            self.assign_client(client)
        return super().get_client()

    def __init__(
        self,
        model_name: str,
        provider: str = "openai",
        *,
        config: LLMConfig | None = None,
        use_completions: bool = False,
    ):
        self.use_completions: bool = (
            use_completions  # TODO: do completions in a separate file
        )

        super().__init__(model_name, provider, config=config)

        self.deep_research = self.provider_config.deep_research
        self.tool_call_mode = self.provider_config.tool_call_mode
        self.verbosity = self.provider_config.verbosity
        self.prompt_cache_retention = self.provider_config.prompt_cache_retention
        self.prompt_cache_key_mode: PromptCacheKeyMode | None = (
            self.provider_config.prompt_cache_key
        )
        self.reasoning_context = self.provider_config.reasoning_context
        self.parallel_tool_calls = self.provider_config.parallel_tool_calls
        self.stream_completions = self.provider_config.stream_completions

        # batch client
        self.supports_batch: bool = self.supports_batch and not self.custom_endpoint
        self.batch: LLMBatchMixin | None = (
            OpenAIBatchMixin(self) if self.supports_batch else None
        )

    async def get_tool_call_ids(self, input: Sequence[InputItem]) -> list[str]:
        raw_responses = [x for x in input if isinstance(x, RawResponse)]
        tool_call_ids: list[str] = []

        if self.use_completions:
            calls = [
                y
                for x in raw_responses
                if isinstance(x.response, ChatCompletionMessage)
                and x.response.tool_calls
                for y in x.response.tool_calls
            ]
            tool_call_ids.extend([x.id for x in calls if x.id])
        else:
            calls = [
                y
                for x in raw_responses
                for y in x.response
                if isinstance(y, ResponseFunctionToolCall)
            ]
            tool_call_ids.extend([x.id for x in calls if x.id])
        return tool_call_ids

    @override
    async def parse_input(
        self,
        input: Sequence[InputItem],
        **kwargs: Any,
    ) -> list[dict[str, Any] | Any]:
        new_input: list[dict[str, Any] | Any] = []

        if isinstance(input[0], SystemInput):
            if self.use_completions:
                new_input.append({"role": "system", "content": input[0].text})
            else:
                new_input.append(
                    {
                        "role": "developer",
                        "content": [{"type": "input_text", "text": input[0].text}],
                    }
                )
            input = input[1:]

        content_user: list[dict[str, Any]] = []

        def flush_content_user():
            if content_user:
                # NOTE: must make new object as we clear()
                new_input.append({"role": "user", "content": content_user.copy()})
                content_user.clear()

        tool_call_ids = await self.get_tool_call_ids(input)

        for item in input:
            if isinstance(item, TextInput):
                if self.use_completions:
                    text_key = "text"
                else:
                    text_key = "input_text"
                content_user.append({"type": text_key, "text": item.text})
                continue

            if isinstance(item, FileBase):
                match item.type:
                    case "image":
                        parsed = await self.parse_image(item)
                    case "file":
                        parsed = await self.parse_file(item)
                content_user.append(parsed)
                continue

            # non content user item
            flush_content_user()

            match item:
                case ToolResult():
                    if item.tool_call.id not in tool_call_ids:
                        raise NoMatchingToolCallError()

                    if self.use_completions:
                        new_input.append(
                            {
                                "role": "tool",
                                "tool_call_id": item.tool_call.id,
                                "content": item.result,
                            }
                        )
                    else:
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
                    if self.use_completions:
                        new_input.append(item.response)
                    else:
                        new_input.extend(item.response)
                case RawInput():
                    new_input.append(item.input)
                case SystemInput():
                    raise UnexpectedSystemInputError()

        # in case content user item is the last item
        flush_content_user()

        return new_input

    @override
    async def parse_image(
        self,
        image: FileInput,
    ) -> dict[str, Any]:
        base_dict: dict[str, Any]
        if self.use_completions:
            base_dict = {
                "type": "image_url",
                "image_url": {
                    "detail": "auto",
                },
            }
            match image:
                case FileWithBase64():
                    base_dict["image_url"]["url"] = (
                        f"data:image/{image.mime};base64,{image.base64}"
                    )
                case FileWithUrl():
                    base_dict["image_url"]["url"] = image.url
                case FileWithId():
                    raise Exception("Completions endpoint does not support file_id")
        else:
            base_dict = {
                "type": "input_image",
                "detail": "auto",
            }
            match image:
                case FileWithBase64():
                    base_dict["image_url"] = (
                        f"data:image/{image.mime};base64,{image.base64}"
                    )
                case FileWithUrl():
                    base_dict["image_url"] = image.url
                case FileWithId():
                    base_dict["file_id"] = image.file_id
        return base_dict

    @override
    async def parse_file(
        self,
        file: FileInput,
    ) -> dict[str, Any]:
        base_dict: dict[str, Any]
        if self.use_completions:
            base_dict = {
                "type": "file",
                "file": {},
            }
            match file:
                case FileWithBase64():
                    base_dict["file"]["file_data"] = (
                        f"data:{file.mime};base64,{file.base64}"
                    )
                case FileWithUrl():
                    raise Exception("Completions endpoint does not support url")
                case FileWithId():
                    base_dict["file"]["file_id"] = file.file_id
        else:
            base_dict = {
                "type": "input_file",
            }
            match file:
                case FileWithBase64():
                    base_dict["filename"] = file.name
                    base_dict["file_data"] = f"data:{file.mime};base64,{file.base64}"
                case FileWithUrl():
                    base_dict["file_url"] = file.url
                case FileWithId():
                    base_dict["file_id"] = file.file_id
        return base_dict

    @property
    @override
    def search_tool(self) -> dict[str, str]:
        return {"type": "web_search_preview"}

    @override
    async def parse_tools(
        self,
        tools: Sequence[ToolDefinition],
    ) -> list[ChatCompletionToolParam | ResponsesToolParam | Any]:

        if self.use_completions and any(is_native_web_search(t.body) for t in tools):
            raise NotImplementedError(
                "Native web search is not supported on the Chat Completions path. "
                "Use the Responses API (use_completions=False)."
            )

        parsed_tools: list[ChatCompletionToolParam | ResponsesToolParam | Any] = []
        for tool in tools:
            body = tool.body
            if is_native_web_search(body):
                parsed_tools.append(self.search_tool)
                continue
            if not isinstance(body, ToolBody):
                parsed_tools.append(body)
                continue

            parameters = {
                "type": "object",
                "properties": body.properties,
                "required": body.required,
                "additionalProperties": body.kwargs.get("additional_properties", False),
            }
            base_payload = {
                "name": body.name,
                "description": body.description,
                "parameters": parameters,
                "strict": body.kwargs.get("strict", False),
            }

            if body.allowed_callers is not None:
                base_payload["allowed_callers"] = body.allowed_callers

            if self.use_completions:
                payload = {
                    "type": "function",
                    "function": base_payload,
                }
                parsed_tools.append(cast(ChatCompletionToolParam, payload))
            else:
                payload = {
                    "type": "function",
                    **base_payload,
                }
                parsed_tools.append(cast(ResponsesToolParam, payload))

        return parsed_tools

    @override
    async def upload_file(
        self,
        name: str,
        mime: str,
        bytes: io.BytesIO,
        type: Literal["image", "file"] = "file",
    ) -> FileWithId:
        response = await self.get_client().files.create(
            file=(name, bytes, mime),
            purpose="file-extract"  # type: ignore[reportArgumentType]
            if self.provider == "kimi" and type == "file"
            else "assistants",
        )

        return FileWithId(
            type=type,
            name=response.filename,
            mime=mime,
            file_id=response.id,
        )

    async def _build_body_completions(
        self,
        input: Sequence[InputItem],
        *,
        tools: list[ToolDefinition],
        output_schema: dict[str, Any] | type[BaseModel] | None = None,
        **kwargs: object,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": await self.parse_input(input),
        }
        if self.stream_completions:
            # enable usage data in streaming responses
            body["stream_options"] = {"include_usage": True}

        if self.max_tokens:
            body["max_tokens"] = self.max_tokens

        parsed_tools = await self.parse_tools(tools)
        if parsed_tools:
            body["tools"] = parsed_tools

        if self.parallel_tool_calls is not None:
            body["parallel_tool_calls"] = self.parallel_tool_calls

        # DeepSeek documents max_tokens for thinking mode, not max_completion_tokens
        if self.reasoning and self.max_tokens and self.provider not in {"deepseek"}:
            del body["max_tokens"]
            body["max_completion_tokens"] = self.max_tokens

        if self.provider == "google":
            if self.reasoning:
                # Google's OpenAI-compat endpoint uses a nested extra_body.google.thinking_config
                # to return visible thinking tokens. reasoning_effort conflicts with thinking_config
                # so only one can be used; map reasoning_effort → thinking_level inside thinking_config.
                thinking_config: dict[str, Any] = {"include_thoughts": True}
                if isinstance(self.reasoning_effort, str):
                    thinking_config["thinking_level"] = self.reasoning_effort
                google_extra = cast(dict[str, Any], body.setdefault("extra_body", {}))
                nested = cast(dict[str, Any], google_extra.setdefault("extra_body", {}))
                google_section = cast(dict[str, Any], nested.setdefault("google", {}))
                google_section["thinking_config"] = thinking_config
        elif self.reasoning_effort is not None:
            # some model endpoints (like `fireworks/deepseek-v3p2`)
            # require explicitly setting reasoning effort to disable thinking
            body["reasoning_effort"] = self.reasoning_effort

        if self.prompt_cache_retention is not None:
            body["prompt_cache_retention"] = self.prompt_cache_retention
        prompt_cache_key = await resolve_prompt_cache_key(
            mode=self.prompt_cache_key_mode,
            model_name=self.model_name,
            input=input,
            parse_prompt_prefix=self.parse_input,
            run_id=kwargs.pop("run_id", None),
            question_id=kwargs.pop("question_id", None),
        )
        if prompt_cache_key is not None:
            body["prompt_cache_key"] = prompt_cache_key

        if self.supports_temperature:
            if self.temperature is not None:
                body["temperature"] = self.temperature
            if self.top_p is not None:
                body["top_p"] = self.top_p
            # top_k isn't a standard chat-completions field; forward it via
            # extra_body so non-OpenAI backends (e.g. Gemini OpenAI-compat)
            # receive it while the OpenAI SDK doesn't reject it client-side.
            if self.top_k is not None:
                extra_body = cast(dict[str, Any], body.setdefault("extra_body", {}))
                extra_body.setdefault("top_k", self.top_k)

        if output_schema is not None:
            schema = (
                output_schema
                if isinstance(output_schema, dict)
                else to_strict_json_schema(output_schema)
            )
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "schema": schema,
                    "strict": True,
                },
            }

        body.update(kwargs)

        return body

    async def _query_completions(
        self,
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
        body = await self.build_body(
            input, tools=tools, output_schema=output_schema, **kwargs
        )

        raw_tool_calls: list[ChatCompletionMessageToolCall] = []
        tool_calls_by_stream_index: dict[int, ChatCompletionMessageToolCall] = {}

        finish_reason: str | None = None
        metadata: QueryResultMetadata = QueryResultMetadata()

        def metadata_from_usage(usage: CompletionUsage) -> QueryResultMetadata:
            reasoning_tokens = (
                usage.completion_tokens_details.reasoning_tokens
                if usage.completion_tokens_details
                else None
            )
            cache_read_tokens = (
                usage.prompt_tokens_details.cached_tokens or 0
                if usage.prompt_tokens_details
                else getattr(usage, "cached_tokens", 0)  # for kimi
            )
            return QueryResultMetadata(
                in_tokens=usage.prompt_tokens - cache_read_tokens,
                out_tokens=usage.completion_tokens - (reasoning_tokens or 0),
                reasoning_tokens=reasoning_tokens,
                cache_read_tokens=cache_read_tokens,
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
                and self.provider == "deepseek"
                and func is not None
                and bool(func.name)
            )
            return (
                self.provider != "poolside" and existing.id != chunk.id
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
            if chunk.id and (not tool_call.id or self.provider != "poolside"):
                tool_call.id = chunk.id
            func = chunk.function
            if func is not None:
                if func.name:
                    tool_call.function.name = func.name
                if func.arguments:
                    tool_call.function.arguments += func.arguments
            if self.provider == "google":
                extra_content = (chunk.model_extra or {}).get("extra_content")
                if extra_content is not None:
                    setattr(tool_call, "extra_content", extra_content)

        result_builder = QueryResultBuilder()
        completion = await self.get_client().chat.completions.create(
            **body,  # pyright: ignore[reportAny]
            stream=self.stream_completions,
        )

        completion_id: str | None = None
        provider_request_id: str | None = None
        if not self.stream_completions:
            completion = cast(ChatCompletion, completion)
            completion_id = completion.id
            provider_request_id = getattr(completion, "_request_id", None)
            query_logger.debug(f"Completion created: {completion.id}")
            if completion.choices:
                choice = completion.choices[0]
                finish_reason = choice.finish_reason
                message = choice.message
                result_builder.set_output_text(message.content)
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
        else:
            stream = cast(AsyncIterator[ChatCompletionChunk], completion)
            async for chunk in stream:
                if not completion_id:
                    completion_id = chunk.id
                    provider_request_id = getattr(chunk, "_request_id", None)
                    query_logger.debug(f"Completion created: {completion_id}")

                if chunk.choices:
                    choice = chunk.choices[0]

                    if choice.finish_reason:
                        finish_reason = choice.finish_reason

                    if choice.delta and choice.delta.content is not None:
                        result_builder.append_content_delta(choice.delta.content)

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

        if self.stream_completions:
            raw_tool_calls = [tool_call for tool_call in raw_tool_calls if tool_call.id]

        mapped_finish_reason = map_openai_completions_finish_reason(finish_reason)

        output_text = result_builder.output_text
        reasoning_text = result_builder.reasoning
        if (
            self.reasoning
            and self.provider not in {"openai", "azure", "deepseek", "openrouter"}
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
        final_message = ChatCompletionMessage(
            role="assistant",
            content=output_text if result_builder.has_output_text else None,
            tool_calls=cast(list[ChatCompletionMessageToolCallUnion], raw_tool_calls)
            if raw_tool_calls
            else None,
        )
        if reasoning_text:
            setattr(final_message, "reasoning_content", reasoning_text)

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

    async def _check_deep_research_args(
        self, tools: Sequence[ToolDefinition], **kwargs: object
    ) -> None:
        min_tokens = 30_000
        if not self.max_tokens or self.max_tokens < min_tokens:
            self.instance_logger.warning(
                f"Recommended to set max_tokens >= {min_tokens} for deep research models"
            )

        if "background" not in kwargs:
            self.instance_logger.warning(
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

    @override
    async def build_body(
        self,
        input: Sequence[InputItem],
        *,
        tools: list[ToolDefinition],
        output_schema: dict[str, Any] | type[BaseModel] | None = None,
        **kwargs: object,
    ) -> dict[str, Any]:
        if self.use_completions:
            return await self._build_body_completions(
                input, tools=tools, output_schema=output_schema, **kwargs
            )

        if self.deep_research:
            await self._check_deep_research_args(tools, **kwargs)

        parsed_tools: list[Any] = cast(list[Any], await self.parse_tools(tools))
        if self.tool_call_mode != "default":
            if self.tool_call_mode == "auto":
                allowed_callers = ["code_mode", "direct"]
            elif self.tool_call_mode == "code_mode":
                allowed_callers = ["code_mode"]
            else:
                raise ValueError(f"Unknown tool_call_mode: {self.tool_call_mode}")

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
            if self.tool_call_mode in {"auto", "code_mode"} and not has_code_mode_tool:
                parsed_tools.append({"type": "code_mode", "language": "javascript"})

        body: dict[str, Any] = {
            "model": self.model_name,
            "input": await self.parse_input(input),
        }

        if self.max_tokens:
            body["max_output_tokens"] = self.max_tokens

        if parsed_tools:
            body["tools"] = parsed_tools

        if self.reasoning:
            body["include"] = ["reasoning.encrypted_content"]
            body["reasoning"] = {"summary": "auto"}
            body["store"] = False
            if self.reasoning_effort is not None:
                body["reasoning"]["effort"] = self.reasoning_effort  # type: ignore[reportArgumentType]
            if self.reasoning_context is not None:
                body["reasoning"]["context"] = self.reasoning_context

        if self.verbosity is not None:
            body["text"] = {"format": {"type": "text"}, "verbosity": self.verbosity}

        if self.prompt_cache_retention is not None:
            body["prompt_cache_retention"] = self.prompt_cache_retention
        prompt_cache_key = await resolve_prompt_cache_key(
            mode=self.prompt_cache_key_mode,
            model_name=self.model_name,
            input=input,
            parse_prompt_prefix=self.parse_input,
            run_id=kwargs.pop("run_id", None),
            question_id=kwargs.pop("question_id", None),
        )
        if prompt_cache_key is not None:
            body["prompt_cache_key"] = prompt_cache_key

        if output_schema is not None:
            schema = (
                output_schema
                if isinstance(output_schema, dict)
                else to_strict_json_schema(output_schema)
            )
            text_block: dict[str, Any] = {
                "format": {
                    "type": "json_schema",
                    "name": "structured_output",
                    "schema": schema,
                    "strict": True,
                }
            }
            if self.verbosity is not None:
                text_block["verbosity"] = self.verbosity
            body["text"] = text_block

        if self.parallel_tool_calls is not None:
            body["parallel_tool_calls"] = self.parallel_tool_calls

        if self.supports_temperature:
            if self.temperature is not None:
                body["temperature"] = self.temperature
            if self.top_p is not None:
                body["top_p"] = self.top_p

        _ = kwargs.pop("stream", None)

        body.update(kwargs)
        return body

    @override
    async def _query_impl(
        self,
        input: Sequence[InputItem],
        *,
        tools: list[ToolDefinition],
        query_logger: logging.Logger,
        output_schema: dict[str, Any] | type[BaseModel] | None = None,
        **kwargs: object,
    ) -> QueryResult:
        if self.use_completions:
            if self.deep_research:
                raise Exception("Use responses endpoint for deep research models")
            return await self._query_completions(
                input,
                tools=tools,
                query_logger=query_logger,
                output_schema=output_schema,
                **kwargs,
            )

        stream_responses = bool(kwargs.pop("stream", True))
        body = await self.build_body(
            input, tools=tools, output_schema=output_schema, **kwargs
        )

        result_builder = QueryResultBuilder()
        response: Response | None = None
        provider_request_id: str | None = None
        recent_stream_events: deque[dict[str, str | None]] = deque(maxlen=5)
        try:
            if stream_responses:
                async with self.get_client().responses.with_streaming_response.create(
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
                                query_logger.debug(
                                    f"Response created: {event.response.id}"
                                )
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
                                if (
                                    getattr(output_item, "type", None)
                                    == "function_call"
                                ):
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
                                if (
                                    getattr(output_item, "type", None)
                                    == "function_call"
                                ):
                                    result_builder.finish_tool_call_segment(
                                        _response_tool_call_event_key(event)
                                    )
                            case _:
                                continue
            else:
                openai_response = await self.get_client().responses.create(
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
        query_logger.debug(f"Response finished: {response.id}")

        finish_reason = (
            None
            if not response.incomplete_details
            else response.incomplete_details.reason
        )

        tool_calls: list[ToolCall] = []
        provider_tool_events: list[ProviderToolEvent] = []
        citations: list[Citation] = []
        parsed_reasoning = None
        code_mode_outputs: list[str] = []
        for i, output in enumerate(response.output):
            if output.type == "message":
                for content in output.content:
                    if not isinstance(content, ResponseOutputText):
                        continue
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
                            provider="openai",
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
        result_builder.set_output_text(
            "\n".join(output_parts) if output_parts else None
        )

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
        if response.usage:
            # see _calculate_cost
            cache_read_tokens = response.usage.input_tokens_details.cached_tokens
            reasoning_tokens = response.usage.output_tokens_details.reasoning_tokens
            result_metadata = QueryResultMetadata(
                in_tokens=response.usage.input_tokens - cache_read_tokens,
                out_tokens=response.usage.output_tokens - reasoning_tokens,
                reasoning_tokens=reasoning_tokens,
                cache_read_tokens=cache_read_tokens,
            )

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

    @override
    async def get_rate_limit(self) -> RateLimit | None:
        headers = {}

        try:
            # NOTE: with_streaming_response doesn't seem to always work
            if self.use_completions:
                response = (
                    await self.get_client().chat.completions.with_raw_response.create(
                        max_completion_tokens=16,
                        model=self.model_name,
                        messages=[
                            {
                                "role": "user",
                                "content": "Do not think. Say 'ok'",
                            }
                        ],
                        stream=True,
                    )
                )
            else:
                response = await self.get_client().responses.with_raw_response.create(
                    max_output_tokens=16,
                    input="Do not think. Say 'ok'",
                    model=self.model_name,
                )
            headers = response.headers

            server_time_str = headers.get("date")
            if server_time_str:
                server_time = datetime.datetime.strptime(
                    server_time_str, "%a, %d %b %Y %H:%M:%S GMT"
                ).replace(tzinfo=datetime.timezone.utc)
                timestamp = server_time.timestamp()
            else:
                timestamp = time.time()

            # NOTE: for openai, max_tokens is used to reject requests if the amount of tokens left is less than the max_tokens

            # we calculate estimated_tokens as (character_count / 4) + max_tokens. Note that OpenAI's rate limiter doesn't tokenize the request using the model's specific tokenizer but relies on a character count-based heuristic.

            return RateLimit(
                raw=headers,
                unix_timestamp=timestamp,
                request_limit=int(
                    headers.get("x-ratelimit-limit-requests", 0)
                    or headers.get("x-ratelimit-limit", 0)
                ),
                request_remaining=int(
                    headers.get("x-ratelimit-remaining-requests", 0)
                    or headers.get("x-ratelimit-remaining", 0)
                ),
                token_limit=int(headers["x-ratelimit-limit-tokens"]),
                token_remaining=int(headers["x-ratelimit-remaining-tokens"]),
            )
        except Exception as e:
            self.instance_logger.warning(f"Failed to get rate limit: {e}")
            return None

    @deprecated("Use query(output_schema=...) instead")
    @override
    async def query_json(
        self,
        input: Sequence[InputItem],
        pydantic_model: type[PydanticT],
        output_schema: dict[str, Any] | type[BaseModel] | None = None,
        **kwargs: object,
    ) -> PydanticT:
        # re-use existing body
        body = await self.build_body(
            input,
            tools=[],
            output_schema=output_schema,
            **kwargs,
        )

        async def _query():
            try:
                return await self.get_client().responses.parse(
                    text_format=pydantic_model,
                    **body,  # pyright: ignore[reportAny]
                )
            except APIConnectionError:
                raise ImmediateRetryException("Failed to connect to OpenAI")

        response = await BaseRetrier.immediate_retry_wrapper(
            func=_query, logger=self.instance_logger
        )

        parsed: PydanticT | None = response.output_parsed
        if parsed is None:
            raise ModelNoOutputError("Model returned empty response")

        return parsed

    async def get_embedding(
        self, text: str, model: str = "text-embedding-3-small"
    ) -> list[float]:
        """Query OpenAI's Embedding endpoint"""

        async def _get_embedding() -> list[float]:
            try:
                response: CreateEmbeddingResponse = (
                    await self.get_client().embeddings.create(
                        input=text,
                        model=model,
                    )
                )
            except APIConnectionError:
                raise ImmediateRetryException("Failed to connect to OpenAI")
            except Exception as e:
                raise Exception("Failed to query OpenAI's Embedding endpoint") from e

            if not response.data:
                raise Exception("No embeddings returned from OpenAI")

            return response.data[0].embedding

        return await BaseRetrier.immediate_retry_wrapper(
            func=_get_embedding, logger=self.instance_logger
        )

    async def moderate_content(self, text: str) -> ModerationCreateResponse:
        """Query OpenAI's Moderation endpoint"""

        async def _moderate_content() -> ModerationCreateResponse:
            try:
                return await self.get_client().moderations.create(input=text)
            except APIConnectionError:
                raise ImmediateRetryException("Failed to connect to OpenAI")
            except Exception as e:
                raise Exception("Failed to query OpenAI's Moderation endpoint") from e

        return await BaseRetrier.immediate_retry_wrapper(
            func=_moderate_content, logger=self.instance_logger
        )

    @override
    async def _calculate_cost(
        self,
        metadata: QueryResultMetadata,
        batch: bool = False,
        bill_reasoning: bool = True,
    ) -> QueryResultCost | None:
        """
        Future Cost considerations
        Per session:
        - Code Interpreter
        Per day:
        - File Search Storage
        Per 1000:
        - File Search Tool Call (responses api)
        - Web Search Tool Call
        """

        # Prompt Caching works automatically on all your API requests (no code changes required) and has no additional fees associated with it
        # cache tokens are included in input tokens

        # reasoning tokens are included in output tokens

        return await super()._calculate_cost(metadata, batch, bill_reasoning=True)
