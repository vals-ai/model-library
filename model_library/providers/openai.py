from __future__ import annotations

import io
import json
from typing import Any, Literal, Sequence, cast

from openai import APIConnectionError, AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallUnion,
)
from openai.types.chat.chat_completion_message_tool_call import Function
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.create_embedding_response import CreateEmbeddingResponse
from openai.types.moderation_create_response import ModerationCreateResponse
from openai.types.responses import (
    ResponseOutputItem,
    ResponseOutputText,
    ResponseStreamEvent,
)
from openai.types.responses.response import Response
from openai.types.responses.tool_param import ToolParam as ResponsesToolParam
from typing_extensions import override

from model_library import model_library_settings
from model_library.base import (
    LLM,
    BatchResult,
    Citation,
    FileInput,
    FileWithBase64,
    FileWithId,
    FileWithUrl,
    InputItem,
    LLMBatchMixin,
    LLMConfig,
    ProviderConfig,
    PydanticT,
    QueryResult,
    QueryResultCost,
    QueryResultExtras,
    QueryResultMetadata,
    RawInputItem,
    TextInput,
    ToolBody,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from model_library.exceptions import (
    ImmediateRetryException,
    MaxOutputTokensExceededError,
    ModelNoOutputError,
)
from model_library.model_utils import get_reasoning_in_tag
from model_library.register_models import register_provider
from model_library.utils import create_openai_client_with_defaults


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
                **kwargs,
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
            self._root.logger.error(f"Couldn't retrieve {batch_id}")
            completed = 0
        return completed

    @override
    async def cancel_batch_request(self, batch_id: str):
        self._root.logger.info(f"Cancelling {batch_id}")
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


class OpenAIConfig(ProviderConfig):
    deep_research: bool = False


@register_provider("openai")
class OpenAIModel(LLM):
    provider_config = OpenAIConfig()

    _client: AsyncOpenAI | None = None

    @override
    def get_client(self) -> AsyncOpenAI:
        if self._delegate_client:
            return self._delegate_client
        if not OpenAIModel._client:
            OpenAIModel._client = create_openai_client_with_defaults(
                api_key=model_library_settings.OPENAI_API_KEY
            )
        return OpenAIModel._client

    def __init__(
        self,
        model_name: str,
        provider: str = "openai",
        *,
        config: LLMConfig | None = None,
        custom_client: AsyncOpenAI | None = None,
        use_completions: bool = False,
    ):
        super().__init__(model_name, provider, config=config)
        self.use_completions: bool = use_completions
        self.deep_research = self.provider_config.deep_research

        # allow custom client to act as delegate (native)
        self._delegate_client: AsyncOpenAI | None = custom_client

        # batch client
        self.supports_batch: bool = self.supports_batch and not custom_client
        self.batch: LLMBatchMixin | None = (
            OpenAIBatchMixin(self) if self.supports_batch else None
        )

    @override
    async def parse_input(
        self,
        input: Sequence[InputItem],
        **kwargs: Any,
    ) -> list[dict[str, Any] | Any]:
        new_input: list[dict[str, Any] | Any] = []
        content_user: list[dict[str, Any]] = []
        for item in input:
            match item:
                case TextInput():
                    if self.use_completions:
                        content_user.append({"type": "text", "text": item.text})
                    else:
                        content_user.append({"type": "input_text", "text": item.text})
                case FileWithBase64() | FileWithUrl() | FileWithId():
                    match item.type:
                        case "image":
                            content_user.append(await self.parse_image(item))
                        case "file":
                            content_user.append(await self.parse_file(item))
                case _:
                    if content_user:
                        new_input.append({"role": "user", "content": content_user})
                        content_user = []
                    match item:
                        case ToolResult():
                            if not (
                                not isinstance(x, dict)
                                and x.type == "function_call"
                                and x.call_id == item.tool_call.call_id
                                for x in new_input
                            ):
                                raise Exception(
                                    "Tool call result provided with no matching tool call"
                                )
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
                                    }
                                )
                        case dict():  # RawInputItem
                            item = cast(RawInputItem, item)
                            new_input.append(item)
                        case _:  # RawResponse
                            if self.use_completions:
                                item = cast(ChatCompletionMessageToolCall, item)
                            else:
                                item = cast(ResponseOutputItem, item)
                            new_input.append(item)

        if content_user:
            new_input.append({"role": "user", "content": content_user})

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

    @override
    async def parse_tools(
        self,
        tools: Sequence[ToolDefinition],
    ) -> list[ChatCompletionToolParam | ResponsesToolParam | Any]:
        parsed_tools: list[ChatCompletionToolParam | ResponsesToolParam | Any] = []
        for tool in tools:
            body = tool.body
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
            purpose="assistants",
        )

        return FileWithId(
            type=type,
            name=response.filename,
            mime=mime,
            file_id=response.id,
        )

    async def _query_completions(
        self,
        input: Sequence[InputItem],
        *,
        tools: list[ToolDefinition],
        **kwargs: object,
    ) -> QueryResult:
        """
        Completions endpoint
        Generally not used for openai models
        Used by some providers using openai as a delegate
        """

        parsed_input: list[dict[str, Any] | ChatCompletionMessage] = []
        if "system_prompt" in kwargs:
            parsed_input.append(
                {"role": "system", "content": kwargs.pop("system_prompt")}
            )

        parsed_input.extend(await self.parse_input(input))

        body: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "messages": parsed_input,
            # enable usage data in streaming responses
            "stream_options": {"include_usage": True},
        }

        if self.supports_tools:
            parsed_tools = await self.parse_tools(tools)
            if parsed_tools:
                body["tools"] = parsed_tools

        if self.reasoning:
            del body["max_tokens"]
            body["max_completion_tokens"] = self.max_tokens
            if self.reasoning_effort:
                body["reasoning_effort"] = self.reasoning_effort

        if self.supports_temperature:
            if self.temperature is not None:
                body["temperature"] = self.temperature
            if self.top_p is not None:
                body["top_p"] = self.top_p

        body.update(kwargs)

        output_text: str = ""
        reasoning_text: str = ""
        metadata: QueryResultMetadata = QueryResultMetadata()
        raw_tool_calls: list[ChatCompletionMessageToolCall] = []

        stream = await self.get_client().chat.completions.create(
            **body,  # pyright: ignore[reportAny]
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices:
                choice = chunk.choices[0]
                if choice.delta and choice.delta.content:
                    output_text += choice.delta.content

                if (
                    hasattr(choice.delta, "reasoning_content")
                    and choice.delta.reasoning_content  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
                ):
                    reasoning_text += cast(str, choice.delta.reasoning_content)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]

                if choice.delta and choice.delta.tool_calls:
                    for tool_call_chunk in choice.delta.tool_calls:
                        func = tool_call_chunk.function
                        # start of new tool call
                        if tool_call_chunk.id and (
                            not raw_tool_calls
                            or raw_tool_calls[-1].id != tool_call_chunk.id
                        ):
                            raw_tool_calls.append(
                                ChatCompletionMessageToolCall(
                                    id=tool_call_chunk.id,
                                    type="function",
                                    function=Function(
                                        name=func.name if func and func.name else "",
                                        arguments=func.arguments
                                        if func and func.arguments
                                        else "",
                                    ),
                                )
                            )
                        # accumulate delta
                        elif func:
                            if func.name:
                                raw_tool_calls[-1].function.name = func.name
                            if func.arguments:
                                raw_tool_calls[-1].function.arguments += func.arguments

                if (
                    choice.finish_reason == "length"
                    and not output_text
                    and not reasoning_text
                    and not raw_tool_calls
                ):
                    raise MaxOutputTokensExceededError()

            if chunk.usage:
                # NOTE: see _calculate_cost
                reasoning_tokens = (
                    chunk.usage.completion_tokens_details.reasoning_tokens or 0
                    if chunk.usage.completion_tokens_details
                    else 0
                )
                cache_read_tokens = (
                    chunk.usage.prompt_tokens_details.cached_tokens or 0
                    if chunk.usage.prompt_tokens_details
                    else getattr(chunk.usage, "cached_tokens", 0)  # for kimi
                )
                metadata = QueryResultMetadata(
                    in_tokens=chunk.usage.prompt_tokens - cache_read_tokens,
                    out_tokens=chunk.usage.completion_tokens - reasoning_tokens,
                    reasoning_tokens=reasoning_tokens,
                    cache_read_tokens=cache_read_tokens,
                )

        if not output_text and not reasoning_text and not raw_tool_calls:
            raise ModelNoOutputError()

        tool_calls: list[ToolCall] = []
        for raw_tool_call in raw_tool_calls:
            tool_calls.append(
                ToolCall(
                    id=raw_tool_call.id,
                    name=raw_tool_call.function.name,
                    args=raw_tool_call.function.arguments,
                )
            )

        if (
            self.reasoning
            and self.provider not in {"openai", "azure", "deepseek"}
            and output_text
            and not reasoning_text
        ):
            output_text, reasoning_text = get_reasoning_in_tag(output_text)

        # build final message for history
        final_message = ChatCompletionMessage(
            role="assistant",
            content=output_text if output_text else None,
            tool_calls=cast(list[ChatCompletionMessageToolCallUnion], raw_tool_calls)
            if raw_tool_calls
            else None,
        )
        if reasoning_text:
            setattr(final_message, "reasoning_content", reasoning_text)

        return QueryResult(
            output_text=output_text,
            reasoning=reasoning_text,
            tool_calls=tool_calls,
            history=[*input, final_message],
            metadata=metadata,
        )

    async def _check_deep_research_args(
        self, tools: Sequence[ToolDefinition], **kwargs: object
    ) -> None:
        min_tokens = 30_000
        if self.max_tokens < min_tokens:
            self.logger.warning(
                f"Recommended to set max_tokens >= {min_tokens} for deep research models"
            )

        if "background" not in kwargs:
            self.logger.warning(
                "Recommended to set background=True for deep research models"
            )

        valid = False
        for tool in tools:
            tool_body = tool.body
            if not isinstance(tool_body, dict):
                continue
            tool_body = cast(dict[str, Any], tool_body)
            tool_type = tool_body.get("type", None)

            if tool_type in {
                "web_search",
                "web_search_preview",
                "web_search_preview_2025_03_11",
            }:
                valid = True
        if not valid:
            raise Exception("Deep research models require web search tools")

    async def build_body(
        self,
        input: Sequence[InputItem],
        *,
        tools: Sequence[ToolDefinition],
        **kwargs: object,
    ) -> dict[str, Any]:
        if self.deep_research:
            await self._check_deep_research_args(tools, **kwargs)

        parsed_input: list[dict[str, Any] | ResponseOutputItem] = []
        if "system_prompt" in kwargs:
            parsed_input.append(
                {
                    "role": "developer",
                    "content": [
                        {"type": "input_text", "text": kwargs.pop("system_prompt")}
                    ],
                }
            )

        parsed_input.extend(await self.parse_input(input))

        parsed_tools = await self.parse_tools(tools)

        body: dict[str, Any] = {
            "model": self.model_name,
            "max_output_tokens": self.max_tokens,
            "input": parsed_input,
        }

        if parsed_tools:
            body["tools"] = parsed_tools
        else:
            body["tool_choice"] = "none"

        if self.reasoning:
            body["reasoning"] = {"summary": "auto"}
            if self.reasoning_effort:
                body["reasoning"]["effort"] = self.reasoning_effort

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
        **kwargs: object,
    ) -> QueryResult:
        if self.use_completions:
            if self.deep_research:
                raise Exception("Use responses endpoint for deep research models")
            return await self._query_completions(input, tools=tools, **kwargs)

        body = await self.build_body(input, tools=tools, **kwargs)

        try:
            stream = await self.get_client().responses.create(
                **body,  # pyright: ignore[reportAny]
                stream=True,
            )
        except APIConnectionError:
            raise ImmediateRetryException("Failed to connect to OpenAI")

        response: Response | None = None
        stream_events: list[ResponseStreamEvent] = []
        async for event in stream:
            stream_events.append(event)
            match event.type:
                case "response.created":
                    self.logger.info(f"Response created: {event.response.id}")
                    response = event.response
                case "response.completed":
                    self.logger.info(f"Response completed: {event.response.id}")
                    response = event.response
                case "response.incomplete":
                    self.logger.warning(f"Response incomplete: {event.response.id}")
                    response = event.response
                case "response.in_progress":
                    self.logger.info(f"Response in progress: {event.response.id}")
                    response = event.response
                case "response.failed":
                    self.logger.error(f"Response failed: {event.response.id}")
                    self.logger.error(f"Error details: {event.response.error}")
                    response = event.response
                case _:
                    continue
        if not response:
            self.logger.error(
                f"Model returned no response. Events: {[e.model_dump(exclude_unset=True, exclude_none=True) for e in stream_events]}"
            )
            raise ImmediateRetryException("Model returned no response")
        self.logger.info(f"Response finished: {response.id}")

        if (
            response.incomplete_details
            and response.incomplete_details.reason == "max_output_tokens"
            and not response.output_text
        ):
            raise MaxOutputTokensExceededError()

        tool_calls: list[ToolCall] = []
        citations: list[Citation] = []
        reasoning = None
        for output in response.output:
            if self.deep_research:
                if output.type == "message":
                    for content in output.content:
                        if not isinstance(content, ResponseOutputText):
                            continue
                        for citation in content.annotations:
                            citations.append(Citation(**citation.model_dump()))

            if output.type == "reasoning":
                reasoning = " ".join([i.text for i in output.summary])
                continue
            if output.type != "function_call":
                continue
            self.logger.info(f"Found tool call for response: {response.id}")
            if not output.id:
                raise Exception(f"Tool call is missing id for response: {response.id}")
            tool_calls.append(
                ToolCall(
                    id=output.id,
                    call_id=output.call_id,
                    name=output.name,
                    args=output.arguments,
                )
            )

        result = QueryResult(
            output_text=response.output_text,
            reasoning=reasoning,
            tool_calls=tool_calls,
            history=[*input, *response.output],
            extras=QueryResultExtras(citations=citations),
        )
        if response.usage:
            # see _calculate_cost
            cache_read_tokens = response.usage.input_tokens_details.cached_tokens
            reasoning_tokens = response.usage.output_tokens_details.reasoning_tokens
            result.metadata = QueryResultMetadata(
                in_tokens=response.usage.input_tokens - cache_read_tokens,
                out_tokens=response.usage.output_tokens - reasoning_tokens,
                reasoning_tokens=reasoning_tokens,
                cache_read_tokens=cache_read_tokens,
            )

        search_results = getattr(response, "search_results", None)
        if search_results is not None:
            result.raw["search_results"] = search_results

        return result

    @override
    async def query_json(
        self,
        input: Sequence[InputItem],
        pydantic_model: type[PydanticT],
        **kwargs: object,
    ) -> PydanticT:
        # re-use existing body
        body = await self.build_body(
            input,
            tools=[],
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

        response = await LLM.immediate_retry_wrapper(func=_query, logger=self.logger)

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

        return await LLM.immediate_retry_wrapper(
            func=_get_embedding, logger=self.logger
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

        return await LLM.immediate_retry_wrapper(
            func=_moderate_content, logger=self.logger
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
