from __future__ import annotations

import io
import logging
from collections.abc import Hashable, Sequence
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel
from typing_extensions import override

from model_library import model_library_settings
from model_library.base import (
    LLM,
    FileBase,
    FileInput,
    FileWithBase64,
    FileWithId,
    FinishReason,
    FinishReasonInfo,
    InputItem,
    LLMConfig,
    QueryResult,
    QueryResultExtras,
    QueryResultMetadata,
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
from model_library.exceptions import (
    BadInputError,
    UnexpectedSystemInputError,
    handle_empty_response,
)
from model_library.file_utils import trim_images
from model_library.agent.tool import is_native_web_search
from model_library.register_models import register_provider
from model_library.utils import default_httpx_client

if TYPE_CHECKING:
    from mistralai.client import Mistral
    from mistralai.client.models.toolcall import ToolCall as MistralToolCall


def _mistral_tool_call_event_key(tool_call: MistralToolCall) -> Hashable | None:
    if "index" in tool_call.model_fields_set and tool_call.index is not None:
        return ("index", tool_call.index)
    if tool_call.id and tool_call.id != "null":
        return tool_call.id
    return None


def map_mistral_finish_reason(
    finish_reason: str | None,
) -> FinishReasonInfo:
    match finish_reason:
        case "stop":
            reason = FinishReason.STOP
        case "length" | "model_length":
            reason = FinishReason.MAX_TOKENS
        case "tool_calls":
            reason = FinishReason.TOOL_CALLS
        case "error":
            reason = FinishReason.ERROR
        case _:
            reason = FinishReason.UNKNOWN

    return FinishReasonInfo(reason=reason, raw=finish_reason)


@register_provider("mistralai")
class MistralModel(LLM):
    @override
    def _get_default_api_key(self) -> str:
        return model_library_settings.MISTRAL_API_KEY

    @override
    def get_client(
        self, api_key: str | None = None, base_url: str | None = None
    ) -> Mistral:
        if not self.has_client():
            assert api_key

            from mistralai.client import Mistral

            if base_url:
                client = Mistral(
                    api_key=api_key,
                    async_client=default_httpx_client(),
                    server_url=base_url,
                )
            else:
                client = Mistral(
                    api_key=api_key,
                    async_client=default_httpx_client(),
                )

            self.assign_client(client)
        return super().get_client()

    def __init__(
        self,
        model_name: str,
        provider: Literal["mistral"] = "mistral",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)

    @override
    async def parse_input(
        self,
        input: Sequence[InputItem],
        **kwargs: Any,
    ) -> list[dict[str, Any] | Any]:
        new_input: list[dict[str, Any] | Any] = []

        if isinstance(input[0], SystemInput):
            new_input.append({"role": "system", "content": input[0].text})
            input = input[1:]

        content_user: list[dict[str, Any]] = []

        def flush_content_user():
            if content_user:
                # NOTE: must make new object as we clear()
                new_input.append({"role": "user", "content": content_user.copy()})
                content_user.clear()

        for item in input:
            if isinstance(item, TextInput):
                content_user.append({"type": "text", "text": item.text})
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
                    new_input.append(
                        {
                            "role": "tool",
                            "name": item.tool_call.name,
                            "content": item.result,
                            "tool_call_id": item.tool_call.id,
                        }
                    )
                case RawResponse():
                    new_input.append(item.response)
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
        """Append images to the request body"""
        match image:
            case FileWithBase64():
                return {
                    "type": "image_url",
                    "image_url": f"data:image/{image.mime};base64,{image.base64}",
                }
            case _:
                raise BadInputError("Unsupported image type")

    @override
    async def parse_file(
        self,
        file: FileInput,
    ) -> Any:
        raise NotImplementedError()

    @override
    async def parse_tools(
        self,
        tools: list[ToolDefinition],
    ) -> Any:
        parsed_tools: list[dict[str, Any]] = []
        for tool in tools:
            body = tool.body
            if is_native_web_search(body):
                parsed_tools.append(self.search_tool)
                continue
            if not isinstance(body, ToolBody):
                parsed_tools.append(body)
                continue

            parsed_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": body.name,
                        "description": body.description,
                        "parameters": {
                            "type": "object",
                            "properties": body.properties,
                            "required": body.required,
                        },
                    },
                },
            )

        return parsed_tools

    @override
    async def upload_file(
        self,
        name: str,
        mime: str,
        bytes: io.BytesIO,
        type: Literal["image", "file"] = "file",
    ) -> FileWithId:
        raise NotImplementedError()

    @override
    async def build_body(
        self,
        input: Sequence[InputItem],
        *,
        tools: list[ToolDefinition],
        output_schema: dict[str, Any] | type[BaseModel] | None = None,
        **kwargs: object,
    ) -> dict[str, Any]:
        from mistralai.client.models import AssistantMessage

        # mistral supports max 8 images, merge extra images into the 8th image
        input = trim_images(input, max_images=8)

        last_message = input[-1]
        if isinstance(last_message, AssistantMessage):
            input.append(TextInput(text="Please Continue."))

        tools = await self.parse_tools(tools)

        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": await self.parse_input(input),
            "tools": tools,
        }

        if self.reasoning:
            if self.reasoning_effort is not None:
                body["reasoning_effort"] = self.reasoning_effort
            else:
                body["prompt_mode"] = "reasoning"

        if self.max_tokens:
            body["max_tokens"] = self.max_tokens

        if self.supports_temperature:
            if self.temperature is not None:
                body["temperature"] = self.temperature
            if self.top_p is not None:
                body["top_p"] = self.top_p

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
        from mistralai.client.models import (
            AssistantMessage,
            ContentChunk,
            TextChunk,
            ThinkChunk,
        )

        body = await self.build_body(
            input, tools=tools, output_schema=output_schema, **kwargs
        )

        result_builder = QueryResultBuilder()
        response = await self.get_client().chat.stream_async(
            **body,  # pyright: ignore[reportAny]
        )

        # Read the content, reasoning, and usage from the streamed chunks.
        # The chunk can be a ThinkChunk (reasoning), TextChunk or str (content), or may contain usage.
        in_tokens = 0
        out_tokens = 0
        finish_reason = None
        response_id: str | None = None
        raw_tool_calls: list[MistralToolCall] = []

        try:
            async for chunk in response:
                data = chunk.data
                if data.id is not None:
                    response_id = data.id
                for choice in data.choices:
                    delta = choice.delta
                    if isinstance(delta.content, list):
                        for content_item in delta.content:
                            if isinstance(content_item, ThinkChunk):
                                for text_chunk in content_item.thinking:
                                    if isinstance(text_chunk, TextChunk):
                                        result_builder.append_reasoning_delta(
                                            text_chunk.text
                                        )
                            elif isinstance(content_item, TextChunk):
                                result_builder.append_content_delta(content_item.text)

                    elif isinstance(delta.content, str):
                        result_builder.append_content_delta(delta.content)

                    if delta.tool_calls:
                        for tool_call in delta.tool_calls:
                            tool_call_key = _mistral_tool_call_event_key(tool_call)
                            if tool_call_key is None:
                                result_builder.start_tool_call_segment().record_tool_call_delta()
                            else:
                                result_builder.record_tool_call_delta(tool_call_key)
                            raw_tool_calls.append(tool_call)
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason

                if hasattr(data, "usage") and data.usage is not None:
                    in_tokens += data.usage.prompt_tokens or 0
                    out_tokens += data.usage.completion_tokens or 0

        except Exception as e:
            query_logger.error(f"Error: {e}", exc_info=True)
            raise e

        no_useful_content = (
            not result_builder.has_output_text
            and not result_builder.has_reasoning
            and not raw_tool_calls
        )
        mapped_finish_reason = map_mistral_finish_reason(finish_reason)
        if no_useful_content:
            handle_empty_response(
                mapped_finish_reason,
                {
                    "in_tokens": in_tokens,
                    "out_tokens": out_tokens,
                },
            )

        tool_calls: list[ToolCall] = []

        for tool_call in raw_tool_calls or []:
            tool_calls.append(
                ToolCall(
                    id=tool_call.id or "",
                    name=tool_call.function.name,
                    args=tool_call.function.arguments,
                )
            )

        output_text = result_builder.output_text
        reasoning = result_builder.reasoning
        content: list[ContentChunk] = []
        if reasoning:
            content.append(
                ThinkChunk(
                    thinking=[
                        TextChunk(
                            text=reasoning,
                            type="text",
                        )
                    ],
                    closed=None,
                    type="thinking",
                )
            )
        if output_text:
            content.append(
                TextChunk(
                    text=output_text,
                    type="text",
                )
            )

        message = AssistantMessage(tool_calls=raw_tool_calls, content=content)

        return result_builder.build(
            finish_reason=mapped_finish_reason,
            history=[*input, RawResponse(response=message)],
            tool_calls=tool_calls,
            metadata=QueryResultMetadata(
                in_tokens=in_tokens,
                out_tokens=out_tokens,
                # Reasoning tokens are not supported by Mistral 09/22/25
            ),
            extras=QueryResultExtras(
                provider_response_id=response_id,
            ),
        )
