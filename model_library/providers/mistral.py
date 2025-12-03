import io
import time
from collections.abc import Sequence
from typing import Any, Literal

from mistralai import AssistantMessage, ContentChunk, Mistral, TextChunk, ThinkChunk
from mistralai.models.completionevent import CompletionEvent
from mistralai.models.toolcall import ToolCall as MistralToolCall
from mistralai.utils.eventstreaming import EventStreamAsync
from typing_extensions import override

from model_library import model_library_settings
from model_library.base import (
    LLM,
    FileInput,
    FileWithBase64,
    FileWithId,
    FileWithUrl,
    InputItem,
    LLMConfig,
    QueryResult,
    QueryResultMetadata,
    TextInput,
    ToolBody,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from model_library.exceptions import (
    BadInputError,
    MaxOutputTokensExceededError,
    ModelNoOutputError,
)
from model_library.file_utils import trim_images
from model_library.register_models import register_provider
from model_library.utils import default_httpx_client


@register_provider("mistralai")
class MistralModel(LLM):
    _client: Mistral | None = None

    @override
    def get_client(self) -> Mistral:
        if not MistralModel._client:
            MistralModel._client = Mistral(
                api_key=model_library_settings.MISTRAL_API_KEY,
                async_client=default_httpx_client(),
            )
        return MistralModel._client

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
        content_user: list[dict[str, Any]] = []

        def flush_content_user():
            nonlocal content_user

            if content_user:
                new_input.append({"role": "user", "content": content_user})
                content_user = []

        for item in input:
            match item:
                case TextInput():
                    content_user.append({"type": "text", "text": item.text})
                case FileWithBase64() | FileWithUrl() | FileWithId():
                    match item.type:
                        case "image":
                            content_user.append(await self.parse_image(item))
                        case "file":
                            content_user.append(await self.parse_file(item))
                case AssistantMessage():
                    flush_content_user()
                    new_input.append(item)
                case ToolResult():
                    flush_content_user()
                    new_input.append(
                        {
                            "role": "tool",
                            "name": item.tool_call.name,
                            "content": item.result,
                            "tool_call_id": item.tool_call.id,
                        }
                    )
                case _:
                    raise BadInputError("Unsupported input type")

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
    async def _query_impl(
        self,
        input: Sequence[InputItem],
        *,
        tools: list[ToolDefinition],
        **kwargs: object,
    ) -> QueryResult:
        # mistral supports max 8 images, merge extra images into the 8th image
        input = trim_images(input, max_images=8)

        last_message = input[-1]
        if isinstance(last_message, AssistantMessage):
            input.append(TextInput(text="Please Continue."))

        messages: list[dict[str, Any]] = []
        if "system_prompt" in kwargs:
            messages.append({"role": "system", "content": kwargs.pop("system_prompt")})

        messages.extend(await self.parse_input(input))

        tools = await self.parse_tools(tools)

        body: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "messages": messages,
            "prompt_mode": "reasoning" if self.reasoning else None,
            "tools": tools,
        }

        if self.supports_temperature:
            if self.temperature is not None:
                body["temperature"] = self.temperature
            if self.top_p is not None:
                body["top_p"] = self.top_p

        body.update(kwargs)

        start = time.time()

        response: EventStreamAsync[
            CompletionEvent
        ] = await self.get_client().chat.stream_async(
            **body,  # pyright: ignore[reportAny]
        )

        # Read the content, reasoning, and usage from the streamed chunks.
        # The chunk can be a ThinkChunk (reasoning), TextChunk or str (content), or may contain usage.
        reasoning: str = ""
        text: str = ""
        in_tokens = 0
        out_tokens = 0
        finish_reason = None
        raw_tool_calls: list[MistralToolCall] = []

        try:
            async for chunk in response:
                data = chunk.data
                for choice in data.choices:
                    delta = choice.delta
                    if isinstance(delta.content, list):
                        for content_item in delta.content:
                            if isinstance(content_item, ThinkChunk):
                                for text_chunk in content_item.thinking:
                                    if isinstance(text_chunk, TextChunk):
                                        reasoning += text_chunk.text

                    else:
                        if isinstance(delta.content, str):
                            text += delta.content
                        if delta.tool_calls:
                            raw_tool_calls.extend(delta.tool_calls)
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason

                if hasattr(data, "usage") and data.usage is not None:
                    in_tokens += data.usage.prompt_tokens or 0
                    out_tokens += data.usage.completion_tokens or 0

            self.logger.info(f"Finished in: {time.time() - start}")

        except Exception as e:
            self.logger.error(f"Error: {e}", exc_info=True)
            raise e

        if (
            finish_reason == "length"
            and not text
            and not reasoning
            and not raw_tool_calls
        ):
            raise MaxOutputTokensExceededError()

        if not text and not reasoning and not raw_tool_calls:
            raise ModelNoOutputError()

        tool_calls: list[ToolCall] = []

        for tool_call in raw_tool_calls or []:
            tool_calls.append(
                ToolCall(
                    id=tool_call.id or "",
                    name=tool_call.function.name,
                    args=tool_call.function.arguments,
                )
            )

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
        if text:
            content.append(
                TextChunk(
                    text=text,
                    type="text",
                )
            )

        message = AssistantMessage(tool_calls=raw_tool_calls, content=content)

        return QueryResult(
            output_text=text,
            reasoning=reasoning or None,
            history=[*input, message],
            tool_calls=tool_calls,
            metadata=QueryResultMetadata(
                in_tokens=in_tokens,
                out_tokens=out_tokens,
                # Reasoning tokens are not supported by Mistral 09/22/25
            ),
        )
