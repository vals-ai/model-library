import asyncio
import io
import logging
from typing import Any, Literal, Sequence, cast

import grpc
from typing_extensions import override
from xai_sdk import AsyncClient, Client
from xai_sdk.aio.chat import Chat as AsyncChat
from xai_sdk.chat import Content, Response, system, tool_result, user
from xai_sdk.chat import image as xai_image
from xai_sdk.chat import tool as xai_tool
from xai_sdk.proto.v6.chat_pb2 import Message, Tool
from xai_sdk.sync.chat import Chat as SyncChat

from model_library import model_library_settings
from model_library.base import (
    LLM,
    FileBase,
    FileInput,
    FileWithBase64,
    FileWithId,
    InputItem,
    LLMConfig,
    ProviderConfig,
    QueryResult,
    QueryResultCost,
    QueryResultMetadata,
    RawInput,
    RawResponse,
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
    NoMatchingToolCallError,
    RateLimitException,
)
from model_library.providers.openai import OpenAIModel
from model_library.register_models import register_provider
from model_library.utils import create_openai_client_with_defaults

Chat = AsyncChat | SyncChat


class XAIConfig(ProviderConfig):
    sync_client: bool = False


@register_provider("grok")
class XAIModel(LLM):
    provider_config = XAIConfig()

    _client: AsyncClient | Client | None = None

    @override
    def get_client(self) -> AsyncClient | Client:
        if self._client:
            return self._client

        ClientClass = Client if self.provider_config.sync_client else AsyncClient
        self._client = ClientClass(
            api_key=model_library_settings.XAI_API_KEY,
        )

        return self._client

    @override
    def __init__(
        self,
        model_name: str,
        provider: Literal["xai"] = "xai",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)

        # https://docs.x.ai/docs/guides/migration
        self.delegate = (
            None
            if self.native
            else OpenAIModel(
                model_name=self.model_name,
                provider=provider,
                config=config,
                custom_client=create_openai_client_with_defaults(
                    api_key=model_library_settings.XAI_API_KEY,
                    base_url=(
                        "https://us-west-1.api.x.ai/v1"
                        if "grok-3-mini-reasoning" in self.model_name
                        else "https://api.x.ai/v1"
                    ),
                ),
                use_completions=True,
            )
        )

    async def get_tool_call_ids(self, input: Sequence[InputItem]) -> list[str]:
        raw_responses = [x for x in input if isinstance(x, RawResponse)]
        tool_call_ids: list[str] = []

        calls = [
            y
            for x in raw_responses
            if isinstance(x.response, Response) and x.response.tool_calls
            for y in x.response.tool_calls
        ]
        tool_call_ids.extend([x.id for x in calls if x.id])
        return tool_call_ids

    @override
    async def parse_input(
        self,
        input: Sequence[InputItem],
        **kwargs: Any,
    ) -> None:
        chat: Chat = kwargs["chat"]

        content_user: list[Any] = []

        def flush_content_user():
            if content_user:
                chat.append(user(*content_user))
                content_user.clear()

        tool_call_ids = await self.get_tool_call_ids(input)

        for item in input:
            if isinstance(item, TextInput):
                content_user.append(item.text)
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

                    chat.append(tool_result(item.result))
                case RawResponse():
                    chat.append(item.response)
                case RawInput():
                    chat.append(item.input)

        # in case content user item is the last item
        flush_content_user()

    @override
    async def parse_image(
        self,
        image: FileInput,
    ) -> Content:
        match image:
            case FileWithBase64():
                image_url = f"data:image/{image.mime};base64,{image.base64}"
                return xai_image(image_url=image_url, detail="high")
            case _:
                raise BadInputError("Unsupported image type")

    @override
    async def parse_file(
        self,
        file: FileInput,
    ) -> Content:
        raise NotImplementedError()

    @override
    async def parse_tools(
        self,
        tools: list[ToolDefinition],
    ) -> list[Tool]:
        parsed_tools: list[Tool] = []
        for tool in tools:
            body = tool.body
            if not isinstance(body, ToolBody):
                parsed_tools.append(body)
                continue
            parsed_tools.append(
                xai_tool(
                    name=body.name,
                    description=body.description,
                    parameters={
                        "type": "object",
                        "properties": body.properties,
                        "required": body.required,
                    },
                )
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

    def fetch_response_sync(
        self,
        chat: SyncChat,
    ) -> Response | None:
        latest_response = None
        for response, _ in chat.stream():
            latest_response = response

        return latest_response

    async def fetch_response_async(
        self,
        chat: AsyncChat,
    ) -> Response | None:
        latest_response = None
        async for response, _ in chat.stream():
            latest_response = response

        return latest_response

    @override
    async def build_body(
        self, input: Sequence[InputItem], *, tools: list[ToolDefinition], **kwargs: Any
    ) -> dict[str, Any]:
        messages: Sequence[Message] = []
        if "system_prompt" in kwargs:
            messages.append(system(str(kwargs.pop("system_prompt"))))

        body: dict[str, Any] = {
            "max_tokens": self.max_tokens,
            "model": self.model_name,
            "tools": await self.parse_tools(tools),
            "messages": messages,
        }

        if self.supports_temperature:
            if self.temperature is not None:
                body["temperature"] = self.temperature
            if self.top_p is not None:
                body["top_p"] = self.top_p

        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort

        body.update(kwargs)
        return body

    @override
    async def _query_impl(
        self,
        input: Sequence[InputItem],
        *,
        tools: list[ToolDefinition],
        query_logger: logging.Logger,
        **kwargs: object,
    ) -> QueryResult:
        if self.delegate:
            return await self.delegate_query(
                input, tools=tools, query_logger=query_logger, **kwargs
            )

        body = await self.build_body(input, tools=tools, **kwargs)

        try:
            chat: Chat = self.get_client().chat.create(**body)  # pyright: ignore[reportAny]
            await self.parse_input(input, chat=chat)

            # Allows users to dynamically swap to a sync client if getting grpc errors
            # Run in a separate thread so we are playing fair with other async processes
            if self.provider_config.sync_client:
                latest_response = await asyncio.to_thread(
                    self.fetch_response_sync, cast(SyncChat, chat)
                )
            else:
                latest_response = await self.fetch_response_async(cast(AsyncChat, chat))

            if not latest_response:
                raise ModelNoOutputError("Model failed to produce a response")

            tool_calls: list[ToolCall] = []
            if (
                latest_response.finish_reason == "REASON_TOOL_CALLS"
                and latest_response.tool_calls
            ):
                for tool_call in latest_response.tool_calls:
                    tool_calls.append(
                        ToolCall(
                            id=tool_call.id,
                            name=tool_call.function.name,
                            args=tool_call.function.arguments,
                        )
                    )

            if (
                latest_response.finish_reason == "REASON_MAX_LEN"
                and not latest_response.content
                and not latest_response.reasoning_content
            ):
                raise MaxOutputTokensExceededError()
        except grpc.RpcError as e:
            raise RateLimitException(e.details())

        return QueryResult(
            output_text=latest_response.content,
            reasoning=latest_response.reasoning_content,
            metadata=QueryResultMetadata(
                # see _calculate_cost
                in_tokens=latest_response.usage.prompt_tokens
                - latest_response.usage.cached_prompt_text_tokens,
                out_tokens=latest_response.usage.completion_tokens,
                reasoning_tokens=latest_response.usage.reasoning_tokens,
                cache_read_tokens=latest_response.usage.cached_prompt_text_tokens,
            ),
            tool_calls=tool_calls,
            history=[*input, RawResponse(response=latest_response)],
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
        Per 1000 calls:
        - Web Search
        - X Search
        - Code Execution
        - Document Search
        Per 1000 sources:
        - Live Search
        Free:
        - File Storage
        - Collections Storage
        """
        # prompt caching automatically enabled
        # reasoning_tokens tokens billed in addition to completion_tokens

        # prompt_tokens include cached_prompt_text_tokens

        return await super()._calculate_cost(metadata, batch, bill_reasoning=True)
