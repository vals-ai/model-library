import base64
import io
import logging
from typing import Any, Literal, Sequence

from pydantic import BaseModel, SecretStr
from typing_extensions import override
from xai_sdk import AsyncClient
from xai_sdk.aio.chat import Chat
from xai_sdk.chat import Content, Response, system, tool_result, user
from xai_sdk.chat import file as xai_file
from xai_sdk.chat import image as xai_image
from xai_sdk.chat import tool as xai_tool
from xai_sdk.proto.v6.chat_pb2 import Message, Tool

from model_library import model_library_settings
from model_library.base import (
    LLM,
    FileBase,
    FileInput,
    FileWithBase64,
    FileWithId,
    FileWithUrl,
    FinishReason,
    FinishReasonInfo,
    InputItem,
    LLMConfig,
    QueryResult,
    QueryResultCost,
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
from model_library.exceptions import (
    ImmediateRetryException,
    ModelNoOutputError,
    NoMatchingToolCallError,
    handle_empty_response,
    UnexpectedSystemInputError,
)
from model_library.providers.openai import OpenAIModel
from model_library.register_models import register_provider


def map_xai_finish_reason(
    finish_reason: str | None,
) -> FinishReasonInfo:
    match finish_reason:
        case "REASON_STOP":
            reason = FinishReason.STOP
        case "REASON_MAX_LEN":
            reason = FinishReason.MAX_TOKENS
        case "REASON_MAX_CONTEXT":
            reason = FinishReason.CONTEXT_WINDOW_EXCEEDED
        case "REASON_TOOL_CALLS":
            reason = FinishReason.TOOL_CALLS
        case "REASON_TIME_LIMIT":
            reason = FinishReason.MAX_TOKENS
        case _:
            reason = FinishReason.UNKNOWN

    return FinishReasonInfo(reason=reason, raw=finish_reason)


@register_provider("grok")
class XAIModel(LLM):
    @override
    def _get_default_api_key(self) -> str:
        return model_library_settings.XAI_API_KEY

    @override
    def get_client(
        self, api_key: str | None = None, base_url: str | None = None
    ) -> AsyncClient:
        if not self.has_client():
            assert api_key

            if base_url:
                client = AsyncClient(
                    api_key=api_key,
                    api_host=base_url,
                    timeout=60 * 60 * 24,
                )
            else:
                client = AsyncClient(api_key=api_key, timeout=60 * 60 * 24)

            self.assign_client(client)
        return super().get_client()

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
        if self.native:
            self.delegate = None
        else:
            default_base_url = (
                "https://us-west-1.api.x.ai/v1"
                if "grok-3-mini-reasoning" in self.model_name
                else "https://api.x.ai/v1"
            )

            config = config or LLMConfig()
            config.custom_endpoint = config.custom_endpoint or default_base_url
            config.custom_api_key = config.custom_api_key or SecretStr(
                model_library_settings.XAI_API_KEY
            )

            # flip native back on for the delegate so it initializes its own
            # client registry; the outer model stays non-native.
            config.native = True
            self.delegate = OpenAIModel(
                model_name=self.model_name,
                provider=provider,
                config=config,
                use_completions=True,
            )
            config.native = False

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
    ) -> list[Message]:
        new_input: list[Message] = []

        if isinstance(input[0], SystemInput):
            new_input.append(system(input[0].text))
            input = input[1:]

        content_user: list[Any] = []

        def flush_content_user():
            if content_user:
                new_input.append(user(*content_user))
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

                    new_input.append(tool_result(item.result))
                case RawResponse():
                    new_input.append(item.response)
                case RawInput():
                    new_input.append(item.input)
                case SystemInput():
                    raise UnexpectedSystemInputError()

        # in case content user item is the last item
        flush_content_user()

        # use Chat object to parse raw Response and other objects into the correct formats
        # see xai's chat.py `class BaseChat` -> `def append`
        chat: Chat = self.get_client().chat.create("dummy")
        for message in new_input:
            chat.append(message)
        return list(chat.messages)

    @override
    async def parse_image(
        self,
        image: FileInput,
    ) -> Content:
        match image:
            case FileWithBase64():
                image_url = f"data:image/{image.mime};base64,{image.base64}"
                return xai_image(image_url=image_url, detail="high")
            case FileWithUrl():
                return xai_image(image_url=image.url, detail="high")
            case FileWithId():
                return xai_file(file_id=image.file_id)

    @override
    async def parse_file(
        self,
        file: FileInput,
    ) -> Content:
        match file:
            case FileWithId():
                return xai_file(file_id=file.file_id)
            case FileWithUrl():
                return xai_file(url=file.url, filename=file.name, mime_type=file.mime)
            case FileWithBase64():
                return xai_file(
                    data=base64.b64decode(file.base64),
                    filename=file.name,
                    mime_type=file.mime,
                )

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
        response = await self.get_client().files.upload(
            bytes,
            filename=name,
        )
        return FileWithId(
            type=type,
            name=response.filename,
            mime=mime,
            file_id=response.id,
        )

    @override
    async def build_body(
        self,
        input: Sequence[InputItem],
        *,
        tools: list[ToolDefinition],
        output_schema: dict[str, Any] | type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model_name,
            "tools": await self.parse_tools(tools),
        }

        if self.max_tokens:
            body["max_tokens"] = self.max_tokens

        if self.supports_temperature:
            if self.temperature is not None:
                body["temperature"] = self.temperature
            if self.top_p is not None:
                body["top_p"] = self.top_p

        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort

        body.update(kwargs)

        body["messages"] = await self.parse_input(input)

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
        if self.delegate:
            return await self.delegate_query(
                input,
                tools=tools,
                query_logger=query_logger,
                output_schema=output_schema,
                **kwargs,
            )

        body = await self.build_body(
            input, tools=tools, output_schema=output_schema, **kwargs
        )

        chat: Chat = self.get_client().chat.create(**body)

        latest_response: Response | None = None
        try:
            async for response, _ in chat.stream():
                latest_response = response
        except OSError as e:
            # Transient gRPC C-core OSError (e.g. [Errno 2]) not caught by retry logic.
            raise ImmediateRetryException(str(e)) from e

        if not latest_response:
            raise ModelNoOutputError("Model failed to produce a response")

        tool_calls: list[ToolCall] = []
        for tool_call in latest_response.tool_calls:
            tool_calls.append(
                ToolCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    args=tool_call.function.arguments,
                )
            )

        content = latest_response.content
        reasoning = latest_response.reasoning_content
        finish_reason = latest_response.finish_reason

        no_useful_content = not content and not reasoning and not tool_calls
        mapped_finish_reason = map_xai_finish_reason(finish_reason)
        if no_useful_content:
            handle_empty_response(mapped_finish_reason, {"raw": latest_response})

        return QueryResult(
            output_text=content,
            reasoning=reasoning,
            finish_reason=mapped_finish_reason,
            metadata=QueryResultMetadata(
                # see _calculate_cost
                # proto3 scalar ints default to 0; CopyFrom on partial
                # streaming chunks can leave prompt_tokens < cached
                in_tokens=max(
                    0,
                    latest_response.usage.prompt_tokens
                    - latest_response.usage.cached_prompt_text_tokens,
                ),
                out_tokens=latest_response.usage.completion_tokens,
                reasoning_tokens=latest_response.usage.reasoning_tokens,
                cache_read_tokens=latest_response.usage.cached_prompt_text_tokens,
            ),
            tool_calls=tool_calls,
            history=[*input, RawResponse(response=latest_response)],
        )

    @override
    async def count_tokens(
        self,
        input: Sequence[InputItem],
        *,
        history: Sequence[InputItem] = [],
        tools: list[ToolDefinition] = [],
        **kwargs: object,
    ) -> int:
        if not input and not history:
            return 0

        string_input = await self.stringify_input(input, history=history, tools=tools)
        self.instance_logger.debug(string_input)

        tokens = await self.get_client().tokenize.tokenize_text(
            string_input, self.model_name
        )
        return len(tokens)

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
