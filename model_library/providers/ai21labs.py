import io
from typing import Any, Literal, Sequence

from ai21 import AsyncAI21Client
from ai21.models.chat import AssistantMessage, ChatMessage, ToolMessage
from ai21.models.chat.chat_completion_response import ChatCompletionResponse
from typing_extensions import override

from model_library import model_library_settings
from model_library.base import (
    LLM,
    FileInput,
    FileWithId,
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
from model_library.register_models import register_provider
from model_library.utils import default_httpx_client


@register_provider("ai21labs")
class AI21LabsModel(LLM):
    _client: AsyncAI21Client | None = None

    @override
    def get_client(self) -> AsyncAI21Client:
        if not AI21LabsModel._client:
            AI21LabsModel._client = AsyncAI21Client(
                api_key=model_library_settings.AI21LABS_API_KEY,
                http_client=default_httpx_client(),
                num_retries=1,
            )
        return AI21LabsModel._client

    def __init__(
        self,
        model_name: str,
        provider: Literal["ai21labs"] = "ai21labs",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)

    @override
    async def parse_input(
        self,
        input: Sequence[InputItem],
        **kwargs: Any,
    ) -> list[ChatMessage | AssistantMessage]:
        new_input: list[ChatMessage | AssistantMessage] = []
        for item in input:
            match item:
                case TextInput():
                    new_input.append(ChatMessage(role="user", content=item.text))
                case AssistantMessage():
                    new_input.append(item)
                case ToolResult():
                    new_input.append(
                        ToolMessage(
                            role="tool",
                            content=item.result,
                            tool_call_id=item.tool_call.id,
                        )
                    )
                case _:
                    raise BadInputError("Unsupported input type")
        return new_input

    @override
    async def parse_image(
        self,
        image: FileInput,
    ) -> Any:
        raise NotImplementedError()

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
        messages: list[ChatMessage] = []
        if "system_prompt" in kwargs:
            messages.append(
                ChatMessage(role="system", content=str(kwargs.pop("system_prompt")))
            )
        messages.extend(await self.parse_input(input))

        body: dict[str, Any] = {
            "max_tokens": self.max_tokens,
            "model": self.model_name,
            "messages": messages,
            "tools": await self.parse_tools(tools),
        }

        if self.supports_temperature:
            if self.temperature is not None:
                body["temperature"] = self.temperature
            if self.top_p is not None:
                body["top_p"] = self.top_p

        body.update(kwargs)

        response: ChatCompletionResponse = (
            await self.get_client().chat.completions.create(**body, stream=False)  # pyright: ignore[reportAny, reportUnknownMemberType]
        )

        if not response or not response.choices or not response.choices[0].message:
            raise ModelNoOutputError("Model returned no completions")
        choice = response.choices[0]

        if choice.finish_reason == "length" and not choice.message.content:
            raise MaxOutputTokensExceededError()

        tool_calls: list[ToolCall] = []
        for tool_call in choice.message.tool_calls or []:
            tool_calls.append(
                ToolCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    args=tool_call.function.arguments,
                )
            )

        output = QueryResult(
            output_text=choice.message.content,
            history=[*input, choice.message],
            metadata=QueryResultMetadata(
                in_tokens=response.usage.prompt_tokens,
                out_tokens=response.usage.completion_tokens,
            ),
            tool_calls=tool_calls,
        )

        return output
