import logging
from typing import Any, Literal, Sequence, cast

from openai.types.chat import ChatCompletionMessage
from pydantic import BaseModel, SecretStr
from typing_extensions import override

from model_library import model_library_settings
from model_library.base import (
    DelegateOnly,
    InputItem,
    LLMConfig,
    QueryResult,
    RawInput,
    RawResponse,
    ToolDefinition,
)
from model_library.register_models import register_provider


@register_provider("nvidia")
class NvidiaModel(DelegateOnly):
    def __init__(
        self,
        model_name: str,
        provider: Literal["nvidia"] = "nvidia",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)

        config = config or LLMConfig()
        config.custom_endpoint = (
            config.custom_endpoint or "https://integrate.api.nvidia.com/v1"
        )
        config.custom_api_key = config.custom_api_key or SecretStr(
            model_library_settings.NVIDIA_API_KEY
        )

        self.init_delegate(
            config=config,
            delegate_provider="openai",
            use_completions=True,
        )

    def _normalize_messages_for_nvidia(self, messages: list[Any]) -> list[Any]:
        """Normalize assistant messages to Nvidia's OpenAI-compatible schema.

        NVIDIA's endpoint rejects ``tool_calls=None`` ("Input should be iterable")
        and assistant ``content=None`` ("Input should be a valid string").
        """
        fixed: list[Any] = []
        for msg in messages:
            if isinstance(msg, ChatCompletionMessage) and msg.role == "assistant":
                updated_msg = msg
                if updated_msg.content is None:
                    updated_msg = updated_msg.model_copy(update={"content": ""})
                if updated_msg.tool_calls is None:
                    fixed.append(updated_msg.model_dump(exclude_none=True))
                else:
                    fixed.append(updated_msg)
            elif isinstance(msg, dict):
                msg_dict = cast(dict[str, Any], msg)
                if msg_dict.get("role") == "assistant" and (
                    msg_dict.get("content") is None
                    or msg_dict.get("tool_calls") is None
                ):
                    updated_dict = dict(msg_dict)
                    if updated_dict.get("content") is None:
                        updated_dict["content"] = ""
                    if updated_dict.get("tool_calls") is None:
                        updated_dict.pop("tool_calls", None)
                    fixed.append(updated_dict)
                else:
                    fixed.append(msg)
            else:
                fixed.append(msg)
        return fixed

    def _normalize_input_items_for_nvidia(
        self, input: Sequence[InputItem]
    ) -> list[InputItem]:
        fixed: list[InputItem] = []
        for item in input:
            if isinstance(item, RawInput):
                normalized = self._normalize_messages_for_nvidia([item.input])[0]
                fixed.append(item.model_copy(update={"input": normalized}))
            elif isinstance(item, RawResponse):
                normalized = self._normalize_messages_for_nvidia([item.response])[0]
                fixed.append(item.model_copy(update={"response": normalized}))
            else:
                fixed.append(item)
        return fixed

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
        return await super()._query_impl(
            self._normalize_input_items_for_nvidia(input),
            tools=tools,
            query_logger=query_logger,
            output_schema=output_schema,
            **kwargs,
        )

    @override
    async def build_body(
        self,
        input: Sequence[InputItem],
        *,
        tools: list[ToolDefinition],
        output_schema: dict[str, Any] | type[BaseModel] | None = None,
        **kwargs: object,
    ) -> dict[str, Any]:
        body = await super().build_body(
            input, tools=tools, output_schema=output_schema, **kwargs
        )
        if "messages" in body:
            body["messages"] = self._normalize_messages_for_nvidia(body["messages"])
        return body
