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


@register_provider("baseten")
class BasetenModel(DelegateOnly):
    """OpenAI-compatible models hosted on a Vals-owned Baseten deployment."""

    def __init__(
        self,
        model_name: str,
        provider: Literal["baseten"] = "baseten",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)

        config = config or LLMConfig()
        config.custom_endpoint = (
            config.custom_endpoint or model_library_settings.BASETEN_API_BASE_URL
        )
        config.custom_api_key = config.custom_api_key or SecretStr(
            model_library_settings.BASETEN_API_KEY
        )

        self.init_delegate(
            config=config,
            delegate_provider="openai",
            use_completions=True,
        )

    @staticmethod
    def _normalize_replayed_assistant_message(message: object) -> object:
        """Make replayed assistant turns acceptable to Baseten's vLLM endpoint."""
        if (
            isinstance(message, ChatCompletionMessage)
            and message.role == "assistant"
            and message.content is None
        ):
            if message.tool_calls is None:
                updated_message = message.model_dump(exclude_unset=True)
                updated_message["content"] = ""
                updated_message.pop("tool_calls", None)
                return updated_message
            return message.model_copy(update={"content": ""})

        if not isinstance(message, dict):
            return message

        message_dict = cast(dict[str, Any], message)
        if (
            message_dict.get("role") == "assistant"
            and "content" in message_dict
            and message_dict["content"] is None
        ):
            updated_message = dict(message_dict)
            updated_message["content"] = ""
            if updated_message.get("tool_calls") is None:
                updated_message.pop("tool_calls", None)
            return updated_message

        return message_dict

    def _normalize_replayed_assistant_history(
        self, input: Sequence[InputItem]
    ) -> list[InputItem]:
        normalized_input: list[InputItem] = []
        for item in input:
            if isinstance(item, RawResponse):
                normalized_response = self._normalize_replayed_assistant_message(
                    item.response
                )
                if normalized_response is not item.response:
                    item = item.model_copy(update={"response": normalized_response})
            elif isinstance(item, RawInput):
                normalized_raw_input = self._normalize_replayed_assistant_message(
                    item.input
                )
                if normalized_raw_input is not item.input:
                    item = item.model_copy(update={"input": normalized_raw_input})
            normalized_input.append(item)
        return normalized_input

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
            self._normalize_replayed_assistant_history(input),
            tools=tools,
            query_logger=query_logger,
            output_schema=output_schema,
            **kwargs,
        )
