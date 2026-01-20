import io
import logging
from typing import Any, Literal, Sequence

from typing_extensions import override

from model_library.base import (
    LLM,
    FileInput,
    FileWithId,
    InputItem,
    LLMConfig,
    QueryResult,
    ToolDefinition,
)
from model_library.utils import (
    create_anthropic_client_with_defaults,
    create_openai_client_with_defaults,
)


class DelegateOnlyException(Exception):
    """
    Raised when native model functionality is performed on a
    delegate-only model.
    """

    DEFAULT_MESSAGE: str = "This model is running in delegate-only mode, certain functionality is not supported."

    def __init__(self, message: str | None = None):
        super().__init__(message or DelegateOnlyException.DEFAULT_MESSAGE)


class DelegateOnly(LLM):
    @override
    def get_client(self) -> None:
        raise DelegateOnlyException()

    def init_delegate(
        self,
        config: LLMConfig | None,
        base_url: str,
        api_key: str,
        delegate_provider: Literal["openai", "anthropic"],
        use_completions: bool = True,
    ) -> None:
        from model_library.providers.anthropic import AnthropicModel
        from model_library.providers.openai import OpenAIModel

        match delegate_provider:
            case "openai":
                self.delegate = OpenAIModel(
                    model_name=self.model_name,
                    provider=self.provider,
                    config=config,
                    custom_client=create_openai_client_with_defaults(
                        base_url=base_url,
                        api_key=api_key,
                    ),
                    use_completions=use_completions,
                )
            case "anthropic":
                self.delegate = AnthropicModel(
                    model_name=self.model_name,
                    provider=self.provider,
                    config=config,
                    custom_client=create_anthropic_client_with_defaults(
                        base_url=base_url,
                        api_key=api_key,
                    ),
                )

    def __init__(
        self,
        model_name: str,
        provider: str,
        *,
        config: LLMConfig | None = None,
    ):
        config = config or LLMConfig()
        config.native = False
        super().__init__(model_name, provider, config=config)

    @override
    async def _query_impl(
        self,
        input: Sequence[InputItem],
        *,
        tools: list[ToolDefinition],
        query_logger: logging.Logger,
        **kwargs: object,
    ) -> QueryResult:
        assert self.delegate
        return await self.delegate_query(
            input, tools=tools, query_logger=query_logger, **kwargs
        )

    @override
    async def build_body(
        self,
        input: Sequence[InputItem],
        *,
        tools: list[ToolDefinition],
        **kwargs: object,
    ) -> dict[str, Any]:
        assert self.delegate
        return await self.delegate.build_body(input, tools=tools, **kwargs)

    @override
    async def parse_input(
        self,
        input: Sequence[InputItem],
        **kwargs: Any,
    ) -> Any:
        assert self.delegate
        return await self.delegate.parse_input(input, **kwargs)

    @override
    async def parse_image(
        self,
        image: FileInput,
    ) -> Any:
        assert self.delegate
        return await self.delegate.parse_image(image)

    @override
    async def parse_file(
        self,
        file: FileInput,
    ) -> Any:
        assert self.delegate
        return await self.delegate.parse_file(file)

    @override
    async def parse_tools(
        self,
        tools: list[ToolDefinition],
    ) -> Any:
        assert self.delegate
        return await self.delegate.parse_tools(tools)

    @override
    async def upload_file(
        self,
        name: str,
        mime: str,
        bytes: io.BytesIO,
        type: Literal["image", "file"] = "file",
    ) -> FileWithId:
        raise DelegateOnlyException()

    @override
    async def count_tokens(
        self,
        input: Sequence[InputItem],
        *,
        history: Sequence[InputItem] = [],
        tools: list[ToolDefinition] = [],
        **kwargs: object,
    ) -> int:
        assert self.delegate
        return await self.delegate.count_tokens(
            input, history=history, tools=tools, **kwargs
        )
