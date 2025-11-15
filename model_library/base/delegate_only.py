import io
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


class DelegateOnlyException(Exception):
    """
    Raised when native model functionality is performed on a
    delegate-only model.
    """

    DEFAULT_MESSAGE: str = "This model supports only delegate-only functionality. Only the query() method should be used."

    def __init__(self, message: str | None = None):
        super().__init__(message or DelegateOnlyException.DEFAULT_MESSAGE)


class DelegateOnly(LLM):
    @override
    def get_client(self) -> None:
        raise DelegateOnlyException()

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
        **kwargs: object,
    ) -> QueryResult:
        assert self.delegate

        return await self.delegate_query(input, tools=tools, **kwargs)

    @override
    async def parse_input(
        self,
        input: Sequence[InputItem],
        **kwargs: Any,
    ) -> Any:
        raise DelegateOnlyException()

    @override
    async def parse_image(
        self,
        image: FileInput,
    ) -> Any:
        raise DelegateOnlyException()

    @override
    async def parse_file(
        self,
        file: FileInput,
    ) -> Any:
        raise DelegateOnlyException()

    @override
    async def parse_tools(
        self,
        tools: list[ToolDefinition],
    ) -> Any:
        raise DelegateOnlyException()

    @override
    async def upload_file(
        self,
        name: str,
        mime: str,
        bytes: io.BytesIO,
        type: Literal["image", "file"] = "file",
    ) -> FileWithId:
        raise DelegateOnlyException()
