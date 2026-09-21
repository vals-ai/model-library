import io
import logging
from typing import Any, ClassVar, Literal, Sequence

from pydantic import BaseModel
from typing_extensions import override

from model_library.base import (
    LLM,
    LLMConfig,
    FileInput,
    FileWithId,
    InputItem,
    QueryResult,
    ToolDefinition,
)


class TranscriptionOnlyException(Exception):
    """
    Raised when text generation functionality is performed on a
    transcription-only model.
    """

    DEFAULT_MESSAGE: str = "This model only supports audio transcription, certain functionality is not supported."

    def __init__(self, message: str | None = None):
        super().__init__(message or TranscriptionOnlyException.DEFAULT_MESSAGE)


class TranscriptionOnly(LLM):
    provider_name: ClassVar[str]

    def __init__(
        self,
        model_name: str,
        provider: str | None = None,
        *,
        config: LLMConfig | None = None,
    ):
        if config is None:
            config = LLMConfig(
                supports_transcription=True,
                supports_temperature=False,
            )
        self._custom_api_key = config.custom_api_key
        super().__init__(
            model_name,
            provider if provider is not None else self.provider_name,
            config=config,
        )

    def _api_key(self) -> str:
        if self._custom_api_key is not None:
            return self._custom_api_key.get_secret_value()
        return self._get_default_api_key()

    @override
    def _client_registry_namespace(self) -> str:
        return f"{self.provider}.transcription"

    @override
    def get_client(
        self, api_key: str | None = None, base_url: str | None = None
    ) -> Any:
        if api_key is not None or base_url is not None:
            raise TranscriptionOnlyException()
        try:
            return super().get_client()
        except (AttributeError, KeyError) as exc:
            raise TranscriptionOnlyException() from exc

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
        raise TranscriptionOnlyException()

    @override
    async def build_body(
        self,
        input: Sequence[InputItem],
        *,
        tools: list[ToolDefinition],
        output_schema: dict[str, Any] | type[BaseModel] | None = None,
        **kwargs: object,
    ) -> dict[str, Any]:
        raise TranscriptionOnlyException()

    @override
    async def parse_input(
        self,
        input: Sequence[InputItem],
        **kwargs: object,
    ) -> Any:
        raise TranscriptionOnlyException()

    @override
    async def parse_image(self, image: FileInput) -> Any:
        raise TranscriptionOnlyException()

    @override
    async def parse_file(self, file: FileInput) -> Any:
        raise TranscriptionOnlyException()

    @override
    async def parse_tools(self, tools: list[ToolDefinition]) -> Any:
        raise TranscriptionOnlyException()

    @override
    async def upload_file(
        self,
        name: str,
        mime: str,
        bytes: io.BytesIO,
        type: Literal["image", "file"] = "file",
    ) -> FileWithId:
        raise TranscriptionOnlyException()
