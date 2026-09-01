import io
import logging
from abc import abstractmethod
from typing import Any, Literal, Sequence

from pydantic import BaseModel
from typing_extensions import override

from model_library.base import (
    LLM,
    FileInput,
    FileWithId,
    InputItem,
    QueryResult,
    ToolDefinition,
    TranscriptionConfig,
    TranscriptionResult,
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
    def __init__(
        self,
        model_name: str,
        provider: str,
        *,
        config: TranscriptionConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config or TranscriptionConfig())

    @override
    @abstractmethod
    async def transcribe_audio(
        self,
        *,
        name: str,
        mime: str,
        audio: bytes,
        language: str | None = None,
    ) -> TranscriptionResult: ...

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
