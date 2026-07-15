from collections.abc import Sequence
from typing import Any, Literal
import io
import logging

import pytest
from pydantic import BaseModel

from model_library.base import LLM, LLMConfig
from model_library.base.input import FileInput, FileWithId, InputItem, ToolDefinition
from model_library.base.output import QueryResult
from model_library.exceptions import InvalidStructuredOutputError


class InvalidJsonLLM(LLM):
    def __init__(self, output_text: str = "SECRET_MODEL_OUTPUT_SHOULD_NOT_LEAK"):
        super().__init__(
            "invalid-json",
            "test",
            config=LLMConfig(native=False, supports_output_schema=True),
        )
        self.output_text = output_text

    def get_client(
        self, api_key: str | None = None, base_url: str | None = None
    ) -> Any:
        return None

    def _get_default_api_key(self) -> str:
        return ""

    async def _query_impl(
        self,
        input: Sequence[InputItem],
        *,
        tools: list[ToolDefinition],
        query_logger: logging.Logger,
        output_schema: dict[str, Any] | type[BaseModel] | None = None,
        **kwargs: object,
    ) -> QueryResult:
        return QueryResult(output_text=self.output_text, history=list(input))

    async def build_body(
        self,
        input: Sequence[InputItem],
        *,
        tools: list[ToolDefinition],
        output_schema: dict[str, Any] | type[BaseModel] | None = None,
        **kwargs: object,
    ) -> dict[str, Any]:
        return {}

    async def parse_input(self, input: Sequence[InputItem], **kwargs: object) -> Any:
        return input

    async def parse_image(self, image: FileInput) -> Any:
        return image

    async def parse_file(self, file: FileInput) -> Any:
        return file

    async def parse_tools(self, tools: list[ToolDefinition]) -> Any:
        return tools

    async def upload_file(
        self,
        name: str,
        mime: str,
        bytes: io.BytesIO,
        type: Literal["image", "file"] = "file",
    ) -> FileWithId:
        raise NotImplementedError


async def test_invalid_structured_output_error_does_not_include_model_output():
    model = InvalidJsonLLM()

    with pytest.raises(InvalidStructuredOutputError) as exc_info:
        await model.query("test", output_schema={"type": "object"})

    exc = exc_info.value
    assert str(exc) == InvalidStructuredOutputError.DEFAULT_MESSAGE
    assert exc.parser_error_type == "JSONDecodeError"
    assert "SECRET_MODEL_OUTPUT_SHOULD_NOT_LEAK" not in str(exc)
    assert exc.__context__ is None


async def test_empty_structured_output_text_normalizes_to_none_and_skips_schema_parse():
    model = InvalidJsonLLM(output_text="")

    result = await model.query("test", output_schema={"type": "object"})

    assert result.output_text is None
    assert result.output_parsed is None
