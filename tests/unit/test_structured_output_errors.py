import io
import logging
from collections.abc import Sequence
from typing import Any, Literal

import pytest
from pydantic import BaseModel, RootModel, ValidationError

from model_library.base import LLM, LLMConfig
from model_library.base.gateway import _parse_query_result
from model_library.base.input import FileInput, FileWithId, InputItem, ToolDefinition
from model_library.base.output import QueryResult, QueryResultCost, QueryResultMetadata
from model_library.exceptions import InvalidStructuredOutputError


class InvalidJsonLLM(LLM):
    def __init__(self, output_text: str = "SECRET_MODEL_OUTPUT_SHOULD_NOT_LEAK"):
        super().__init__(
            "invalid-json",
            "test",
            config=LLMConfig(native=False, supports_output_schema=True),
        )
        self.output_text = output_text
        self.calls = 0
        self.received_schema = None
        self._test_usage_metadata = QueryResultMetadata(
            in_tokens=17, out_tokens=0, cache_read_tokens=5
        )

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
        self.calls += 1
        self.received_schema = output_schema
        return QueryResult(
            output_text=self.output_text,
            history=list(input),
            metadata=self._test_usage_metadata,
        )

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


class Answer(BaseModel):
    answer: int


@pytest.mark.parametrize("schema", [Answer, {"type": "object"}])
async def test_parse_failure_retains_completed_usage_without_parser_context(
    schema, caplog
):
    model = InvalidJsonLLM()
    with pytest.raises(InvalidStructuredOutputError) as caught:
        await model.query("test", output_schema=schema)
    error = caught.value
    assert model.calls == 1 and model.received_schema is schema
    assert error.query_result is not None
    assert error.query_result.output_text == model.output_text
    assert error.usage_metadata is error.query_result.metadata
    assert error.usage_metadata.in_tokens == 17
    assert error.usage_metadata.out_tokens == 0
    assert error.usage_metadata.cache_read_tokens == 5
    assert "cache_write_tokens" not in error.usage_metadata.model_fields_set
    assert error.__context__ is None and error.__cause__ is None
    assert model.output_text not in caplog.text
    assert model.output_text not in repr(error)


async def test_parse_failure_without_provider_usage_has_no_estimated_cost():
    model = InvalidJsonLLM()
    model._registry_key = "openai/gpt-4o"
    model._test_usage_metadata = QueryResultMetadata()

    with pytest.raises(InvalidStructuredOutputError) as caught:
        await model.query("test", output_schema={"type": "object"})

    assert caught.value.query_result is not None
    assert caught.value.query_result.metadata.cost is None


@pytest.mark.parametrize(
    ("metadata", "expected_zero"),
    [
        (QueryResultMetadata(in_tokens=0, out_tokens=0), True),
        (QueryResultMetadata(in_tokens=7, out_tokens=3), False),
    ],
)
async def test_parse_failure_prices_explicit_complete_usage(
    metadata: QueryResultMetadata, expected_zero: bool
):
    model = InvalidJsonLLM()
    model._registry_key = "openai/gpt-4o"
    model._test_usage_metadata = metadata

    with pytest.raises(InvalidStructuredOutputError) as caught:
        await model.query("test", output_schema={"type": "object"})

    assert caught.value.query_result is not None
    cost = caught.value.query_result.metadata.cost
    assert cost is not None
    assert (cost.total == 0) is expected_zero


@pytest.mark.parametrize(
    "metadata",
    [
        QueryResultMetadata(cache_read_tokens=0),
        QueryResultMetadata(reasoning_tokens=3),
        QueryResultMetadata(in_tokens=7),
        QueryResultMetadata(out_tokens=7),
    ],
)
async def test_parse_failure_does_not_price_partial_usage(
    metadata: QueryResultMetadata,
):
    model = InvalidJsonLLM()
    model._registry_key = "openai/gpt-4o"
    model._test_usage_metadata = metadata

    with pytest.raises(InvalidStructuredOutputError) as caught:
        await model.query("test", output_schema={"type": "object"})

    assert caught.value.query_result is not None
    assert caught.value.query_result.metadata.cost is None


@pytest.mark.parametrize(
    "metadata",
    [
        QueryResultMetadata(),
        QueryResultMetadata(in_tokens=7),
        QueryResultMetadata(in_tokens=7, out_tokens=3),
    ],
)
async def test_parse_failure_retains_provider_supplied_cost(
    metadata: QueryResultMetadata,
):
    model = InvalidJsonLLM()
    model._registry_key = "openai/gpt-4o"
    provider_cost = QueryResultCost(input=2.5, output=0)
    metadata.cost = provider_cost
    model._test_usage_metadata = metadata

    with pytest.raises(InvalidStructuredOutputError) as caught:
        await model.query("test", output_schema={"type": "object"})

    assert caught.value.query_result is not None
    assert caught.value.query_result.metadata.cost is provider_cost


@pytest.mark.parametrize("parsed", [{"answer": "PRIVATE"}, ["PRIVATE"], "PRIVATE"])
def test_gateway_schema_validation_keeps_validation_error_and_usage(parsed):
    with pytest.raises(ValidationError) as caught:
        _parse_query_result(
            {"output_parsed": parsed, "metadata": {"out_tokens": 0}}, Answer
        )
    metadata = getattr(caught.value, "usage_metadata")
    assert metadata.out_tokens == 0
    assert "in_tokens" not in metadata.model_fields_set
    assert not isinstance(caught.value, InvalidStructuredOutputError)


def test_gateway_valid_root_schema_keeps_array_success():
    result = _parse_query_result(
        {"output_parsed": [1, 2], "metadata": {"in_tokens": 3}}, RootModel[list[int]]
    )
    assert result.output_parsed.root == [1, 2]
