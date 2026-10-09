from collections.abc import Sequence
from typing import Any, Literal
import io
import logging
import json
import traceback

import httpx
import model_library
import pytest
from pydantic import BaseModel, RootModel

from model_gateway.types import query_result_response_body
import model_gateway.model_helpers as model_helpers
from tests.unit.model_gateway._support import HEADERS, _make_client
from model_library import telemetry
from model_library.base import LLM, LLMConfig
from model_library.base.gateway import GatewayLLM
from model_library.base.input import FileInput, FileWithId, InputItem, ToolDefinition
from model_library.base.output import QueryResult, QueryResultMetadata
from model_library.exceptions import GatewayProviderError, InvalidStructuredOutputError
from sentry_sdk.utils import event_from_exception
from sentry_sdk.consts import DEFAULT_OPTIONS


class InvalidJsonLLM(LLM):
    def __init__(self, output_text: str = "SECRET_MODEL_OUTPUT_SHOULD_NOT_LEAK"):
        super().__init__(
            "invalid-json",
            "test",
            config=LLMConfig(native=False, supports_output_schema=True),
        )
        self.output_text = output_text
        self.calls = 0

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
        return QueryResult(
            output_text=self.output_text,
            history=list(input),
            metadata=QueryResultMetadata(in_tokens=17, out_tokens=0),
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


class RequiredAnswer(BaseModel):
    answer: int


@pytest.mark.parametrize("schema", [{"type": "object"}, RequiredAnswer])
async def test_invalid_structured_output_error_does_not_include_model_output(
    schema: dict[str, Any] | type[BaseModel],
):
    model = InvalidJsonLLM()

    with pytest.raises(InvalidStructuredOutputError) as exc_info:
        await model.query("test", output_schema=schema)

    exc = exc_info.value
    assert str(exc) == InvalidStructuredOutputError.DEFAULT_MESSAGE
    assert exc.parser_error_type == (
        "JSONDecodeError" if isinstance(schema, dict) else "ValidationError"
    )
    assert "SECRET_MODEL_OUTPUT_SHOULD_NOT_LEAK" not in str(exc)
    assert exc.__context__ is None
    assert exc.__cause__ is None
    assert model.calls == 1
    assert exc.query_result is not None
    assert exc.query_result.output_text == model.output_text
    metadata = exc.query_result.metadata.model_dump(
        exclude_unset=True, exclude_computed_fields=True
    )
    assert metadata["in_tokens"] == 17
    assert metadata["out_tokens"] == 0
    assert "cache_read_tokens" not in metadata
    assert model.output_text not in "".join(traceback.format_exception(exc))
    assert model.output_text not in repr(exc)
    assert model.output_text not in json.dumps(exc.to_provider_error())
    event, _ = event_from_exception(
        exc, client_options={**DEFAULT_OPTIONS, "include_local_variables": True}
    )
    sanitized = telemetry._before_send(
        event, {"exc_info": (type(exc), exc, exc.__traceback__)}
    )
    assert sanitized is not None
    assert model.output_text not in json.dumps(sanitized)


async def test_empty_structured_output_text_normalizes_to_none_and_skips_schema_parse():
    model = InvalidJsonLLM(output_text="")

    result = await model.query("test", output_schema={"type": "object"})

    assert result.output_text is None
    assert result.output_parsed is None


@pytest.mark.parametrize("parsed", [False, True])
async def test_gateway_failed_schema_preserves_wire_response(monkeypatch, parsed):
    wire = {
        "output_text": "SECRET_MODEL_OUTPUT_SHOULD_NOT_LEAK",
        "output_parsed": {"answer": "SECRET_MODEL_OUTPUT_SHOULD_NOT_LEAK"}
        if parsed
        else None,
        "metadata": {"in_tokens": 17, "out_tokens": 0},
    }
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=wire)

    monkeypatch.setattr(
        model_library.model_library_settings,
        "MODEL_GATEWAY_URL",
        "https://synthetic.invalid",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        model = GatewayLLM("fixture", "test")
        monkeypatch.setattr(model, "get_client", lambda: client)
        with pytest.raises(InvalidStructuredOutputError) as caught:
            await model.query("test", output_schema=RequiredAnswer)
    exc = caught.value
    assert len(requests) == 1
    assert requests[0]["output_schema"]["required"] == ["answer"]
    assert exc.query_result is not None
    assert exc.query_result.output_text == wire["output_text"]
    assert exc.query_result.output_parsed == wire["output_parsed"]
    metadata = exc.query_result.metadata.model_dump(
        exclude_unset=True, exclude_computed_fields=True
    )
    assert metadata == {"in_tokens": 17, "out_tokens": 0}
    assert exc.__context__ is None
    assert exc.__cause__ is None
    assert "SECRET_MODEL_OUTPUT_SHOULD_NOT_LEAK" not in "".join(
        traceback.format_exception(exc)
    )
    assert "SECRET_MODEL_OUTPUT_SHOULD_NOT_LEAK" not in json.dumps(
        exc.to_provider_error()
    )
    event, hint = event_from_exception(exc, client_options=DEFAULT_OPTIONS)
    assert "SECRET_MODEL_OUTPUT_SHOULD_NOT_LEAK" not in json.dumps(
        telemetry._before_send(event, hint)
    )


@pytest.mark.parametrize(
    "schema, value, valid",
    [
        (RootModel[list[int]], [1, 2], True),
        (RootModel[int], 7, True),
        (RootModel[list[int]], ["PRIVATE_INVALID_VALUE"], False),
        (RootModel[int], "PRIVATE_INVALID_VALUE", False),
        (RequiredAnswer, ["PRIVATE_INVALID_VALUE"], False),
        (RequiredAnswer, "PRIVATE_INVALID_VALUE", False),
    ],
)
async def test_gateway_json_shapes_and_usage_survive_wire(
    monkeypatch, schema, value, valid
):
    reported_usage = (
        {}
        if valid and isinstance(value, list)
        else ({"in_tokens": 0, "out_tokens": 0} if valid else {"out_tokens": 0})
    )
    wire = query_result_response_body(
        QueryResult(
            output_parsed=value, metadata=QueryResultMetadata(**reported_usage)
        ),
        signed_history="[]",
    )
    assert wire["metadata"] == reported_usage
    monkeypatch.setattr(
        model_library.model_library_settings,
        "MODEL_GATEWAY_URL",
        "https://synthetic.invalid",
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=wire))
    ) as client:
        model = GatewayLLM("fixture", "test")
        monkeypatch.setattr(model, "get_client", lambda: client)
        if valid:
            result = await model.query("test", output_schema=schema)
            assert result.output_parsed.model_dump() == value
        else:
            with pytest.raises(InvalidStructuredOutputError) as caught:
                await model.query("test", output_schema=schema)
            exc = caught.value
            assert exc.query_result is not None
            result = exc.query_result
            assert result.output_parsed == value
            assert exc.__context__ is None
            assert exc.__cause__ is None
            assert "PRIVATE_INVALID_VALUE" not in "".join(
                traceback.format_exception(exc)
            )
            event, hint = event_from_exception(exc, client_options=DEFAULT_OPTIONS)
            assert "PRIVATE_INVALID_VALUE" not in json.dumps(
                telemetry._before_send(event, hint)
            )
        assert (
            result.metadata.model_dump(exclude_unset=True, exclude_computed_fields=True)
            == reported_usage
        )


async def test_real_gateway_error_route_retains_completed_response(monkeypatch, caplog):
    provider = InvalidJsonLLM()
    server = _make_client()
    monkeypatch.setattr(
        model_helpers, "get_registry_model", lambda *args, **kwargs: provider
    )
    monkeypatch.setattr(
        model_library.model_library_settings,
        "MODEL_GATEWAY_URL",
        "https://synthetic.invalid",
    )
    envelopes = []

    def route(request):
        response = server.post(
            "/query", json=json.loads(request.content), headers=HEADERS
        )
        envelopes.append(response.json())
        return httpx.Response(response.status_code, json=response.json())

    async with httpx.AsyncClient(transport=httpx.MockTransport(route)) as client:
        gateway = GatewayLLM("gpt-4o", "openai")
        monkeypatch.setattr(gateway, "get_client", lambda: client)
        with pytest.raises(GatewayProviderError) as caught:
            await gateway.query("test", output_schema=RequiredAnswer)
    error = caught.value
    assert provider.calls == 1
    assert error.exception_type == "InvalidStructuredOutputError"
    assert error.query_result is not None
    assert error.query_result.output_text == provider.output_text
    metadata = error.query_result.metadata.model_dump(
        exclude_unset=True, exclude_computed_fields=True
    )
    assert metadata["in_tokens"] == 17
    assert metadata["out_tokens"] == 0
    assert "cache_read_tokens" not in metadata
    assert envelopes[0]["failed_query_result"]["signed_history"] == "[]"
    assert provider.output_text not in json.dumps(envelopes[0]["error"])
    assert provider.output_text not in json.dumps(error.raw_error)
    assert provider.output_text not in str(error)
    assert provider.output_text not in caplog.text
    event, hint = event_from_exception(error, client_options=DEFAULT_OPTIONS)
    assert provider.output_text not in json.dumps(telemetry._before_send(event, hint))
