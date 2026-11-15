"""openai>=3.38 keeps only `error.message` in `str(APIStatusError)`; classifiers must still see the body."""

import json

import httpx2
import pytest
from openai import APIStatusError, AsyncOpenAI

from model_gateway.errors import map_exception_to_error
from model_library.exceptions import (
    MaxContextWindowExceededError,
    is_content_filter_error,
    is_context_window_error,
    is_retriable_error,
)

pytestmark = pytest.mark.unit


async def _status_error(body: object) -> APIStatusError:
    content = body if isinstance(body, str) else json.dumps(body)

    def respond(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            400, content=content.encode(), headers={"content-type": "application/json"}
        )

    client = AsyncOpenAI(
        api_key="test",
        base_url="http://provider.test/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(respond)),
    )
    with pytest.raises(APIStatusError) as raised:
        await client.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "hi"}]
        )
    return raised.value


@pytest.mark.parametrize(
    "body",
    [
        {"error": "Input length 300000 exceeds the maximum allowed input length"},
        {"detail": "prompt is too long: 250000 tokens > 200000 maximum"},
        {"msg": "too many tokens: size limit exceeded", "code": 40001},
    ],
)
async def test_context_window_error_outside_error_message(body: object) -> None:
    assert is_context_window_error(await _status_error(body))


@pytest.mark.parametrize(
    "body",
    [
        {"error": {"message": "Slow down", "type": "rate_limit_exceeded"}},
        {"error": {"message": "Try later", "type": "overloaded_error"}},
        {"error": {"message": "oops", "code": "server_error"}},
    ],
)
async def test_retriable_code_outside_error_message(body: object) -> None:
    assert is_retriable_error(await _status_error(body))


async def test_content_filter_code_outside_error_message() -> None:
    error = await _status_error(
        {"code": "DataInspectionFailed", "message": "Input may be inappropriate."}
    )
    assert is_content_filter_error(error)


async def test_gateway_maps_rate_limit_type_outside_error_message() -> None:
    error = await _status_error(
        {"error": {"message": "Slow down", "type": "rate_limit_exceeded"}}
    )
    assert map_exception_to_error(error).body.code == "provider_rate_limit"


async def test_gateway_message_keeps_plain_text_body() -> None:
    error = await _status_error("The input you provided is invalid")
    assert (
        "The input you provided is invalid"
        in map_exception_to_error(error).body.message
    )


async def test_context_window_code_without_matching_message() -> None:
    error = await _status_error(
        {"error": {"message": "Request rejected", "code": "context_length_exceeded"}}
    )
    assert is_context_window_error(error)
    assert not is_retriable_error(error)
    assert map_exception_to_error(error).body.code == "context_window_exceeded"


@pytest.mark.parametrize(
    "error",
    [
        Exception(
            "Error code: 400 - Invalid request: Your request exceeded model token limit: 262144"
        ),
        MaxContextWindowExceededError(
            "Error code: 400 - The input you provided is invalid"
        ),
    ],
    ids=["kimi-provider-error", "stepfun-wrapped-error"],
)
def test_gateway_maps_library_context_window_errors(error: Exception) -> None:
    assert map_exception_to_error(error).body.code == "context_window_exceeded"
