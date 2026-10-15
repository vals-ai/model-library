from unittest.mock import patch

import httpx
import pytest
from openai import AsyncOpenAI, BadRequestError
from pydantic import SecretStr

from model_library import model
from model_library.base import LLMConfig, TextInput
from model_library.exceptions import MaxContextWindowExceededError

CONTEXT_OVERFLOW = {
    "error": {"message": "The input you provided is invalid", "type": "input_invalid"}
}
BAD_TOOL_SCHEMA = {
    "error": {
        "message": "Toolcall params are invalid, detail:invalid tool type ",
        "type": "input_invalid",
    }
}
BAD_TEMPERATURE = {
    "error": {
        "message": "Request param: temperature is invalid, recommended val is: >=0 and <=2",
        "type": "input_invalid",
    }
}


async def _query_with_400(body: dict[str, object]) -> None:
    llm = model(
        "stepfun/step-5-preview",
        override_config=LLMConfig(max_tokens=16, custom_api_key=SecretStr("test")),
    )
    assert llm.delegate is not None

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json=body, request=request)

    async with AsyncOpenAI(
        api_key="test",
        base_url="https://api.stepfun.ai/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        with patch.object(llm.delegate, "get_client", return_value=client):
            await llm.query([TextInput(text="hi")])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_context_overflow_400_is_context_window_error() -> None:
    with pytest.raises(MaxContextWindowExceededError):
        await _query_with_400(CONTEXT_OVERFLOW)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("body", [BAD_TOOL_SCHEMA, BAD_TEMPERATURE])
async def test_other_input_invalid_400_is_not_context_window_error(
    body: dict[str, object],
) -> None:
    with pytest.raises(BadRequestError):
        await _query_with_400(body)


