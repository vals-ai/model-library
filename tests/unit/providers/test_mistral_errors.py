from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from mistralai.client.errors import SDKError

from model_library.exceptions import BackoffRetryException, is_retriable_error
from model_library.providers.mistral import MistralModel
from tests.unit.provider_response_helpers import _INPUT, _LOGGER


async def _query(body: str):
    response = httpx.Response(
        400,
        request=httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions"),
        headers={"content-type": "application/json"},
        text=body,
    )
    client = MagicMock()
    client.chat.stream_async = AsyncMock(
        side_effect=SDKError("API error occurred", response, body)
    )
    model = MistralModel("mistral-test")

    with (
        patch.object(model, "get_client", return_value=client),
        patch.object(model, "build_body", new_callable=AsyncMock, return_value={}),
    ):
        return await model._query_impl(_INPUT, tools=[], query_logger=_LOGGER)


async def test_mistral_parse_body_400_is_retried():
    with pytest.raises(BackoffRetryException) as exc_info:
        await _query('{"detail":"There was an error parsing the body"}')

    assert is_retriable_error(exc_info.value)
    assert isinstance(exc_info.value.__cause__, SDKError)


@pytest.mark.parametrize(
    "body",
    [
        '{"object":"error","message":"Prompt contains 300000 tokens and 0 draft tokens, too large for model with 262144 maximum context length","type":"invalid_request_prompt_too_long","param":null,"code":"3059"}',
        '{"object":"error","message":"Unexpected role \'user\' after role \'tool\'","type":"invalid_request_message_order","param":null,"code":"3230"}',
    ],
)
async def test_other_mistral_400s_are_not_retried(body: str):
    with pytest.raises(SDKError) as exc_info:
        await _query(body)

    assert not is_retriable_error(exc_info.value)
