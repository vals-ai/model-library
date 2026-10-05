from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from mistralai.client.models import UsageInfo

from model_library.providers.mistral import MistralModel
from tests.unit.provider_response_helpers import _INPUT, _LOGGER


async def _query(usage: UsageInfo):
    async def stream() -> AsyncIterator[Any]:
        yield SimpleNamespace(
            data=SimpleNamespace(
                id="mistral-response-1",
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="hello", tool_calls=None),
                        finish_reason="stop",
                    )
                ],
                usage=usage,
            )
        )

    client = MagicMock()
    client.chat.stream_async = AsyncMock(return_value=stream())
    model = MistralModel("mistral-test")

    with (
        patch.object(model, "get_client", return_value=client),
        patch.object(model, "build_body", new_callable=AsyncMock, return_value={}),
    ):
        return await model._query_impl(_INPUT, tools=[], query_logger=_LOGGER)


async def test_mistral_query_reports_cached_prompt_tokens():
    usage = UsageInfo.model_validate(
        {
            "prompt_tokens": 38026,
            "completion_tokens": 63,
            "total_tokens": 38089,
            "prompt_tokens_details": {"cached_tokens": 37888},
        }
    )

    result = await _query(usage)

    assert result.metadata.in_tokens == 138
    assert result.metadata.cache_read_tokens == 37888
    assert result.metadata.out_tokens == 63


async def test_mistral_query_without_cache_details_reports_zero_cache_reads():
    usage = UsageInfo.model_validate(
        {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105}
    )

    result = await _query(usage)

    assert result.metadata.in_tokens == 100
    assert result.metadata.cache_read_tokens == 0
