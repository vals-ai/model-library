from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from google.genai.types import GenerateContentResponse

from model_library.base import LLMConfig, QueryResult, RawResponse
from model_library.base.output import FinishReason
from model_library.exceptions import (
    ContentFilterError,
    ImmediateRetryExhaustedError,
    ModelNoOutputError,
)
from model_library.providers.google.google import GoogleModel


def _answer(*, usage: dict[str, int] | None = None) -> GenerateContentResponse:
    return GenerateContentResponse.model_validate(
        {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": "hello"}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": usage,
        }
    )


_FINAL_USAGE = {
    "promptTokenCount": 100,
    "cachedContentTokenCount": 20,
    "candidatesTokenCount": 5,
    "thoughtsTokenCount": 7,
    "totalTokenCount": 112,
}


async def _query(
    monkeypatch: pytest.MonkeyPatch,
    chunks: list[GenerateContentResponse],
) -> QueryResult:
    async def stream() -> AsyncIterator[GenerateContentResponse]:
        for chunk in chunks:
            yield chunk

    create = AsyncMock(side_effect=lambda **_kwargs: stream())
    client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=create))
    )
    model = GoogleModel(
        "gemini-2.5-flash",
        config=LLMConfig(registry_key="google/gemini-2.5-flash"),
    )
    monkeypatch.setattr(model, "get_client", lambda: client)
    result = await model.query("Hello")
    create.assert_awaited_once()
    return result


@pytest.mark.parametrize("candidates", [None, []], ids=["absent", "empty"])
@pytest.mark.parametrize(
    "initial_usage",
    [None, {"promptTokenCount": 60, "candidatesTokenCount": 1}],
    ids=["no-earlier-usage", "earlier-usage"],
)
async def test_google_query_preserves_usage_only_tail(
    monkeypatch: pytest.MonkeyPatch,
    candidates,
    initial_usage,
) -> None:
    result = await _query(
        monkeypatch,
        [
            _answer(usage=initial_usage),
            GenerateContentResponse.model_validate(
                {
                    "candidates": candidates,
                    "usageMetadata": _FINAL_USAGE,
                    "responseId": "usage-only-response",
                }
            ),
            GenerateContentResponse(),
        ],
    )

    assert result.output_text == "hello"
    assert result.finish_reason.reason == FinishReason.STOP
    assert result.metadata.in_tokens == 80
    assert result.metadata.cache_read_tokens == 20
    assert result.metadata.total_input_tokens == 100
    assert result.metadata.out_tokens == 5
    assert result.metadata.reasoning_tokens == 7
    assert result.metadata.total_output_tokens == 12
    assert result.metadata.cost is not None and result.metadata.cost.total > 0
    assert result.extras.provider_response_id == "usage-only-response"
    assert len(result.history) == 2
    assert isinstance(result.history[-1], RawResponse)
    assert len(result.history[-1].response) == 1


async def test_google_usage_on_content_chunk_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = await _query(monkeypatch, [_answer(usage=_FINAL_USAGE)])

    assert result.metadata.total_input_tokens == 100
    assert result.metadata.total_output_tokens == 12


async def test_google_metadata_only_response_is_still_no_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ImmediateRetryExhaustedError) as caught:
        await _query(
            monkeypatch,
            [GenerateContentResponse(usage_metadata=_FINAL_USAGE)],
        )
    assert isinstance(caught.value.original, ModelNoOutputError)


async def test_google_usage_does_not_hide_blocked_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ContentFilterError, match="prompt blocked"):
        await _query(
            monkeypatch,
            [
                GenerateContentResponse.model_validate(
                    {
                        "promptFeedback": {"blockReason": "SAFETY"},
                        "usageMetadata": _FINAL_USAGE,
                    }
                )
            ],
        )
