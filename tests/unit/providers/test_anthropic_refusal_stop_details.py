"""Anthropic refusal `stop_details` in result metadata."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic.types.beta import BetaMessage

from model_library.base import TextInput
from model_library.base.output import FinishReason, QueryResult
from model_library.exceptions import ContentFilterError
from model_library.providers.anthropic import AnthropicModel

_TOKEN = "fallback-credit-token-value"
_STOP_DETAILS = {
    "type": "refusal",
    "category": "cyber",
    "explanation": "declined",
    "fallback_credit_token": _TOKEN,
    "recommended_model": "claude-fallback-test",
}
_RECORDED = {
    "category": "cyber",
    "recommended_model": "claude-fallback-test",
    "has_fallback_credit_token": True,
}


def _message(
    content: list[dict[str, Any]],
    stop_reason: str = "refusal",
    stop_details: dict[str, Any] | None = _STOP_DETAILS,
) -> BetaMessage:
    return BetaMessage.model_validate(
        {
            "id": "msg_refusal",
            "type": "message",
            "role": "assistant",
            "model": "claude-primary-test",
            "stop_reason": stop_reason,
            "stop_details": stop_details,
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "content": content,
        }
    )


async def _query(message: BetaMessage) -> QueryResult:
    model = AnthropicModel("claude-primary-test")
    stream = AsyncMock()
    stream.__aenter__ = AsyncMock(return_value=stream)
    stream.__aexit__ = AsyncMock(return_value=None)
    stream.get_final_message = AsyncMock(return_value=message)
    client = MagicMock()
    client.beta.messages.stream.return_value = stream
    body: dict[str, Any] = {
        "model": "claude-primary-test",
        "messages": [],
        "max_tokens": 512,
    }

    with (
        patch.object(model, "get_client", return_value=client),
        patch.object(model, "build_body", AsyncMock(return_value=body)),
    ):
        return await model._query_impl(  # pyright: ignore[reportPrivateUsage]
            [TextInput(text="reverse engineer this binary")],
            tools=[],
            query_logger=MagicMock(),
        )


async def test_refusal_after_output_records_stop_details_without_the_token():
    result = await _query(_message([{"type": "text", "text": "partial answer"}]))

    assert result.finish_reason.reason == FinishReason.CONTENT_FILTER
    assert result.metadata.extra["anthropic_refusal_stop_details"] == _RECORDED
    assert _TOKEN not in result.metadata.model_dump_json()
    assert _TOKEN not in str(result.history)


async def test_refusal_without_output_raises_with_stop_details_on_the_result():
    with pytest.raises(ContentFilterError) as caught:
        await _query(_message([]))

    query_result = caught.value.query_result
    assert query_result is not None
    assert query_result.finish_reason.reason == FinishReason.CONTENT_FILTER
    assert query_result.metadata.in_tokens == 10
    assert query_result.metadata.extra["anthropic_refusal_stop_details"] == _RECORDED
    assert _TOKEN not in str(caught.value)
    assert _TOKEN not in str(query_result.history)
    assert "'has_fallback_credit_token': True" in str(caught.value)


async def test_refusal_without_a_credit_token_reports_false():
    result = await _query(
        _message(
            [{"type": "text", "text": "partial answer"}],
            stop_details={"type": "refusal", "category": None},
        )
    )

    assert result.metadata.extra["anthropic_refusal_stop_details"] == {
        "category": None,
        "recommended_model": None,
        "has_fallback_credit_token": False,
    }


async def test_non_refusal_turn_has_no_stop_details():
    result = await _query(
        _message(
            [{"type": "text", "text": "answer"}],
            stop_reason="end_turn",
            stop_details=None,
        )
    )

    assert "anthropic_refusal_stop_details" not in result.metadata.extra
