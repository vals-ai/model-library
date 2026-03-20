"""Unit tests for model_library/query_utils.py"""

from unittest.mock import AsyncMock, patch

import pytest

from model_library.base import LLM
from model_library.base.output import FinishReason, FinishReasonInfo, QueryResult, QueryResultMetadata
from model_library.exceptions import MaxContextWindowExceededError
from model_library.query_utils import query_with_truncation_retry

CONTEXT_WINDOW = 1000


def stop_result(text: str = "answer") -> QueryResult:
    return QueryResult(
        output_text=text,
        finish_reason=FinishReasonInfo(reason=FinishReason.STOP, raw="stop"),
        metadata=QueryResultMetadata(in_tokens=10, out_tokens=5),
    )


def max_tokens_result() -> QueryResult:
    return QueryResult(
        output_text="truncated",
        finish_reason=FinishReasonInfo(reason=FinishReason.MAX_TOKENS, raw="length"),
        metadata=QueryResultMetadata(in_tokens=10, out_tokens=5),
    )


@pytest.fixture(autouse=True)
def patch_context_window():
    with patch("model_library.query_utils.get_model_input_context_window", return_value=CONTEXT_WINDOW):
        yield


class TestQueryWithTruncationRetry:
    async def test_no_truncation_needed(self, mock_llm: LLM):
        mock_llm._registry_key = "openai/gpt-4o-mini"  # pyright: ignore[reportAttributeAccessIssue]
        mock_llm.count_tokens = AsyncMock(return_value=100)  # pyright: ignore[reportAttributeAccessIssue]
        mock_llm.query = AsyncMock(return_value=stop_result())  # pyright: ignore[reportAttributeAccessIssue]

        result, record = await query_with_truncation_retry(mock_llm, "doc", lambda d: f"prompt {d}")

        assert result.output_text == "answer"
        assert record == {}
        mock_llm.query.assert_called_once()

    async def test_initial_truncation(self, mock_llm: LLM):
        mock_llm._registry_key = "openai/gpt-4o-mini"  # pyright: ignore[reportAttributeAccessIssue]
        mock_llm.count_tokens = AsyncMock(return_value=2000)  # pyright: ignore[reportAttributeAccessIssue]
        mock_llm.query = AsyncMock(return_value=stop_result())  # pyright: ignore[reportAttributeAccessIssue]

        result, record = await query_with_truncation_retry(mock_llm, "a" * 1000, lambda d: d)

        assert record == {"initial_context_window_truncation": 1}
        called_prompt = mock_llm.query.call_args[0][0]
        assert len(called_prompt) == 500  # 1000 * (1000/2000)

    async def test_max_context_window_error_retry(self, mock_llm: LLM):
        mock_llm._registry_key = "openai/gpt-4o-mini"  # pyright: ignore[reportAttributeAccessIssue]
        mock_llm.count_tokens = AsyncMock(return_value=100)  # pyright: ignore[reportAttributeAccessIssue]
        mock_llm.query = AsyncMock(side_effect=[MaxContextWindowExceededError("too long"), stop_result()])  # pyright: ignore[reportAttributeAccessIssue]

        result, record = await query_with_truncation_retry(mock_llm, "a" * 1000, lambda d: d)

        assert record == {"max_context_window_exceeded_truncation": 1}
        assert mock_llm.query.call_count == 2

    async def test_max_tokens_retry(self, mock_llm: LLM):
        mock_llm._registry_key = "openai/gpt-4o-mini"  # pyright: ignore[reportAttributeAccessIssue]
        mock_llm.count_tokens = AsyncMock(return_value=100)  # pyright: ignore[reportAttributeAccessIssue]
        mock_llm.query = AsyncMock(side_effect=[max_tokens_result(), stop_result()])  # pyright: ignore[reportAttributeAccessIssue]

        result, record = await query_with_truncation_retry(mock_llm, "a" * 1000, lambda d: d)

        assert record == {"max_output_tokens_exceeded_truncation": 1}
        assert mock_llm.query.call_count == 2

    async def test_unknown_finish_reason_raises(self, mock_llm: LLM):
        mock_llm._registry_key = "openai/gpt-4o-mini"  # pyright: ignore[reportAttributeAccessIssue]
        mock_llm.count_tokens = AsyncMock(return_value=100)  # pyright: ignore[reportAttributeAccessIssue]
        mock_llm.query = AsyncMock(return_value=QueryResult(  # pyright: ignore[reportAttributeAccessIssue]
            finish_reason=FinishReasonInfo(reason=FinishReason.UNKNOWN, raw="unknown"),
        ))

        with pytest.raises(ValueError, match="Unknown finish reason"):
            await query_with_truncation_retry(mock_llm, "doc", lambda d: d)

    async def test_no_registry_key_raises(self, mock_llm: LLM):
        mock_llm._registry_key = None  # pyright: ignore[reportAttributeAccessIssue]

        with pytest.raises(ValueError, match="no registry key"):
            await query_with_truncation_retry(mock_llm, "doc", lambda d: d)

    async def test_truncation_record_only_nonzero_keys(self, mock_llm: LLM):
        mock_llm._registry_key = "openai/gpt-4o-mini"  # pyright: ignore[reportAttributeAccessIssue]
        mock_llm.count_tokens = AsyncMock(return_value=100)  # pyright: ignore[reportAttributeAccessIssue]
        mock_llm.query = AsyncMock(return_value=stop_result())  # pyright: ignore[reportAttributeAccessIssue]

        _, record = await query_with_truncation_retry(mock_llm, "doc", lambda d: d)

        assert record == {}
