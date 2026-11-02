"""Unit tests for model_library/query_utils.py"""

from unittest.mock import AsyncMock, PropertyMock, call, patch

import pytest

from model_library.base import LLM
from model_library.base.output import (
    FinishReason,
    FinishReasonInfo,
    QueryResult,
    QueryResultMetadata,
)
from model_library.exceptions import (
    MaxContextWindowExceededError,
    MaxOutputTokensExceededError,
)
from model_library.query_utils import query_with_truncation_retry

CONTEXT_WINDOW = 1000


# Helper functions begin.
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
    with patch.object(
        LLM,
        "input_context_window",
        new_callable=PropertyMock,
        return_value=CONTEXT_WINDOW,
    ):
        yield


# Helper functions end.


class TestQueryWithTruncationRetry:
    @pytest.mark.parametrize("doc_text", ["", "a"])
    @pytest.mark.parametrize(
        "error_type", [MaxContextWindowExceededError, MaxOutputTokensExceededError]
    )
    async def test_exhausted_document_preserves_terminal_error(
        self, mock_llm: LLM, doc_text: str, error_type: type[Exception]
    ):
        mock_llm._registry_key = "openai/gpt-4o-mini"
        mock_llm.count_tokens = AsyncMock(return_value=100)
        error = error_type("fixed instructions still exceed the limit")
        expected_prompts = (
            ["Instructions: a", "Instructions: "] if doc_text else ["Instructions: "]
        )
        mock_llm.query = AsyncMock(
            side_effect=[error_type("document too long")] * (len(expected_prompts) - 1)
            + [error, stop_result()]
        )

        with pytest.raises(error_type) as exc_info:
            await query_with_truncation_retry(
                mock_llm,
                doc_text,
                lambda d: f"Instructions: {d}",
                question_id="question",
                run_id="run",
            )

        assert exc_info.value is error
        assert mock_llm.query.call_args_list == [
            call(prompt, question_id="question", run_id="run")
            for prompt in expected_prompts
        ]

    @pytest.mark.parametrize(
        "doc_text, token_count", [("", 100), ("a", 100), ("a", 2000)]
    )
    async def test_max_tokens_stops_when_document_exhausted(
        self, mock_llm: LLM, doc_text: str, token_count: int
    ):
        mock_llm._registry_key = "openai/gpt-4o-mini"
        mock_llm.count_tokens = AsyncMock(return_value=token_count)
        expected_prompts = (
            ["Instructions: a", "Instructions: "]
            if doc_text and token_count == 100
            else ["Instructions: "]
        )
        mock_llm.query = AsyncMock(
            side_effect=[max_tokens_result()] * len(expected_prompts) + [stop_result()]
        )

        with pytest.raises(MaxOutputTokensExceededError, match="finish_reason: length"):
            await query_with_truncation_retry(
                mock_llm, doc_text, lambda d: f"Instructions: {d}"
            )

        assert [
            args[0][0] for args in mock_llm.query.call_args_list
        ] == expected_prompts

    async def test_empty_document_can_succeed_after_mixed_retries(self, mock_llm: LLM):
        mock_llm._registry_key = "openai/gpt-4o-mini"
        mock_llm.count_tokens = AsyncMock(return_value=2000)
        expected = stop_result()
        mock_llm.query = AsyncMock(
            side_effect=[
                MaxContextWindowExceededError("too long"),
                max_tokens_result(),
                expected,
            ]
        )

        result, record = await query_with_truncation_retry(
            mock_llm,
            "abcde",
            lambda d: f"Instructions: {d}",
            question_id="question",
            run_id="run",
        )

        assert result is expected
        assert record == {
            "initial_context_window_truncation": 1,
            "max_context_window_exceeded_truncation": 1,
            "max_output_tokens_exceeded_truncation": 1,
        }
        assert mock_llm.query.call_args_list == [
            call(f"Instructions: {doc}", question_id="question", run_id="run")
            for doc in ["ab", "a", ""]
        ]
        mock_llm.count_tokens.assert_awaited_once()

    async def test_unrelated_query_error_propagates(self, mock_llm: LLM):
        mock_llm._registry_key = "openai/gpt-4o-mini"
        mock_llm.count_tokens = AsyncMock(return_value=100)
        error = RuntimeError("unrelated failure")
        mock_llm.query = AsyncMock(side_effect=error)

        with pytest.raises(RuntimeError) as exc_info:
            await query_with_truncation_retry(mock_llm, "doc", lambda d: d)

        assert exc_info.value is error
        mock_llm.query.assert_awaited_once()

    @pytest.mark.parametrize(
        "reason",
        [
            reason
            for reason in FinishReason
            if reason not in {FinishReason.MAX_TOKENS, FinishReason.UNKNOWN}
        ],
    )
    async def test_other_finish_reasons_are_returned_unchanged(
        self, mock_llm: LLM, reason: FinishReason
    ):
        mock_llm._registry_key = "openai/gpt-4o-mini"
        mock_llm.count_tokens = AsyncMock(return_value=100)
        expected = stop_result()
        expected.finish_reason = FinishReasonInfo(reason=reason, raw=reason.value)
        mock_llm.query = AsyncMock(return_value=expected)

        result, record = await query_with_truncation_retry(mock_llm, "", lambda d: d)

        assert result is expected
        assert record == {}
        mock_llm.query.assert_awaited_once()

    async def test_missing_context_window_raises_before_query(self, mock_llm: LLM):
        mock_llm._registry_key = "openai/gpt-4o-mini"
        mock_llm.query = AsyncMock(return_value=stop_result())

        with (
            patch.object(
                LLM,
                "input_context_window",
                new_callable=PropertyMock,
                return_value=None,
            ),
            patch(
                "model_library.query_utils.get_model_input_context_window",
                return_value=None,
            ),
            pytest.raises(ValueError, match="no input context window"),
        ):
            await query_with_truncation_retry(mock_llm, "doc", lambda d: d)

        mock_llm.query.assert_not_awaited()

    async def test_no_truncation_needed(self, mock_llm: LLM):
        mock_llm._registry_key = "openai/gpt-4o-mini"  # pyright: ignore[reportAttributeAccessIssue]
        mock_llm.count_tokens = AsyncMock(return_value=100)  # pyright: ignore[reportAttributeAccessIssue]
        mock_llm.query = AsyncMock(return_value=stop_result())  # pyright: ignore[reportAttributeAccessIssue]

        result, record = await query_with_truncation_retry(
            mock_llm, "doc", lambda d: f"prompt {d}"
        )

        assert result.output_text == "answer"
        assert record == {}
        mock_llm.query.assert_called_once()

    async def test_initial_truncation(self, mock_llm: LLM):
        mock_llm._registry_key = "openai/gpt-4o-mini"  # pyright: ignore[reportAttributeAccessIssue]
        mock_llm.count_tokens = AsyncMock(return_value=2000)  # pyright: ignore[reportAttributeAccessIssue]
        mock_llm.query = AsyncMock(return_value=stop_result())  # pyright: ignore[reportAttributeAccessIssue]

        result, record = await query_with_truncation_retry(
            mock_llm, "a" * 1000, lambda d: d
        )

        assert record == {"initial_context_window_truncation": 1}
        called_prompt = mock_llm.query.call_args[0][0]
        assert len(called_prompt) == 500  # 1000 * (1000/2000)

    async def test_max_context_window_error_retry(self, mock_llm: LLM):
        mock_llm._registry_key = "openai/gpt-4o-mini"  # pyright: ignore[reportAttributeAccessIssue]
        mock_llm.count_tokens = AsyncMock(return_value=100)  # pyright: ignore[reportAttributeAccessIssue]
        mock_llm.query = AsyncMock(
            side_effect=[MaxContextWindowExceededError("too long"), stop_result()]
        )  # pyright: ignore[reportAttributeAccessIssue]

        result, record = await query_with_truncation_retry(
            mock_llm, "a" * 1000, lambda d: d
        )

        assert record == {"max_context_window_exceeded_truncation": 1}
        assert mock_llm.query.call_count == 2

    async def test_max_tokens_retry(self, mock_llm: LLM):
        mock_llm._registry_key = "openai/gpt-4o-mini"  # pyright: ignore[reportAttributeAccessIssue]
        mock_llm.count_tokens = AsyncMock(return_value=100)  # pyright: ignore[reportAttributeAccessIssue]
        mock_llm.query = AsyncMock(side_effect=[max_tokens_result(), stop_result()])  # pyright: ignore[reportAttributeAccessIssue]

        result, record = await query_with_truncation_retry(
            mock_llm, "a" * 1000, lambda d: d
        )

        assert record == {"max_output_tokens_exceeded_truncation": 1}
        assert mock_llm.query.call_count == 2

    async def test_unknown_finish_reason_raises(self, mock_llm: LLM):
        mock_llm._registry_key = "openai/gpt-4o-mini"  # pyright: ignore[reportAttributeAccessIssue]
        mock_llm.count_tokens = AsyncMock(return_value=100)  # pyright: ignore[reportAttributeAccessIssue]
        mock_llm.query = AsyncMock(
            return_value=QueryResult(  # pyright: ignore[reportAttributeAccessIssue]
                finish_reason=FinishReasonInfo(
                    reason=FinishReason.UNKNOWN, raw="unknown"
                ),
            )
        )

        with pytest.raises(ValueError, match="Unknown finish reason"):
            await query_with_truncation_retry(mock_llm, "doc", lambda d: d)

    async def test_no_registry_key_raises(self, mock_llm: LLM):
        mock_llm._registry_key = None  # pyright: ignore[reportAttributeAccessIssue]

        with pytest.raises(ValueError, match="no registry key"):
            await query_with_truncation_retry(mock_llm, "doc", lambda d: d)

    async def test_registry_key_fallback_when_metadata_missing(self, mock_llm: LLM):
        mock_llm._registry_key = "openai/gpt-4o-mini"  # pyright: ignore[reportAttributeAccessIssue]
        mock_llm.count_tokens = AsyncMock(return_value=100)  # pyright: ignore[reportAttributeAccessIssue]
        mock_llm.query = AsyncMock(return_value=stop_result())  # pyright: ignore[reportAttributeAccessIssue]

        with (
            patch.object(
                LLM,
                "input_context_window",
                new_callable=PropertyMock,
                return_value=None,
            ),
            patch(
                "model_library.query_utils.get_model_input_context_window",
                return_value=CONTEXT_WINDOW,
            ) as fallback,
        ):
            result, record = await query_with_truncation_retry(
                mock_llm, "doc", lambda d: f"prompt {d}"
            )

        assert result.output_text == "answer"
        assert record == {}
        fallback.assert_called_once_with("openai/gpt-4o-mini")

    async def test_truncation_record_only_nonzero_keys(self, mock_llm: LLM):
        mock_llm._registry_key = "openai/gpt-4o-mini"  # pyright: ignore[reportAttributeAccessIssue]
        mock_llm.count_tokens = AsyncMock(return_value=100)  # pyright: ignore[reportAttributeAccessIssue]
        mock_llm.query = AsyncMock(return_value=stop_result())  # pyright: ignore[reportAttributeAccessIssue]

        _, record = await query_with_truncation_retry(mock_llm, "doc", lambda d: d)

        assert record == {}
