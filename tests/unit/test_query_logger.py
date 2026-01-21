"""Unit tests for query_logger parameter handling in LLM.query()"""

import logging
from unittest.mock import AsyncMock, MagicMock

from model_library.base import LLM, QueryResult


async def test_query_uses_provided_query_logger(mock_llm: LLM):
    """
    Test that when query_logger is passed as a kwarg, it is used for logging
    instead of generating a fresh logger.
    """
    custom_logger = MagicMock()

    query_impl_mock = AsyncMock(return_value=QueryResult(output_text="success"))
    mock_llm._query_impl = query_impl_mock  # pyright: ignore[reportPrivateUsage]

    await mock_llm.query("Test input", query_logger=custom_logger)

    # Verify the custom logger was used for logging
    assert custom_logger.info.call_count


async def test_query_logger_passed_to_query_impl(mock_llm: LLM):
    """
    Test that the query_logger passed to query() is forwarded to _query_impl.
    """
    custom_logger = MagicMock(spec=logging.Logger)

    query_impl_mock = AsyncMock(return_value=QueryResult(output_text="success"))
    mock_llm._query_impl = query_impl_mock  # pyright: ignore[reportPrivateUsage]

    await mock_llm.query("Test input", query_logger=custom_logger)

    # Verify _query_impl was called with the custom logger
    query_impl_mock.assert_called_once()
    call_kwargs = query_impl_mock.call_args.kwargs
    assert call_kwargs["query_logger"] is custom_logger
