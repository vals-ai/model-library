"""Unit tests for query_logger parameter handling in LLM.query()"""

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
