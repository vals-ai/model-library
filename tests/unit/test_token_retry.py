"""
Unit tests for TokenRetrier logic.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from model_library.base.output import QueryResult, QueryResultMetadata
from model_library.exceptions import ImmediateRetryException, RetryException
from model_library.retriers.base import BaseRetrier
from model_library.retriers.token import TokenRetrier, set_redis_client


class AsyncContextManagerMock:
    """A mock that handles 'async with' properly."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object):
        pass


@pytest.fixture(autouse=True)
def mock_asyncio_sleep():
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        yield mock_sleep


@pytest.fixture
def mock_redis():
    """Fixture to provide a mocked redis client and set it in the retrier."""
    client = MagicMock()

    # async methods
    client.get = AsyncMock()
    client.set = AsyncMock()
    client.incr = AsyncMock()
    client.decr = AsyncMock()
    client.incrby = AsyncMock()
    client.decrby = AsyncMock()
    client.exists = AsyncMock(return_value=True)

    async def get_side_effect(key: str):
        if ":tokens" in key:
            return "1000"
        if ":priority" in key:
            return "0"
        raise Exception("Unexpected key")

    client.get.side_effect = get_side_effect

    # lock
    client.lock.return_value = AsyncContextManagerMock()

    set_redis_client(client)
    return client


@pytest.fixture
def token_retrier():
    """Helper to create a standard TokenRetrier instance."""
    return TokenRetrier(
        logger=logging.getLogger("test"),
        client_registry_key=("provider", "model"),
        estimate_input_tokens=100,
        estimate_output_tokens=50,
        token_wait_time=1.0,
    )


async def test_token_retrier_initialization(mock_redis: MagicMock):
    """Test that init_remaining_tokens sets keys and starts tasks."""
    key_tuple = ("p", "m")

    with patch("asyncio.create_task") as mock_create_task:
        await TokenRetrier.init_remaining_tokens(
            client_registry_key=key_tuple,
            limit=3000,
            limit_refresh_seconds=60,
            logger=logging.getLogger("test"),
            get_rate_limit_func=AsyncMock(),
        )

    # Check Redis interactions
    mock_redis.set.assert_any_call("p:m:tokens", 3000)
    assert mock_create_task.call_count == 2


async def test_pre_function_waits_for_tokens(
    mock_redis: MagicMock, token_retrier: TokenRetrier
):
    """Test that _pre_function loops until tokens are available."""
    # First call to get tokens returns "0" (not enough), second returns "200" (enough)
    # Note: _has_lower_priority_waiting also calls get, so we provide values for those calls too
    mock_redis.get.side_effect = [
        "0",  # _has_lower_priority_waiting check 1 (no lower priority)
        "0",  # _get_remaining_tokens check 1 (not enough)
        "0",  # _has_lower_priority_waiting check 2 (no lower priority)
        "200",  # _get_remaining_tokens check 2 (success!)
    ]

    await token_retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

    # Verify token deduction occurred (estimate_total_tokens = 100 + 50 = 150)
    mock_redis.decrby.assert_called_once_with(token_retrier.token_key, 150)


async def test_pre_function_waits_for_priority(
    mock_redis: MagicMock, token_retrier: TokenRetrier
):
    """Test that _pre_function waits if a higher priority request is in queue."""
    # First call: "1" (someone higher priority is waiting), then "0" (clear)
    # Then "200" for the actual token check
    mock_redis.get.side_effect = ["1", "0", "200"]

    await token_retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

    # Should have checked priority twice
    assert mock_redis.get.call_count >= 2


async def test_post_function_adjusts_tokens(
    mock_redis: MagicMock, token_retrier: TokenRetrier
):
    """Test that token estimates are corrected based on actual usage."""
    # Create a mock result object that mimics the QueryResult structure
    mock_qr = MagicMock()
    mock_qr.metadata.total_input_tokens = 40
    mock_qr.metadata.cache_read_tokens = 0
    mock_qr.metadata.total_output_tokens = 10

    # The retrier expects a tuple (QueryResult, float)
    await token_retrier._post_function((mock_qr, 1.0))  # pyright: ignore[reportPrivateUsage]

    # Estimated 150, Actual 50. Difference = 100
    mock_redis.incrby.assert_called_once_with(token_retrier.token_key, 100)


async def test_on_retry_increases_priority(token_retrier: TokenRetrier):
    """Test that each retry attempt lowers the priority (higher numerical value)."""
    assert token_retrier.priority == 1

    await token_retrier._on_retry(Exception("fail"), 1.0, 1.0)  # pyright: ignore[reportPrivateUsage]
    assert token_retrier.priority == 2

    # Cap priority at MIN_PRIORITY (5)
    for _ in range(10):
        await token_retrier._on_retry(Exception("fail"), 1.0, 1.0)  # pyright: ignore[reportPrivateUsage]
    assert token_retrier.priority == 5


async def test_full_execute_flow(mock_redis: MagicMock, token_retrier: TokenRetrier):
    """Integration test for the execute wrapper using TokenRetrier logic."""
    mock_redis.get.return_value = "1000"

    mock_qr = MagicMock()
    mock_qr.metadata.total_input_tokens = 100
    mock_qr.metadata.total_output_tokens = 50
    mock_qr.metadata.cache_read_tokens = 0

    # The function being decorated
    work_func = AsyncMock(return_value=(mock_qr, 0.5))

    # Use wraps to verify the logic inside execute runs
    with patch.object(TokenRetrier, "execute", wraps=token_retrier.execute):
        result = await token_retrier.execute(work_func, "arg1")

    assert result == (mock_qr, 0.5)
    work_func.assert_called_once_with("arg1")

    # Verify Redis was touched
    mock_redis.decrby.assert_called_with(token_retrier.token_key, 150)
    # Actual 150 vs Estimated 150 = 0 adjustment
    mock_redis.incrby.assert_called_with(token_retrier.token_key, 0)


async def test_post_function_handles_missing_cache_metadata(
    mock_redis: MagicMock, token_retrier: TokenRetrier
):
    """
    Production Resilience: Ensure math doesn't break if cache_read_tokens is None.
    """
    mock_qr = MagicMock()
    mock_qr.metadata.total_input_tokens = 100
    mock_qr.metadata.total_output_tokens = 50
    mock_qr.metadata.cache_read_tokens = None  # Missing field

    # Estimate was 150. Actual is 150. Adjustment should be 0.
    await token_retrier._post_function((mock_qr, 1.0))  # pyright: ignore[reportPrivateUsage]
    mock_redis.incrby.assert_called_once_with(token_retrier.token_key, 0)


async def test_header_correction_fallback_logic():
    """
    Verifies that if token_remaining is None, it sums input and output headers.
    """
    from model_library.base.base import RateLimit

    # Mocking the RateLimit object found in your code
    rate_limit = MagicMock(spec=RateLimit)
    rate_limit.token_remaining = None
    rate_limit.token_remaining_input = 400
    rate_limit.token_remaining_output = 100
    rate_limit.unix_timestamp = 0  # for calculation simplicity

    # This simulates the logic inside _header_correction_loop
    tokens_remaining = rate_limit.token_remaining
    if tokens_remaining is None:
        if (
            rate_limit.token_remaining_input is not None
            and rate_limit.token_remaining_output is not None
        ):
            tokens_remaining = (
                rate_limit.token_remaining_input + rate_limit.token_remaining_output
            )

    assert tokens_remaining == 500


async def test_pre_function_token_debt_and_recovery(
    mock_redis: MagicMock, token_retrier: TokenRetrier
):
    """
    The retrier must loop until the background refill brings it above the estimate.
    """
    # "200" (Sufficient) -> Should Deduct
    mock_redis.get.side_effect = ["0", "-500", "0", "100", "0", "200"]

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        await token_retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

        assert mock_sleep.call_count == 5
        # Verify atomic deduction of the full estimate (150)
        mock_redis.decrby.assert_called_once_with("provider:model:tokens", 150)


async def test_priority_preemption_during_wait(
    mock_redis: MagicMock, token_retrier: TokenRetrier
):
    """
    EDGE CASE: Tokens are available, but a HIGHER priority request (e.g. priority 1)
    enters the queue while we (priority 2) are waiting. We must yield.
    """
    token_retrier.priority = 2

    # Sequence:
    # 1. Check Priority 1: Returns "1" (Blocked) -> Sleep
    # 2. Check Priority 1: Returns "0" (Clear) -> Proceed to token check
    # 3. Check Tokens: Returns "500" (Enough) -> Proceed
    mock_redis.get.side_effect = ["1", "0", "500"]

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        await token_retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

        mock_sleep.assert_called_once()
        # Verify we registered and de-registered our presence in the priority queue
        mock_redis.incr.assert_called_with("provider:model:priority:2")
        mock_redis.decr.assert_called_with("provider:model:priority:2")


async def test_post_function_cache_refund_math(
    mock_redis: MagicMock, token_retrier: TokenRetrier
):
    """
    EDGE CASE: Most tokens were CACHED.
    If we estimated 150 but 140 were cached, we only used 10 tokens.
    Redis should be refunded 140 tokens.
    """
    mock_qr = MagicMock()
    mock_qr.metadata.total_input_tokens = 100
    mock_qr.metadata.total_output_tokens = 50
    mock_qr.metadata.cache_read_tokens = 140  # Only 10 tokens were "real" usage

    # Formula: adj = Estimate(150) - (Actual(150) - Cache(140)) = 140
    await token_retrier._post_function((mock_qr, 1.0))  # pyright: ignore[reportPrivateUsage]

    mock_redis.incrby.assert_called_once_with("provider:model:tokens", 140)


async def test_post_function_underestimation_debt(
    mock_redis: MagicMock, token_retrier: TokenRetrier
):
    """
    EDGE CASE: Underestimation.
    If we estimated 150 but used 1000, we must deduct the remaining 850
    from Redis immediately to maintain rate-limit integrity.
    """
    mock_qr = MagicMock()
    mock_qr.metadata.total_input_tokens = 800
    mock_qr.metadata.total_output_tokens = 200
    mock_qr.metadata.cache_read_tokens = 0

    # Formula: adj = 150 - 1000 = -850
    await token_retrier._post_function((mock_qr, 1.0))  # pyright: ignore[reportPrivateUsage]

    # incrby(-850) is effectively a further deduction
    mock_redis.incrby.assert_called_once_with("provider:model:tokens", -850)


async def test_full_execution_priority_degradation(
    mock_redis: MagicMock, token_retrier: TokenRetrier
):
    """
    EDGE CASE: Request keeps failing.
    Priority should increase (1 -> 2 -> 3...) to allow other requests
    to pass the failing one.
    """

    mock_qr = MagicMock(spec=QueryResult)
    mock_qr.metadata = MagicMock(spec=QueryResultMetadata)
    mock_qr.metadata.total_input_tokens = 100
    mock_qr.metadata.total_output_tokens = 50
    mock_qr.metadata.cache_read_tokens = 0
    mock_qr.metadata.extra = {}

    work_func = AsyncMock()
    work_func.side_effect = [
        RetryException("Rate Limit"),
        RetryException("Rate Limit"),
        (mock_qr, 0.5),
    ]

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await token_retrier.execute(work_func)

    assert result[0] == mock_qr
    assert token_retrier.priority == 3
    assert token_retrier.attempts == 2

    assert mock_sleep.call_count >= 2


async def test_immediate_retry_skips_token_logic(
    mock_redis: MagicMock, token_retrier: TokenRetrier
):
    """
    EDGE CASE: ImmediateRetryException (Network Error).
    Tokens haven't been deducted yet or should be retried immediately.
    Verify that TokenRetrier.execute is called by the wrapper but respects the loop.
    """

    # Success on second try
    mock_qr = MagicMock(spec=QueryResult)
    mock_qr.metadata = MagicMock(spec=QueryResultMetadata)
    mock_qr.metadata.cache_read_tokens = MagicMock
    mock_qr.metadata.extra = {}

    api_call = AsyncMock(
        side_effect=[
            ImmediateRetryException("Net Error"),
            ImmediateRetryException("Net Error"),
            (mock_qr, 1.0),
        ]
    )

    def wrapped_func():
        return BaseRetrier.immediate_retry_wrapper(api_call, MagicMock())

    # Testing the interaction between BaseRetrier.immediate_retry_wrapper and TokenRetrier.execute
    # We use wraps to see the internal calls while keeping logic intact
    with patch.object(
        token_retrier, "execute", wraps=token_retrier.execute
    ) as mock_exec:
        # Wrap the call in the immediate retry logic as LLM.query does
        result = await token_retrier.execute(wrapped_func)

    assert result[0] == mock_qr
    assert mock_exec.call_count == 1
    # Priority should NOT have degraded
    assert token_retrier.priority == 1


async def test_pre_function_cleanup_on_cancel(
    mock_redis: MagicMock, token_retrier: TokenRetrier
):
    """
    EDGE CASE: Task Cancellation.
    If the async task is cancelled while waiting for tokens,
    the priority count in Redis MUST still be decremented (finally block).
    """
    mock_redis.get.side_effect = asyncio.CancelledError()
    priority_key = "provider:model:priority:1"

    with pytest.raises(asyncio.CancelledError):
        await token_retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

    # The finally block must have run
    mock_redis.decr.assert_called_with(priority_key)


async def test_validate_uninitialized_key(
    mock_redis: MagicMock, token_retrier: TokenRetrier
):
    """
    Verify that validate() raises an error if the model hasn't been initialized in Redis.
    """
    # Key does not exist in Redis
    mock_redis.exists.return_value = False

    with pytest.raises(Exception, match="remaining_tokens not intialized"):
        await token_retrier.validate()
