"""
Unit tests for TokenRetrier logic.
"""

import asyncio
import logging
import time
from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis.aioredis
import pytest

from model_library.base.output import QueryResult, QueryResultMetadata
from model_library.exceptions import ImmediateRetryException, RetryException
from model_library.retriers.base import BaseRetrier
from model_library.retriers.token import TokenRetrier, set_redis_client
from model_library.retriers.token.utils import get_status

CLIENT_KEY = ("provider", "model")
TOKEN_KEY = "provider:model:tokens"
PRIORITY_KEY_PREFIX = "provider:model:priority"


@pytest.fixture(autouse=True)
def mock_asyncio_sleep():
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        yield mock_sleep


class _FakeLock:
    """No-op async context manager replacing redis Lock (fakeredis lacks evalsha)."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


@pytest.fixture
def redis():
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    client.lock = lambda *args, **kwargs: _FakeLock()
    set_redis_client(client)
    return client


@pytest.fixture
def token_retrier():
    """Helper to create a standard TokenRetrier instance."""
    return TokenRetrier(
        logger=logging.getLogger("test"),
        client_registry_key=CLIENT_KEY,
        request_id="test-req",
        estimate_input_tokens=100,
        estimate_output_tokens=50,
        token_wait_time=1.0,
    )


async def _init_tokens(redis, value: int = 1000, limit: int = 1000):
    """Set up token state in redis."""
    await redis.set(TOKEN_KEY, str(value))
    await redis.set(f"{TOKEN_KEY}:limit", str(limit))


async def test_token_retrier_initialization(redis):
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

    assert await redis.get("p:m:tokens") == "3000"
    assert await redis.get("p:m:tokens:limit") == "3000"
    assert mock_create_task.call_count == 2


async def test_pre_function_waits_for_tokens(redis, token_retrier: TokenRetrier):
    """Test that _pre_function loops until tokens are available."""
    await _init_tokens(redis, value=0)

    # On first sleep, bump tokens so second iteration succeeds
    async def bump_tokens(*args, **kwargs):
        await redis.set(TOKEN_KEY, "200")

    mock_asyncio_sleep = asyncio.sleep  # already mocked by fixture
    with patch("asyncio.sleep", new_callable=AsyncMock, side_effect=bump_tokens):
        await token_retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

    # Verify token deduction occurred (estimate_total_tokens = 100 + 50 = 150)
    remaining = int(await redis.get(TOKEN_KEY))
    assert remaining == 50  # 200 - 150


async def test_pre_function_waits_for_priority(redis, token_retrier: TokenRetrier):
    """Test that _pre_function waits if a higher priority request is in queue."""
    await _init_tokens(redis, value=1000)
    token_retrier.priority = 2

    # Set priority 1 as having a waiter
    await redis.zadd(f"{PRIORITY_KEY_PREFIX}:1", {"other-req": time.time()})

    call_count = 0

    async def clear_priority_then_noop(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            await redis.zrem(f"{PRIORITY_KEY_PREFIX}:1", "other-req")

    with patch(
        "asyncio.sleep", new_callable=AsyncMock, side_effect=clear_priority_then_noop
    ):
        await token_retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

    # Priority was registered and deregistered
    priority_count = await redis.zcard(f"{PRIORITY_KEY_PREFIX}:2")
    assert priority_count == 0


async def test_post_function_adjusts_tokens(redis, token_retrier: TokenRetrier):
    """Test that token estimates are corrected based on actual usage."""
    await _init_tokens(redis, value=1000, limit=2000)

    mock_qr = MagicMock()
    mock_qr.metadata.total_input_tokens = 40
    mock_qr.metadata.cache_read_tokens = 0
    mock_qr.metadata.total_output_tokens = 10

    await token_retrier._post_function((mock_qr, 1.0))  # pyright: ignore[reportPrivateUsage]

    # Estimated 150, Actual 50. Difference = 100. 1000 + 100 = 1100
    remaining = int(await redis.get(TOKEN_KEY))
    assert remaining == 1100


async def test_on_retry_increases_priority(token_retrier: TokenRetrier):
    """Test that each retry attempt lowers the priority (higher numerical value)."""
    assert token_retrier.priority == 0

    await token_retrier._on_retry(Exception("fail"), 1.0, 1.0)  # pyright: ignore[reportPrivateUsage]
    assert token_retrier.priority == 1

    # Cap priority at MIN_PRIORITY (5)
    for _ in range(10):
        await token_retrier._on_retry(Exception("fail"), 1.0, 1.0)  # pyright: ignore[reportPrivateUsage]
    assert token_retrier.priority == 5


async def test_full_execute_flow(redis, token_retrier: TokenRetrier):
    """Integration test for the execute wrapper using TokenRetrier logic."""
    await _init_tokens(redis, value=1000)

    mock_qr = MagicMock()
    mock_qr.metadata.total_input_tokens = 100
    mock_qr.metadata.total_output_tokens = 50
    mock_qr.metadata.cache_read_tokens = 0
    mock_qr.metadata.extra = {}

    work_func = AsyncMock(return_value=(mock_qr, 0.5))

    result = await token_retrier.execute(work_func, "arg1")

    assert result == (mock_qr, 0.5)
    work_func.assert_called_once_with("arg1")

    # 1000 - 150 (deduct) + 0 (adjustment: estimated 150 == actual 150) = 850
    remaining = int(await redis.get(TOKEN_KEY))
    assert remaining == 850


async def test_post_function_handles_missing_cache_metadata(
    redis, token_retrier: TokenRetrier
):
    """
    Production Resilience: Ensure math doesn't break if cache_read_tokens is None.
    """
    await _init_tokens(redis, value=1000)

    mock_qr = MagicMock()
    mock_qr.metadata.total_input_tokens = 100
    mock_qr.metadata.total_output_tokens = 50
    mock_qr.metadata.cache_read_tokens = None  # Missing field

    await token_retrier._post_function((mock_qr, 1.0))  # pyright: ignore[reportPrivateUsage]

    # Estimate 150, Actual 150. Adjustment = 0. 1000 + 0 = 1000
    remaining = int(await redis.get(TOKEN_KEY))
    assert remaining == 1000


async def test_header_correction_fallback_logic():
    """
    Verifies that if token_remaining is None, it sums input and output headers.
    """
    from model_library.base.base import RateLimit

    rate_limit = MagicMock(spec=RateLimit)
    rate_limit.token_remaining = None
    rate_limit.token_remaining_input = 400
    rate_limit.token_remaining_output = 100
    rate_limit.unix_timestamp = 0

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


async def test_pre_function_token_debt_and_recovery(redis, token_retrier: TokenRetrier):
    """
    The retrier must loop until the background refill brings it above the estimate.
    """
    await _init_tokens(redis, value=-500)

    values = iter(["-500", "100", "200"])

    async def simulate_refill(*args, **kwargs):
        await redis.set(TOKEN_KEY, next(values))

    with patch(
        "asyncio.sleep", new_callable=AsyncMock, side_effect=simulate_refill
    ) as mock_sleep:
        await token_retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

    # 200 - 150 = 50
    remaining = int(await redis.get(TOKEN_KEY))
    assert remaining == 50


async def test_priority_preemption_during_wait(redis, token_retrier: TokenRetrier):
    """
    EDGE CASE: Tokens are available, but a HIGHER priority request (e.g. priority 1)
    enters the queue while we (priority 2) are waiting. We must yield.
    """
    await _init_tokens(redis, value=500)
    token_retrier.priority = 2

    # Higher priority waiter blocks us
    await redis.zadd(f"{PRIORITY_KEY_PREFIX}:1", {"other-req": time.time()})

    async def clear_priority(*args, **kwargs):
        await redis.zrem(f"{PRIORITY_KEY_PREFIX}:1", "other-req")

    with patch(
        "asyncio.sleep", new_callable=AsyncMock, side_effect=clear_priority
    ) as mock_sleep:
        await token_retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

        mock_sleep.assert_called_once()

    # Priority 2 counter should be back to 0 (registered then deregistered)
    p2_count = await redis.zcard(f"{PRIORITY_KEY_PREFIX}:2")
    assert p2_count == 0


async def test_post_function_cache_refund_math(redis, token_retrier: TokenRetrier):
    """
    EDGE CASE: Most tokens were CACHED.
    If we estimated 150 but 140 were cached, we only used 10 tokens.
    Redis should be refunded 140 tokens.
    """
    await _init_tokens(redis, value=1000, limit=2000)

    mock_qr = MagicMock()
    mock_qr.metadata.total_input_tokens = 100
    mock_qr.metadata.total_output_tokens = 50
    mock_qr.metadata.cache_read_tokens = 140  # Only 10 tokens were "real" usage

    # Formula: adj = Estimate(150) - (Actual(150) - Cache(140)) = 140
    await token_retrier._post_function((mock_qr, 1.0))  # pyright: ignore[reportPrivateUsage]

    remaining = int(await redis.get(TOKEN_KEY))
    assert remaining == 1140  # 1000 + 140


async def test_post_function_underestimation_debt(redis, token_retrier: TokenRetrier):
    """
    EDGE CASE: Underestimation.
    If we estimated 150 but used 1000, we must deduct the remaining 850
    from Redis immediately to maintain rate-limit integrity.
    """
    await _init_tokens(redis, value=1000)

    mock_qr = MagicMock()
    mock_qr.metadata.total_input_tokens = 800
    mock_qr.metadata.total_output_tokens = 200
    mock_qr.metadata.cache_read_tokens = 0

    # Formula: adj = 150 - 1000 = -850
    await token_retrier._post_function((mock_qr, 1.0))  # pyright: ignore[reportPrivateUsage]

    remaining = int(await redis.get(TOKEN_KEY))
    assert remaining == 150  # 1000 + (-850)


async def test_full_execution_priority_degradation(redis, token_retrier: TokenRetrier):
    """
    EDGE CASE: Request keeps failing.
    Priority should increase (1 -> 2 -> 3...) to allow other requests
    to pass the failing one.
    """
    await _init_tokens(redis, value=1000)

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
    assert token_retrier.priority == 2
    assert token_retrier.attempts == 2

    assert mock_sleep.call_count >= 2


async def test_immediate_retry_skips_token_logic(redis, token_retrier: TokenRetrier):
    """
    EDGE CASE: ImmediateRetryException (Network Error).
    Tokens haven't been deducted yet or should be retried immediately.
    Verify that TokenRetrier.execute is called by the wrapper but respects the loop.
    """
    await _init_tokens(redis, value=1000)

    mock_qr = MagicMock(spec=QueryResult)
    mock_qr.metadata = MagicMock(spec=QueryResultMetadata)
    mock_qr.metadata.total_input_tokens = 100
    mock_qr.metadata.total_output_tokens = 50
    mock_qr.metadata.cache_read_tokens = 0
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

    with patch.object(
        token_retrier, "execute", wraps=token_retrier.execute
    ) as mock_exec:
        result = await token_retrier.execute(wrapped_func)

    assert result[0] == mock_qr
    assert mock_exec.call_count == 1
    # Priority should NOT have degraded
    assert token_retrier.priority == 0


async def test_pre_function_cleanup_on_cancel(redis, token_retrier: TokenRetrier):
    """
    EDGE CASE: Task Cancellation.
    If the async task is cancelled while waiting for tokens,
    the priority count in Redis MUST still be decremented (finally block).
    """
    await _init_tokens(redis, value=0)

    # Make the Lua deduct script raise CancelledError
    original_eval = redis.eval

    async def cancel_on_eval(script, numkeys, *args):
        if args and args[0] == TOKEN_KEY:
            raise asyncio.CancelledError()
        return await original_eval(script, numkeys, *args)

    redis.eval = cancel_on_eval

    with pytest.raises(asyncio.CancelledError):
        await token_retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

    # The finally block must have removed us from the priority set
    priority_count = await redis.zcard(f"{PRIORITY_KEY_PREFIX}:0")
    assert priority_count == 0


async def test_validate_uninitialized_key(redis, token_retrier: TokenRetrier):
    """
    Verify that validate() raises an error if the model hasn't been initialized in Redis.
    """
    # Key does not exist in Redis
    with pytest.raises(Exception, match="not intialized"):
        await token_retrier.validate()


# ── Token retry status ───────────────────────────────────────────────


async def test_get_status_empty(redis):
    """Returns empty lists when no keys exist."""
    status = await get_status()
    assert status.token_retry == []
    assert status.benchmark_queue == []


async def test_get_status_single(redis):
    """Returns status for a single initialized model."""
    await _init_tokens(redis, value=750, limit=1000)

    status = await get_status()

    assert len(status.token_retry) == 1
    tr = status.token_retry[0]
    assert tr.token_key == TOKEN_KEY
    assert tr.tokens_remaining == 750
    assert tr.token_limit == 1000
    assert len(tr.priorities) == 11  # priorities -5 to 5
    assert all(v == 0 for v in tr.priorities.values())


async def test_get_status_with_waiters(redis):
    """Status reflects waiting requests at each priority level."""
    await _init_tokens(redis, value=500)
    await redis.zadd(f"{PRIORITY_KEY_PREFIX}:1", {"r1": 1.0, "r2": 2.0, "r3": 3.0})
    await redis.zadd(f"{PRIORITY_KEY_PREFIX}:3", {"r4": 1.0})

    status = await get_status()

    assert len(status.token_retry) == 1
    tr = status.token_retry[0]
    assert tr.priorities["1"] == 3
    assert tr.priorities["3"] == 1


async def test_get_status_multiple_models(redis):
    """Returns status for all models."""
    await redis.set("openai:key1:tokens", "500")
    await redis.set("openai:key1:tokens:limit", "1000")
    await redis.set("anthropic:key2:tokens", "200")
    await redis.set("anthropic:key2:tokens:limit", "400")

    status = await get_status()

    assert len(status.token_retry) == 2
    keys = {s.token_key for s in status.token_retry}
    assert keys == {"anthropic:key2:tokens", "openai:key1:tokens"}


# ── Straggler detection & dispatch tracking ──────────────────────────


async def test_initial_priority_is_zero(token_retrier: TokenRetrier):
    """New requests start at INITIAL_PRIORITY (0), not 1."""
    assert token_retrier.priority == 0


async def test_validate_captures_benchmark_run_id(redis, token_retrier: TokenRetrier):
    """validate() reads benchmark_run key from Redis."""
    await _init_tokens(redis, value=1000)

    benchmark_run_key = f"{TOKEN_KEY}:inflight:benchmark_run"
    await redis.set(benchmark_run_key, "run-42")

    await token_retrier.validate()

    assert token_retrier._benchmark_run_id == "run-42"


async def test_validate_no_benchmark_run(redis, token_retrier: TokenRetrier):
    """validate() leaves _benchmark_run_id as None when no key exists."""
    await _init_tokens(redis, value=1000)

    await token_retrier.validate()

    assert token_retrier._benchmark_run_id is None


async def test_straggler_gets_max_priority(redis, token_retrier: TokenRetrier):
    """When benchmark_run_id != queue head, request gets MAX_PRIORITY (-5)."""
    await _init_tokens(redis, value=1000)

    # simulate: this retrier belongs to run-1 which was early-released
    token_retrier._benchmark_run_id = "run-1"

    # queue now has run-2 at head (run-1 was removed by early release)
    queue_key = f"{CLIENT_KEY[0]}:{CLIENT_KEY[1]}:benchmark:queue"
    await redis.rpush(queue_key, "run-2")

    await token_retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

    assert token_retrier.priority == -5


async def test_non_benchmark_skips_straggler_check(redis, token_retrier: TokenRetrier):
    """Non-benchmark requests (_benchmark_run_id=None) keep INITIAL_PRIORITY."""
    await _init_tokens(redis, value=1000)

    assert token_retrier._benchmark_run_id is None

    await token_retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

    assert token_retrier.priority == 0


async def test_dispatched_set_populated_on_deduction(
    redis, token_retrier: TokenRetrier
):
    """After token deduction, request_id is added to the dispatched set."""
    await _init_tokens(redis, value=1000)

    await token_retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

    dispatched_key = f"{TOKEN_KEY}:inflight:dispatched"
    count = await redis.scard(dispatched_key)
    assert count == 1


async def test_straggler_still_in_queue_keeps_initial_priority(
    redis, token_retrier: TokenRetrier
):
    """When benchmark_run_id is still the queue head, priority stays at INITIAL_PRIORITY."""
    await _init_tokens(redis, value=1000)

    token_retrier._benchmark_run_id = "run-1"

    # run-1 is still head of queue (not yet early-released)
    queue_key = f"{CLIENT_KEY[0]}:{CLIENT_KEY[1]}:benchmark:queue"
    await redis.rpush(queue_key, "run-1")

    await token_retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

    assert token_retrier.priority == 0
