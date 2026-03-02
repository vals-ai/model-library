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
from model_library.retriers.token.utils import KEY_PREFIX, RunContext, current_run, get_status

CLIENT_KEY = ("provider", "model")
TOKEN_KEY = f"{KEY_PREFIX}:provider:model:tokens"
PRIORITY_KEY_PREFIX = f"{KEY_PREFIX}:provider:model:priority"


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

    assert await redis.get(f"{KEY_PREFIX}:p:m:tokens") == "3000"
    assert await redis.get(f"{KEY_PREFIX}:p:m:tokens:limit") == "3000"
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
    """Returns empty models list when no keys exist."""
    status = await get_status()
    assert status.models == []


async def test_get_status_single(redis):
    """Returns status for a single initialized model."""
    await _init_tokens(redis, value=750, limit=1000)

    status = await get_status()

    assert len(status.models) == 1
    tr = status.models[0].token
    assert tr is not None
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

    assert len(status.models) == 1
    tr = status.models[0].token
    assert tr is not None
    assert tr.priorities["1"] == 3
    assert tr.priorities["3"] == 1


async def test_get_status_multiple_models(redis):
    """Returns status for all models."""
    await redis.set(f"{KEY_PREFIX}:openai:key1:tokens", "500")
    await redis.set(f"{KEY_PREFIX}:openai:key1:tokens:limit", "1000")
    await redis.set(f"{KEY_PREFIX}:anthropic:key2:tokens", "200")
    await redis.set(f"{KEY_PREFIX}:anthropic:key2:tokens:limit", "400")

    status = await get_status()

    assert len(status.models) == 2
    keys = {m.token.token_key for m in status.models if m.token}
    assert keys == {f"{KEY_PREFIX}:anthropic:key2:tokens", f"{KEY_PREFIX}:openai:key1:tokens"}


# ── Straggler detection & dispatch tracking ──────────────────────────


async def test_initial_priority_is_zero(token_retrier: TokenRetrier):
    """New requests start at INITIAL_PRIORITY (0), not 1."""
    assert token_retrier.priority == 0


async def test_contextvar_sets_run_id():
    """TokenRetrier reads run_id from contextvar when set."""
    token = current_run.set(RunContext(run_id="run-42", is_queued=True))
    try:
        retrier = TokenRetrier(
            logger=logging.getLogger("test"),
            client_registry_key=CLIENT_KEY,
            request_id="req-1",
            estimate_input_tokens=100,
            estimate_output_tokens=50,
        )
        assert retrier._run_id == "run-42"
        assert retrier._is_queued is True
    finally:
        current_run.reset(token)


async def test_contextvar_fallback_to_instance_id():
    """Without contextvar, _run_id falls back to dynamic_estimate_instance_id."""
    retrier = TokenRetrier(
        logger=logging.getLogger("test"),
        client_registry_key=CLIENT_KEY,
        request_id="req-1",
        estimate_input_tokens=100,
        estimate_output_tokens=50,
        dynamic_estimate_instance_id="inst-abc",
    )
    assert retrier._run_id == "inst-abc"
    assert retrier._is_queued is False


async def test_contextvar_no_fallback():
    """Without contextvar or instance_id, _run_id is None."""
    retrier = TokenRetrier(
        logger=logging.getLogger("test"),
        client_registry_key=CLIENT_KEY,
        request_id="req-1",
        estimate_input_tokens=100,
        estimate_output_tokens=50,
    )
    assert retrier._run_id is None
    assert retrier._is_queued is False
    assert retrier.dynamic_estimate_key is None


async def test_dynamic_estimate_key_uses_run_id():
    """Dynamic estimate key is built from contextvar run_id, not instance_id."""
    token = current_run.set(RunContext(run_id="run-99", is_queued=True))
    try:
        retrier = TokenRetrier(
            logger=logging.getLogger("test"),
            client_registry_key=CLIENT_KEY,
            request_id="req-1",
            estimate_input_tokens=100,
            estimate_output_tokens=50,
            dynamic_estimate_instance_id="inst-abc",  # should be ignored
        )
        assert retrier.dynamic_estimate_key == f"{TOKEN_KEY}:dynamic_estimate:run-99"
    finally:
        current_run.reset(token)


async def test_validate_straggler_when_queued(redis):
    """validate() detects straggler: is_queued but active benchmark != our run."""
    await _init_tokens(redis, value=1000)

    benchmark_run_key = f"{TOKEN_KEY}:inflight:benchmark_run"
    await redis.set(benchmark_run_key, "run-2")  # another run has the slot

    token = current_run.set(RunContext(run_id="run-1", is_queued=True))
    try:
        retrier = TokenRetrier(
            logger=logging.getLogger("test"),
            client_registry_key=CLIENT_KEY,
            request_id="req-1",
            estimate_input_tokens=100,
            estimate_output_tokens=50,
        )
        await retrier.validate()
        assert retrier._is_straggler is True
    finally:
        current_run.reset(token)


async def test_validate_not_straggler_when_active(redis):
    """validate() is not straggler when our run is the active benchmark."""
    await _init_tokens(redis, value=1000)

    benchmark_run_key = f"{TOKEN_KEY}:inflight:benchmark_run"
    await redis.set(benchmark_run_key, "run-1")

    token = current_run.set(RunContext(run_id="run-1", is_queued=True))
    try:
        retrier = TokenRetrier(
            logger=logging.getLogger("test"),
            client_registry_key=CLIENT_KEY,
            request_id="req-1",
            estimate_input_tokens=100,
            estimate_output_tokens=50,
        )
        await retrier.validate()
        assert retrier._is_straggler is False
    finally:
        current_run.reset(token)


async def test_validate_not_straggler_when_not_queued(redis):
    """validate() skips straggler detection for non-queued runs."""
    await _init_tokens(redis, value=1000)

    retrier = TokenRetrier(
        logger=logging.getLogger("test"),
        client_registry_key=CLIENT_KEY,
        request_id="req-1",
        estimate_input_tokens=100,
        estimate_output_tokens=50,
    )
    await retrier.validate()
    assert retrier._is_straggler is False


async def test_straggler_gets_max_priority(redis):
    """Straggler request gets MAX_PRIORITY (-5) in _pre_function."""
    await _init_tokens(redis, value=1000)

    token = current_run.set(RunContext(run_id="run-1", is_queued=True))
    try:
        retrier = TokenRetrier(
            logger=logging.getLogger("test"),
            client_registry_key=CLIENT_KEY,
            request_id="req-straggler",
            estimate_input_tokens=100,
            estimate_output_tokens=50,
        )
        retrier._is_straggler = True

        await retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

        assert retrier.priority == -5
    finally:
        current_run.reset(token)


async def test_non_queued_skips_straggler_check(redis, token_retrier: TokenRetrier):
    """Non-queued requests keep INITIAL_PRIORITY."""
    await _init_tokens(redis, value=1000)

    assert token_retrier._is_queued is False

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


async def test_per_request_metadata_stores_run_id(redis):
    """Per-request metadata hash stores run_id field from contextvar."""
    await _init_tokens(redis, value=1000)

    token = current_run.set(RunContext(run_id="run-77", is_queued=True))
    try:
        retrier = TokenRetrier(
            logger=logging.getLogger("test"),
            client_registry_key=CLIENT_KEY,
            request_id="req-meta",
            estimate_input_tokens=100,
            estimate_output_tokens=50,
        )
        await retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

        meta = await redis.hgetall(f"{TOKEN_KEY}:inflight:req-meta")
        assert meta["run_id"] == "run-77"
        assert "benchmark_run" not in meta
    finally:
        current_run.reset(token)


async def test_per_run_dispatched_counter_incremented(redis):
    """Per-run dispatched counter is incremented for queued runs."""
    await _init_tokens(redis, value=1000)

    token = current_run.set(RunContext(run_id="run-dispatch", is_queued=True))
    try:
        retrier = TokenRetrier(
            logger=logging.getLogger("test"),
            client_registry_key=CLIENT_KEY,
            request_id="req-d1",
            estimate_input_tokens=100,
            estimate_output_tokens=50,
        )
        await retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

        counter_key = f"{KEY_PREFIX}:{CLIENT_KEY[0]}:{CLIENT_KEY[1]}:benchmark:run:run-dispatch:dispatched"
        count = await redis.get(counter_key)
        assert count == "1"
    finally:
        current_run.reset(token)


async def test_per_run_dispatched_counter_skipped_for_non_queued(redis, token_retrier: TokenRetrier):
    """Per-run dispatched counter is NOT incremented for non-queued runs."""
    await _init_tokens(redis, value=1000)

    await token_retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

    # no benchmark:run:*:dispatched keys should exist
    keys = await redis.keys(f"{KEY_PREFIX}:*:benchmark:run:*:dispatched")
    assert keys == []


async def test_dynamic_estimate_ratio_written_to_run_key(redis):
    """After execute, the EMA ratio is stored under the run_id key, not instance_id."""
    await _init_tokens(redis, value=1000, limit=2000)

    token = current_run.set(RunContext(run_id="run-ema", is_queued=True))
    try:
        retrier = TokenRetrier(
            logger=logging.getLogger("test"),
            client_registry_key=CLIENT_KEY,
            request_id="req-ema",
            estimate_input_tokens=100,
            estimate_output_tokens=50,
            dynamic_estimate_instance_id="inst-should-be-ignored",
        )

        mock_qr = MagicMock()
        mock_qr.metadata.total_input_tokens = 200
        mock_qr.metadata.total_output_tokens = 100
        mock_qr.metadata.cache_read_tokens = 0
        mock_qr.metadata.extra = {}

        await retrier.execute(AsyncMock(return_value=(mock_qr, 0.5)))

        # ratio stored under run_id key
        run_key = f"{TOKEN_KEY}:dynamic_estimate:run-ema"
        ratio = await redis.get(run_key)
        assert ratio is not None
        assert float(ratio) > 1.0  # actual 300 > estimate 150

        # no key under instance_id
        inst_key = f"{TOKEN_KEY}:dynamic_estimate:inst-should-be-ignored"
        assert await redis.get(inst_key) is None
    finally:
        current_run.reset(token)


# ── Full tokens shutdown ──────────────────────────────────────────────


async def test_refill_loop_exits_on_full_tokens_shutdown(redis, mock_asyncio_sleep):
    """Refill loop shuts down after tokens sit at limit for FULL_TOKENS_SHUTDOWN."""
    from model_library.retriers.token.token import refill_tasks

    key_tuple = ("idle", "refill")
    token_key = f"{KEY_PREFIX}:idle:refill:tokens"
    time_val = [0.0]

    async def advance_time(*args):
        time_val[0] += 301.0

    mock_asyncio_sleep.side_effect = advance_time

    with patch("model_library.retriers.token.token.time") as mock_time_mod:
        mock_time_mod.time = lambda: time_val[0]

        await TokenRetrier.init_remaining_tokens(
            client_registry_key=key_tuple,
            limit=1000,
            limit_refresh_seconds=60,
            logger=logging.getLogger("test"),
            get_rate_limit_func=AsyncMock(return_value=None),
        )

        _, refill_task = refill_tasks[f"refill:{token_key}"]
        await asyncio.wait_for(refill_task, timeout=1.0)

    assert refill_task.done()


async def test_correction_loop_exits_on_full_tokens_shutdown(
    redis, mock_asyncio_sleep
):
    """Header correction loop shuts down after tokens sit at limit for FULL_TOKENS_SHUTDOWN."""
    from model_library.retriers.token.token import refill_tasks

    key_tuple = ("idle", "correction")
    token_key = f"{KEY_PREFIX}:idle:correction:tokens"
    time_val = [0.0]

    async def advance_time(*args):
        time_val[0] += 301.0

    mock_asyncio_sleep.side_effect = advance_time

    rate_limit = MagicMock()
    rate_limit.token_remaining_total = 1000
    rate_limit.unix_timestamp = 0

    with patch("model_library.retriers.token.token.time") as mock_time_mod:
        mock_time_mod.time = lambda: time_val[0]

        await TokenRetrier.init_remaining_tokens(
            client_registry_key=key_tuple,
            limit=1000,
            limit_refresh_seconds=60,
            logger=logging.getLogger("test"),
            get_rate_limit_func=AsyncMock(return_value=rate_limit),
        )

        _, correction_task = refill_tasks[f"correction:{token_key}"]
        await asyncio.wait_for(correction_task, timeout=1.0)

    assert correction_task.done()


async def test_refill_loop_continues_when_tokens_not_full(redis, mock_asyncio_sleep):
    """Refill loop resets idle timer each iteration when tokens are below limit."""
    from model_library.retriers.token.token import refill_tasks

    key_tuple = ("active", "refill")
    token_key = f"{KEY_PREFIX}:active:refill:tokens"
    iteration_count = [0]
    time_val = [0.0]

    async def advance_time_and_deduct(*args):
        time_val[0] += 301.0
        iteration_count[0] += 1
        # keep tokens below limit so last_not_full resets each iteration
        await redis.decrby(token_key, 500)
        # after 3 iterations, change version to stop the loop
        if iteration_count[0] >= 3:
            await redis.set(f"{token_key}:version", "force-exit")

    mock_asyncio_sleep.side_effect = advance_time_and_deduct

    with patch("model_library.retriers.token.token.time") as mock_time_mod:
        mock_time_mod.time = lambda: time_val[0]

        await TokenRetrier.init_remaining_tokens(
            client_registry_key=key_tuple,
            limit=1000,
            limit_refresh_seconds=60,
            logger=logging.getLogger("test"),
            get_rate_limit_func=AsyncMock(return_value=None),
        )

        _, refill_task = refill_tasks[f"refill:{token_key}"]
        await asyncio.wait_for(refill_task, timeout=1.0)

    # loop ran 3 iterations (exited via version change, not shutdown)
    assert iteration_count[0] >= 3


async def test_refill_loop_continues_when_inflight_requests(redis, mock_asyncio_sleep):
    """Refill loop stays alive when tokens are full but requests are inflight."""
    from model_library.retriers.token.token import refill_tasks

    key_tuple = ("inflight", "refill")
    token_key = f"{KEY_PREFIX}:inflight:refill:tokens"
    inflight_key = f"{token_key}:inflight"
    iteration_count = [0]
    time_val = [0.0]

    async def advance_time(*args):
        time_val[0] += 301.0
        iteration_count[0] += 1
        # after 3 iterations, remove inflight entry so loop can exit
        if iteration_count[0] >= 3:
            await redis.zrem(inflight_key, "long-running-req")

    mock_asyncio_sleep.side_effect = advance_time

    with patch("model_library.retriers.token.token.time") as mock_time_mod:
        mock_time_mod.time = lambda: time_val[0]

        await TokenRetrier.init_remaining_tokens(
            client_registry_key=key_tuple,
            limit=1000,
            limit_refresh_seconds=60,
            logger=logging.getLogger("test"),
            get_rate_limit_func=AsyncMock(return_value=None),
        )

        # simulate a long-running inflight request
        await redis.zadd(inflight_key, {"long-running-req": 0.0})

        _, refill_task = refill_tasks[f"refill:{token_key}"]
        await asyncio.wait_for(refill_task, timeout=1.0)

    # loop survived past the shutdown threshold because of inflight request
    assert iteration_count[0] >= 3
