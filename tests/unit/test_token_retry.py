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
from model_library.retriers.token.utils import KEY_PREFIX, get_status

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


def _make_retrier(
    *,
    run_id: str = "test-instance",
    question_id: str = "test-qid",
    is_queued: bool = False,
    estimate_input_tokens: int = 100,
    estimate_output_tokens: int = 50,
    use_dynamic_estimate: bool = True,
    token_wait_time: float = 1.0,
) -> TokenRetrier:
    """Helper to create a TokenRetrier with sensible defaults."""
    return TokenRetrier(
        logger=logging.getLogger("test"),
        client_registry_key=CLIENT_KEY,
        run_id=run_id,
        question_id=question_id,
        is_queued=is_queued,
        estimate_input_tokens=estimate_input_tokens,
        estimate_output_tokens=estimate_output_tokens,
        use_dynamic_estimate=use_dynamic_estimate,
        token_wait_time=token_wait_time,
    )


@pytest.fixture
def token_retrier():
    """Standard non-queued TokenRetrier instance."""
    return _make_retrier()


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


# ── Constructor & identity ───────────────────────────────────────────


async def test_initial_priority_is_zero():
    """New requests start at INITIAL_PRIORITY (0)."""
    retrier = _make_retrier()
    assert retrier.priority == 0


async def test_run_id_stored():
    """run_id is stored as _run_id."""
    retrier = _make_retrier(run_id="run-42", is_queued=True)
    assert retrier._run_id == "run-42"
    assert retrier._is_queued is True


async def test_non_queued_defaults():
    """Non-queued retrier uses provided run_id and is not queued."""
    retrier = _make_retrier(run_id="inst-abc")
    assert retrier._run_id == "inst-abc"
    assert retrier._is_queued is False


async def test_dynamic_estimate_disabled():
    """use_dynamic_estimate=False disables dynamic estimate key but keeps run tracking."""
    retrier = _make_retrier(run_id="inst-abc", use_dynamic_estimate=False)
    assert retrier._run_id == "inst-abc"
    assert retrier.dynamic_estimate_key is None
    assert retrier._run_inflight_key is not None


async def test_dynamic_estimate_key_uses_run_id():
    """Dynamic estimate key is built from run_id."""
    retrier = _make_retrier(run_id="run-99", is_queued=True)
    assert retrier.dynamic_estimate_key == f"{TOKEN_KEY}:dynamic_estimate:run-99"


# ── Straggler detection ──────────────────────────────────────────────


async def test_straggler_detected_in_pre_function(redis):
    """Queued retrier gets MAX_PRIORITY when another run is queue head."""
    await _init_tokens(redis, value=1000)

    queue_key = f"{KEY_PREFIX}:provider:model:benchmark:queue"
    await redis.rpush(queue_key, "run-2")  # another run is at head

    retrier = _make_retrier(run_id="run-1", question_id="q1", is_queued=True)
    await retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

    assert retrier.priority == -5  # MAX_PRIORITY


async def test_not_straggler_when_queue_head(redis):
    """Queued retrier keeps normal priority when it's the queue head."""
    await _init_tokens(redis, value=1000)

    queue_key = f"{KEY_PREFIX}:provider:model:benchmark:queue"
    await redis.rpush(queue_key, "run-1")  # our run is at head

    retrier = _make_retrier(run_id="run-1", question_id="q1", is_queued=True)
    await retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

    assert retrier.priority == 0  # INITIAL_PRIORITY


async def test_not_straggler_when_not_queued(redis, token_retrier: TokenRetrier):
    """Non-queued requests skip straggler check, keep INITIAL_PRIORITY."""
    await _init_tokens(redis, value=1000)

    assert token_retrier._is_queued is False
    await token_retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

    assert token_retrier.priority == 0


# ── Per-question metadata ────────────────────────────────────────────


async def test_metadata_stores_run_id(redis):
    """Per-question metadata hash stores run_id field."""
    await _init_tokens(redis, value=1000)

    retrier = _make_retrier(run_id="run-77", question_id="q-meta", is_queued=True)
    await retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

    meta = await redis.hgetall(f"{TOKEN_KEY}:inflight:q-meta")
    assert meta["run_id"] == "run-77"


async def test_metadata_hash_has_ttl(redis):
    """Per-question metadata hash gets a TTL after deduction."""
    await _init_tokens(redis, value=1000)

    retrier = _make_retrier(question_id="q-ttl")
    await retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

    ttl = await redis.ttl(f"{TOKEN_KEY}:inflight:q-ttl")
    assert ttl > 0  # has a TTL


# ── Per-run dispatched counter ────────────────────────────────────────


async def test_dispatched_counter_incremented(redis):
    """Per-run dispatched counter tracks question_ids via SET for queued runs."""
    await _init_tokens(redis, value=1000)

    retrier = _make_retrier(run_id="run-dispatch", question_id="q-d1", is_queued=True)
    await retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

    counter_key = f"{KEY_PREFIX}:{CLIENT_KEY[0]}:{CLIENT_KEY[1]}:benchmark:run:run-dispatch:dispatched"
    count = await redis.scard(counter_key)
    assert count == 1


async def test_dispatched_counter_skipped_for_non_queued(redis, token_retrier: TokenRetrier):
    """Per-run dispatched counter is NOT incremented for non-queued runs."""
    await _init_tokens(redis, value=1000)

    await token_retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

    # no benchmark:run:*:dispatched keys should exist
    keys = await redis.keys(f"{KEY_PREFIX}:*:benchmark:run:*:dispatched")
    assert keys == []


# ── question_id dispatch tracking ─────────────────────────────────────


async def test_question_id_in_dispatched_set(redis):
    """question_id is used as the member in the dispatched SET."""
    await _init_tokens(redis, value=1000)

    retrier = _make_retrier(run_id="run-qid", question_id="question-1", is_queued=True)
    await retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

    dispatched_key = f"{KEY_PREFIX}:{CLIENT_KEY[0]}:{CLIENT_KEY[1]}:benchmark:run:run-qid:dispatched"
    assert await redis.sismember(dispatched_key, "question-1")


async def test_question_id_idempotent_across_turns(redis):
    """Multiple queries with the same question_id only count as 1 in dispatched SET."""
    await _init_tokens(redis, value=10000)

    for _ in range(5):
        retrier = _make_retrier(
            run_id="run-qid-idemp",
            question_id="question-1",
            is_queued=True,
        )
        await retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

    dispatched_key = f"{KEY_PREFIX}:{CLIENT_KEY[0]}:{CLIENT_KEY[1]}:benchmark:run:run-qid-idemp:dispatched"
    count = await redis.scard(dispatched_key)
    assert count == 1  # all 5 turns share question_id


async def test_question_id_in_inflight_zset(redis):
    """question_id is used as the member in the per-run inflight ZSET."""
    await _init_tokens(redis, value=1000)

    retrier = _make_retrier(run_id="run-inf", question_id="q-check")
    await retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

    inflight_key = f"{TOKEN_KEY}:run:run-inf:inflight"
    members = await redis.zrangebyscore(inflight_key, "-inf", "+inf")
    assert "q-check" in members


# ── Dynamic estimate ratio ────────────────────────────────────────────


async def test_dynamic_estimate_ratio_written_to_run_key(redis):
    """After execute, the EMA ratio is stored under the run_id key."""
    await _init_tokens(redis, value=1000, limit=2000)

    retrier = _make_retrier(run_id="run-ema", question_id="q-ema", is_queued=True)

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


async def test_dynamic_estimate_has_ttl(redis):
    """Dynamic estimate key gets a TTL after ratio update."""
    await _init_tokens(redis, value=1000, limit=2000)

    retrier = _make_retrier(run_id="run-ttl", question_id="q-ttl")

    mock_qr = MagicMock()
    mock_qr.metadata.total_input_tokens = 100
    mock_qr.metadata.total_output_tokens = 50
    mock_qr.metadata.cache_read_tokens = 0
    mock_qr.metadata.extra = {}

    await retrier.execute(AsyncMock(return_value=(mock_qr, 0.5)))

    de_key = f"{TOKEN_KEY}:dynamic_estimate:run-ttl"
    ttl = await redis.ttl(de_key)
    assert ttl > 0  # has a TTL


# ── Execute cleanup ──────────────────────────────────────────────────


async def test_execute_cleans_up_inflight(redis):
    """execute finally removes question from inflight ZSET and deletes metadata."""
    await _init_tokens(redis, value=1000, limit=2000)

    retrier = _make_retrier(run_id="run-cleanup", question_id="q-cleanup")

    mock_qr = MagicMock()
    mock_qr.metadata.total_input_tokens = 100
    mock_qr.metadata.total_output_tokens = 50
    mock_qr.metadata.cache_read_tokens = 0
    mock_qr.metadata.extra = {}

    await retrier.execute(AsyncMock(return_value=(mock_qr, 0.5)))

    # inflight entry removed
    inflight_key = f"{TOKEN_KEY}:run:run-cleanup:inflight"
    assert await redis.zcard(inflight_key) == 0

    # metadata hash deleted
    meta_key = f"{TOKEN_KEY}:inflight:q-cleanup"
    assert not await redis.exists(meta_key)

    # active_runs cleaned (was last inflight)
    assert not await redis.sismember(f"{TOKEN_KEY}:active_runs", "run-cleanup")


async def test_execute_keeps_active_runs_when_others_inflight(redis):
    """execute finally does NOT remove run from active_runs if other questions are inflight."""
    await _init_tokens(redis, value=10000, limit=10000)

    # simulate another question already inflight for same run
    inflight_key = f"{TOKEN_KEY}:run:run-shared:inflight"
    await redis.zadd(inflight_key, {"other-question": time.time()})
    await redis.sadd(f"{TOKEN_KEY}:active_runs", "run-shared")

    retrier = _make_retrier(run_id="run-shared", question_id="q-mine")

    mock_qr = MagicMock()
    mock_qr.metadata.total_input_tokens = 100
    mock_qr.metadata.total_output_tokens = 50
    mock_qr.metadata.cache_read_tokens = 0
    mock_qr.metadata.extra = {}

    await retrier.execute(AsyncMock(return_value=(mock_qr, 0.5)))

    # our question removed from inflight
    members = await redis.zrangebyscore(inflight_key, "-inf", "+inf")
    assert "q-mine" not in members
    assert "other-question" in members

    # active_runs still contains run (other question still inflight)
    assert await redis.sismember(f"{TOKEN_KEY}:active_runs", "run-shared")


async def test_execute_cleans_up_on_exception(redis):
    """execute finally cleans up even when func raises a non-retriable exception."""
    await _init_tokens(redis, value=1000, limit=2000)

    retrier = _make_retrier(run_id="run-err", question_id="q-err")

    with pytest.raises(ValueError, match="boom"):
        await retrier.execute(AsyncMock(side_effect=ValueError("boom")))

    # inflight entry removed
    assert await redis.zcard(f"{TOKEN_KEY}:run:run-err:inflight") == 0
    # metadata deleted
    assert not await redis.exists(f"{TOKEN_KEY}:inflight:q-err")
    # active_runs cleaned
    assert not await redis.sismember(f"{TOKEN_KEY}:active_runs", "run-err")


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
        mock_time_mod.monotonic = lambda: time_val[0]

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
        mock_time_mod.monotonic = lambda: time_val[0]

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
        # must exceed per-iteration refill: floor(16 * 301) = 4816
        await redis.decrby(token_key, 5000)
        # after 3 iterations, change version to stop the loop
        if iteration_count[0] >= 3:
            await redis.set(f"{token_key}:version", "force-exit")

    mock_asyncio_sleep.side_effect = advance_time_and_deduct

    with patch("model_library.retriers.token.token.time") as mock_time_mod:
        mock_time_mod.time = lambda: time_val[0]
        mock_time_mod.monotonic = lambda: time_val[0]

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


async def test_refill_loop_continues_when_active_runs(redis, mock_asyncio_sleep):
    """Refill loop stays alive when tokens are full but runs have active requests."""
    from model_library.retriers.token.token import refill_tasks

    key_tuple = ("inflight", "refill")
    token_key = f"{KEY_PREFIX}:inflight:refill:tokens"
    active_runs_key = f"{token_key}:active_runs"
    iteration_count = [0]
    time_val = [0.0]

    run_inflight_key = f"{token_key}:run:some-run:inflight"

    async def advance_time(*args):
        time_val[0] += 301.0
        iteration_count[0] += 1
        # after 3 iterations, remove active run so loop can exit
        if iteration_count[0] >= 3:
            await redis.zrem(run_inflight_key, "long-running-req")
            # reaper will srem from active_runs when it sees empty ZSET

    mock_asyncio_sleep.side_effect = advance_time

    with patch("model_library.retriers.token.token.time") as mock_time_mod:
        mock_time_mod.time = lambda: time_val[0]
        mock_time_mod.monotonic = lambda: time_val[0]

        await TokenRetrier.init_remaining_tokens(
            client_registry_key=key_tuple,
            limit=1000,
            limit_refresh_seconds=60,
            logger=logging.getLogger("test"),
            get_rate_limit_func=AsyncMock(return_value=None),
        )

        # simulate an active run with an inflight request
        await redis.sadd(active_runs_key, "some-run")
        await redis.zadd(run_inflight_key, {"long-running-req": time_val[0]})

        _, refill_task = refill_tasks[f"refill:{token_key}"]
        await asyncio.wait_for(refill_task, timeout=1.0)

    # loop survived past the shutdown threshold because of active run
    assert iteration_count[0] >= 3


# ── Refill drift compensation ─────────────────────────────────────────


async def test_refill_compensates_for_loop_drift(redis, mock_asyncio_sleep):
    """Refill loop scales amount by elapsed time when loop takes longer than 1s."""
    from model_library.retriers.token.token import refill_tasks

    key_tuple = ("drift", "refill")
    token_key = f"{KEY_PREFIX}:drift:refill:tokens"
    mono_val = [0.0]
    time_val = [0.0]
    iteration_count = [0]

    async def advance_time_and_drain(*args):
        # each iteration takes 2s instead of 1s (simulates slow Redis)
        mono_val[0] += 2.0
        time_val[0] += 2.0
        iteration_count[0] += 1
        if iteration_count[0] == 1:
            # drain tokens during sleep so refill starts from 0
            # only on first call — correction loop also triggers asyncio.sleep
            await redis.set(token_key, "0")
        await redis.set(f"{token_key}:version", "force-exit")

    mock_asyncio_sleep.side_effect = advance_time_and_drain

    with patch("model_library.retriers.token.token.time") as mock_time_mod:
        mock_time_mod.time = lambda: time_val[0]
        mock_time_mod.monotonic = lambda: mono_val[0]

        await TokenRetrier.init_remaining_tokens(
            client_registry_key=key_tuple,
            limit=1000,
            limit_refresh_seconds=60,
            logger=logging.getLogger("test"),
            get_rate_limit_func=AsyncMock(return_value=None),
        )

        _, refill_task = refill_tasks[f"refill:{token_key}"]
        await asyncio.wait_for(refill_task, timeout=1.0)

    # tokens_per_second = floor(1000/60) = 16
    # tokens drained to 0 during sleep, elapsed 2s → refill_amount = floor(16 * 2) = 32
    remaining = int(await redis.get(token_key))
    assert remaining == 32


async def test_refill_clamps_on_clock_rollback(redis, mock_asyncio_sleep):
    """Refill amount is clamped to 0 when monotonic clock appears to go backward."""
    from model_library.retriers.token.token import refill_tasks

    key_tuple = ("rollback", "refill")
    token_key = f"{KEY_PREFIX}:rollback:refill:tokens"
    mono_val = [100.0]
    time_val = [100.0]
    iteration_count = [0]

    async def rollback_then_exit(*args):
        iteration_count[0] += 1
        if iteration_count[0] == 1:
            # simulate clock going backward (shouldn't happen with monotonic, but clamp protects)
            mono_val[0] -= 5.0
            time_val[0] -= 5.0
        else:
            await redis.set(f"{token_key}:version", "force-exit")

    mock_asyncio_sleep.side_effect = rollback_then_exit

    with patch("model_library.retriers.token.token.time") as mock_time_mod:
        mock_time_mod.time = lambda: time_val[0]
        mock_time_mod.monotonic = lambda: mono_val[0]

        await TokenRetrier.init_remaining_tokens(
            client_registry_key=key_tuple,
            limit=1000,
            limit_refresh_seconds=60,
            logger=logging.getLogger("test"),
            get_rate_limit_func=AsyncMock(return_value=None),
        )

        _, refill_task = refill_tasks[f"refill:{token_key}"]
        await asyncio.wait_for(refill_task, timeout=1.0)

    # init sets tokens to 1000 (limit). With clamp, negative elapsed → refill 0 → stays at 1000.
    # Without clamp (the bug), refill would be -80 → tokens would drop below 1000.
    remaining = int(await redis.get(token_key))
    assert remaining == 1000
