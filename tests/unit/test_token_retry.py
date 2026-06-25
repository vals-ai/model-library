"""
Unit tests for TokenRetrier logic.
"""

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis.aioredis
import pytest

from model_library.base.output import QueryResult, QueryResultMetadata
from model_library.exceptions import ImmediateRetryException, RetryException
from model_library.retriers.base import BaseRetrier
from model_library.retriers.token import TokenRetrier, set_redis_client
from model_library.retriers.token import token as token_module
from model_library.retriers.token.utils import KEY_PREFIX, get_status

CLIENT_KEY = ("provider", "model")
TOKEN_KEY = f"{KEY_PREFIX}:provider:model:tokens"
PRIORITY_KEY_PREFIX = f"{KEY_PREFIX}:provider:model:priority"


@pytest.fixture(autouse=True)
def mock_asyncio_sleep():
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        yield mock_sleep


@pytest.fixture(autouse=True)
def mock_background_loops():
    with patch(
        "model_library.retriers.token.background.background_loops",
        new_callable=AsyncMock,
    ):
        yield


@pytest.fixture(autouse=True)
def clear_background_loop_registry():
    token_module._BACKGROUND_LOOP_TASKS.clear()  # pyright: ignore[reportPrivateUsage]
    yield
    token_module._BACKGROUND_LOOP_TASKS.clear()  # pyright: ignore[reportPrivateUsage]


class _FakeTask:
    def __init__(self) -> None:
        self.callbacks: list[Callable[["_FakeTask"], None]] = []
        self._done = False

    def done(self) -> bool:
        return self._done

    def add_done_callback(self, callback: Callable[["_FakeTask"], None]) -> None:
        self.callbacks.append(callback)

    def finish(self) -> None:
        self._done = True
        for callback in self.callbacks:
            callback(self)


def _fake_create_task_factory(
    tasks: list[_FakeTask],
) -> Callable[[Coroutine[Any, Any, object]], _FakeTask]:
    task_iter = iter(tasks)

    def _fake_create_task(coro: Coroutine[Any, Any, object]) -> _FakeTask:
        coro.close()
        return next(task_iter)

    return _fake_create_task


class _FakeLock:
    """No-op async context manager replacing redis Lock (fakeredis lacks evalsha)."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


def _close_created_task(coro):
    """Consume a coroutine passed to mocked create_task."""
    coro.close()


@pytest.fixture
def redis():
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    client.lock = lambda *args, **kwargs: _FakeLock()  # pyright: ignore[reportAttributeAccessIssue]
    set_redis_client(client)
    return client


def _make_retrier(
    *,
    run_id: str = "test-instance",
    question_id: str = "test-qid",
    estimate_input_tokens: int = 100,
    estimate_output_tokens: int = 50,
    use_dynamic_estimate: bool = True,
) -> TokenRetrier:
    """Helper to create a TokenRetrier with sensible defaults."""
    return TokenRetrier(
        logger=logging.getLogger("test"),
        client_registry_key=CLIENT_KEY,
        run_id=run_id,
        question_id=question_id,
        estimate_input_tokens=estimate_input_tokens,
        estimate_output_tokens=estimate_output_tokens,
        use_dynamic_estimate=use_dynamic_estimate,
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

    with patch(
        "asyncio.create_task", side_effect=_fake_create_task_factory([_FakeTask()])
    ) as mock_create_task:
        await TokenRetrier.init_remaining_tokens(
            client_registry_key=key_tuple,
            limit=3000,
            limit_refresh_seconds=60,
            logger=logging.getLogger("test"),
            get_rate_limit_func=AsyncMock(),
        )

    assert await redis.get(f"{KEY_PREFIX}:p:m:tokens") == "3000"
    assert await redis.get(f"{KEY_PREFIX}:p:m:tokens:limit") == "3000"
    assert mock_create_task.call_count == 1


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

    with patch("asyncio.sleep", new_callable=AsyncMock, side_effect=simulate_refill):
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
    now = time.time()
    await redis.zadd(f"{PRIORITY_KEY_PREFIX}:1", {"r1": now, "r2": now, "r3": now})
    await redis.zadd(f"{PRIORITY_KEY_PREFIX}:3", {"r4": now})
    for request_id in ["r1", "r2", "r3", "r4"]:
        await redis.hset(
            f"{TOKEN_KEY}:inflight:{request_id}",
            mapping={"run_id": f"run-{request_id}"},
        )

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
    assert keys == {
        f"{KEY_PREFIX}:anthropic:key2:tokens",
        f"{KEY_PREFIX}:openai:key1:tokens",
    }


# ── Constructor & identity ───────────────────────────────────────────


async def test_initial_priority_is_zero():
    """New requests start at INITIAL_PRIORITY (0)."""
    retrier = _make_retrier()
    assert retrier.priority == 0


async def test_run_id_stored():
    """run_id is stored as _run_id."""
    retrier = _make_retrier(run_id="run-42")
    assert retrier._run_id == "run-42"
    assert retrier._is_queued is None  # lazy: not yet detected


async def test_non_queued_defaults():
    """Non-queued retrier uses provided run_id; _is_queued starts unset."""
    retrier = _make_retrier(run_id="inst-abc")
    assert retrier._run_id == "inst-abc"
    assert retrier._is_queued is None  # lazy: not yet detected


async def test_dynamic_estimate_disabled():
    """use_dynamic_estimate=False disables dynamic estimate key but keeps run tracking."""
    retrier = _make_retrier(run_id="inst-abc", use_dynamic_estimate=False)
    assert retrier._run_id == "inst-abc"
    assert retrier.dynamic_estimate_key is None
    assert retrier._run_inflight_key is not None


async def test_dynamic_estimate_key_uses_run_id():
    """Dynamic estimate key is built from run_id."""
    retrier = _make_retrier(run_id="run-99")
    assert retrier.dynamic_estimate_key == f"{TOKEN_KEY}:dynamic_estimate:run-99"


# ── Straggler detection ──────────────────────────────────────────────


async def test_straggler_detected_in_pre_function(redis):
    """Queued retrier gets MAX_PRIORITY when its run is not an active head."""
    await _init_tokens(redis, value=1000)

    queue_key = f"{KEY_PREFIX}:provider:model:benchmark:queue"
    await redis.rpush(queue_key, "run-2")
    await redis.rpush(queue_key, "run-1")
    await redis.zadd(
        f"{KEY_PREFIX}:provider:model:benchmark:active_heads", {"run-2": 1}
    )

    retrier = _make_retrier(run_id="run-1", question_id="q1")
    await retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

    assert retrier.priority == -5  # MAX_PRIORITY


async def test_not_straggler_when_active_head(redis):
    """Queued retrier keeps normal priority when its run is an active head."""
    await _init_tokens(redis, value=1000)

    queue_key = f"{KEY_PREFIX}:provider:model:benchmark:queue"
    active_heads_key = f"{KEY_PREFIX}:provider:model:benchmark:active_heads"
    await redis.rpush(queue_key, "run-2", "run-1")
    await redis.zadd(active_heads_key, {"run-2": 1, "run-1": 2})

    retrier = _make_retrier(run_id="run-1", question_id="q1")
    await retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

    assert retrier.priority == 0  # INITIAL_PRIORITY


async def test_not_straggler_when_not_queued(redis, token_retrier: TokenRetrier):
    """Non-queued requests skip straggler check, keep INITIAL_PRIORITY."""
    await _init_tokens(redis, value=1000)

    assert token_retrier._is_queued is None
    await token_retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

    assert token_retrier.priority == 0


# ── Per-question metadata ────────────────────────────────────────────


async def test_metadata_stores_run_id(redis):
    """Per-question metadata hash stores run_id field."""
    await _init_tokens(redis, value=1000)

    retrier = _make_retrier(run_id="run-77", question_id="q-meta")
    await retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

    meta = await redis.hgetall(f"{TOKEN_KEY}:inflight:run-77:q-meta")
    assert meta["run_id"] == "run-77"


async def test_metadata_hash_has_ttl(redis):
    """Per-question metadata hash gets a TTL after deduction."""
    await _init_tokens(redis, value=1000)

    retrier = _make_retrier(question_id="q-ttl")
    await retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

    ttl = await redis.ttl(f"{TOKEN_KEY}:inflight:test-instance:q-ttl")
    assert ttl > 0  # has a TTL


async def test_waiting_metadata_hash_has_ttl(redis):
    """Queued metadata hash gets a TTL even before token deduction succeeds."""
    await _init_tokens(redis, value=0)
    retrier = _make_retrier(question_id="q-waiting")
    meta_key = f"{TOKEN_KEY}:inflight:test-instance:q-waiting"
    sleeping = asyncio.Event()
    blocker = asyncio.Event()

    async def block_sleep(*_args, **_kwargs):
        sleeping.set()
        await blocker.wait()

    with patch("asyncio.sleep", new_callable=AsyncMock, side_effect=block_sleep):
        task = asyncio.create_task(retrier._pre_function())  # pyright: ignore[reportPrivateUsage]
        await asyncio.wait_for(sleeping.wait(), timeout=1.0)

        assert await redis.ttl(meta_key) > 0

        task.cancel()
        blocker.set()
        with pytest.raises(asyncio.CancelledError):
            await task


# ── Per-run dispatched counter ────────────────────────────────────────


async def test_dispatched_counter_incremented(redis):
    """Per-run dispatched counter tracks question_ids via SET for queued runs."""
    await _init_tokens(redis, value=1000)

    queue_key = f"{KEY_PREFIX}:provider:model:benchmark:queue"
    await redis.rpush(queue_key, "run-dispatch")

    retrier = _make_retrier(run_id="run-dispatch", question_id="q-d1")
    await retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

    counter_key = f"{KEY_PREFIX}:{CLIENT_KEY[0]}:{CLIENT_KEY[1]}:benchmark:run:run-dispatch:dispatched"
    count = await redis.scard(counter_key)
    assert count == 1


async def test_dispatched_counter_incremented_for_non_queued(
    redis, token_retrier: TokenRetrier
):
    """Per-run dispatched counter is incremented for all runs, including non-queued."""
    await _init_tokens(redis, value=1000)

    await token_retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

    keys = await redis.keys(f"{KEY_PREFIX}:*:benchmark:run:*:dispatched")
    assert len(keys) == 1


# ── question_id dispatch tracking ─────────────────────────────────────


async def test_question_id_in_dispatched_set(redis):
    """question_id is used as the member in the dispatched SET."""
    await _init_tokens(redis, value=1000)

    queue_key = f"{KEY_PREFIX}:provider:model:benchmark:queue"
    await redis.rpush(queue_key, "run-qid")

    retrier = _make_retrier(run_id="run-qid", question_id="question-1")
    await retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

    dispatched_key = (
        f"{KEY_PREFIX}:{CLIENT_KEY[0]}:{CLIENT_KEY[1]}:benchmark:run:run-qid:dispatched"
    )
    assert await redis.sismember(dispatched_key, "run-qid:question-1")


async def test_question_id_idempotent_across_turns(redis):
    """Multiple queries with the same question_id only count as 1 in dispatched SET."""
    await _init_tokens(redis, value=10000)

    queue_key = f"{KEY_PREFIX}:provider:model:benchmark:queue"
    await redis.rpush(queue_key, "run-qid-idemp")

    for _ in range(5):
        retrier = _make_retrier(
            run_id="run-qid-idemp",
            question_id="question-1",
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
    assert "run-inf:q-check" in members


# ── Dynamic estimate ratio ────────────────────────────────────────────


async def test_dynamic_estimate_ratio_written_to_run_key(redis):
    """After execute, the EMA ratio is stored under the run_id key."""
    await _init_tokens(redis, value=1000, limit=2000)

    retrier = _make_retrier(run_id="run-ema", question_id="q-ema")

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
    meta_key = f"{TOKEN_KEY}:inflight:run-cleanup:q-cleanup"
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
    assert "run-shared:q-mine" not in members
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
    assert not await redis.exists(f"{TOKEN_KEY}:inflight:run-err:q-err")
    # active_runs cleaned
    assert not await redis.sismember(f"{TOKEN_KEY}:active_runs", "run-err")


# ── Config-change triggers immediate takeover ───────────────────────


async def test_init_with_changed_config_starts_active(redis):
    """When config changes between init calls, background_loops starts with standby=False."""
    key_tuple = ("p", "m")
    key = f"{KEY_PREFIX}:p:m:tokens"
    fake_tasks = [_FakeTask(), _FakeTask()]

    with (
        patch(
            "model_library.retriers.token.background.background_loops",
            new_callable=AsyncMock,
        ) as mock_bg,
        patch(
            "asyncio.create_task", side_effect=_fake_create_task_factory(fake_tasks)
        ) as mock_create_task,
    ):
        # first init: establishes config in redis
        await TokenRetrier.init_remaining_tokens(
            client_registry_key=key_tuple,
            limit=3000,
            limit_refresh_seconds=60,
            logger=logging.getLogger("test"),
            get_rate_limit_func=AsyncMock(),
        )

        # simulate an active loop and stale config (old limit=3000) in redis
        await redis.set(f"{key}:task:active", "some-task-id")
        await redis.hset(
            f"{key}:config",
            mapping={"limit": "3000", "tokens_per_second": "50", "burst_limit": "2400"},
        )

        # second init with a DIFFERENT limit
        await TokenRetrier.init_remaining_tokens(
            client_registry_key=key_tuple,
            limit=5000,
            limit_refresh_seconds=60,
            logger=logging.getLogger("test"),
            get_rate_limit_func=AsyncMock(),
        )

    # the second create_task call should have standby=False (config changed)
    assert mock_create_task.call_count == 2
    # background_loops was awaited via create_task; check the mock call args
    assert mock_bg.call_count == 2
    _, second_kwargs = mock_bg.call_args_list[1]
    assert second_kwargs["standby"] is False


async def test_init_with_same_config_reuses_existing_background_loop(redis):
    """When config is unchanged between init calls, reuse the in-process background loop."""
    key_tuple = ("p", "m")
    key = f"{KEY_PREFIX}:p:m:tokens"
    fake_task = _FakeTask()

    with (
        patch(
            "model_library.retriers.token.background.background_loops",
            new_callable=AsyncMock,
        ) as mock_bg,
        patch(
            "asyncio.create_task", side_effect=_fake_create_task_factory([fake_task])
        ) as mock_create_task,
    ):
        # first init
        await TokenRetrier.init_remaining_tokens(
            client_registry_key=key_tuple,
            limit=3000,
            limit_refresh_seconds=60,
            logger=logging.getLogger("test"),
            get_rate_limit_func=AsyncMock(),
        )

        # simulate an active loop with the SAME config
        await redis.set(f"{key}:task:active", "some-task-id")
        await redis.hset(
            f"{key}:config",
            mapping={"limit": "3000", "tokens_per_second": "50", "burst_limit": "2400"},
        )

        # second init with SAME limit
        await TokenRetrier.init_remaining_tokens(
            client_registry_key=key_tuple,
            limit=3000,
            limit_refresh_seconds=60,
            logger=logging.getLogger("test"),
            get_rate_limit_func=AsyncMock(),
        )

    assert mock_create_task.call_count == 1
    assert mock_bg.call_count == 1


async def test_init_starts_new_background_loop_after_existing_task_finishes(redis):
    """A completed registered background task does not block a later init."""
    key_tuple = ("p", "m")
    fake_tasks = [_FakeTask(), _FakeTask()]

    with (
        patch(
            "model_library.retriers.token.background.background_loops",
            new_callable=AsyncMock,
        ) as mock_bg,
        patch(
            "asyncio.create_task", side_effect=_fake_create_task_factory(fake_tasks)
        ) as mock_create_task,
    ):
        await TokenRetrier.init_remaining_tokens(
            client_registry_key=key_tuple,
            limit=3000,
            limit_refresh_seconds=60,
            logger=logging.getLogger("test"),
            get_rate_limit_func=AsyncMock(),
        )

        fake_tasks[0].finish()

        await TokenRetrier.init_remaining_tokens(
            client_registry_key=key_tuple,
            limit=3000,
            limit_refresh_seconds=60,
            logger=logging.getLogger("test"),
            get_rate_limit_func=AsyncMock(),
        )

    assert mock_create_task.call_count == 2
    assert mock_bg.call_count == 2


BURST_KEY = f"{TOKEN_KEY}:burst"
CONFIG_KEY = f"{TOKEN_KEY}:config"


# ── Burst limit ─────────────────────────────────────────────────────


async def test_burst_blocks_when_exceeded(redis):
    """Deduction fails when per-second burst usage + required > burst_limit."""
    from model_library.retriers.token.token import DEDUCT_TOKENS_LUA

    await _init_tokens(redis, value=1000, limit=1000)
    burst_limit = 200

    # first deduction: 150 tokens, within limit
    result = await redis.eval(
        DEDUCT_TOKENS_LUA, 2, TOKEN_KEY, BURST_KEY, 150, burst_limit
    )  # noqa: S307
    assert result == 1
    assert int(await redis.get(TOKEN_KEY)) == 850

    # second deduction: 150 more would put burst at 300 > 200
    result = await redis.eval(
        DEDUCT_TOKENS_LUA, 2, TOKEN_KEY, BURST_KEY, 150, burst_limit
    )  # noqa: S307
    assert result == 0
    assert int(await redis.get(TOKEN_KEY)) == 850  # unchanged


async def test_burst_allows_within_cap(redis):
    """Deduction succeeds when per-second burst usage + required <= burst_limit."""
    from model_library.retriers.token.token import DEDUCT_TOKENS_LUA

    await _init_tokens(redis, value=1000, limit=1000)
    burst_limit = 200

    result = await redis.eval(
        DEDUCT_TOKENS_LUA, 2, TOKEN_KEY, BURST_KEY, 100, burst_limit
    )  # noqa: S307
    assert result == 1
    result = await redis.eval(
        DEDUCT_TOKENS_LUA, 2, TOKEN_KEY, BURST_KEY, 100, burst_limit
    )  # noqa: S307
    assert result == 1

    assert int(await redis.get(TOKEN_KEY)) == 800
    assert int(await redis.get(BURST_KEY)) == 200


async def test_burst_resets_after_ttl(redis):
    """Burst counter resets after 1-second TTL expires."""
    from model_library.retriers.token.token import DEDUCT_TOKENS_LUA

    await _init_tokens(redis, value=1000, limit=1000)
    burst_limit = 200

    # fill burst
    await redis.eval(DEDUCT_TOKENS_LUA, 2, TOKEN_KEY, BURST_KEY, 200, burst_limit)  # noqa: S307
    assert int(await redis.get(BURST_KEY)) == 200

    # verify TTL was set
    ttl = await redis.pttl(BURST_KEY)
    assert 0 < ttl <= 1000

    # simulate expiry
    await redis.delete(BURST_KEY)

    # fresh window — should pass
    result = await redis.eval(
        DEDUCT_TOKENS_LUA, 2, TOKEN_KEY, BURST_KEY, 150, burst_limit
    )  # noqa: S307
    assert result == 1
    assert int(await redis.get(BURST_KEY)) == 150


async def test_burst_ttl_not_extended(redis):
    """Subsequent deductions don't extend the burst window TTL."""
    from model_library.retriers.token.token import DEDUCT_TOKENS_LUA

    await _init_tokens(redis, value=1000, limit=1000)
    burst_limit = 500

    # first deduction sets TTL
    await redis.eval(DEDUCT_TOKENS_LUA, 2, TOKEN_KEY, BURST_KEY, 100, burst_limit)  # noqa: S307
    ttl1 = await redis.pttl(BURST_KEY)

    # second deduction should not reset TTL (TTL > 0, not -1)
    await redis.eval(DEDUCT_TOKENS_LUA, 2, TOKEN_KEY, BURST_KEY, 100, burst_limit)  # noqa: S307
    ttl2 = await redis.pttl(BURST_KEY)

    assert ttl2 <= ttl1  # TTL only decreases, never extended


async def test_burst_limit_reads_from_config(redis):
    """_burst_limit is lazily read from the config hash on first _pre_function call."""
    await _init_tokens(redis, value=1000, limit=1000)
    await redis.hset(CONFIG_KEY, mapping={"burst_limit": "800"})

    retrier = _make_retrier()
    assert retrier._burst_limit is None  # pyright: ignore[reportPrivateUsage]

    await retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

    assert retrier._burst_limit == 800  # pyright: ignore[reportPrivateUsage]


# ── Dispatched cycling (agentic multi-turn) ─────────────────────────


async def test_dispatched_removed_on_reentry(redis):
    """On re-entry into _pre_function (next agentic turn), question is removed from dispatched."""
    await _init_tokens(redis, value=10000)

    dispatched_key = f"{KEY_PREFIX}:{CLIENT_KEY[0]}:{CLIENT_KEY[1]}:benchmark:run:run-cycle:dispatched"

    # Turn 1: deducts tokens, question added to dispatched
    r1 = _make_retrier(run_id="run-cycle", question_id="q-agent")
    await r1._pre_function()  # pyright: ignore[reportPrivateUsage]
    assert await redis.sismember(dispatched_key, "run-cycle:q-agent")

    # Turn 2: new retrier (same question_id/run_id), simulating next agentic turn
    # srem happens before deduction, so check state mid-flow
    removed_before_deduct = None
    original_eval = redis.eval  # noqa: S307

    async def capture_mid_pre(script, numkeys, *args):
        nonlocal removed_before_deduct
        if numkeys == 2 and removed_before_deduct is None:
            removed_before_deduct = not await redis.sismember(
                dispatched_key, "run-cycle:q-agent"
            )
        return await original_eval(script, numkeys, *args)  # noqa: S307

    redis.eval = capture_mid_pre  # noqa: S307

    r2 = _make_retrier(run_id="run-cycle", question_id="q-agent")
    await r2._pre_function()  # pyright: ignore[reportPrivateUsage]

    assert removed_before_deduct is True
    assert await redis.sismember(dispatched_key, "run-cycle:q-agent")


async def test_dispatched_cycling_across_turns(redis):
    """Simulate 3 agentic turns: dispatched count drops then recovers each turn."""
    await _init_tokens(redis, value=100000)

    dispatched_key = f"{KEY_PREFIX}:{CLIENT_KEY[0]}:{CLIENT_KEY[1]}:benchmark:run:run-multi:dispatched"

    for turn in range(3):
        retrier = _make_retrier(run_id="run-multi", question_id="q-agentic")

        removed_during_turn = None
        original_eval = redis.eval  # noqa: S307

        async def capture_mid_pre(script, numkeys, *args, _orig=original_eval):
            nonlocal removed_during_turn
            if numkeys == 2 and removed_during_turn is None:
                removed_during_turn = not await redis.sismember(
                    dispatched_key, "q-agentic"
                )
            return await _orig(script, numkeys, *args)  # noqa: S307

        redis.eval = capture_mid_pre  # noqa: S307

        await retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

        if turn > 0:
            assert removed_during_turn is True, (
                f"Turn {turn}: question not removed before deduction"
            )

        assert await redis.sismember(dispatched_key, "run-multi:q-agentic"), (
            f"Turn {turn}: question not in dispatched after deduction"
        )
        assert await redis.scard(dispatched_key) == 1

        redis.eval = original_eval  # noqa: S307


# ── Shield cleanup on cancellation ──────────────────────────────────


async def test_pre_function_shield_completes_on_cancel(redis):
    """Cancelling during _pre_function still cleans up priority entry via shield."""
    await _init_tokens(redis, value=1000)
    retrier = _make_retrier()
    priority_key = TokenRetrier.get_priority_key(CLIENT_KEY, 0)

    # run _pre_function in a task, cancel it after it registers priority
    started = asyncio.Event()
    original_zadd = redis.zadd

    async def zadd_then_signal(name, mapping, **kwargs):
        result = await original_zadd(name, mapping, **kwargs)
        if "priority" in name:
            started.set()
        return result

    redis.zadd = zadd_then_signal

    task = asyncio.create_task(retrier._pre_function())  # pyright: ignore[reportPrivateUsage]
    await started.wait()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # shield should have cleaned up priority entry
    await asyncio.sleep(0)
    assert await redis.zcard(priority_key) == 0


async def test_execute_shield_completes_on_cancel(redis):
    """Cancelling during execute still cleans up inflight tracking via shield."""
    await _init_tokens(redis, value=1000)
    retrier = _make_retrier()
    inflight_key = f"{TOKEN_KEY}:run:test-instance:inflight"

    work_done = asyncio.Event()

    async def work_func():
        work_done.set()
        await asyncio.sleep(999)  # hang until cancelled

    task = asyncio.create_task(retrier.execute(work_func))
    await work_done.wait()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # shield should have cleaned up inflight
    await asyncio.sleep(0)
    assert await redis.zcard(inflight_key) == 0
