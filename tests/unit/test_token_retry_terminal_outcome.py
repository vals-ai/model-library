import logging
from importlib import import_module
from unittest.mock import AsyncMock, patch

import pytest

from model_library.exceptions import NoRetryException
from model_library.retriers.token import set_redis_client
from model_library.retriers.token import token as token_module
from model_library.retriers.token.utils import KEY_PREFIX

fakeredis = import_module("fakeredis")

CLIENT_KEY = ("provider", "model")
TOKEN_KEY = f"{KEY_PREFIX}:provider:model:tokens"
BURST_KEY = f"{TOKEN_KEY}:burst"
PRIORITY_KEY = f"{KEY_PREFIX}:provider:model:priority:0"
REQUEST_KEY = f"{KEY_PREFIX}:provider:model:requests"
CONFIG_KEY = f"{TOKEN_KEY}:config"


@pytest.fixture
def redis():
    client = fakeredis.FakeAsyncRedis(decode_responses=True)
    set_redis_client(client)
    return client


def make_retrier(run_id: str, question_id: str = "question"):
    return token_module.TokenRetrier(
        logger=logging.getLogger("test"),
        client_registry_key=CLIENT_KEY,
        run_id=run_id,
        question_id=question_id,
        estimate_input_tokens=100,
        estimate_output_tokens=50,
        use_dynamic_estimate=False,
    )


async def initialize_tokens(redis) -> None:
    await redis.set(TOKEN_KEY, "1000")
    await redis.set(f"{TOKEN_KEY}:limit", "1000")
    await redis.hset(f"{TOKEN_KEY}:config", mapping={"burst_limit": "800"})


@pytest.mark.parametrize("outcome", ["cancelled", "failed"])
async def test_terminal_outcome_atomically_prevents_token_deduction(
    redis, outcome: str
):
    await initialize_tokens(redis)
    run_meta_key = f"{KEY_PREFIX}:provider:model:benchmark:run:terminal"
    await redis.hset(run_meta_key, mapping={"outcome": outcome})

    result = await redis.eval(
        token_module.ADMIT_REQUEST_LUA,
        5,
        TOKEN_KEY,
        BURST_KEY,
        REQUEST_KEY,
        CONFIG_KEY,
        run_meta_key,
        600,
        800,
        "terminal-member",
        token_module.REQUEST_WINDOW_MILLISECONDS,
        token_module.REQUEST_LOG_TTL_MILLISECONDS,
    )

    assert result == [0, 4, 0, 0]
    assert await redis.get(TOKEN_KEY) == "1000"
    assert not await redis.exists(BURST_KEY)


@pytest.mark.parametrize("outcome", ["cancelled", "failed"])
@pytest.mark.parametrize("requests_per_minute", [None, "1"])
async def test_terminal_waiter_exits_without_retry_and_cleans_metadata(
    redis,
    outcome: str,
    requests_per_minute: str | None,
):
    await initialize_tokens(redis)
    if requests_per_minute is not None:
        await redis.hset(CONFIG_KEY, "requests_per_minute", requests_per_minute)
    run_id = f"run-{outcome}"
    question_id = "terminal"
    run_meta_key = f"{KEY_PREFIX}:provider:model:benchmark:run:{run_id}"
    question_meta_key = f"{TOKEN_KEY}:inflight:{run_id}:{question_id}"
    await redis.hset(run_meta_key, mapping={"outcome": outcome})
    retrier = make_retrier(run_id, question_id)
    provider = AsyncMock()

    with pytest.raises(NoRetryException) as exc_info:
        await retrier.execute(provider)

    provider.assert_not_awaited()
    assert type(exc_info.value).__name__ == "BenchmarkRunTerminated"
    assert await redis.get(TOKEN_KEY) == "1000"
    assert not await redis.exists(BURST_KEY)
    assert await redis.zcard(REQUEST_KEY) == 0
    assert await redis.zscore(PRIORITY_KEY, f"{run_id}:{question_id}") is None
    assert not await redis.exists(question_meta_key)


@pytest.mark.parametrize("outcome", [None, "completed"])
async def test_nonterminal_outcome_preserves_token_deduction(
    redis, outcome: str | None
):
    await initialize_tokens(redis)
    run_id = "run-normal"
    run_meta_key = f"{KEY_PREFIX}:provider:model:benchmark:run:{run_id}"
    if outcome is not None:
        await redis.hset(run_meta_key, mapping={"outcome": outcome})
    retrier = make_retrier(run_id)

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await retrier._pre_function()  # pyright: ignore[reportPrivateUsage]

    assert await redis.get(TOKEN_KEY) == "850"
