"""
Unit tests for background_loops in model_library.retriers.token.background.

Tests cover: refill loop, correction loop, reaper loop, heartbeat,
standby -> takeover, watchdog -> demotion, and idle shutdown.
"""

import asyncio
import importlib
import logging
import time as real_time_module
from math import floor
from types import ModuleType
from unittest.mock import AsyncMock, patch

import fakeredis.aioredis
import pytest

bg_module = importlib.import_module("model_library.retriers.token.background")
from model_library.base.output import RateLimit
from model_library.retriers.token.background import (
    FULL_TOKENS_SHUTDOWN,
    LOOP_POLL_INTERVAL,
    REFILL_TASK_TTL,
    LoopConfig,
    background_loops,
)
from model_library.retriers.token.token import (
    INFLIGHT_MAX_AGE,
    MAX_PRIORITY,
    MIN_PRIORITY,
    PRIORITY_STALE_AGE,
    REAP_INTERVAL,
)
from model_library.retriers.token.utils import set_redis_client

logger = logging.getLogger("test_background_loops")

TOKEN_KEY = "model_library:provider:model:tokens"

# save the real asyncio.sleep before any patching -- used inside FakeClock
_real_sleep = asyncio.sleep


# -- Helpers -----------------------------------------------------------------


class FakeClock:
    """Shared fake clock for time.time, time.monotonic, and asyncio.sleep.

    Uses the real asyncio.sleep(0) internally to yield control without recursion.
    """

    def __init__(self, start: float = 1_000_000.0):
        self.now = start

    def time(self) -> float:
        return self.now

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds
        await _real_sleep(0)


class FakeTimeModule(ModuleType):
    """A fake `time` module that delegates time()/monotonic() to a FakeClock
    while forwarding everything else to the real time module.

    Patched onto bg_module.time so only the background module sees fake
    timestamps, leaving fakeredis (which needs real time for TTLs) unaffected.
    """

    def __init__(self, clock: FakeClock):
        super().__init__("time")
        self._clock = clock

    def time(self) -> float:
        return self._clock.time()

    def monotonic(self) -> float:
        return self._clock.monotonic()

    def __getattr__(self, name: str):
        return getattr(real_time_module, name)


class _FakeAsyncioModule(ModuleType):
    """Fake asyncio module that overrides sleep() but delegates everything else."""

    def __init__(self, clock: FakeClock):
        super().__init__("asyncio")
        self._clock = clock

    async def sleep(self, seconds: float) -> None:
        self._clock.now += seconds
        await _real_sleep(0)

    def __getattr__(self, name: str):
        return getattr(asyncio, name)


async def _yield(n: int = 1) -> None:
    """Yield control to other tasks n times using the real sleep."""
    for _ in range(n):
        await _real_sleep(0)


def _cfg(limit: int = 10_000, tps: int = 100) -> LoopConfig:
    return LoopConfig(key=TOKEN_KEY, limit=limit, tokens_per_second=tps)


async def _init_redis(redis, limit: int = 10_000, tps: int = 100) -> None:
    """Set up Redis state matching what init_remaining_tokens would create."""
    await redis.set(TOKEN_KEY, limit)
    await redis.set(f"{TOKEN_KEY}:limit", limit)
    await redis.hset(
        f"{TOKEN_KEY}:config",
        mapping={
            "limit": limit,
            "tokens_per_second": tps,
            "limit_refresh_seconds": limit // tps if tps > 0 else 0,
            "burst_limit": floor(limit * 0.2),
            "initialized_at": 0,
        },
    )


@pytest.fixture
def redis():
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    set_redis_client(client)
    return client


@pytest.fixture
def clock():
    return FakeClock()


def _make_patches(clock: FakeClock, **constant_overrides):
    """Build patch context managers for bg_module.time, bg_module.asyncio,
    and any module-level constants (e.g. LOOP_POLL_INTERVAL, REAP_INTERVAL)."""
    fake_time = FakeTimeModule(clock)
    fake_asyncio = _FakeAsyncioModule(clock)
    patches = [
        patch.object(bg_module, "time", fake_time),
        patch.object(bg_module, "asyncio", fake_asyncio),
    ]
    for k, v in constant_overrides.items():
        patches.append(patch.object(bg_module, k, v))
    return patches


# -- Refill loop -------------------------------------------------------------


async def test_refill_increases_tokens_after_drain(redis, clock):
    """After draining tokens below limit, the refill loop adds tokens back."""
    await _init_redis(redis)
    drained = 5_000
    await redis.set(TOKEN_KEY, drained)
    cfg = _cfg()

    patches = _make_patches(clock, LOOP_POLL_INTERVAL=0.0)
    for p in patches:
        p.start()
    try:
        task = asyncio.create_task(
            background_loops(cfg, AsyncMock(return_value=None), logger)
        )
        await _yield(20)

        tokens = int(await redis.get(TOKEN_KEY))
        assert tokens > drained, f"expected tokens > {drained}, got {tokens}"

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        for p in reversed(patches):
            p.stop()


async def test_refill_caps_at_limit(redis, clock):
    """Refill loop never pushes tokens above the configured limit."""
    limit = 10_000
    await _init_redis(redis, limit=limit)
    await redis.set(TOKEN_KEY, limit)
    cfg = _cfg(limit=limit)

    patches = _make_patches(clock, LOOP_POLL_INTERVAL=0.0)
    for p in patches:
        p.start()
    try:
        task = asyncio.create_task(
            background_loops(cfg, AsyncMock(return_value=None), logger)
        )
        await _yield(30)

        tokens = int(await redis.get(TOKEN_KEY))
        assert tokens == limit

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        for p in reversed(patches):
            p.stop()


async def test_refill_resets_burst_counter(redis, clock):
    """Burst counter is reset to 0 every refill tick."""
    await _init_redis(redis)
    await redis.set(f"{TOKEN_KEY}:burst", 500)
    cfg = _cfg()

    patches = _make_patches(clock, LOOP_POLL_INTERVAL=0.0)
    for p in patches:
        p.start()
    try:
        task = asyncio.create_task(
            background_loops(cfg, AsyncMock(return_value=None), logger)
        )
        await _yield(20)

        burst = int(await redis.get(f"{TOKEN_KEY}:burst"))
        assert burst == 0

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        for p in reversed(patches):
            p.stop()


# -- Correction loop ---------------------------------------------------------


async def test_correction_corrects_down(redis, clock):
    """When headers report fewer tokens than current, correction reduces tokens."""
    limit = 10_000
    await _init_redis(redis, limit=limit, tps=0)
    await redis.set(TOKEN_KEY, limit)
    cfg = _cfg(limit=limit, tps=0)  # no refill so correction effect is visible

    header_remaining = 3_000

    async def get_rate_limit():
        # always return a fresh timestamp so elapsed is ~0
        return RateLimit(
            token_remaining=header_remaining,
            unix_timestamp=clock.now,
            raw={},
        )

    patches = _make_patches(clock, LOOP_POLL_INTERVAL=0.0)
    for p in patches:
        p.start()
    try:
        task = asyncio.create_task(
            background_loops(cfg, get_rate_limit, logger)
        )
        await _yield(80)

        tokens = int(await redis.get(TOKEN_KEY))
        assert tokens <= header_remaining, f"expected tokens <= {header_remaining}, got {tokens}"

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        for p in reversed(patches):
            p.stop()


async def test_correction_does_not_correct_up(redis, clock):
    """Correction only adjusts tokens downward, never upward."""
    limit = 10_000
    await _init_redis(redis, limit=limit, tps=0)
    await redis.set(TOKEN_KEY, 2000)
    cfg = _cfg(limit=limit, tps=0)

    async def get_rate_limit():
        # 8000 > 2000 current -> should NOT correct up
        return RateLimit(
            token_remaining=8000,
            unix_timestamp=clock.now,
            raw={},
        )

    patches = _make_patches(clock, LOOP_POLL_INTERVAL=0.0)
    for p in patches:
        p.start()
    try:
        task = asyncio.create_task(
            background_loops(cfg, get_rate_limit, logger)
        )
        await _yield(80)

        tokens = int(await redis.get(TOKEN_KEY))
        assert tokens == 2000

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        for p in reversed(patches):
            p.stop()


async def test_correction_exits_when_none(redis, clock):
    """Correction loop exits (returns) when get_rate_limit returns None."""
    await _init_redis(redis)
    cfg = _cfg()

    call_count = 0

    async def rate_limit_returns_none():
        nonlocal call_count
        call_count += 1
        return None

    patches = _make_patches(clock, LOOP_POLL_INTERVAL=0.0)
    for p in patches:
        p.start()
    try:
        task = asyncio.create_task(
            background_loops(cfg, rate_limit_returns_none, logger)
        )
        await _yield(80)

        # should have been called once then exited
        assert call_count == 1

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        for p in reversed(patches):
            p.stop()


# -- Reaper loop -------------------------------------------------------------


async def test_reaper_removes_stale_inflight(redis, clock):
    """Reaper removes inflight entries older than INFLIGHT_MAX_AGE."""
    await _init_redis(redis)
    cfg = _cfg()

    run_id = "run-1"
    inflight_key = f"{TOKEN_KEY}:run:{run_id}:inflight"
    active_runs_key = f"{TOKEN_KEY}:active_runs"
    meta_key = f"{TOKEN_KEY}:inflight:q-stale"

    stale_ts = clock.now - INFLIGHT_MAX_AGE - 10
    await redis.zadd(inflight_key, {"q-stale": stale_ts})
    await redis.sadd(active_runs_key, run_id)
    await redis.hset(meta_key, mapping={"run_id": run_id})

    patches = _make_patches(clock, LOOP_POLL_INTERVAL=0.0, REAP_INTERVAL=1)
    for p in patches:
        p.start()
    try:
        task = asyncio.create_task(
            background_loops(cfg, AsyncMock(return_value=None), logger)
        )
        await _yield(40)

        remaining = await redis.zrangebyscore(inflight_key, "-inf", "+inf")
        assert "q-stale" not in remaining

        assert await redis.exists(meta_key) == 0

        # run should be removed from active_runs since its inflight is empty
        assert await redis.scard(active_runs_key) == 0

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        for p in reversed(patches):
            p.stop()


async def test_reaper_keeps_fresh_inflight(redis, clock):
    """Reaper does NOT remove inflight entries that are still fresh."""
    await _init_redis(redis)
    cfg = _cfg()

    run_id = "run-1"
    inflight_key = f"{TOKEN_KEY}:run:{run_id}:inflight"
    active_runs_key = f"{TOKEN_KEY}:active_runs"

    fresh_ts = clock.now - 10  # 10 seconds old, well within INFLIGHT_MAX_AGE
    await redis.zadd(inflight_key, {"q-fresh": fresh_ts})
    await redis.sadd(active_runs_key, run_id)

    patches = _make_patches(clock, LOOP_POLL_INTERVAL=0.0, REAP_INTERVAL=1)
    for p in patches:
        p.start()
    try:
        task = asyncio.create_task(
            background_loops(cfg, AsyncMock(return_value=None), logger)
        )
        await _yield(40)

        remaining = await redis.zrangebyscore(inflight_key, "-inf", "+inf")
        assert "q-fresh" in remaining

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        for p in reversed(patches):
            p.stop()


async def test_reaper_removes_stale_priority(redis, clock):
    """Reaper removes priority entries older than PRIORITY_STALE_AGE."""
    await _init_redis(redis)
    cfg = _cfg()

    base = TOKEN_KEY.removesuffix(":tokens")
    pkey = f"{base}:priority:0"
    stale_ts = clock.now - PRIORITY_STALE_AGE - 10
    await redis.zadd(pkey, {"q-stale-priority": stale_ts})

    patches = _make_patches(clock, LOOP_POLL_INTERVAL=0.0, REAP_INTERVAL=1)
    for p in patches:
        p.start()
    try:
        task = asyncio.create_task(
            background_loops(cfg, AsyncMock(return_value=None), logger)
        )
        await _yield(40)

        remaining = await redis.zrangebyscore(pkey, "-inf", "+inf")
        assert "q-stale-priority" not in remaining

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        for p in reversed(patches):
            p.stop()


async def test_reaper_cleans_all_priority_levels(redis, clock):
    """Reaper iterates MAX_PRIORITY..MIN_PRIORITY and removes stale entries from each."""
    await _init_redis(redis)
    cfg = _cfg()
    base = TOKEN_KEY.removesuffix(":tokens")

    stale_ts = clock.now - PRIORITY_STALE_AGE - 10
    for p in range(MAX_PRIORITY, MIN_PRIORITY + 1):
        pkey = f"{base}:priority:{p}"
        await redis.zadd(pkey, {f"stale-at-{p}": stale_ts})

    patches = _make_patches(clock, LOOP_POLL_INTERVAL=0.0, REAP_INTERVAL=1)
    for ps in patches:
        ps.start()
    try:
        task = asyncio.create_task(
            background_loops(cfg, AsyncMock(return_value=None), logger)
        )
        await _yield(40)

        for p in range(MAX_PRIORITY, MIN_PRIORITY + 1):
            pkey = f"{base}:priority:{p}"
            remaining = await redis.zrangebyscore(pkey, "-inf", "+inf")
            assert f"stale-at-{p}" not in remaining, f"stale entry at priority {p} not reaped"

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        for ps in reversed(patches):
            ps.stop()


async def test_standby_takeover_refreshes_config(redis, clock):
    """Standby re-reads config from Redis on takeover, picking up changes."""
    await _init_redis(redis, limit=10_000, tps=100)
    cfg = _cfg(limit=10_000, tps=100)

    # simulate an active loop that will expire
    await redis.set(f"{TOKEN_KEY}:task:active", "old-loop", ex=1)

    # update config to new values (as if a new init_remaining_tokens was called)
    await redis.hset(
        f"{TOKEN_KEY}:config",
        mapping={"limit": 20_000, "tokens_per_second": 200},
    )

    patches = _make_patches(clock, LOOP_POLL_INTERVAL=0.0)
    for p in patches:
        p.start()
    try:
        task = asyncio.create_task(
            background_loops(cfg, AsyncMock(return_value=None), logger, standby=True)
        )
        # let the active key expire and standby take over
        await asyncio.sleep(0)
        await redis.delete(f"{TOKEN_KEY}:task:active")
        await _yield(40)

        assert cfg.limit == 20_000
        assert cfg.tokens_per_second == 200

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        for p in reversed(patches):
            p.stop()


async def test_reaper_cleans_empty_active_runs(redis, clock):
    """Reaper removes a run from active_runs when its inflight set is empty."""
    await _init_redis(redis)
    cfg = _cfg()

    run_id = "run-empty"
    active_runs_key = f"{TOKEN_KEY}:active_runs"

    await redis.sadd(active_runs_key, run_id)
    # inflight ZSET is empty (no members added)

    patches = _make_patches(clock, LOOP_POLL_INTERVAL=0.0, REAP_INTERVAL=1)
    for p in patches:
        p.start()
    try:
        task = asyncio.create_task(
            background_loops(cfg, AsyncMock(return_value=None), logger)
        )
        await _yield(40)

        members = await redis.smembers(active_runs_key)
        assert run_id not in members

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        for p in reversed(patches):
            p.stop()


# -- Heartbeat ---------------------------------------------------------------


async def test_heartbeat_writes_task_keys(redis, clock):
    """Heartbeat writes task keys for alive workers."""
    await _init_redis(redis)
    cfg = _cfg()

    patches = _make_patches(clock, LOOP_POLL_INTERVAL=1.0)
    for p in patches:
        p.start()
    try:
        task = asyncio.create_task(
            background_loops(cfg, AsyncMock(return_value=None), logger)
        )
        await _yield(60)

        refill_key = await redis.get(f"{TOKEN_KEY}:task:refill")
        assert refill_key == "1"

        reaper_key = await redis.get(f"{TOKEN_KEY}:task:reaper")
        assert reaper_key == "1"

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        for p in reversed(patches):
            p.stop()


async def test_heartbeat_writes_active_key(redis, clock):
    """Heartbeat writes active key with loop_id when is_active is True."""
    await _init_redis(redis)
    cfg = _cfg()

    patches = _make_patches(clock, LOOP_POLL_INTERVAL=1.0)
    for p in patches:
        p.start()
    try:
        task = asyncio.create_task(
            background_loops(cfg, AsyncMock(return_value=None), logger)
        )
        await _yield(60)

        active_val = await redis.get(f"{TOKEN_KEY}:task:active")
        assert active_val is not None
        # should be a UUID
        assert len(active_val) == 36

        ttl = await redis.ttl(f"{TOKEN_KEY}:task:active")
        assert 0 < ttl <= REFILL_TASK_TTL

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        for p in reversed(patches):
            p.stop()


# -- Standby -> Takeover -----------------------------------------------------


async def test_standby_waits_then_takes_over(redis, clock):
    """A loop started in standby waits for active key to expire, then takes over."""
    await _init_redis(redis)
    cfg = _cfg()

    # simulate an existing active loop
    await redis.set(f"{TOKEN_KEY}:task:active", "other-loop-id", ex=9999)

    patches = _make_patches(clock, LOOP_POLL_INTERVAL=1.0)
    for p in patches:
        p.start()
    try:
        task = asyncio.create_task(
            background_loops(cfg, AsyncMock(return_value=None), logger, standby=True)
        )

        # first few ticks: active key exists, standby should poll
        await _yield(10)

        # simulate the active key expiring (fakeredis doesn't expire on fake time)
        await redis.delete(f"{TOKEN_KEY}:task:active")

        # now standby should detect absence and take over
        await _yield(40)

        # after takeover, it should write its own active key
        active_val = await redis.get(f"{TOKEN_KEY}:task:active")
        assert active_val is not None
        assert active_val != "other-loop-id"

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        for p in reversed(patches):
            p.stop()


async def test_standby_re_reads_config(redis, clock):
    """On takeover, standby re-reads config from Redis."""
    await _init_redis(redis, limit=10_000, tps=100)
    cfg = _cfg(limit=10_000, tps=100)

    # no active loop -- standby takes over on first check
    await redis.hset(
        f"{TOKEN_KEY}:config",
        mapping={"limit": 50_000, "tokens_per_second": 500},
    )

    patches = _make_patches(clock, LOOP_POLL_INTERVAL=1.0)
    for p in patches:
        p.start()
    try:
        task = asyncio.create_task(
            background_loops(cfg, AsyncMock(return_value=None), logger, standby=True)
        )
        await _yield(40)

        assert cfg.limit == 50_000
        assert cfg.tokens_per_second == 500

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        for p in reversed(patches):
            p.stop()


async def test_standby_does_not_write_active_key(redis, clock):
    """While in standby, the heartbeat does NOT write the active key."""
    await _init_redis(redis)
    cfg = _cfg()

    # keep an active key so standby never takes over
    await redis.set(f"{TOKEN_KEY}:task:active", "permanent-owner", ex=9999)

    patches = _make_patches(clock, LOOP_POLL_INTERVAL=1.0)
    for p in patches:
        p.start()
    try:
        task = asyncio.create_task(
            background_loops(cfg, AsyncMock(return_value=None), logger, standby=True)
        )
        await _yield(30)

        # active key should still be the permanent owner, not overwritten
        active_val = await redis.get(f"{TOKEN_KEY}:task:active")
        assert active_val == "permanent-owner"

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        for p in reversed(patches):
            p.stop()


# -- Watchdog -> Demotion ----------------------------------------------------


async def test_watchdog_detects_takeover_and_demotes(redis, clock):
    """When another loop writes the active key, watchdog raises _Demoted, group shuts down,
    and the loop goes to standby."""
    await _init_redis(redis)
    # drain tokens so idle shutdown in reaper doesn't trigger
    await redis.set(TOKEN_KEY, 0)
    cfg = _cfg()

    patches = _make_patches(clock, LOOP_POLL_INTERVAL=1.0)
    for p in patches:
        p.start()
    try:
        task = asyncio.create_task(
            background_loops(cfg, AsyncMock(return_value=None), logger)
        )

        # let it become active
        await _yield(30)

        first_loop_id = await redis.get(f"{TOKEN_KEY}:task:active")
        assert first_loop_id is not None

        # another loop takes over
        await redis.set(f"{TOKEN_KEY}:task:active", "intruder-loop-id", ex=9999)

        # let watchdog detect the change and demote
        await _yield(60)

        # after demotion, it goes to standby; remove the intruder so it can take over again
        await redis.delete(f"{TOKEN_KEY}:task:active")

        await _yield(60)

        # should have taken over with a new loop_id
        new_loop_id = await redis.get(f"{TOKEN_KEY}:task:active")
        assert new_loop_id is not None
        assert new_loop_id != first_loop_id
        assert new_loop_id != "intruder-loop-id"

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        for p in reversed(patches):
            p.stop()


async def test_demotion_clears_worker_tasks(redis, clock):
    """After demotion, worker_tasks dict is cleared so heartbeat no longer
    writes stale task keys."""
    await _init_redis(redis)
    # drain tokens so idle shutdown doesn't trigger
    await redis.set(TOKEN_KEY, 0)
    cfg = _cfg()

    patches = _make_patches(clock, LOOP_POLL_INTERVAL=1.0)
    for p in patches:
        p.start()
    try:
        task = asyncio.create_task(
            background_loops(cfg, AsyncMock(return_value=None), logger)
        )

        # let it become active and heartbeat write task keys
        await _yield(30)

        # verify task keys exist while active
        assert await redis.get(f"{TOKEN_KEY}:task:refill") == "1"

        # another loop takes over -> demotion
        await redis.set(f"{TOKEN_KEY}:task:active", "intruder-loop-id", ex=9999)

        # let watchdog fire, demote, go to standby
        await _yield(60)

        # delete task keys manually; heartbeat should NOT recreate them
        # because worker_tasks was cleared on demotion
        await redis.delete(f"{TOKEN_KEY}:task:refill")
        await redis.delete(f"{TOKEN_KEY}:task:reaper")
        await redis.delete(f"{TOKEN_KEY}:task:correction")

        await _yield(30)

        # in standby: worker_tasks is empty, so heartbeat doesn't write these
        assert await redis.get(f"{TOKEN_KEY}:task:refill") is None
        assert await redis.get(f"{TOKEN_KEY}:task:reaper") is None

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        for p in reversed(patches):
            p.stop()


# -- Idle shutdown (via reaper) ----------------------------------------------


async def test_idle_shutdown_triggers_demotion(redis, clock):
    """Tokens at limit + no active runs for FULL_TOKENS_SHUTDOWN -> reaper raises
    _Demoted which exits the TaskGroup. The outer loop then goes to standby."""
    limit = 10_000
    await _init_redis(redis, limit=limit)
    await redis.set(TOKEN_KEY, limit)
    cfg = _cfg(limit=limit)

    patches = _make_patches(
        clock,
        LOOP_POLL_INTERVAL=1.0,
        REAP_INTERVAL=1,
        FULL_TOKENS_SHUTDOWN=0,
    )
    for p in patches:
        p.start()
    try:
        task = asyncio.create_task(
            background_loops(cfg, AsyncMock(return_value=None), logger)
        )

        await _yield(120)

        # after idle shutdown the reaper raises _Demoted, group exits,
        # loop goes to standby (it does NOT return -- it stays in standby)
        assert not task.done(), "background_loops should be in standby, not exited"

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        for p in reversed(patches):
            p.stop()


async def test_no_idle_shutdown_with_active_runs(redis, clock):
    """Idle shutdown does NOT trigger when there are active runs."""
    limit = 10_000
    await _init_redis(redis, limit=limit)
    await redis.set(TOKEN_KEY, limit)
    await redis.sadd(f"{TOKEN_KEY}:active_runs", "run-1")
    # add a fresh inflight entry so the run isn't cleaned up by reaper
    await redis.zadd(f"{TOKEN_KEY}:run:run-1:inflight", {"q1": clock.now})
    cfg = _cfg(limit=limit)

    patches = _make_patches(
        clock,
        LOOP_POLL_INTERVAL=1.0,
        REAP_INTERVAL=1,
        FULL_TOKENS_SHUTDOWN=0,
    )
    for p in patches:
        p.start()
    try:
        task = asyncio.create_task(
            background_loops(cfg, AsyncMock(return_value=None), logger)
        )
        await _yield(60)

        # should NOT have triggered idle shutdown -- reaper sees active_runs > 0
        assert not task.done()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        for p in reversed(patches):
            p.stop()


async def test_no_idle_shutdown_when_tokens_below_limit(redis, clock):
    """Idle shutdown does NOT trigger when tokens are below limit."""
    limit = 10_000
    await _init_redis(redis, limit=limit, tps=0)
    await redis.set(TOKEN_KEY, limit - 1)
    cfg = _cfg(limit=limit, tps=0)

    patches = _make_patches(
        clock,
        LOOP_POLL_INTERVAL=1.0,
        REAP_INTERVAL=1,
        FULL_TOKENS_SHUTDOWN=0,
    )
    for p in patches:
        p.start()
    try:
        task = asyncio.create_task(
            background_loops(cfg, AsyncMock(return_value=None), logger)
        )
        await _yield(60)

        assert not task.done()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        for p in reversed(patches):
            p.stop()


async def test_idle_timer_resets_when_tokens_drop(redis, clock):
    """idle_since resets when tokens drop below limit, preventing premature shutdown."""
    limit = 10_000
    await _init_redis(redis, limit=limit, tps=0)
    await redis.set(TOKEN_KEY, limit)
    cfg = _cfg(limit=limit, tps=0)

    reap_tick = 0

    async def custom_sleep(seconds):
        nonlocal reap_tick
        clock.now += seconds
        # after a few reaper ticks (idle_since accumulating), drop tokens below limit
        reap_tick += 1
        if reap_tick == 5:
            await redis.set(TOKEN_KEY, limit - 500)
        await _real_sleep(0)

    fake_time = FakeTimeModule(clock)
    fake_asyncio = _FakeAsyncioModule(clock)
    # override sleep with our custom version
    fake_asyncio.sleep = custom_sleep  # type: ignore[assignment]

    patches = [
        patch.object(bg_module, "time", fake_time),
        patch.object(bg_module, "asyncio", fake_asyncio),
        patch.object(bg_module, "LOOP_POLL_INTERVAL", 1.0),
        patch.object(bg_module, "REAP_INTERVAL", 1),
        patch.object(bg_module, "FULL_TOKENS_SHUTDOWN", 20),
    ]
    for p in patches:
        p.start()
    try:
        task = asyncio.create_task(
            background_loops(cfg, AsyncMock(return_value=None), logger)
        )
        await _yield(100)

        # should NOT have triggered idle shutdown since tokens dropped mid-way
        assert not task.done()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        for p in reversed(patches):
            p.stop()


# -- Heartbeat cancellation on exit ------------------------------------------


async def test_heartbeat_cancelled_on_exit(redis, clock):
    """The heartbeat task is cancelled when background_loops exits (via cancellation)."""
    await _init_redis(redis)
    cfg = _cfg()

    patches = _make_patches(clock, LOOP_POLL_INTERVAL=1.0)
    for p in patches:
        p.start()
    try:
        task = asyncio.create_task(
            background_loops(cfg, AsyncMock(return_value=None), logger)
        )
        await _yield(30)

        # verify active key exists while running
        assert await redis.get(f"{TOKEN_KEY}:task:active") is not None

        # cancel the background_loops task
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        for p in reversed(patches):
            p.stop()

    # after exit, heartbeat should be cancelled; no more writes
    await redis.delete(f"{TOKEN_KEY}:task:active")
    await _yield(10)
    assert await redis.get(f"{TOKEN_KEY}:task:active") is None


# -- Active loop started without standby -------------------------------------


async def test_non_standby_immediately_active(redis, clock):
    """A loop started without standby=True becomes active immediately."""
    await _init_redis(redis)
    cfg = _cfg()

    patches = _make_patches(clock, LOOP_POLL_INTERVAL=1.0)
    for p in patches:
        p.start()
    try:
        task = asyncio.create_task(
            background_loops(cfg, AsyncMock(return_value=None), logger, standby=False)
        )
        await _yield(40)

        active_val = await redis.get(f"{TOKEN_KEY}:task:active")
        assert active_val is not None

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        for p in reversed(patches):
            p.stop()
