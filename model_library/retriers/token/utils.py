from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Awaitable, Protocol, cast

from pydantic import BaseModel
from redis.asyncio import Redis
from redis.asyncio.lock import Lock

# ── Run identification ──────────────────────────────────────────────
#
# RunContext propagates run identity from benchmark_queue into TokenRetrier.
# benchmark_queue sets it before yield, resets in finally. When unset,
# TokenRetrier falls back to LLM.run_id (from LLMConfig or auto-generated).
#
# Fields:
#   run_id    — benchmark run ID (queued) or LLM.run_id (fallback)
#   is_queued — True inside benchmark_queue context manager. Controls:
#     - Straggler detection: only queued runs check the queue head via lindex.
#       If it differs from run_id, the run is demoted to MAX_PRIORITY.
#     - Per-run dispatched counter: only incremented for queued runs.
#     Does NOT affect: dynamic estimates, inflight tracking, priority queues,
#     or token deduction — those work identically for all runs.
#
# Dynamic estimate scoping:
#   - Queued: keyed by run_id — each benchmark run starts at ratio 1.0
#   - Non-queued: keyed by LLM.run_id — cross-run learning preserved
#
# Edge case — nested benchmark contexts:
#   All TokenRetrier instances inside one benchmark_queue context share the
#   same run_id. If sub-run isolation is needed, use separate contexts.
#


@dataclass
class RunContext:
    """Identifies the current run for TokenRetrier requests.

    Set via contextvar by benchmark_queue (queued runs) or left unset
    to fall back to LLM.run_id (non-benchmark runs).
    """

    run_id: str
    is_queued: bool = False


current_run: ContextVar[RunContext | None] = ContextVar("current_run", default=None)


class AsyncRedisClient(Protocol):
    """Typed protocol for the subset of async Redis commands we use."""

    async def get(self, name: str) -> str | None: ...
    async def set(
        self, name: str, value: str | int | float, ex: int | None = None
    ) -> bool: ...
    async def exists(self, *names: str) -> int: ...
    async def delete(self, *names: str) -> int: ...
    async def expire(self, name: str, time: int) -> bool: ...
    async def incr(self, name: str) -> int: ...
    async def lpos(self, name: str, value: str) -> int | None: ...
    async def rpush(self, name: str, *values: str) -> int: ...
    async def lrem(self, name: str, count: int, value: str) -> int: ...
    async def lindex(self, name: str, index: int) -> str | None: ...
    async def lrange(self, name: str, start: int, end: int) -> list[str]: ...
    async def blpop(
        self, keys: list[str], timeout: int | float | None = 0
    ) -> list[str] | None: ...
    async def ttl(self, name: str) -> int: ...
    async def zadd(
        self, name: str, mapping: dict[str, float], nx: bool = False
    ) -> int: ...
    async def zrem(self, name: str, *members: str) -> int: ...
    async def zrangebyscore(
        self, name: str, min: float | str, max: float | str, withscores: bool = False
    ) -> list[str]: ...
    async def sadd(self, name: str, *values: str) -> int: ...
    async def srem(self, name: str, *values: str) -> int: ...
    async def scard(self, name: str) -> int: ...
    async def smembers(self, name: str) -> set[str]: ...
    async def zcard(self, name: str) -> int: ...
    async def hexists(self, name: str, key: str) -> bool: ...
    async def hset(
        self, name: str, mapping: Mapping[str, str | int | float]
    ) -> int: ...
    async def hgetall(self, name: str) -> dict[str, str]: ...
    async def keys(self, pattern: str) -> list[str]: ...
    async def eval(
        self, script: str, numkeys: int, *keys_and_args: str | int | float
    ) -> Any: ...
    def lock(self, name: str, timeout: float | None = None) -> Lock: ...


KEY_PREFIX = "model_library"

# Redis key schema: see docs/token-retry.md

redis_client: AsyncRedisClient = None  # pyright: ignore[reportAssignmentType]


def set_redis_client(client: Redis):
    global redis_client
    redis_client = client  # pyright: ignore[reportAssignmentType]


async def validate_redis_client(
    key: str | None = None, missing_key_message: str | None = None
):
    try:
        assert redis_client
    except Exception as e:
        raise Exception(f"redis client not set, run `set_redis_client`. Exception: {e}")

    if key:
        if not await redis_client.exists(key):
            raise Exception(f"{key} not intialized. {missing_key_message}")


async def get_token_keys() -> list[str]:
    """Return all valid token retry base keys"""

    all_keys = await redis_client.keys(f"{KEY_PREFIX}:*:*:tokens")
    if not all_keys:
        return []
    values = await asyncio.gather(*(redis_client.get(key) for key in all_keys))
    keys: list[str] = []
    for key, val in zip(all_keys, values):
        try:
            int(val) if val else 0
        except ValueError:
            continue
        keys.append(key)
    return sorted(keys)


async def cleanup_all_keys() -> int:
    """Delete all model_library:* keys (token retry, priority, benchmark queue)"""

    to_delete = await redis_client.keys(f"{KEY_PREFIX}:*")
    if not to_delete:
        return 0
    return await redis_client.delete(*to_delete)


# ── Status models ────────────────────────────────────────────────────


class InflightRequest(BaseModel):
    question_id: str
    elapsed_seconds: float
    estimate_input: int | None
    estimate_output: int | None
    estimate_total: int | None
    priority: int | None
    attempts: int | None
    run_id: str | None
    dispatched_at: float | None


class DynamicEstimate(BaseModel):
    run_id: str
    ratio: float


class TokenRetryConfig(BaseModel):
    limit: int | None
    limit_refresh_seconds: int | None
    tokens_per_second: int | None
    version: str | None
    initialized_at: float | None


class TokenRetryStatus(BaseModel):
    token_key: str
    tokens_remaining: int
    token_limit: int
    config: TokenRetryConfig
    inflight: list[InflightRequest]
    priorities: dict[str, int]
    refill_alive: bool
    correction_alive: bool
    active_benchmark_run: str | None
    dynamic_estimates: list[DynamicEstimate]


class QueueEntry(BaseModel):
    run_id: str
    alive: bool
    heartbeat_ttl: int
    position: int
    notified: bool
    total_requests: int | None
    dispatched_count: int
    inflight_count: int
    queued_by_priority: dict[str, int]
    slot_acquired: bool
    enqueued_at: float | None
    slot_acquired_at: float | None
    popped_at: float | None
    completed_at: float | None
    popped: bool = False
    is_queued: bool = True


class QueueStatus(BaseModel):
    queue_key: str
    length: int
    entries: list[QueueEntry]


class ModelStatus(BaseModel):
    key: str
    token: TokenRetryStatus | None
    queue: QueueStatus | None


class Status(BaseModel):
    models: list[ModelStatus]


# ── Status function ──────────────────────────────────────────────────


async def get_status() -> Status:
    """Get combined token retry and benchmark queue status, grouped by model key."""

    token_retry, queued_by_run, queued_run_to_key = await _get_token_retry_status()

    # compute per-run inflight counts from already-fetched inflight data
    inflight_by_run: dict[str, int] = {}
    for t in token_retry:
        for r in t.inflight:
            if r.run_id:
                inflight_by_run[r.run_id] = inflight_by_run.get(r.run_id, 0) + 1

    # derive base keys from token keys so popped detection works even when queue list is gone
    token_base_keys = [t.token_key.removesuffix(":tokens") for t in token_retry]
    queue_statuses = await _get_queue_status(
        inflight_by_run, queued_by_run, token_base_keys
    )

    # collect all run_ids already covered by queue entries
    covered_run_ids: set[str] = set()
    for q in queue_statuses:
        for e in q.entries:
            covered_run_ids.add(e.run_id)

    # find non-queued runs (in token retrier but not in benchmark queue)
    all_token_run_ids = set(inflight_by_run.keys()) | set(queued_by_run.keys())
    unqueued_run_ids = all_token_run_ids - covered_run_ids

    # group by model key (strip prefix + suffixes for clean display key)
    prefix_and_colon = f"{KEY_PREFIX}:"
    grouped: dict[str, ModelStatus] = {}

    for t in token_retry:
        display_key = t.token_key.removeprefix(prefix_and_colon).removesuffix(":tokens")
        if display_key in grouped:
            grouped[display_key].token = t
        else:
            grouped[display_key] = ModelStatus(key=display_key, token=t, queue=None)

    for q in queue_statuses:
        display_key = q.queue_key.removeprefix(prefix_and_colon).removesuffix(
            ":benchmark:queue"
        )
        if display_key in grouped:
            grouped[display_key].queue = q
        else:
            grouped[display_key] = ModelStatus(key=display_key, token=None, queue=q)

    # attach non-queued runs to their model's queue status
    if unqueued_run_ids:
        # map run_id → display_key via inflight metadata and priority set data
        run_to_model: dict[str, str] = {}
        for t in token_retry:
            display_key = t.token_key.removeprefix(prefix_and_colon).removesuffix(
                ":tokens"
            )
            for r in t.inflight:
                if r.run_id and r.run_id in unqueued_run_ids:
                    run_to_model[r.run_id] = display_key
        # also map from priority sets (covers runs with no inflight requests yet)
        for rid in unqueued_run_ids:
            if rid not in run_to_model and rid in queued_run_to_key:
                display_key = (
                    queued_run_to_key[rid]
                    .removeprefix(prefix_and_colon)
                    .removesuffix(":tokens")
                )
                run_to_model[rid] = display_key

        for rid in sorted(unqueued_run_ids):
            display_key = run_to_model.get(rid)
            if not display_key:
                continue
            entry = QueueEntry(
                run_id=rid,
                alive=False,
                heartbeat_ttl=-1,
                position=-1,
                notified=False,
                total_requests=None,
                dispatched_count=0,
                inflight_count=inflight_by_run.get(rid, 0),
                queued_by_priority=queued_by_run.get(rid, {}),
                slot_acquired=False,
                enqueued_at=None,
                slot_acquired_at=None,
                popped_at=None,
                completed_at=None,
                is_queued=False,
            )
            model = grouped.get(display_key)
            if model and model.queue:
                model.queue.entries.append(entry)
            elif model:
                model.queue = QueueStatus(
                    queue_key=f"{prefix_and_colon}{display_key}:benchmark:queue",
                    length=0,
                    entries=[entry],
                )

    return Status(models=list(grouped.values()))


async def _get_token_retry_status() -> tuple[
    list[TokenRetryStatus], dict[str, dict[str, int]], dict[str, str]
]:
    from model_library.retriers.token.token import (
        MAX_PRIORITY,
        MIN_PRIORITY,
    )

    token_keys = await get_token_keys()
    if not token_keys:
        return [], {}, {}

    results = await asyncio.gather(
        *(
            _get_status_for_token_key(tk, MAX_PRIORITY, MIN_PRIORITY)
            for tk in token_keys
        )
    )

    statuses: list[TokenRetryStatus] = []
    merged_queued_by_run: dict[str, dict[str, int]] = {}
    # map run_id → token_key for run_ids found in priority sets
    queued_run_to_key: dict[str, str] = {}
    for status, queued_by_run in results:
        statuses.append(status)
        for rid, priorities in queued_by_run.items():
            queued_run_to_key.setdefault(rid, status.token_key)
            merged = merged_queued_by_run.setdefault(rid, {})
            for p_str, count in priorities.items():
                merged[p_str] = merged.get(p_str, 0) + count

    return statuses, merged_queued_by_run, queued_run_to_key


async def _get_status_for_token_key(
    token_key: str, max_priority: int, min_priority: int
) -> tuple[TokenRetryStatus, dict[str, dict[str, int]]]:
    """Fetch status for a single token key with parallelized Redis calls."""
    limit_key = f"{token_key}:limit"
    base = token_key.removesuffix(":tokens")

    # parallel: base data (split for typed gather overloads)
    tokens_raw, limit_raw, config_raw, active_benchmark_run = await asyncio.gather(
        redis_client.get(token_key),
        redis_client.get(limit_key),
        redis_client.hgetall(f"{token_key}:config"),
        redis_client.lindex(f"{base}:benchmark:queue", 0),
    )
    (
        active_run_ids,
        refill_alive_raw,
        correction_alive_raw,
        dynamic_estimate_keys,
    ) = await asyncio.gather(
        redis_client.smembers(f"{token_key}:active_runs"),
        redis_client.exists(f"{token_key}:task:refill"),
        redis_client.exists(f"{token_key}:task:correction"),
        redis_client.keys(f"{token_key}:dynamic_estimate:*"),
    )

    tokens_remaining = int(tokens_raw) if tokens_raw else 0
    token_limit = int(limit_raw) if limit_raw else 0
    refill_alive = refill_alive_raw > 0
    correction_alive = correction_alive_raw > 0
    config = TokenRetryConfig(
        limit=int(config_raw["limit"]) if "limit" in config_raw else None,
        limit_refresh_seconds=int(config_raw["limit_refresh_seconds"])
        if "limit_refresh_seconds" in config_raw
        else None,
        tokens_per_second=int(config_raw["tokens_per_second"])
        if "tokens_per_second" in config_raw
        else None,
        version=config_raw.get("version"),
        initialized_at=float(config_raw["initialized_at"])
        if "initialized_at" in config_raw
        else None,
    )

    now = time.time()

    # parallel: fetch inflight entries per active run
    sorted_run_ids = sorted(active_run_ids)
    inflight_entries_list = (
        await asyncio.gather(
            *(
                redis_client.zrangebyscore(
                    f"{token_key}:run:{rid}:inflight", "-inf", "+inf", withscores=True
                )
                for rid in sorted_run_ids
            )
        )
        if sorted_run_ids
        else []
    )

    all_inflight_members: list[tuple[str, float]] = []
    for entries in inflight_entries_list:
        # withscores=True returns list[tuple[str, float]] despite protocol typing
        all_inflight_members.extend(cast(list[tuple[str, float]], entries))

    # parallel: fetch all inflight metadata
    inflight_metas = (
        await asyncio.gather(
            *(
                redis_client.hgetall(f"{token_key}:inflight:{member}")
                for member, _ in all_inflight_members
            )
        )
        if all_inflight_members
        else []
    )

    inflight: list[InflightRequest] = []
    for (member, score), meta in zip(all_inflight_members, inflight_metas):
        run_id_val = meta.get("run_id", "")
        inflight.append(
            InflightRequest(
                question_id=member,
                elapsed_seconds=round(now - float(score), 1),
                estimate_input=int(meta["estimate_input"])
                if "estimate_input" in meta
                else None,
                estimate_output=int(meta["estimate_output"])
                if "estimate_output" in meta
                else None,
                estimate_total=int(meta["estimate_total"])
                if "estimate_total" in meta
                else None,
                priority=int(meta["priority"]) if "priority" in meta else None,
                attempts=int(meta["attempts"]) if "attempts" in meta else None,
                run_id=run_id_val if run_id_val else None,
                dispatched_at=float(meta["dispatched_at"])
                if "dispatched_at" in meta
                else None,
            )
        )
    inflight.sort(key=lambda r: r.elapsed_seconds, reverse=True)

    # parallel: fetch all priority sets
    priority_range = list(range(max_priority, min_priority + 1))
    priority_members_list = await asyncio.gather(
        *(
            redis_client.zrangebyscore(f"{base}:priority:{p}", "-inf", "+inf")
            for p in priority_range
        )
    )

    all_priority_members: list[str] = []
    for members in priority_members_list:
        all_priority_members.extend(members)

    # parallel: fetch priority member metadata
    priority_metas = (
        await asyncio.gather(
            *(
                redis_client.hgetall(f"{token_key}:inflight:{member}")
                for member in all_priority_members
            )
        )
        if all_priority_members
        else []
    )

    priorities: dict[str, int] = {}
    queued_by_run: dict[str, dict[str, int]] = {}
    meta_idx = 0
    for priority, members in zip(priority_range, priority_members_list):
        priorities[str(priority)] = len(members)
        for _member in members:
            meta = priority_metas[meta_idx]
            meta_idx += 1
            rid = meta.get("run_id", "")
            if rid:
                queued_by_run.setdefault(rid, {})
                p_str = str(priority)
                queued_by_run[rid][p_str] = queued_by_run[rid].get(p_str, 0) + 1

    # parallel: fetch dynamic estimates
    sorted_de_keys = sorted(dynamic_estimate_keys)
    de_values = (
        await asyncio.gather(*(redis_client.get(de_key) for de_key in sorted_de_keys))
        if sorted_de_keys
        else []
    )
    dynamic_estimates = [
        DynamicEstimate(
            run_id=de_key.removeprefix(f"{token_key}:dynamic_estimate:"),
            ratio=float(ratio_raw),
        )
        for de_key, ratio_raw in zip(sorted_de_keys, de_values)
        if ratio_raw
    ]

    return TokenRetryStatus(
        token_key=token_key,
        tokens_remaining=tokens_remaining,
        token_limit=token_limit,
        config=config,
        inflight=inflight,
        priorities=priorities,
        refill_alive=refill_alive,
        correction_alive=correction_alive,
        active_benchmark_run=active_benchmark_run,
        dynamic_estimates=dynamic_estimates,
    ), queued_by_run


async def _get_queue_entry(
    base_key: str,
    run_id: str,
    position: int,
    inflight_by_run: dict[str, int],
    queued_by_run: dict[str, dict[str, int]],
) -> QueueEntry:
    """Fetch a single queue entry with parallelized Redis calls."""
    alive_key = f"{base_key}:alive:{run_id}"
    notify_key = f"{base_key}:notify:{run_id}"
    run_meta_key = f"{base_key}:run:{run_id}"

    alive_raw, ttl_raw, notified_raw, meta, dispatched_count = await asyncio.gather(
        redis_client.exists(alive_key),
        redis_client.ttl(alive_key),
        redis_client.exists(notify_key),
        redis_client.hgetall(run_meta_key),
        redis_client.scard(f"{run_meta_key}:dispatched"),
    )

    alive = alive_raw > 0
    total_raw = meta.get("total_requests", "0")
    return QueueEntry(
        run_id=run_id,
        alive=alive,
        heartbeat_ttl=ttl_raw if alive else -1,
        position=position,
        notified=notified_raw > 0,
        total_requests=int(total_raw) if total_raw != "0" else None,
        dispatched_count=dispatched_count,
        inflight_count=inflight_by_run.get(run_id, 0),
        queued_by_priority=queued_by_run.get(run_id, {}),
        slot_acquired=meta.get("slot_acquired") == "1",
        enqueued_at=float(meta["enqueued_at"]) if "enqueued_at" in meta else None,
        slot_acquired_at=float(meta["slot_acquired_at"])
        if "slot_acquired_at" in meta
        else None,
        popped_at=float(meta["popped_at"]) if "popped_at" in meta else None,
        completed_at=float(meta["completed_at"]) if "completed_at" in meta else None,
    )


async def _get_queue_status(
    inflight_by_run: dict[str, int],
    queued_by_run: dict[str, dict[str, int]],
    token_base_keys: list[str] | None = None,
) -> list[QueueStatus]:
    queue_keys = sorted(await redis_client.keys(f"{KEY_PREFIX}:*:benchmark:queue"))

    seen_base_keys: set[str] = set()

    # parallel: fetch run IDs for all queues
    run_ids_per_queue = (
        await asyncio.gather(*(redis_client.lrange(qk, 0, -1) for qk in queue_keys))
        if queue_keys
        else []
    )

    # parallel: fetch all queue entries across all queues
    entry_coros: list[Awaitable[QueueEntry]] = []
    entry_queue_idx: list[int] = []
    for q_idx, (run_queue_key, run_ids) in enumerate(
        zip(queue_keys, run_ids_per_queue)
    ):
        base_key = run_queue_key.removesuffix(":queue")
        seen_base_keys.add(base_key)
        for i, run_id in enumerate(run_ids):
            entry_coros.append(
                _get_queue_entry(base_key, run_id, i, inflight_by_run, queued_by_run)
            )
            entry_queue_idx.append(q_idx)

    all_entries = await asyncio.gather(*entry_coros) if entry_coros else []

    entries_by_queue: dict[int, list[QueueEntry]] = {}
    for entry, q_idx in zip(all_entries, entry_queue_idx):
        entries_by_queue.setdefault(q_idx, []).append(entry)

    statuses: list[QueueStatus] = []
    for q_idx, (run_queue_key, run_ids) in enumerate(
        zip(queue_keys, run_ids_per_queue)
    ):
        entries = entries_by_queue.get(q_idx, [])
        if entries:
            statuses.append(
                QueueStatus(
                    queue_key=run_queue_key,
                    length=len(run_ids),
                    entries=entries,
                )
            )

    # discover popped/completed runs from metadata hashes (survives queue LIST deletion)
    covered_run_ids: set[str] = set()
    for s in statuses:
        for e in s.entries:
            covered_run_ids.add(e.run_id)

    all_benchmark_bases = set(seen_base_keys)
    if token_base_keys:
        for base_key in token_base_keys:
            all_benchmark_bases.add(f"{base_key}:benchmark")

    # parallel: scan metadata keys for all benchmark bases
    sorted_bases = sorted(all_benchmark_bases)
    run_meta_keys_per_base = (
        await asyncio.gather(*(redis_client.keys(f"{bb}:run:*") for bb in sorted_bases))
        if sorted_bases
        else []
    )

    # collect uncovered popped runs needing metadata
    popped_lookups: list[tuple[str, str, str]] = []  # (benchmark_base, run_id, rmk)
    for benchmark_base, run_meta_keys in zip(sorted_bases, run_meta_keys_per_base):
        for rmk in sorted(run_meta_keys):
            if rmk.endswith(":dispatched"):
                continue
            run_id = rmk.removeprefix(f"{benchmark_base}:run:")
            if run_id in covered_run_ids:
                continue
            popped_lookups.append((benchmark_base, run_id, rmk))
            covered_run_ids.add(run_id)

    # parallel: fetch metadata + dispatched count for all popped runs
    popped_results = (
        await asyncio.gather(
            *(
                asyncio.gather(
                    redis_client.hgetall(rmk),
                    redis_client.scard(f"{rmk}:dispatched"),
                )
                for _, _, rmk in popped_lookups
            )
        )
        if popped_lookups
        else []
    )

    popped_by_base: dict[str, list[QueueEntry]] = {}
    for (benchmark_base, run_id, _), (meta, dispatched_count) in zip(
        popped_lookups, popped_results
    ):
        if not meta:
            continue
        total_raw = meta.get("total_requests", "0")
        entry = QueueEntry(
            run_id=run_id,
            alive=False,
            heartbeat_ttl=-1,
            position=-1,
            notified=False,
            total_requests=int(total_raw) if total_raw != "0" else None,
            dispatched_count=dispatched_count,
            inflight_count=inflight_by_run.get(run_id, 0),
            queued_by_priority=queued_by_run.get(run_id, {}),
            slot_acquired=meta.get("slot_acquired") == "1",
            enqueued_at=float(meta["enqueued_at"]) if "enqueued_at" in meta else None,
            slot_acquired_at=float(meta["slot_acquired_at"])
            if "slot_acquired_at" in meta
            else None,
            popped_at=float(meta["popped_at"]) if "popped_at" in meta else None,
            completed_at=float(meta["completed_at"])
            if "completed_at" in meta
            else None,
            popped=True,
        )
        popped_by_base.setdefault(benchmark_base, []).append(entry)

    for benchmark_base, entries in popped_by_base.items():
        queue_key = f"{benchmark_base}:queue"
        existing = next((s for s in statuses if s.queue_key == queue_key), None)
        if existing:
            existing.entries.extend(entries)
        else:
            statuses.append(
                QueueStatus(
                    queue_key=queue_key,
                    length=0,
                    entries=entries,
                )
            )

    return statuses
