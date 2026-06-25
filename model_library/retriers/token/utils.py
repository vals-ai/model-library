from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Mapping
from typing import Any, Protocol, cast

from pydantic import BaseModel, Field
from redis.asyncio import Redis
from redis.asyncio.client import Pipeline
from redis.asyncio.lock import Lock

from model_library.base.output import RateLimit
from model_library.utils import SecondsMetric, ValsModel


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
    async def lpop(self, name: str) -> str | None: ...
    async def lrange(self, name: str, start: int, end: int) -> list[str]: ...
    async def blpop(
        self, keys: list[str], timeout: int | float | None = 0
    ) -> list[str] | None: ...
    async def ttl(self, name: str) -> int: ...
    async def zadd(
        self, name: str, mapping: dict[str, float], nx: bool = False
    ) -> int: ...
    async def zrem(self, name: str, *members: str) -> int: ...
    async def zscore(self, name: str, value: str) -> float | None: ...
    async def zrange(self, name: str, start: int, end: int) -> list[str]: ...
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
    async def hdel(self, name: str, *keys: str) -> int: ...
    async def hgetall(self, name: str) -> dict[str, str]: ...
    async def keys(self, pattern: str) -> list[str]: ...
    def scan_iter(
        self, match: str | None = None, count: int | None = None
    ) -> AsyncIterator[str]: ...
    async def eval(
        self, script: str, numkeys: int, *keys_and_args: str | int | float
    ) -> Any: ...
    def lock(self, name: str, timeout: float | None = None) -> Lock: ...
    def pipeline(self, transaction: bool = True) -> Pipeline: ...


KEY_PREFIX = "model_library"
DEFAULT_STATUS_QUEUE_ENTRY_LIMIT = 100
STATUS_PIPELINE_COMMAND_LIMIT = 200
StatusPipelineCommand = tuple[str, tuple[Any, ...], dict[str, Any]]

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


async def scan_keys(pattern: str) -> list[str]:
    """Collect keys matching pattern using SCAN (non-blocking, cursor-based)."""
    keys = list({k async for k in redis_client.scan_iter(match=pattern, count=500)})
    return keys


async def _execute_status_pipeline(commands: list[StatusPipelineCommand]) -> list[Any]:
    """Execute status-read Redis commands in bounded non-transactional pipelines.

    Status polling can need hundreds of Redis reads. Unbounded asyncio.gather
    exhausts the Redis client's connection pool; one giant pipeline can monopolize
    Redis. Chunked pipelines keep reads fast while limiting connection fan-out.
    """
    results: list[Any] = []
    chunks = 0
    for start in range(0, len(commands), STATUS_PIPELINE_COMMAND_LIMIT):
        chunks += 1
        pipe = redis_client.pipeline(transaction=False)
        for method_name, args, kwargs in commands[
            start : start + STATUS_PIPELINE_COMMAND_LIMIT
        ]:
            getattr(pipe, method_name)(*args, **kwargs)
        results.extend(await pipe.execute())
    return results


async def get_token_keys() -> list[str]:
    """Return all valid token retry base keys"""

    all_keys = await scan_keys(f"{KEY_PREFIX}:*:*:tokens")
    if not all_keys:
        return []
    values = await _execute_status_pipeline([("get", (key,), {}) for key in all_keys])
    keys: list[str] = []
    for key, val in zip(all_keys, values):
        try:
            int(val) if val else 0
        except ValueError:
            continue
        keys.append(key)
    sorted_keys = sorted(keys)
    return sorted_keys


async def cleanup_all_keys() -> int:
    """Delete all model_library:* keys (token retry, priority, benchmark queue)"""

    to_delete = await scan_keys(f"{KEY_PREFIX}:*")
    if not to_delete:
        return 0
    return await redis_client.delete(*to_delete)


# ── Status models ────────────────────────────────────────────────────


class InflightRequest(ValsModel):
    question_id: str
    elapsed_seconds: SecondsMetric
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
    # Valid, non-stale priority waiters with metadata. This is the user-facing
    # queued count; raw/orphaned/stale counts are exposed separately for debugging.
    priorities: dict[str, int]
    raw_priorities: dict[str, int]
    orphaned_priorities: dict[str, int]
    stale_priorities: dict[str, int]
    refill_alive: bool
    correction_alive: bool
    active_benchmark_run: str | None
    active_benchmark_runs: list[str]
    dynamic_estimates: list[DynamicEstimate]
    last_header: RateLimit | None = None


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
    is_active_head: bool = False
    enqueued_at: float | None
    slot_acquired_at: float | None
    popped_at: float | None
    completed_at: float | None
    popped: bool = False
    is_queued: bool = True
    display_state: str = "UNKNOWN"
    is_historical: bool = False
    has_live_queued_requests: bool = False
    has_live_inflight_requests: bool = False


class QueueStatus(BaseModel):
    queue_key: str
    length: int
    entries: list[QueueEntry]
    active_head_window: int = 1
    active_heads: list[str] = Field(default_factory=list)


class ModelStatus(BaseModel):
    key: str
    token: TokenRetryStatus | None
    queue: QueueStatus | None


class Status(BaseModel):
    models: list[ModelStatus]


# ── Status function ──────────────────────────────────────────────────


async def get_status(
    queue_entry_limit: int | None = None,
    include_historical: bool = True,
) -> Status:
    """Get combined token retry and benchmark queue status, grouped by model key.

    Args:
        queue_entry_limit: Maximum queued run entries to fetch per model queue.
            ``None`` returns the full queue for diagnostic callers.
        include_historical: Whether to scan popped/completed run metadata that is no
            longer present in the queue LIST. Disable for high-frequency polling.
    """

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
        inflight_by_run,
        queued_by_run,
        token_base_keys,
        queue_entry_limit=queue_entry_limit,
        include_historical=include_historical,
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

        unqueued_meta_commands: list[StatusPipelineCommand] = []
        unqueued_lookups: list[tuple[str, str]] = []
        for rid in sorted(unqueued_run_ids):
            display_key = run_to_model.get(rid)
            if not display_key:
                continue
            benchmark_base = f"{prefix_and_colon}{display_key}:benchmark"
            run_meta_key = f"{benchmark_base}:run:{rid}"
            unqueued_lookups.append((rid, display_key))
            unqueued_meta_commands.append(("hgetall", (run_meta_key,), {}))
            unqueued_meta_commands.append(
                ("scard", (f"{run_meta_key}:dispatched",), {})
            )

        unqueued_meta_results = (
            await _execute_status_pipeline(unqueued_meta_commands)
            if unqueued_meta_commands
            else []
        )

        for index, (rid, display_key) in enumerate(unqueued_lookups):
            meta = unqueued_meta_results[index * 2]
            dispatched_count = unqueued_meta_results[index * 2 + 1]
            total_raw = meta.get("total_requests", "0")
            entry = _make_queue_entry(
                run_id=rid,
                alive=False,
                heartbeat_ttl=-1,
                position=-1,
                notified=False,
                total_requests=int(total_raw) if total_raw != "0" else None,
                dispatched_count=dispatched_count,
                inflight_count=inflight_by_run.get(rid, 0),
                queued_by_priority=queued_by_run.get(rid, {}),
                slot_acquired=meta.get("slot_acquired") == "1",
                is_active_head=False,
                enqueued_at=float(meta["enqueued_at"])
                if "enqueued_at" in meta
                else None,
                slot_acquired_at=float(meta["slot_acquired_at"])
                if "slot_acquired_at" in meta
                else None,
                popped_at=float(meta["popped_at"]) if "popped_at" in meta else None,
                completed_at=float(meta["completed_at"])
                if "completed_at" in meta
                else None,
                popped="popped_at" in meta,
                is_queued=bool(meta),
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

    models = list(grouped.values())
    return Status(models=models)


async def _get_token_retry_status() -> tuple[
    list[TokenRetryStatus], dict[str, dict[str, int]], dict[str, str]
]:
    from model_library.retriers.token.token import (
        MAX_PRIORITY,
        MIN_PRIORITY,
        PRIORITY_STALE_AGE,
    )

    token_keys = await get_token_keys()
    if not token_keys:
        return [], {}, {}

    results = await asyncio.gather(
        *(
            _get_status_for_token_key(
                tk, MAX_PRIORITY, MIN_PRIORITY, PRIORITY_STALE_AGE
            )
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
    token_key: str, max_priority: int, min_priority: int, priority_stale_age: int
) -> tuple[TokenRetryStatus, dict[str, dict[str, int]]]:
    """Fetch status for a single token key with parallelized Redis calls."""
    limit_key = f"{token_key}:limit"
    base = token_key.removesuffix(":tokens")

    # Pipeline fixed-key status reads to avoid connection fan-out during polling.
    (
        tokens_raw,
        limit_raw,
        config_raw,
        active_benchmark_queue_head,
        active_benchmark_runs_raw,
        last_header_raw,
        active_run_ids,
        refill_alive_raw,
        correction_alive_raw,
    ) = await _execute_status_pipeline(
        [
            ("get", (token_key,), {}),
            ("get", (limit_key,), {}),
            ("hgetall", (f"{token_key}:config",), {}),
            ("lindex", (f"{base}:benchmark:queue", 0), {}),
            ("zrange", (f"{base}:benchmark:active_heads", 0, -1), {}),
            ("hgetall", (f"{token_key}:last_header",), {}),
            ("smembers", (f"{token_key}:active_runs",), {}),
            ("exists", (f"{token_key}:task:refill",), {}),
            ("exists", (f"{token_key}:task:correction",), {}),
        ]
    )

    dynamic_estimate_keys = await scan_keys(f"{token_key}:dynamic_estimate:*")

    tokens_remaining = int(tokens_raw) if tokens_raw else 0
    token_limit = int(limit_raw) if limit_raw else 0
    active_benchmark_runs = cast(list[str], active_benchmark_runs_raw)
    active_benchmark_run = (
        active_benchmark_runs[0]
        if active_benchmark_runs
        else active_benchmark_queue_head
    )
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

    # Pipeline inflight entries per active run.
    sorted_run_ids = sorted(active_run_ids)
    inflight_entries_list = (
        await _execute_status_pipeline(
            [
                (
                    "zrangebyscore",
                    (f"{token_key}:run:{rid}:inflight", "-inf", "+inf"),
                    {"withscores": True},
                )
                for rid in sorted_run_ids
            ]
        )
        if sorted_run_ids
        else []
    )

    all_inflight_members: list[tuple[str, float]] = []
    for entries in inflight_entries_list:
        # withscores=True returns list[tuple[str, float]] despite protocol typing
        all_inflight_members.extend(cast(list[tuple[str, float]], entries))

    # Pipeline all inflight metadata.
    inflight_metas = (
        await _execute_status_pipeline(
            [
                ("hgetall", (f"{token_key}:inflight:{member}",), {})
                for member, _ in all_inflight_members
            ]
        )
        if all_inflight_members
        else []
    )

    inflight: list[InflightRequest] = []
    for (member, score), meta in zip(all_inflight_members, inflight_metas):
        run_id_val = meta.get("run_id", "")
        inflight.append(
            InflightRequest(
                question_id=member.partition(":")[2] or member,
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

    # Pipeline all priority sets.
    priority_range = list(range(max_priority, min_priority + 1))
    priority_members_list = await _execute_status_pipeline(
        [
            (
                "zrangebyscore",
                (f"{base}:priority:{p}", "-inf", "+inf"),
                {"withscores": True},
            )
            for p in priority_range
        ]
    )

    all_priority_members: list[str] = []
    priority_members_by_score: list[list[tuple[str, float]]] = []
    for members in priority_members_list:
        typed_members = cast(list[tuple[str, float]], members)
        priority_members_by_score.append(typed_members)
        all_priority_members.extend(member for member, _ in typed_members)

    # Pipeline priority member metadata.
    priority_metas = (
        await _execute_status_pipeline(
            [
                ("hgetall", (f"{token_key}:inflight:{member}",), {})
                for member in all_priority_members
            ]
        )
        if all_priority_members
        else []
    )

    priorities: dict[str, int] = {}
    raw_priorities: dict[str, int] = {}
    orphaned_priorities: dict[str, int] = {}
    stale_priorities: dict[str, int] = {}
    queued_by_run: dict[str, dict[str, int]] = {}
    meta_idx = 0
    stale_cutoff = now - priority_stale_age
    for priority, members in zip(priority_range, priority_members_by_score):
        p_str = str(priority)
        raw_priorities[p_str] = len(members)
        valid_count = 0
        orphaned_count = 0
        stale_count = 0
        for member, score in members:
            meta = priority_metas[meta_idx]
            meta_idx += 1
            if float(score) < stale_cutoff:
                stale_count += 1
                continue

            rid = meta.get("run_id", "")
            if not rid:
                orphaned_count += 1
                continue

            valid_count += 1
            queued_by_run.setdefault(rid, {})
            queued_by_run[rid][p_str] = queued_by_run[rid].get(p_str, 0) + 1

        priorities[p_str] = valid_count
        orphaned_priorities[p_str] = orphaned_count
        stale_priorities[p_str] = stale_count

    # Pipeline dynamic estimates.
    sorted_de_keys = sorted(dynamic_estimate_keys)
    de_values = (
        await _execute_status_pipeline(
            [("get", (de_key,), {}) for de_key in sorted_de_keys]
        )
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

    # reconstruct RateLimit from last_header hash
    last_header: RateLimit | None = None
    if last_header_raw:

        def _parse_optional_int(v: str | None) -> int | None:
            return int(v) if v is not None and v != "None" else None

        last_header = RateLimit(
            request_limit=_parse_optional_int(last_header_raw.get("request_limit")),
            request_remaining=_parse_optional_int(
                last_header_raw.get("request_remaining")
            ),
            token_limit=_parse_optional_int(last_header_raw.get("token_limit")),
            token_limit_input=_parse_optional_int(
                last_header_raw.get("token_limit_input")
            ),
            token_limit_output=_parse_optional_int(
                last_header_raw.get("token_limit_output")
            ),
            token_remaining=_parse_optional_int(last_header_raw.get("token_remaining")),
            token_remaining_input=_parse_optional_int(
                last_header_raw.get("token_remaining_input")
            ),
            token_remaining_output=_parse_optional_int(
                last_header_raw.get("token_remaining_output")
            ),
            unix_timestamp=float(last_header_raw["unix_timestamp"]),
            raw=None,
        )

    status = TokenRetryStatus(
        token_key=token_key,
        tokens_remaining=tokens_remaining,
        token_limit=token_limit,
        config=config,
        inflight=inflight,
        priorities=priorities,
        raw_priorities=raw_priorities,
        orphaned_priorities=orphaned_priorities,
        stale_priorities=stale_priorities,
        refill_alive=refill_alive,
        correction_alive=correction_alive,
        active_benchmark_run=active_benchmark_run,
        active_benchmark_runs=active_benchmark_runs,
        dynamic_estimates=dynamic_estimates,
        last_header=last_header,
    )
    return status, queued_by_run


def _make_queue_entry(
    *,
    run_id: str,
    alive: bool,
    heartbeat_ttl: int,
    position: int,
    notified: bool,
    total_requests: int | None,
    dispatched_count: int,
    inflight_count: int,
    queued_by_priority: dict[str, int],
    slot_acquired: bool,
    is_active_head: bool,
    enqueued_at: float | None,
    slot_acquired_at: float | None,
    popped_at: float | None,
    completed_at: float | None,
    popped: bool = False,
    is_queued: bool = True,
) -> QueueEntry:
    has_live_queued_requests = any(count > 0 for count in queued_by_priority.values())
    has_live_inflight_requests = inflight_count > 0
    is_historical = popped and completed_at is not None

    if is_historical and (has_live_queued_requests or has_live_inflight_requests):
        display_state = "HISTORY_WITH_LIVE_TOKEN_STATE"
    elif is_historical and slot_acquired:
        display_state = "HISTORY_DONE"
    elif is_historical:
        display_state = "HISTORY_RELEASED"
    elif not is_queued:
        display_state = "DIRECT"
    elif popped:
        display_state = "POPPED"
    elif is_active_head:
        display_state = "ACTIVE_HEAD"
    elif has_live_inflight_requests:
        display_state = "STRAGGLER"
    elif has_live_queued_requests:
        display_state = "WAITING_FOR_TOKENS"
    elif alive:
        display_state = "QUEUE_WAITING"
    else:
        display_state = "QUEUE_DEAD"

    return QueueEntry(
        run_id=run_id,
        alive=alive,
        heartbeat_ttl=heartbeat_ttl,
        position=position,
        notified=notified,
        total_requests=total_requests,
        dispatched_count=dispatched_count,
        inflight_count=inflight_count,
        queued_by_priority=queued_by_priority,
        slot_acquired=slot_acquired,
        is_active_head=is_active_head,
        enqueued_at=enqueued_at,
        slot_acquired_at=slot_acquired_at,
        popped_at=popped_at,
        completed_at=completed_at,
        popped=popped,
        is_queued=is_queued,
        display_state=display_state,
        is_historical=is_historical,
        has_live_queued_requests=has_live_queued_requests,
        has_live_inflight_requests=has_live_inflight_requests,
    )


async def _get_queue_entries(
    base_key: str,
    run_ids: list[str],
    inflight_by_run: dict[str, int],
    queued_by_run: dict[str, dict[str, int]],
    active_heads: set[str],
) -> list[QueueEntry]:
    """Fetch queue entries using chunked pipelines instead of connection fan-out."""
    commands: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
    for run_id in run_ids:
        alive_key = f"{base_key}:alive:{run_id}"
        notify_key = f"{base_key}:notify:{run_id}"
        run_meta_key = f"{base_key}:run:{run_id}"
        commands.extend(
            [
                ("ttl", (alive_key,), {}),
                ("exists", (notify_key,), {}),
                ("hgetall", (run_meta_key,), {}),
                ("scard", (f"{run_meta_key}:dispatched",), {}),
            ]
        )

    raw_results = await _execute_status_pipeline(commands) if commands else []
    entries: list[QueueEntry] = []
    for position, run_id in enumerate(run_ids):
        offset = position * 4
        ttl_raw, notified_raw, meta, dispatched_count = raw_results[offset : offset + 4]
        alive = ttl_raw >= 0 or ttl_raw == -1
        total_raw = meta.get("total_requests", "0")
        entries.append(
            _make_queue_entry(
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
                is_active_head=run_id in active_heads,
                enqueued_at=float(meta["enqueued_at"])
                if "enqueued_at" in meta
                else None,
                slot_acquired_at=float(meta["slot_acquired_at"])
                if "slot_acquired_at" in meta
                else None,
                popped_at=float(meta["popped_at"]) if "popped_at" in meta else None,
                completed_at=float(meta["completed_at"])
                if "completed_at" in meta
                else None,
            )
        )
    return entries


async def _get_queue_status(
    inflight_by_run: dict[str, int],
    queued_by_run: dict[str, dict[str, int]],
    token_base_keys: list[str] | None = None,
    *,
    queue_entry_limit: int | None = None,
    include_historical: bool = True,
) -> list[QueueStatus]:
    queue_keys = sorted(await scan_keys(f"{KEY_PREFIX}:*:benchmark:queue"))

    seen_base_keys: set[str] = set()

    # Pipeline queue lengths and only the visible queue prefix. High-frequency
    # platform polling should not read every queued run just to render the first
    # page; callers that need a full diagnostic dump can pass queue_entry_limit=None.
    queue_lengths = (
        await _execute_status_pipeline([("llen", (qk,), {}) for qk in queue_keys])
        if queue_keys
        else []
    )
    queue_end = -1 if queue_entry_limit is None else max(queue_entry_limit - 1, -1)
    run_ids_per_queue = (
        await _execute_status_pipeline(
            [("lrange", (qk, 0, queue_end), {}) for qk in queue_keys]
        )
        if queue_keys and queue_end != -1
        else (
            await _execute_status_pipeline(
                [("lrange", (qk, 0, -1), {}) for qk in queue_keys]
            )
            if queue_keys and queue_entry_limit is None
            else []
        )
    )

    queue_control_commands: list[StatusPipelineCommand] = []
    for qk in queue_keys:
        benchmark_base = qk.removesuffix(":queue")
        queue_control_commands.append(
            ("zrange", (f"{benchmark_base}:active_heads", 0, -1), {})
        )
        queue_control_commands.append(
            ("get", (f"{benchmark_base}:active_head_window",), {})
        )
    queue_control_results = (
        await _execute_status_pipeline(queue_control_commands)
        if queue_control_commands
        else []
    )

    active_heads_by_queue: dict[int, list[str]] = {}
    active_window_by_queue: dict[int, int] = {}
    for q_idx, _ in enumerate(queue_keys):
        active_heads_by_queue[q_idx] = cast(list[str], queue_control_results[q_idx * 2])
        window_raw = queue_control_results[(q_idx * 2) + 1]
        active_window_by_queue[q_idx] = int(window_raw) if window_raw else 1

    entries_by_queue: dict[int, list[QueueEntry]] = {}
    for q_idx, (run_queue_key, run_ids) in enumerate(
        zip(queue_keys, run_ids_per_queue)
    ):
        base_key = run_queue_key.removesuffix(":queue")
        seen_base_keys.add(base_key)
        entries_by_queue[q_idx] = await _get_queue_entries(
            base_key,
            run_ids,
            inflight_by_run,
            queued_by_run,
            set(active_heads_by_queue.get(q_idx, [])),
        )

    statuses: list[QueueStatus] = []
    for q_idx, (run_queue_key, run_ids) in enumerate(
        zip(queue_keys, run_ids_per_queue)
    ):
        entries = entries_by_queue.get(q_idx, [])
        if entries:
            statuses.append(
                QueueStatus(
                    queue_key=run_queue_key,
                    length=cast(int, queue_lengths[q_idx]),
                    entries=entries,
                    active_head_window=active_window_by_queue.get(q_idx, 1),
                    active_heads=active_heads_by_queue.get(q_idx, []),
                )
            )

    if not include_historical:
        return statuses

    # discover popped/completed runs from metadata hashes (survives queue LIST deletion)
    covered_run_ids: set[str] = set()
    for s in statuses:
        for e in s.entries:
            covered_run_ids.add(e.run_id)

    all_benchmark_bases = set(seen_base_keys)
    if token_base_keys:
        for base_key in token_base_keys:
            all_benchmark_bases.add(f"{base_key}:benchmark")

    # discover popped/completed runs from metadata hashes (survives queue LIST deletion)
    sorted_bases = sorted(all_benchmark_bases)
    run_meta_keys_per_base = (
        await asyncio.gather(*(scan_keys(f"{bb}:run:*") for bb in sorted_bases))
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

    popped_commands: list[StatusPipelineCommand] = []
    for _, _, rmk in popped_lookups:
        popped_commands.append(("hgetall", (rmk,), {}))
        popped_commands.append(("scard", (f"{rmk}:dispatched",), {}))

    # Pipeline metadata + dispatched count for all popped runs.
    popped_raw_results = (
        await _execute_status_pipeline(popped_commands) if popped_commands else []
    )

    popped_by_base: dict[str, list[QueueEntry]] = {}
    for index, (benchmark_base, run_id, _) in enumerate(popped_lookups):
        meta = popped_raw_results[index * 2]
        dispatched_count = popped_raw_results[index * 2 + 1]
        if not meta:
            continue
        total_raw = meta.get("total_requests", "0")
        entry = _make_queue_entry(
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
            is_active_head=False,
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
