"""Atomic Redis state for the shared rate-limit monitor."""

from __future__ import annotations

import math
import secrets
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

from pydantic import ValidationError

from model_gateway.rate_limit_monitor.types import (
    MonitorActivationResponse,
    MonitorErrorCode,
    MonitorFacts,
    MonitorListResponse,
    MonitorSourceFacts,
    MonitorSourceName,
    MonitorSourceState,
    MonitorState,
    derive_monitor_status,
)
from model_library.rate_limits import RateLimit

MONITOR_KEY_PREFIX = "model_gateway:rate_limit_monitor"
ACTIVE_KEY = f"{MONITOR_KEY_PREFIX}:active"
RETAINED_KEY = f"{MONITOR_KEY_PREFIX}:retained"
OWNER_KEY_PREFIX = f"{MONITOR_KEY_PREFIX}:owner:"
SNAPSHOT_KEY_PREFIX = f"{MONITOR_KEY_PREFIX}:snapshot:"
ACTIVATION_SECONDS = 10 * 60
RETENTION_SECONDS = 24 * 60 * 60
OWNER_TTL_MILLISECONDS = 60 * 1000
_CAS_RETRIES = 4


class MonitorRedis(Protocol):
    async def eval(
        self,
        script: str,
        numkeys: int,
        *keys_and_args: str | int | float,
    ) -> Any: ...


class MonitorStateCorrupt(RuntimeError):
    pass


@dataclass(frozen=True)
class MonitorSourceUpdate:
    source: MonitorSourceName
    status: Literal["ok", "unsupported", "error"]
    rate_limit: RateLimit | None = None


@dataclass(frozen=True)
class _StoredSnapshot:
    now: float
    active_score: str
    retained_score: str
    retention_until_milliseconds: int
    raw: str
    facts: MonitorFacts


_REDIS_TIME_LUA = r"""
local function redis_time()
  local value = redis.call("TIME")
  return tonumber(value[1]) + tonumber(value[2]) / 1000000
end
"""

_INDEX_STRUCTURE_LUA = r"""
local function indexes_are_structurally_consistent(active_key, retained_key, snapshot_prefix)
  local active = redis.call("ZRANGE", active_key, 0, -1)
  for _, model in ipairs(active) do
    if not redis.call("ZSCORE", retained_key, model)
        or not redis.call("GET", snapshot_prefix .. model) then
      return false
    end
  end
  local retained = redis.call("ZRANGE", retained_key, 0, -1)
  for _, model in ipairs(retained) do
    if not redis.call("ZSCORE", active_key, model)
        or not redis.call("GET", snapshot_prefix .. model) then
      return false
    end
  end
  return true
end
"""

_TIME_SCRIPT = _REDIS_TIME_LUA + "return {tostring(redis_time())}"

_READ_SCRIPT = (
    _REDIS_TIME_LUA
    + r"""
local now = redis_time()
local active_score = redis.call("ZSCORE", KEYS[1], ARGV[1]) or ""
local retained_score = redis.call("ZSCORE", KEYS[2], ARGV[1]) or ""
if retained_score ~= "" and tonumber(retained_score) <= now then
  redis.call("ZREM", KEYS[1], ARGV[1])
  redis.call("ZREM", KEYS[2], ARGV[1])
  redis.call("DEL", KEYS[3])
  return {tostring(now), "", "", ""}
end
return {
  tostring(now),
  active_score,
  retained_score,
  redis.call("GET", KEYS[3]) or ""
}
"""
)

_ACTIVATE_COMMIT_SCRIPT = (
    _REDIS_TIME_LUA
    + r"""
local now = redis_time()
local current_raw = redis.call("GET", KEYS[4]) or ""
local active_score = redis.call("ZSCORE", KEYS[1], ARGV[1]) or ""
local retained_score = redis.call("ZSCORE", KEYS[2], ARGV[1]) or ""
if current_raw ~= ARGV[2] or active_score ~= ARGV[3] or retained_score ~= ARGV[4] then
  return {"conflict"}
end
local was_active = active_score ~= "" and tonumber(active_score) > now
if was_active ~= (ARGV[9] == "1") then return {"conflict"} end
if tonumber(ARGV[6]) <= now or tonumber(ARGV[7]) <= tonumber(ARGV[6]) then
  return {"conflict"}
end
if not was_active or ARGV[10] == "1" then redis.call("DEL", KEYS[3]) end
redis.call("ZADD", KEYS[1], ARGV[6], ARGV[1])
redis.call("ZADD", KEYS[2], ARGV[7], ARGV[1])
redis.call("SET", KEYS[4], ARGV[5])
redis.call("PEXPIREAT", KEYS[4], ARGV[8])
return {"ok", tostring(now)}
"""
)

_PUBLISH_COMMIT_SCRIPT = (
    _REDIS_TIME_LUA
    + r"""
local now = redis_time()
if redis.call("GET", KEYS[3]) ~= ARGV[2] then return {"rejected"} end
local active_score = redis.call("ZSCORE", KEYS[1], ARGV[1]) or ""
if active_score == "" or tonumber(active_score) <= now then return {"rejected"} end
local current_raw = redis.call("GET", KEYS[4]) or ""
local retained_score = redis.call("ZSCORE", KEYS[2], ARGV[1]) or ""
if current_raw ~= ARGV[3] or active_score ~= ARGV[4] or retained_score ~= ARGV[5] then
  return {"conflict"}
end
if retained_score == "" or tonumber(retained_score) <= now then return {"rejected"} end
redis.call("SET", KEYS[4], ARGV[6])
redis.call("PEXPIREAT", KEYS[4], ARGV[7])
return {"ok"}
"""
)

_LIST_SCRIPT = (
    _REDIS_TIME_LUA
    + _INDEX_STRUCTURE_LUA
    + r"""
local now = redis_time()
local expired = redis.call("ZRANGEBYSCORE", KEYS[2], "-inf", now)
for _, model in ipairs(expired) do
  redis.call("ZREM", KEYS[1], model)
  redis.call("ZREM", KEYS[2], model)
  redis.call("DEL", ARGV[1] .. model)
end
if not indexes_are_structurally_consistent(KEYS[1], KEYS[2], ARGV[1]) then
  return {"corrupt"}
end
local retained = redis.call("ZRANGE", KEYS[2], 0, -1, "WITHSCORES")
local result = {tostring(now)}
for index = 1, #retained, 2 do
  local model = retained[index]
  table.insert(result, model)
  table.insert(result, retained[index + 1])
  table.insert(result, redis.call("ZSCORE", KEYS[1], model) or "")
  table.insert(result, redis.call("GET", ARGV[1] .. model) or "")
end
return result
"""
)

_DISCOVER_SCRIPT = (
    _REDIS_TIME_LUA
    + r"""
local now = redis_time()
local expired = redis.call("ZRANGEBYSCORE", KEYS[2], "-inf", now)
for _, model in ipairs(expired) do
  redis.call("ZREM", KEYS[1], model)
  redis.call("ZREM", KEYS[2], model)
  redis.call("DEL", ARGV[1] .. model)
end
return redis.call("ZRANGEBYSCORE", KEYS[1], "(" .. tostring(now), "+inf")
"""
)

_CLAIM_SCRIPT = (
    _REDIS_TIME_LUA
    + r"""
local now = redis_time()
local active_score = redis.call("ZSCORE", KEYS[1], ARGV[1]) or ""
local raw = redis.call("GET", KEYS[3]) or ""
local retained_score = redis.call("ZSCORE", KEYS[4], ARGV[1]) or ""
if raw ~= ARGV[4] or active_score ~= ARGV[5] or retained_score ~= ARGV[6] then
  return {"conflict"}
end
if active_score == "" or tonumber(active_score) <= now then return {"lost"} end
local owner = redis.call("GET", KEYS[2])
if owner == ARGV[2] then return {"ok", tostring(now)} end
if owner then return {"lost"} end
if not redis.call("SET", KEYS[2], ARGV[2], "NX", "PX", ARGV[3]) then
  return {"lost"}
end
return {"ok", tostring(now)}
"""
)

_RENEW_SCRIPT = (
    _REDIS_TIME_LUA
    + r"""
local now = redis_time()
if redis.call("GET", KEYS[2]) ~= ARGV[2] then return {"lost"} end
local active_score = redis.call("ZSCORE", KEYS[1], ARGV[1]) or ""
local raw = redis.call("GET", KEYS[3]) or ""
local retained_score = redis.call("ZSCORE", KEYS[4], ARGV[1]) or ""
if raw ~= ARGV[4] or active_score ~= ARGV[5] or retained_score ~= ARGV[6] then
  return {"conflict"}
end
if active_score == "" or tonumber(active_score) <= now then
  redis.call("DEL", KEYS[2])
  return {"lost"}
end
redis.call("PEXPIRE", KEYS[2], ARGV[3])
return {"ok", tostring(math.floor(now * 1000) / 1000)}
"""
)

_RELEASE_SCRIPT = r"""
if redis.call("GET", KEYS[1]) == ARGV[1] then redis.call("DEL", KEYS[1]) end
"""


def owner_key(model: str) -> str:
    return f"{OWNER_KEY_PREFIX}{model}"


def snapshot_key(model: str) -> str:
    return f"{SNAPSHOT_KEY_PREFIX}{model}"


def _redis_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise MonitorStateCorrupt("Redis monitor scalar is invalid") from exc
    if not math.isfinite(number) or number < 0:
        raise MonitorStateCorrupt("Redis monitor scalar is invalid")
    return number


def _parse_snapshot(value: str) -> MonitorFacts:
    try:
        return MonitorFacts.model_validate_json(value)
    except ValidationError as exc:
        raise MonitorStateCorrupt("Redis monitor snapshot is invalid") from exc


def _snapshot_json(facts: MonitorFacts) -> str:
    return facts.model_dump_json(exclude_computed_fields=True)


def _canonical_milliseconds(value: float) -> int:
    milliseconds = value * 1000
    if not math.isfinite(milliseconds):
        raise MonitorStateCorrupt("Redis monitor timestamp is invalid")
    rounded = round(milliseconds)
    if not math.isclose(milliseconds, rounded, rel_tol=0, abs_tol=1e-6):
        raise MonitorStateCorrupt(
            "Redis monitor timestamp is not millisecond canonical"
        )
    return rounded


def _validate_snapshot_indexes(
    facts: MonitorFacts,
    model: str,
    now: float,
    active_score: str,
    retained_score: str,
) -> int:
    if facts.model != model or not active_score or not retained_score:
        raise MonitorStateCorrupt("Redis monitor snapshot indexes are invalid")
    active_until = _redis_float(active_score)
    retained_until = _redis_float(retained_score)
    active_milliseconds = _canonical_milliseconds(active_until)
    retained_milliseconds = _canonical_milliseconds(retained_until)
    if (
        retained_until <= now
        or retained_milliseconds != active_milliseconds + RETENTION_SECONDS * 1000
    ):
        raise MonitorStateCorrupt("Redis monitor snapshot indexes are invalid")
    return retained_milliseconds


def _source_state_from_facts(
    facts: MonitorSourceFacts,
    generation: str,
    active: bool,
) -> MonitorSourceState:
    has_last_good = facts.last_success_at is not None and facts.rate_limit is not None
    attempted_in_generation = facts.last_attempt_generation == generation
    if not attempted_in_generation and not has_last_good:
        return MonitorSourceState(source=facts.source, status="starting")

    if not attempted_in_generation:
        status: Literal["stale", "unsupported", "error", "ok"] = "stale"
    elif facts.last_error_code is None:
        status = "ok" if active else "stale"
    elif has_last_good:
        status = "stale"
    elif facts.last_error_code == "unsupported":
        status = "unsupported"
    else:
        status = "error"

    return MonitorSourceState(
        source=facts.source,
        status=status,
        last_attempt_at=facts.last_attempt_at,
        last_success_at=facts.last_success_at,
        rate_limit=facts.rate_limit,
        error_code=facts.last_error_code,
    )


def _state_from_snapshot(
    facts: MonitorFacts,
    server_time: float,
    active_score: str,
    retained_score: str,
) -> MonitorState:
    active_until = _redis_float(active_score)
    retention_until = _redis_float(retained_score)
    active = active_until > server_time
    sources = [
        _source_state_from_facts(source, facts.generation, active)
        for source in facts.sources
    ]
    return MonitorState(
        model=facts.model,
        active=active,
        active_until=active_until,
        retention_until=retention_until,
        status=derive_monitor_status(sources),
        sources=sources,
    )


class RateLimitMonitorStore:
    def __init__(self, redis_client: MonitorRedis):
        self.redis = redis_client

    async def _read_snapshot(self, model: str) -> _StoredSnapshot | None:
        now_raw, active_score_raw, retained_score_raw, snapshot_raw = cast(
            list[str],
            await self.redis.eval(
                _READ_SCRIPT,
                3,
                ACTIVE_KEY,
                RETAINED_KEY,
                snapshot_key(model),
                model,
            ),
        )
        now = _redis_float(now_raw)
        active_score = active_score_raw
        retained_score = retained_score_raw
        if not snapshot_raw:
            if active_score or retained_score:
                raise MonitorStateCorrupt("Redis monitor snapshot is missing")
            return None
        facts = _parse_snapshot(snapshot_raw)
        retention_until_milliseconds = _validate_snapshot_indexes(
            facts,
            model,
            now,
            active_score,
            retained_score,
        )
        return _StoredSnapshot(
            now,
            active_score,
            retained_score,
            retention_until_milliseconds,
            snapshot_raw,
            facts,
        )

    async def activate(
        self,
        model: str,
        expected_sources: tuple[MonitorSourceName, ...],
    ) -> MonitorActivationResponse:
        for _ in range(_CAS_RETRIES):
            stored = await self._read_snapshot(model)
            previous_facts = stored.facts if stored is not None else None
            now = stored.now if stored is not None else await self._server_time()
            source_set_changed = (
                previous_facts is not None
                and tuple(source.source for source in previous_facts.sources)
                != expected_sources
            )

            previous_active_until = (
                _redis_float(stored.active_score) if stored is not None else 0
            )
            was_active = previous_active_until > now
            generation = (
                previous_facts.generation
                if was_active and previous_facts is not None and not source_set_changed
                else secrets.token_hex(16)
            )
            active_until_milliseconds = math.ceil(
                max(
                    now + ACTIVATION_SECONDS, previous_active_until if was_active else 0
                )
                * 1000
            )
            retention_until_milliseconds = (
                active_until_milliseconds + RETENTION_SECONDS * 1000
            )
            active_until = active_until_milliseconds / 1000
            retention_until = retention_until_milliseconds / 1000
            sources: list[MonitorSourceFacts] = []
            for index, source_name in enumerate(expected_sources):
                previous = (
                    previous_facts.sources[index]
                    if previous_facts is not None and not source_set_changed
                    else None
                )
                if previous is not None and (
                    was_active
                    or (
                        previous.last_success_at is not None
                        and previous.rate_limit is not None
                    )
                ):
                    sources.append(previous)
                else:
                    sources.append(MonitorSourceFacts(source=source_name))
            facts = MonitorFacts(
                generation=generation,
                model=model,
                sources=sources,
            )
            marker, *payload = cast(
                list[str],
                await self.redis.eval(
                    _ACTIVATE_COMMIT_SCRIPT,
                    4,
                    ACTIVE_KEY,
                    RETAINED_KEY,
                    owner_key(model),
                    snapshot_key(model),
                    model,
                    stored.raw if stored is not None else "",
                    stored.active_score if stored is not None else "",
                    stored.retained_score if stored is not None else "",
                    _snapshot_json(facts),
                    active_until,
                    retention_until,
                    retention_until_milliseconds,
                    "1" if was_active else "0",
                    "1" if source_set_changed else "0",
                ),
            )
            if marker == "conflict":
                continue
            assert marker == "ok"
            [server_time_raw] = payload
            server_time = _redis_float(server_time_raw)
            return MonitorActivationResponse(
                server_time=server_time,
                state=_state_from_snapshot(
                    facts,
                    server_time,
                    str(active_until),
                    str(retention_until),
                ),
            )
        raise MonitorStateCorrupt("Redis monitor activation conflicted repeatedly")

    async def _server_time(self) -> float:
        [server_time] = cast(list[str], await self.redis.eval(_TIME_SCRIPT, 0))
        return _redis_float(server_time)

    async def list_states(self) -> MonitorListResponse:
        now_raw, *rows = cast(
            list[str],
            await self.redis.eval(
                _LIST_SCRIPT, 2, ACTIVE_KEY, RETAINED_KEY, SNAPSHOT_KEY_PREFIX
            ),
        )
        if now_raw == "corrupt":
            raise MonitorStateCorrupt("Redis monitor indexes are invalid")
        now = _redis_float(now_raw)
        states: list[MonitorState] = []
        for index in range(0, len(rows), 4):
            model, retained_score_raw, active_score_raw, raw = rows[index : index + 4]
            retained_score = retained_score_raw
            active_score = active_score_raw
            facts = _parse_snapshot(raw)
            _validate_snapshot_indexes(facts, model, now, active_score, retained_score)
            states.append(
                _state_from_snapshot(facts, now, active_score, retained_score)
            )
        return MonitorListResponse(
            server_time=now, states=sorted(states, key=lambda state: state.model)
        )

    async def discover_active(self) -> set[str]:
        return set(
            cast(
                list[str],
                await self.redis.eval(
                    _DISCOVER_SCRIPT,
                    2,
                    ACTIVE_KEY,
                    RETAINED_KEY,
                    SNAPSHOT_KEY_PREFIX,
                ),
            )
        )

    async def _owner_operation(
        self,
        script: str,
        model: str,
        token: str,
    ) -> tuple[_StoredSnapshot, float] | None:
        for _ in range(_CAS_RETRIES):
            stored = await self._read_snapshot(model)
            if stored is None or _redis_float(stored.active_score) <= stored.now:
                return None
            marker, *payload = cast(
                list[str],
                await self.redis.eval(
                    script,
                    4,
                    ACTIVE_KEY,
                    owner_key(model),
                    snapshot_key(model),
                    RETAINED_KEY,
                    model,
                    token,
                    OWNER_TTL_MILLISECONDS,
                    stored.raw,
                    stored.active_score,
                    stored.retained_score,
                ),
            )
            if marker == "lost":
                return None
            if marker == "conflict":
                continue
            assert marker == "ok"
            [server_time] = payload
            return stored, _redis_float(server_time)
        raise MonitorStateCorrupt("Redis monitor lease conflicted repeatedly")

    async def claim_owner(
        self,
        model: str,
        token: str,
    ) -> tuple[MonitorSourceName, ...] | None:
        result = await self._owner_operation(_CLAIM_SCRIPT, model, token)
        if result is None:
            return None
        stored, _ = result
        return tuple(source.source for source in stored.facts.sources)

    async def renew_owner(self, model: str, token: str) -> tuple[str, float] | None:
        result = await self._owner_operation(_RENEW_SCRIPT, model, token)
        if result is None:
            return None
        stored, server_time = result
        return stored.facts.generation, server_time

    async def release_owner(self, model: str, token: str) -> None:
        await self.redis.eval(_RELEASE_SCRIPT, 1, owner_key(model), token)

    async def publish_source(
        self,
        model: str,
        token: str,
        captured_generation: str,
        attempted_at: float,
        source: MonitorSourceUpdate,
    ) -> bool:
        for _ in range(_CAS_RETRIES):
            stored = await self._read_snapshot(model)
            if stored is None or _redis_float(stored.active_score) <= stored.now:
                return False
            facts = stored.facts
            if facts.generation != captured_generation:
                return False
            position = [item.source for item in facts.sources].index(source.source)
            previous = facts.sources[position]
            if (
                previous.last_attempt_generation == captured_generation
                and previous.last_attempt_at == attempted_at
            ):
                update = previous
            elif source.status == "ok":
                update = MonitorSourceFacts(
                    source=source.source,
                    last_attempt_at=attempted_at,
                    last_attempt_generation=captured_generation,
                    last_success_at=attempted_at,
                    rate_limit=source.rate_limit,
                )
            else:
                error_code: MonitorErrorCode = (
                    "unsupported"
                    if source.status == "unsupported"
                    else "provider_error"
                )
                update = MonitorSourceFacts(
                    source=source.source,
                    last_attempt_at=attempted_at,
                    last_attempt_generation=captured_generation,
                    last_success_at=previous.last_success_at,
                    rate_limit=previous.rate_limit,
                    last_error_code=error_code,
                )
            sources = list(facts.sources)
            sources[position] = update
            next_facts = MonitorFacts(
                generation=facts.generation,
                model=facts.model,
                sources=sources,
            )
            [marker] = cast(
                list[str],
                await self.redis.eval(
                    _PUBLISH_COMMIT_SCRIPT,
                    4,
                    ACTIVE_KEY,
                    RETAINED_KEY,
                    owner_key(model),
                    snapshot_key(model),
                    model,
                    token,
                    stored.raw,
                    stored.active_score,
                    stored.retained_score,
                    _snapshot_json(next_facts),
                    stored.retention_until_milliseconds,
                ),
            )
            if marker == "ok":
                return True
            if marker == "rejected":
                return False
            if marker == "conflict":
                continue
            raise AssertionError(f"Unexpected publication result: {marker}")
        raise MonitorStateCorrupt("Redis monitor publication conflicted repeatedly")
