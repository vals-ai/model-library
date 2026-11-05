import asyncio
import logging
from collections import defaultdict
from datetime import UTC, datetime
from typing import Protocol, cast, overload
from uuid import UUID, uuid4

from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError
from sqlalchemy import and_, func, select
from sqlalchemy.exc import DBAPIError, TimeoutError as SQLAlchemyTimeoutError
from sqlalchemy.ext.asyncio import AsyncEngine

from model_gateway.managed_auth import tables as t
from model_gateway.managed_auth.reconciliation import HumanReconciler
from model_gateway.managed_auth.runtime import (
    RuntimeAuthority,
    RuntimeProjectLimit,
    TemporaryProjectIncrease,
    authority_hash_key,
    serialize_authority,
)
from model_gateway.managed_auth.types import (
    HumanIdentity,
    KeyLimit,
    PersonalOwner,
    ProjectReference,
    ServiceOwner,
)
from model_gateway.run_token_types import ModelPermissions

logger = logging.getLogger(__name__)
_OWNER_TTL_SECONDS = 60
_RENEW_SECONDS = 20
_PUBLICATION_SECONDS = 60
_STAGING_TTL_SECONDS = 3600
_MARKER_FIELD = "__published__"

_STAGE_SCRIPT = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
redis.call('HSET', KEYS[2], '__published__', 'staging')
redis.call('EXPIRE', KEYS[2], ARGV[2])
return 1
"""

_COMMIT_SCRIPT = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
if redis.call('HGET', KEYS[2], '__published__') ~= ARGV[2] then return -1 end
if redis.call('HLEN', KEYS[2]) ~= tonumber(ARGV[3]) then return -1 end
redis.call('PERSIST', KEYS[2])
redis.call('RENAME', KEYS[2], KEYS[3])
return 1
"""

_RENEW_SCRIPT = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
redis.call('EXPIRE', KEYS[1], ARGV[2])
return 1
"""

_RELEASE_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then redis.call('DEL', KEYS[1]) end
"""


class AuthorityPublisherRedis(Protocol):
    async def set(self, name: str, value: str, *, nx: bool, ex: int) -> bool | None: ...

    async def eval(
        self, script: str, numkeys: int, *keys_and_args: str | int
    ) -> int | None: ...

    @overload
    async def hset(self, name: str, key: str, value: str) -> int: ...

    @overload
    async def hset(self, name: str, *, mapping: dict[str, str]) -> int: ...

    async def delete(self, name: str) -> int: ...


class ManagedAuthorityPublisher:
    def __init__(
        self,
        engine: AsyncEngine,
        redis_client: Redis,
        reconciler: HumanReconciler,
        *,
        namespace: str,
    ):
        self.engine = engine
        self.redis: AuthorityPublisherRedis = cast(
            AuthorityPublisherRedis, redis_client
        )
        self.reconciler = reconciler
        self._active_key = authority_hash_key(namespace)
        self._owner_key = f"gateway:managed-auth:{namespace}:owner"
        self._stage_prefix = f"gateway:managed-auth:{namespace}:staging:"
        self._stop = asyncio.Event()

    async def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        while not self._stop.is_set():
            try:
                token = uuid4().hex
                acquired = await self.redis.set(
                    self._owner_key, token, nx=True, ex=_OWNER_TTL_SECONDS
                )
                if acquired:
                    try:
                        await self._serve_owner(token)
                    finally:
                        await self.redis.eval(
                            _RELEASE_SCRIPT, 1, self._owner_key, token
                        )
                else:
                    await self._sleep(10)
            except (
                DBAPIError,
                SQLAlchemyTimeoutError,
                RedisConnectionError,
                RedisTimeoutError,
            ):
                logger.exception("Managed authority owner operation failed")
                await self._sleep(_PUBLICATION_SECONDS)

    async def _serve_owner(self, token: str) -> None:
        owner_stop = asyncio.Event()
        reconciliation = asyncio.create_task(self.reconciler.run(owner_stop))
        renewal = asyncio.create_task(
            self._renew_owner(token, owner_stop, reconciliation)
        )
        try:
            while not self._stop.is_set():
                if renewal.done():
                    await renewal
                    break
                try:
                    await self.publish_once(token)
                except (
                    DBAPIError,
                    SQLAlchemyTimeoutError,
                    RedisConnectionError,
                    RedisTimeoutError,
                ):
                    logger.exception("Managed authority publication failed")
                if not await self._wait_for_next_pass(renewal):
                    break
        finally:
            owner_stop.set()
            renewal.cancel()
            reconciliation.cancel()
            await asyncio.gather(renewal, reconciliation, return_exceptions=True)

    async def _renew_owner(
        self,
        token: str,
        owner_stop: asyncio.Event,
        reconciliation: asyncio.Task[None],
    ) -> None:
        try:
            while not owner_stop.is_set() and not self._stop.is_set():
                try:
                    await asyncio.wait_for(owner_stop.wait(), timeout=_RENEW_SECONDS)
                    return
                except TimeoutError:
                    pass
                if not await self.redis.eval(
                    _RENEW_SCRIPT, 1, self._owner_key, token, _OWNER_TTL_SECONDS
                ):
                    return
        finally:
            owner_stop.set()
            reconciliation.cancel()

    async def _wait_for_next_pass(self, renewal: asyncio.Task[None]) -> bool:
        stop_wait = asyncio.create_task(self._stop.wait())
        try:
            done, _ = await asyncio.wait(
                (stop_wait, renewal),
                timeout=_PUBLICATION_SECONDS,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if renewal in done:
                await renewal
                return False
            return not self._stop.is_set()
        finally:
            stop_wait.cancel()
            await asyncio.gather(stop_wait, return_exceptions=True)

    async def _sleep(self, seconds: int) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except TimeoutError:
            pass

    async def project(self) -> list[tuple[str, RuntimeAuthority]]:
        async with self.engine.connect() as connection:
            await connection.execution_options(isolation_level="REPEATABLE READ")
            async with connection.begin():
                keys = (
                    await connection.execute(
                        select(
                            t.keys.c.id,
                            t.keys.c.verifier_hash,
                            t.keys.c.owner_kind,
                            t.keys.c.owner_issuer,
                            t.keys.c.owner_subject,
                            t.keys.c.service_id,
                            t.keys.c.project_id,
                            t.keys.c.key_limit_period,
                            t.keys.c.key_limit_usd,
                            t.project_policies.c.project_organization,
                            t.project_policies.c.project_environment,
                            t.project_policies.c.period,
                            t.project_policies.c.base_limit_usd,
                            func.coalesce(
                                t.project_policies.c.total_limit_usd,
                                t.project_policies.c.base_limit_usd,
                            ).label("total_limit_usd"),
                            t.humans.c.enabled,
                        )
                        .select_from(
                            t.keys.join(
                                t.project_policies,
                                t.keys.c.project_id == t.project_policies.c.project_id,
                            ).outerjoin(
                                t.humans,
                                and_(
                                    t.keys.c.owner_issuer == t.humans.c.issuer,
                                    t.keys.c.owner_subject == t.humans.c.subject,
                                ),
                            )
                        )
                        .where(t.keys.c.status == "active")
                    )
                ).all()
                temporary_rows = (
                    await connection.execute(
                        select(
                            t.project_limit_requests.c.project_id,
                            t.project_limit_requests.c.period_start,
                            t.project_limit_requests.c.period_end,
                            t.project_limit_requests.c.increase_usd,
                        )
                        .where(
                            t.project_limit_requests.c.kind == "temporary",
                            t.project_limit_requests.c.status == "approved",
                            t.project_limit_requests.c.cancelled_at.is_(None),
                        )
                        .order_by(
                            t.project_limit_requests.c.project_id,
                            t.project_limit_requests.c.period_start,
                            t.project_limit_requests.c.id,
                        )
                    )
                ).all()

        temporary_by_project: dict[UUID, list[TemporaryProjectIncrease]] = defaultdict(
            list
        )
        for row in temporary_rows:
            temporary_by_project[row.project_id].append(
                TemporaryProjectIncrease(
                    increase_usd=row.increase_usd,
                    period_start=row.period_start,
                    period_end=row.period_end,
                )
            )
        projection: list[tuple[str, RuntimeAuthority]] = []
        for row in keys:
            if row.owner_kind == "personal":
                if not row.enabled:
                    continue
                owner = PersonalOwner(
                    HumanIdentity(row.owner_issuer, row.owner_subject)
                )
            else:
                owner = ServiceOwner(row.service_id)

            project = ProjectReference(
                row.project_id,
                row.project_organization,
                row.project_environment,
            )
            project_limit = RuntimeProjectLimit(
                period=row.period,
                base_limit_usd=row.base_limit_usd,
                temporary_increases=tuple(temporary_by_project[row.project_id]),
                total_limit_usd=row.total_limit_usd,
            )
            key_limit = (
                KeyLimit(period=row.key_limit_period, limit_usd=row.key_limit_usd)
                if row.key_limit_period is not None
                else None
            )
            authority = RuntimeAuthority(
                key_id=row.id,
                owner=owner,
                permissions=ModelPermissions(
                    allowed_models=None,
                    allowed_embedding_models=[],
                    allowed_operations=None,
                ),
                project=project,
                project_limit=project_limit,
                key_limit=key_limit,
            )
            projection.append((row.verifier_hash, authority))
        return projection

    async def publish_once(self, token: str) -> bool:
        projection = await self.project()
        generation = uuid4().hex
        staging_key = f"{self._stage_prefix}{generation}"
        marker = f"v1:{generation}:{datetime.now(UTC).isoformat()}"
        if not await self.redis.eval(
            _STAGE_SCRIPT,
            2,
            self._owner_key,
            staging_key,
            token,
            _STAGING_TTL_SECONDS,
        ):
            return False
        published = False
        try:
            fields: dict[str, str] = {}
            for verifier, authority in projection:
                record = serialize_authority(authority)
                fields[f"v:{verifier}"] = record
                fields[f"k:{authority.key_id}"] = record
                if len(fields) >= 500:
                    await self.redis.hset(staging_key, mapping=fields)
                    fields.clear()
            if fields:
                await self.redis.hset(staging_key, mapping=fields)
            await self.redis.hset(staging_key, _MARKER_FIELD, marker)
            result = await self.redis.eval(
                _COMMIT_SCRIPT,
                3,
                self._owner_key,
                staging_key,
                self._active_key,
                token,
                marker,
                1 + 2 * len(projection),
            )
            if result == -1:
                raise RuntimeError("Managed authority staging hash is incomplete")
            published = result == 1
            if published:
                logger.info("Published managed authority generation %s", generation)
            return published
        finally:
            if not published:
                await self.redis.delete(staging_key)
