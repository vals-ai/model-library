"""Background transfer of recorded spend into a local query-time budget view."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import TYPE_CHECKING, Protocol, TypedDict, cast

import boto3
from botocore.exceptions import ClientError
from redis.asyncio import Redis

from model_gateway.budgets.bumps import effective_limit, grant_amount
from model_gateway.budgets.schema import BudgetPolicy, SpendSnapshot, period_bounds

if TYPE_CHECKING:
    from model_gateway.budgets.bump_service import BumpService


class BudgetS3Body(Protocol):
    def read(self) -> bytes: ...


class BudgetS3Response(TypedDict):
    Body: BudgetS3Body


class BudgetS3Client(Protocol):
    def get_object(self, *, Bucket: str, Key: str) -> BudgetS3Response: ...


class BudgetRedis(Protocol):
    async def get(self, name: str) -> bytes | str | None: ...

    async def eval(
        self, script: str, numkeys: int, *keys_and_args: str | int | bytes
    ) -> object: ...


logger = logging.getLogger("model_proxy_server.budgets")

REPLACE_IF_NEWER = """
local previous = redis.call('GET', KEYS[1])
if previous and tonumber(cjson.decode(previous).revision) >= tonumber(ARGV[1]) then
    return 0
end
redis.call('SET', KEYS[1], ARGV[2])
return 1
"""


class BudgetService:
    def __init__(
        self,
        *,
        stage: str,
        redis: Redis | None,
        bucket: str | None = None,
        key: str | None = None,
    ) -> None:
        self.stage = stage
        self.redis: BudgetRedis | None = cast(BudgetRedis | None, redis)
        self.bucket = bucket
        self.key = key
        self.redis_key = f"gateway:budgets:{stage}:spend"
        self.snapshot: SpendSnapshot | None = None
        self._s3: BudgetS3Client | None = None
        self.bump_service: BumpService | None = None

    def blocked(
        self, policy: BudgetPolicy | None, email: str | None, now: datetime
    ) -> bool:
        if policy is None or email is None or self.snapshot is None:
            return False
        if self.bump_service is not None and self.bump_service.snapshot is None:
            return False
        period = self.snapshot.periods[policy.period]
        start, end = period_bounds(policy.period, now)
        if period.start != start or period.end != end:
            return False
        limit = policy.email_overrides.get(email, policy.default_limit_usd)
        if self.bump_service is not None:
            snapshot = self.bump_service.snapshot
            if snapshot is not None:
                grant = snapshot.bumps.get(email)
                if (
                    grant is not None
                    and grant.period == policy.period
                    and grant.period_start == start
                    and grant.period_end == end
                ):
                    limit = effective_limit(limit, grant_amount(grant))
        return period.spend_usd.get(email, 0) >= limit

    async def refresh_control_once(self) -> None:
        if self.bucket is None or self.key is None or self.redis is None:
            return
        if self._s3 is None:
            self._s3 = await asyncio.to_thread(
                cast(Callable[[str], BudgetS3Client], boto3.client), "s3"
            )
        try:
            response = await asyncio.to_thread(
                self._s3.get_object, Bucket=self.bucket, Key=self.key
            )
        except ClientError as error:
            if error.response["Error"]["Code"] in {"NoSuchKey", "404"}:
                return
            raise
        payload = await asyncio.to_thread(response["Body"].read)
        snapshot = self._parse_snapshot(payload)
        await self.redis.eval(
            REPLACE_IF_NEWER, 1, self.redis_key, snapshot.revision, payload
        )
        await self.read_shared_snapshot()

    async def run_control(self, stop: asyncio.Event) -> None:
        await self._run(stop, self.refresh_control_once)

    async def refresh_query_once(self) -> None:
        await self.read_shared_snapshot()

    async def run_query(self, stop: asyncio.Event) -> None:
        await self._run(stop, self.refresh_query_once)

    async def read_shared_snapshot(self) -> SpendSnapshot | None:
        if self.redis is None:
            return self.snapshot
        payload = await self.redis.get(self.redis_key)
        if payload is not None:
            self._retain_snapshot(self._parse_snapshot(payload))
        return self.snapshot

    def _parse_snapshot(self, payload: bytes | str) -> SpendSnapshot:
        snapshot = SpendSnapshot.model_validate_json(payload)
        if snapshot.stage != self.stage:
            raise ValueError("budget snapshot stage does not match gateway stage")
        return snapshot

    def _retain_snapshot(self, snapshot: SpendSnapshot) -> None:
        if self.snapshot is None or snapshot.revision >= self.snapshot.revision:
            self.snapshot = snapshot

    async def _run(
        self, stop: asyncio.Event, refresh: Callable[[], Awaitable[None]]
    ) -> None:
        while not stop.is_set():
            try:
                await refresh()
            except Exception:
                logger.exception("Budget snapshot refresh failed stage=%s", self.stage)
            try:
                await asyncio.wait_for(stop.wait(), timeout=5)
            except TimeoutError:
                pass
