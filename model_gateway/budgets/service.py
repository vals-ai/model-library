"""Managed spend admission from Redis and control-side S3 snapshot publication."""

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Protocol, TypedDict, cast
from uuid import UUID

import boto3
from botocore.exceptions import ClientError
from pydantic import ValidationError
from redis.asyncio import Redis
from redis.exceptions import RedisError

from model_gateway.budgets.schema import (
    BudgetPeriod,
    SpendSnapshot,
    managed_key_spend_id,
    period_bounds,
)
from model_gateway.managed_auth.runtime import RuntimeAuthority


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


def _current_spend(
    snapshot: SpendSnapshot, period: BudgetPeriod, spend_id: str, now: datetime
) -> Decimal:
    """Recorded spend this period; $0 until the snapshot covers the period or scope."""
    recorded = snapshot.periods[period]
    if (recorded.start, recorded.end) != period_bounds(period, now):
        return Decimal(0)
    return recorded.spend_usd.get(spend_id, Decimal(0))


class BudgetDecision(StrEnum):
    ALLOWED = "allowed"
    EXHAUSTED = "exhausted"
    KEY_EXHAUSTED = "key_exhausted"
    MISSING = "missing"
    UNREADABLE = "unreadable"


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
        self.redis_key = f"gateway:budgets:{stage}:managed-spend-v2"
        self._s3: BudgetS3Client | None = None

    async def managed_decision(
        self,
        authority: RuntimeAuthority,
        now: datetime,
    ) -> BudgetDecision:
        if self.redis is None:
            return BudgetDecision.UNREADABLE
        try:
            payload = await self.redis.get(self.redis_key)
            if payload is None:
                return BudgetDecision.MISSING
            snapshot = self._parse_snapshot(payload)
        except (RedisError, ValidationError, ValueError):
            logger.exception("Managed budget snapshot unreadable stage=%s", self.stage)
            return BudgetDecision.UNREADABLE

        project_limit = authority.project_limit
        if project_limit is not None and project_limit.total_limit_usd is not None:
            if snapshot.total_spend_usd is None:
                return BudgetDecision.MISSING
            spend = snapshot.total_spend_usd.get(
                str(authority.project.project_id), Decimal(0)
            )
            if spend >= project_limit.total_limit_usd:
                return BudgetDecision.EXHAUSTED
        elif project_limit is not None:
            spend = _current_spend(
                snapshot,
                project_limit.period,
                str(authority.project.project_id),
                now,
            )
            limit = project_limit.base_limit_usd + sum(
                (
                    increase.increase_usd
                    for increase in project_limit.temporary_increases
                    if increase.period_start <= now < increase.period_end
                ),
                Decimal(0),
            )
            if spend >= limit:
                return BudgetDecision.EXHAUSTED

        key_limit = authority.key_limit
        if key_limit is not None:
            spend = _current_spend(
                snapshot, key_limit.period, managed_key_spend_id(authority.key_id), now
            )
            if spend >= key_limit.limit_usd:
                return BudgetDecision.KEY_EXHAUSTED
        return BudgetDecision.ALLOWED

    async def project_total_spend(self, project_id: UUID) -> Decimal | None:
        if self.redis is None:
            return None
        try:
            payload = await self.redis.get(self.redis_key)
            if payload is None:
                return None
            totals = self._parse_snapshot(payload).total_spend_usd
        except (RedisError, ValidationError, ValueError):
            logger.exception("Managed budget snapshot unreadable stage=%s", self.stage)
            return None
        if totals is None:
            return None
        return totals.get(str(project_id), Decimal(0))

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

    async def run_control(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await self.refresh_control_once()
            except Exception:
                logger.exception("Budget snapshot refresh failed stage=%s", self.stage)
            try:
                await asyncio.wait_for(stop.wait(), timeout=5)
            except TimeoutError:
                pass

    def _parse_snapshot(self, payload: bytes | str) -> SpendSnapshot:
        snapshot = SpendSnapshot.model_validate_json(payload)
        if snapshot.stage != self.stage:
            raise ValueError("budget snapshot stage does not match gateway stage")
        return snapshot
