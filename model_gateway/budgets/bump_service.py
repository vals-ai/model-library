"""Control-owned S3 allowances distributed to query workers through Redis."""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from decimal import Decimal
from typing import Protocol, TypedDict, cast

import boto3
from botocore.exceptions import ClientError
from redis.asyncio import Redis

from model_gateway.budgets.bumps import (
    BUMP_USD,
    BumpDocument,
    BumpSnapshot,
    BudgetActor,
    BudgetAuditEvent,
    BudgetGrant,
    LegacyBumpSnapshot,
    effective_limit,
    grant_amount,
    exact_amount_difference,
)
from model_gateway.budgets.schema import BudgetPeriod, period_bounds
from model_gateway.budgets.service import REPLACE_IF_NEWER, BudgetRedis

logger = logging.getLogger("model_proxy_server.budgets.bumps")


class S3Body(Protocol):
    def read(self) -> bytes: ...


class S3Object(TypedDict):
    Body: S3Body
    ETag: str


class BumpS3(Protocol):
    def get_object(self, *, Bucket: str, Key: str) -> S3Object: ...

    def put_object(self, **kwargs: object) -> object: ...


class BumpConflict(Exception):
    """A concurrent grant or stale operator view requires a new read."""


class BumpService:
    def __init__(
        self,
        stage: str,
        redis: Redis | None,
        bucket: str | None = None,
        key: str | None = None,
    ) -> None:
        self.stage = stage
        self.redis: BudgetRedis | None = cast(BudgetRedis | None, redis)
        self.bucket = bucket
        self.key = key
        self.redis_key = f"gateway:budgets:{stage}:bumps"
        self.snapshot: BumpSnapshot | LegacyBumpSnapshot | None = None
        self._s3: BumpS3 | None = None

    def _read_s3(self) -> tuple[BumpDocument | LegacyBumpSnapshot, str | None]:
        assert self.bucket is not None and self.key is not None
        if self._s3 is None:
            self._s3 = cast(Callable[[str], BumpS3], boto3.client)("s3")
        client = self._s3
        try:
            response = client.get_object(Bucket=self.bucket, Key=self.key)
        except ClientError as error:
            if error.response["Error"]["Code"] in {"NoSuchKey", "404"}:
                return LegacyBumpSnapshot(stage=self.stage, revision=0, bumps={}), None
            raise
        payload = response["Body"].read()
        model = (
            BumpDocument
            if "history_started_at" in json.loads(payload)
            else LegacyBumpSnapshot
        )
        snapshot = model.model_validate_json(payload)
        if snapshot.stage != self.stage:
            raise ValueError("bump snapshot stage does not match Gateway stage")
        return snapshot, response["ETag"]

    async def read_control(self) -> BumpSnapshot | LegacyBumpSnapshot:
        document, _ = await asyncio.to_thread(self._read_s3)
        snapshot = (
            BumpSnapshot(
                stage=document.stage, revision=document.revision, bumps=document.bumps
            )
            if isinstance(document, BumpDocument)
            else document
        )
        self._retain(snapshot)
        return cast(BumpSnapshot | LegacyBumpSnapshot, self.snapshot)

    async def read_history(self) -> BumpDocument | LegacyBumpSnapshot:
        document, _ = await asyncio.to_thread(self._read_s3)
        return document

    async def change(
        self,
        email: str,
        expected_revision: int,
        period: BudgetPeriod,
        start: datetime,
        end: datetime,
        base: Decimal,
        actor: BudgetActor,
        reason: str,
        limit: Decimal | None,
        remove: bool = False,
    ) -> tuple[BumpSnapshot, Decimal]:
        assert self.bucket is not None and self.key is not None

        def commit() -> tuple[BumpSnapshot, Decimal]:
            current, etag = self._read_s3()
            if current.revision != expected_revision or (
                self.snapshot is not None and current.revision < self.snapshot.revision
            ):
                raise BumpConflict
            old = current.bumps.get(email)
            old_amount = (
                grant_amount(old)
                if old is not None
                and old.period == period
                and old.period_start == start
                and old.period_end == end
                else Decimal(0)
            )
            if remove:
                amount = Decimal(0)
            elif limit is not None:
                amount = exact_amount_difference(limit, base)
            else:
                amount = effective_limit(old_amount, Decimal(BUMP_USD))
            now = datetime.now(UTC)
            if period_bounds(period, now) != (start, end):
                raise BumpConflict
            bumps = {
                key: BudgetGrant(
                    period=grant.period,
                    period_start=grant.period_start,
                    period_end=grant.period_end,
                    bump_usd=grant_amount(grant),
                )
                for key, grant in current.bumps.items()
            }
            if amount == 0:
                bumps.pop(email, None)
            else:
                bumps[email] = BudgetGrant(
                    period=period, period_start=start, period_end=end, bump_usd=amount
                )
            event = BudgetAuditEvent(
                revision=current.revision + 1,
                email=email,
                period=period,
                period_start=start,
                period_end=end,
                changed_at=now,
                actor=actor,
                reason=reason,
                base_limit_usd=base,
                old_bump_usd=old_amount,
                new_bump_usd=amount,
            )
            updated = BumpDocument(
                stage=self.stage,
                revision=current.revision + 1,
                bumps=bumps,
                history_started_at=(
                    current.history_started_at
                    if isinstance(current, BumpDocument)
                    else now
                ),
                events=[
                    *(current.events if isinstance(current, BumpDocument) else []),
                    event,
                ],
            )
            client = cast(BumpS3, self._s3)
            condition = {"IfMatch": etag} if etag is not None else {"IfNoneMatch": "*"}
            try:
                client.put_object(
                    Bucket=self.bucket,
                    Key=self.key,
                    Body=updated.model_dump_json().encode("utf-8"),
                    ContentType="application/json",
                    **condition,
                )
            except ClientError as error:
                if error.response["Error"]["Code"] in {
                    "PreconditionFailed",
                    "ConditionalRequestConflict",
                }:
                    raise BumpConflict from error
                raise
            return BumpSnapshot(
                stage=self.stage, revision=updated.revision, bumps=bumps
            ), amount

        snapshot, amount = await asyncio.to_thread(commit)
        self._retain(snapshot)
        return snapshot, amount

    async def refresh_control_once(self) -> None:
        if self.bucket is None or self.key is None or self.redis is None:
            return
        snapshot = await self.read_control()
        await self.redis.eval(
            REPLACE_IF_NEWER,
            1,
            self.redis_key,
            snapshot.revision,
            snapshot.model_dump_json().encode("utf-8"),
        )

    async def refresh_query_once(self) -> None:
        if self.redis is None:
            return
        payload = await self.redis.get(self.redis_key)
        if payload is not None:
            data = json.loads(payload)
            first_grant = next(iter(data["bumps"].values()), None)
            model = (
                LegacyBumpSnapshot
                if first_grant is None or "count" in first_grant
                else BumpSnapshot
            )
            snapshot = model.model_validate_json(payload)
            if snapshot.stage != self.stage:
                raise ValueError(
                    "bump Redis snapshot stage does not match Gateway stage"
                )
            self._retain(snapshot)

    def _retain(self, snapshot: BumpSnapshot | LegacyBumpSnapshot) -> None:
        if self.snapshot is None or snapshot.revision >= self.snapshot.revision:
            self.snapshot = snapshot

    async def run_control(self, stop: asyncio.Event) -> None:
        await self._run(stop, self.refresh_control_once)

    async def run_query(self, stop: asyncio.Event) -> None:
        await self._run(stop, self.refresh_query_once)

    async def _run(
        self, stop: asyncio.Event, refresh: Callable[[], Awaitable[None]]
    ) -> None:
        while not stop.is_set():
            try:
                await refresh()
            except Exception:
                logger.exception("Bump snapshot refresh failed stage=%s", self.stage)
            try:
                await asyncio.wait_for(stop.wait(), timeout=5)
            except TimeoutError:
                pass
