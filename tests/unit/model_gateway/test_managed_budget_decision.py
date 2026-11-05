import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import cast
from uuid import UUID, uuid4

import pytest
from fastapi import Request
from redis.asyncio import Redis

from model_gateway.budgets.schema import (
    BudgetPeriod,
    PeriodSpend,
    SpendSnapshot,
    managed_key_spend_id,
    period_bounds,
)
from model_gateway.budgets.service import BudgetDecision, BudgetService
from model_gateway.managed_auth.runtime import (
    RuntimeAuthority,
    RuntimeProjectLimit,
    deserialize_authority,
    serialize_authority,
)
from model_gateway.request_budget import BudgetRefusal, budget_refusal
from model_gateway.run_token_types import ModelPermissions
from model_gateway.managed_auth.types import (
    HumanIdentity,
    KeyLimit,
    PersonalOwner,
    ProjectReference,
)

pytestmark = pytest.mark.unit

STAGE = "test"
NOW = datetime(2026, 11, 1, 0, 1, tzinfo=UTC)


class FakeRedis:
    def __init__(self, payload: str | None) -> None:
        self.payload = payload

    async def get(self, name: str) -> str | None:
        assert name == f"gateway:budgets:{STAGE}:managed-spend-v2"
        return self.payload


def authority(
    project_limit: Decimal | None = None,
    key_limit: Decimal | None = None,
    total: Decimal | None = None,
) -> RuntimeAuthority:
    return RuntimeAuthority(
        key_id=uuid4(),
        owner=PersonalOwner(HumanIdentity("https://issuer.example", "owner")),
        permissions=ModelPermissions(),
        project=ProjectReference(uuid4(), "organization", "dev"),
        project_limit=None
        if project_limit is None
        else RuntimeProjectLimit("monthly", project_limit, (), total),
        key_limit=None if key_limit is None else KeyLimit("monthly", key_limit),
    )


def snapshot(
    period_moment: datetime,
    generated_at: datetime,
    spend: dict[str, Decimal],
    totals: dict[str, Decimal] | None = None,
) -> str:
    periods: dict[BudgetPeriod, PeriodSpend] = {
        period: PeriodSpend(
            start=period_bounds(period, period_moment)[0],
            end=period_bounds(period, period_moment)[1],
            spend_usd=spend,
        )
        for period in ("daily", "weekly", "monthly")
    }
    return SpendSnapshot(
        schema_version=1,
        stage=STAGE,
        revision=1,
        generated_at=generated_at,
        data_through=generated_at,
        periods=periods,
        total_spend_usd=totals,
    ).model_dump_json()


async def decide(payload: str | None, auth: RuntimeAuthority) -> str:
    service = BudgetService(stage=STAGE, redis=None)
    service.redis = FakeRedis(payload)  # pyright: ignore[reportAttributeAccessIssue]
    return await service.managed_decision(auth, NOW)


async def test_previous_period_snapshot_counts_as_zero_after_rollover():
    auth = authority(project_limit=Decimal("10"), key_limit=Decimal("5"))
    october = NOW - timedelta(minutes=6)
    payload = snapshot(
        october,
        october,
        {
            str(auth.project.project_id): Decimal("10"),
            managed_key_spend_id(auth.key_id): Decimal("5"),
        },
    )
    assert await decide(payload, auth) == "allowed"


async def test_absent_entry_in_recent_snapshot_counts_as_zero():
    auth = authority(project_limit=Decimal("10"), key_limit=Decimal("5"))
    assert await decide(snapshot(NOW, NOW, {}), auth) == "allowed"


async def test_long_stale_previous_period_snapshot_counts_as_zero():
    auth = authority(project_limit=Decimal("10"))
    october = NOW - timedelta(days=3)
    payload = snapshot(october, october, {str(auth.project.project_id): Decimal("10")})
    assert await decide(payload, auth) == "allowed"


async def test_no_snapshot_is_missing():
    assert await decide(None, authority(project_limit=Decimal("10"))) == "missing"


async def test_current_spend_at_limit_is_still_exhausted():
    auth = authority(project_limit=Decimal("10"), key_limit=Decimal("5"))
    project = str(auth.project.project_id)
    key = managed_key_spend_id(auth.key_id)
    assert (
        await decide(snapshot(NOW, NOW, {project: Decimal("10")}), auth) == "exhausted"
    )
    assert (
        await decide(
            snapshot(NOW, NOW, {project: Decimal("1"), key: Decimal("5")}), auth
        )
        == "key_exhausted"
    )


async def test_total_under_cap_allows_even_when_this_month_is_over_the_base():
    auth = authority(project_limit=Decimal("10"), total=Decimal("25"))
    project = str(auth.project.project_id)
    payload = snapshot(NOW, NOW, {project: Decimal("12")}, {project: Decimal("24")})
    assert await decide(payload, auth) == "allowed"


async def test_total_at_cap_is_exhausted_even_when_this_month_is_under_the_base():
    auth = authority(project_limit=Decimal("10"), total=Decimal("25"))
    project = str(auth.project.project_id)
    payload = snapshot(NOW, NOW, {project: Decimal("1")}, {project: Decimal("25")})
    assert await decide(payload, auth) == "exhausted"


async def test_total_without_published_totals_is_missing():
    auth = authority(project_limit=Decimal("10"), total=Decimal("25"))
    assert await decide(snapshot(NOW, NOW, {}), auth) == "missing"


async def test_refusal_names_the_project_total():
    auth = authority(project_limit=Decimal("10"), total=Decimal("25"))
    project = str(auth.project.project_id)
    service = BudgetService(
        stage=STAGE,
        redis=cast(Redis, FakeRedis(snapshot(NOW, NOW, {}, {project: Decimal("30")}))),
    )
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(budget_service=service)),
        state=SimpleNamespace(managed_authority=auth),
    )
    refusal = await budget_refusal(cast(Request, request))
    assert refusal == BudgetRefusal(
        402, "budget_exceeded", "Project total budget of $25 is spent"
    )


def test_authority_round_trips_the_total_and_reads_records_without_one():
    auth = authority(project_limit=Decimal("10"), total=Decimal("25"))
    record = serialize_authority(auth)
    assert deserialize_authority(record).project_limit == auth.project_limit
    older = json.loads(record)
    del older["project_limit"]["total_limit_usd"]
    project_limit = deserialize_authority(json.dumps(older)).project_limit
    assert project_limit is not None and project_limit.total_limit_usd is None


async def spent(payload: str | None, project: str) -> Decimal | None:
    service = BudgetService(stage=STAGE, redis=cast(Redis, FakeRedis(payload)))
    return await service.project_total_spend(UUID(project))


async def test_project_total_spend_reads_the_published_total():
    project, other = str(uuid4()), str(uuid4())
    payload = snapshot(NOW, NOW, {}, {project: Decimal("30")})
    assert await spent(payload, project) == Decimal("30")
    assert await spent(payload, other) == Decimal("0")


async def test_project_total_spend_is_unknown_without_published_totals():
    project = str(uuid4())
    assert await spent(None, project) is None
    assert await spent(snapshot(NOW, NOW, {}), project) is None


async def test_project_total_spend_is_unknown_when_the_snapshot_is_unreadable():
    assert await spent("not a snapshot", str(uuid4())) is None


class FixedDecision:
    def __init__(self, decision: BudgetDecision) -> None:
        self.decision = decision

    async def managed_decision(self, auth: RuntimeAuthority, now: datetime) -> BudgetDecision:
        return self.decision


@pytest.mark.parametrize("decision", list(BudgetDecision))
async def test_budget_refusal_handles_every_decision(decision: BudgetDecision):
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(budget_service=FixedDecision(decision))),
        state=SimpleNamespace(
            managed_authority=authority(project_limit=Decimal("10"), total=Decimal("25"))
        ),
    )
    refusal = await budget_refusal(cast(Request, request))
    expected = {
        BudgetDecision.ALLOWED: None,
        BudgetDecision.EXHAUSTED: BudgetRefusal(
            402, "budget_exceeded", "Project total budget of $25 is spent"
        ),
        BudgetDecision.KEY_EXHAUSTED: BudgetRefusal(
            402, "budget_exceeded", "Managed key budget exceeded"
        ),
        BudgetDecision.MISSING: BudgetRefusal(
            503, "budget_unavailable", "Managed budget spend unavailable"
        ),
        BudgetDecision.UNREADABLE: BudgetRefusal(
            503, "budget_unavailable", "Managed budget spend unavailable"
        ),
    }
    assert refusal == expected[decision]


def test_budget_decisions_keep_their_wire_values():
    assert [str(decision) for decision in BudgetDecision] == [
        "allowed", "exhausted", "key_exhausted", "missing", "unreadable",
    ]
