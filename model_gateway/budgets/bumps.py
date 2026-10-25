"""Durable temporary allowances for the active email budget period."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from model_gateway.budgets.schema import BudgetPeriod, BudgetPolicy, period_bounds
from model_gateway.usage_ledger.schema import canonical_email_value

BUMP_USD = 5000


class GrantBounds(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    period: BudgetPeriod
    period_start: datetime
    period_end: datetime

    @field_validator("period_start", "period_end")
    @classmethod
    def utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("grant bounds must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def ordered_bounds(self) -> "GrantBounds":
        if self.period_start >= self.period_end:
            raise ValueError("grant start must precede end")
        return self


class LegacyBudgetGrant(GrantBounds):
    count: int = Field(ge=1)


class BudgetGrant(GrantBounds):
    bump_usd: Decimal = Field(ge=0, allow_inf_nan=False)

    @field_serializer("bump_usd")
    def serialize_bump(self, value: Decimal) -> str:
        return format(value, "f")


class SnapshotIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: str
    revision: int = Field(ge=0)


class LegacyBumpSnapshot(SnapshotIdentity):
    bumps: dict[str, LegacyBudgetGrant]

    @field_validator("bumps")
    @classmethod
    def canonical_keys(
        cls, bumps: dict[str, LegacyBudgetGrant]
    ) -> dict[str, LegacyBudgetGrant]:
        for email in bumps:
            if canonical_email_value(email) != email:
                raise ValueError(f"noncanonical bump email: {email!r}")
        return bumps


class BumpSnapshot(SnapshotIdentity):
    bumps: dict[str, BudgetGrant]

    @field_validator("bumps")
    @classmethod
    def canonical_keys(cls, bumps: dict[str, BudgetGrant]) -> dict[str, BudgetGrant]:
        for email in bumps:
            if canonical_email_value(email) != email:
                raise ValueError(f"noncanonical bump email: {email!r}")
        return bumps


class BudgetActor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: str = Field(min_length=1)
    email: str | None


class BudgetAuditEvent(GrantBounds):
    revision: int = Field(ge=1)
    email: str
    changed_at: datetime
    actor: BudgetActor
    reason: str = Field(min_length=1)
    base_limit_usd: Decimal
    old_bump_usd: Decimal
    new_bump_usd: Decimal

    @field_validator("changed_at")
    @classmethod
    def utc_change(cls, value: datetime) -> datetime:
        return GrantBounds.utc_timestamp(value)

    @field_serializer("base_limit_usd", "old_bump_usd", "new_bump_usd")
    def serialize_amount(self, value: Decimal) -> str:
        return format(value, "f")


class BumpDocument(BumpSnapshot):
    history_started_at: datetime
    events: list[BudgetAuditEvent]


def grant_amount(grant: BudgetGrant | LegacyBudgetGrant) -> Decimal:
    return (
        grant.bump_usd
        if isinstance(grant, BudgetGrant)
        else Decimal(BUMP_USD * grant.count)
    )


def active_grants(
    snapshot: BumpSnapshot | LegacyBumpSnapshot, policy: BudgetPolicy, now: datetime
) -> tuple[datetime, datetime, dict[str, BudgetGrant | LegacyBudgetGrant]]:
    start, end = period_bounds(policy.period, now)
    grants = {
        email: grant
        for email, grant in snapshot.bumps.items()
        if grant.period == policy.period
        and grant.period_start == start
        and grant.period_end == end
        and grant_amount(grant) > 0
    }
    return start, end, grants


def exact_amount_difference(left: Decimal, right: Decimal) -> Decimal:
    """Subtract finite decimal amounts without applying the process Decimal context."""
    a, b = left.as_tuple(), right.as_tuple()
    exponent = min(cast(int, a.exponent), cast(int, b.exponent))
    a_units = 0
    for digit in a.digits:
        a_units = a_units * 10 + digit
    a_units *= 10 ** (cast(int, a.exponent) - exponent)
    b_units = 0
    for digit in b.digits:
        b_units = b_units * 10 + digit
    b_units *= 10 ** (cast(int, b.exponent) - exponent)
    units = (-a_units if a.sign else a_units) - (-b_units if b.sign else b_units)
    digits: list[int] = []
    magnitude = abs(units)
    while magnitude:
        magnitude, digit = divmod(magnitude, 10)
        digits.append(digit)
    return Decimal(
        (int(units < 0), tuple(reversed(digits)) if digits else (0,), exponent)
    )


def effective_limit(base: Decimal, bump: Decimal) -> Decimal:
    return exact_amount_difference(base, bump.copy_negate())
