"""Shared policy and spend snapshot contracts for email budgets."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from model_gateway.usage_ledger.schema import canonical_email_value

BudgetPeriod = Literal["daily", "weekly", "monthly"]


def _utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("budget timestamps must be timezone-aware")
    return value.astimezone(UTC)


def period_bounds(period: BudgetPeriod, now: datetime) -> tuple[datetime, datetime]:
    """Return the containing half-open UTC calendar period."""
    current = _utc_datetime(now)
    start = current.replace(hour=0, minute=0, second=0, microsecond=0)
    if period == "daily":
        return start, start + timedelta(days=1)
    if period == "weekly":
        start -= timedelta(days=start.weekday())
        return start, start + timedelta(days=7)
    start = start.replace(day=1)
    if start.month == 12:
        end = start.replace(year=start.year + 1, month=1)
    else:
        end = start.replace(month=start.month + 1)
    return start, end


class BudgetPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    period: BudgetPeriod
    default_limit_usd: Decimal = Field(ge=0, allow_inf_nan=False)
    email_overrides: dict[str, Decimal]

    @field_validator("email_overrides")
    @classmethod
    def _canonical_overrides(cls, overrides: dict[str, Decimal]) -> dict[str, Decimal]:
        canonical_overrides: dict[str, Decimal] = {}
        for email, limit in overrides.items():
            canonical = canonical_email_value(email)
            if canonical is None or canonical in canonical_overrides:
                raise ValueError(
                    f"invalid or duplicate budget email override: {email!r}"
                )
            if not limit.is_finite() or limit < 0:
                raise ValueError(
                    f"budget email override must be nonnegative: {email!r}"
                )
            canonical_overrides[canonical] = limit
        return canonical_overrides


class PeriodSpend(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    start: datetime
    end: datetime
    spend_usd: dict[str, Decimal]

    @field_validator("start", "end")
    @classmethod
    def _utc_bounds(cls, value: datetime) -> datetime:
        return _utc_datetime(value)

    @model_validator(mode="after")
    def _ordered_bounds(self) -> Self:
        if self.start >= self.end:
            raise ValueError("spend period start must precede end")
        return self


class SpendSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    stage: str
    revision: int = Field(ge=0)
    generated_at: datetime
    data_through: datetime
    periods: dict[BudgetPeriod, PeriodSpend]

    @field_validator("generated_at", "data_through")
    @classmethod
    def _utc_timestamps(cls, value: datetime) -> datetime:
        return _utc_datetime(value)

    @field_validator("periods")
    @classmethod
    def _complete_periods(
        cls, periods: dict[BudgetPeriod, PeriodSpend]
    ) -> dict[BudgetPeriod, PeriodSpend]:
        if periods.keys() != {"daily", "weekly", "monthly"}:
            raise ValueError("spend snapshot must contain all three budget periods")
        return periods
