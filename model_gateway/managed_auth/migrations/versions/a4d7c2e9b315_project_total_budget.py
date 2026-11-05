"""Give each project one overall total and a fixed start: every recorded allowance since the policy began plus its approved temporary increases."""

import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Sequence, Union, cast
from uuid import UUID

from alembic import op
import sqlalchemy as sa
from sqlalchemy import RowMapping

from model_gateway.budgets.schema import BudgetPeriod, period_bounds
from model_gateway.managed_auth.types import (
    LimitRequestStatus,
    PROJECT_POLICY_AUDIT,
    TEMPORARY_INCREASE,
)

revision: str = "a4d7c2e9b315"
down_revision: Union[str, Sequence[str], None] = "f2b3c4d5e6a7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

logger = logging.getLogger("alembic.runtime.migration")

policies = sa.table(
    "managed_project_policies",
    sa.column("project_id", sa.Uuid()),
    sa.column("period", sa.String()),
    sa.column("base_limit_usd", sa.Numeric()),
    sa.column("created_at", sa.DateTime(timezone=True)),
    sa.column("total_limit_usd", sa.Numeric()),
    sa.column("total_started_at", sa.DateTime(timezone=True)),
)
requests = sa.table(
    "managed_project_limit_requests",
    sa.column("project_id", sa.Uuid()),
    sa.column("kind", sa.String()),
    sa.column("status", sa.String()),
    sa.column("increase_usd", sa.Numeric()),
)
audit = sa.table(
    "managed_audit",
    sa.column("id", sa.Uuid()),
    sa.column("target_kind", sa.String()),
    sa.column("target_id", sa.Uuid()),
    sa.column("detail", sa.JSON()),
    sa.column("occurred_at", sa.DateTime(timezone=True)),
)

State = tuple[datetime, BudgetPeriod, Decimal]


def _periods_touched(period: BudgetPeriod, start: datetime, end: datetime) -> int:
    window, last = period_bounds(period, start)[0], period_bounds(period, end)[0]
    count = 1
    while window < last:
        window = period_bounds(period, window)[1]
        count += 1
    return count


def _state(recorded: Mapping[str, Any]) -> tuple[BudgetPeriod, Decimal]:
    return cast(BudgetPeriod, recorded["period"]), Decimal(recorded["base_limit_usd"])


def _states(policy: RowMapping, events: list[RowMapping], now: datetime) -> list[State]:
    current = (cast(BudgetPeriod, policy["period"]), policy["base_limit_usd"])
    first: Mapping[str, Any] = events[0]["detail"] if events else {}
    recorded = first.get("before") or first.get("after")
    initial = _state(recorded) if recorded else current
    states: list[State] = [(policy["created_at"], initial[0], initial[1])]
    for event in events:
        after = event["detail"].get("after")
        if after and _state(after) != states[-1][1:]:
            period, base = _state(after)
            states.append(
                (max(event["occurred_at"], policy["created_at"]), period, base)
            )
    if states[-1][1:] != current:
        states.append((now, current[0], current[1]))
    return states


def upgrade() -> None:
    op.add_column(
        "managed_project_policies", sa.Column("total_limit_usd", sa.Numeric())
    )
    op.add_column(
        "managed_project_policies",
        sa.Column("total_started_at", sa.DateTime(timezone=True)),
    )
    op.create_check_constraint(
        "project_policy_positive_total",
        "managed_project_policies",
        "total_limit_usd IS NULL OR total_limit_usd > 0",
    )
    connection = op.get_bind()
    now = datetime.now(UTC)
    temporary = (
        sa.select(sa.func.coalesce(sa.func.sum(requests.c.increase_usd), 0))
        .where(
            requests.c.project_id == policies.c.project_id,
            requests.c.kind == TEMPORARY_INCREASE,
            requests.c.status == LimitRequestStatus.APPROVED,
        )
        .scalar_subquery()
    )
    events: dict[UUID, list[RowMapping]] = {}
    for event in connection.execute(
        sa.select(audit.c.target_id, audit.c.detail, audit.c.occurred_at)
        .where(audit.c.target_kind == PROJECT_POLICY_AUDIT, audit.c.detail.is_not(None))
        .order_by(audit.c.target_id, audit.c.occurred_at, audit.c.id)
    ).mappings():
        events.setdefault(event["target_id"], []).append(event)
    totals: list[dict[str, Any]] = []
    for policy in connection.execute(
        sa.select(
            policies.c.project_id,
            policies.c.period,
            policies.c.base_limit_usd,
            policies.c.created_at,
            temporary.label("temporary_usd"),
        ).order_by(policies.c.project_id)
    ).mappings():
        states = _states(policy, events.get(policy["project_id"], []), now)
        allowance = sum(
            (
                base
                * _periods_touched(
                    period,
                    begins,
                    states[index + 1][0] if index + 1 < len(states) else now,
                )
                for index, (begins, period, base) in enumerate(states)
            ),
            Decimal(0),
        )
        total = allowance + policy["temporary_usd"]
        started = period_bounds(states[0][1], policy["created_at"])[0]
        logger.info(
            "project total project_id=%s states=%s allowance=%s temporary=%s total=%s start=%s",
            policy["project_id"],
            "|".join(f"{period}:{base}" for _, period, base in states),
            allowance,
            policy["temporary_usd"],
            total,
            started.isoformat(),
        )
        totals.append(
            {"policy_id": policy["project_id"], "total": total, "started": started}
        )
    if totals:
        connection.execute(
            sa.update(policies)
            .where(policies.c.project_id == sa.bindparam("policy_id", type_=sa.Uuid()))
            .values(
                total_limit_usd=sa.bindparam("total", type_=sa.Numeric()),
                total_started_at=sa.bindparam(
                    "started", type_=sa.DateTime(timezone=True)
                ),
            ),
            totals,
        )


def downgrade() -> None:
    op.drop_constraint(
        "project_policy_positive_total", "managed_project_policies", type_="check"
    )
    op.drop_column("managed_project_policies", "total_started_at")
    op.drop_column("managed_project_policies", "total_limit_usd")
