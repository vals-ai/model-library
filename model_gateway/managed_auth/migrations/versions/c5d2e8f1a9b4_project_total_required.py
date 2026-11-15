"""Make each project's overall total required, and stop requiring its recurring period and base or a period window on temporary increases."""

from typing import Any, Sequence, Union, cast

from alembic import op
import sqlalchemy as sa

from model_gateway.budgets.schema import BudgetPeriod, period_bounds

revision: str = "c5d2e8f1a9b4"
down_revision: Union[str, Sequence[str], None] = "b8e1f3a6c2d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

policies = sa.table(
    "managed_project_policies",
    sa.column("project_id", sa.Uuid()),
    sa.column("period", sa.String()),
    sa.column("base_limit_usd", sa.Numeric()),
    sa.column("created_at", sa.DateTime(timezone=True)),
    sa.column("total_limit_usd", sa.Numeric()),
    sa.column("total_started_at", sa.DateTime(timezone=True)),
)


def upgrade() -> None:
    connection = op.get_bind()
    missing: list[dict[str, Any]] = [
        {
            "policy_id": policy["project_id"],
            "total": policy["total_limit_usd"] or policy["base_limit_usd"],
            "started": policy["total_started_at"]
            or period_bounds(
                cast(BudgetPeriod, policy["period"]), policy["created_at"]
            )[0],
        }
        for policy in connection.execute(
            sa.select(
                policies.c.project_id,
                policies.c.period,
                policies.c.base_limit_usd,
                policies.c.created_at,
                policies.c.total_limit_usd,
                policies.c.total_started_at,
            )
            .where(
                sa.or_(
                    policies.c.total_limit_usd.is_(None),
                    policies.c.total_started_at.is_(None),
                )
            )
            .order_by(policies.c.project_id)
        ).mappings()
    ]
    if missing:
        connection.execute(
            sa.update(policies)
            .where(policies.c.project_id == sa.bindparam("policy_id", type_=sa.Uuid()))
            .values(
                total_limit_usd=sa.bindparam("total", type_=sa.Numeric()),
                total_started_at=sa.bindparam(
                    "started", type_=sa.DateTime(timezone=True)
                ),
            ),
            missing,
        )
    op.alter_column(
        "managed_project_policies",
        "total_limit_usd",
        existing_type=sa.Numeric(),
        nullable=False,
    )
    op.alter_column(
        "managed_project_policies",
        "total_started_at",
        existing_type=sa.DateTime(timezone=True),
        nullable=False,
    )
    op.drop_constraint(
        "project_policy_positive_base", "managed_project_policies", type_="check"
    )
    op.alter_column(
        "managed_project_policies",
        "period",
        existing_type=sa.String(),
        nullable=True,
    )
    op.alter_column(
        "managed_project_policies",
        "base_limit_usd",
        existing_type=sa.Numeric(),
        nullable=True,
    )
    op.drop_constraint(
        "limit_request_period", "managed_project_limit_requests", type_="check"
    )


def downgrade() -> None:
    raise RuntimeError(
        "Downgrade requires a period and base for every project approved after this revision"
    )
