"""Require a positive recurring base for each approved project policy."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "e1a7b8c9d012"
down_revision: Union[str, Sequence[str], None] = "c6f10b5ea842"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "managed_project_policies",
        "period",
        existing_type=sa.String(),
        nullable=False,
    )
    op.alter_column(
        "managed_project_policies",
        "base_limit_usd",
        existing_type=sa.Numeric(),
        nullable=False,
    )
    op.create_check_constraint(
        "project_policy_positive_base",
        "managed_project_policies",
        "period IN ('daily', 'weekly', 'monthly') AND base_limit_usd > 0",
    )


def downgrade() -> None:
    op.drop_constraint(
        "project_policy_positive_base", "managed_project_policies", type_="check"
    )
    op.alter_column(
        "managed_project_policies",
        "base_limit_usd",
        existing_type=sa.Numeric(),
        nullable=True,
    )
    op.alter_column(
        "managed_project_policies",
        "period",
        existing_type=sa.String(),
        nullable=True,
    )
