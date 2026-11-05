"""Give each project policy an editable budget reason and an archive time, and each budget raise its target total and source revision."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "b8e1f3a6c2d4"
down_revision: Union[str, Sequence[str], None] = "a4d7c2e9b315"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("managed_project_policies", sa.Column("budget_reason", sa.String()))
    op.add_column(
        "managed_project_policies",
        sa.Column("archived_at", sa.DateTime(timezone=True)),
    )
    op.add_column(
        "managed_project_limit_requests", sa.Column("target_total_usd", sa.Numeric())
    )
    op.add_column(
        "managed_project_limit_requests", sa.Column("source_revision", sa.Integer())
    )


def downgrade() -> None:
    op.drop_column("managed_project_limit_requests", "source_revision")
    op.drop_column("managed_project_limit_requests", "target_total_usd")
    op.drop_column("managed_project_policies", "archived_at")
    op.drop_column("managed_project_policies", "budget_reason")
