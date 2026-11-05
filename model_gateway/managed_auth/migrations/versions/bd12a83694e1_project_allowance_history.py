"""Record project allowance history and approved-extra cancellations."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "bd12a83694e1"
down_revision: Union[str, Sequence[str], None] = "aa78b6d901f4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("managed_audit", sa.Column("detail", sa.JSON(), nullable=True))
    op.add_column(
        "managed_project_limit_requests",
        sa.Column("cancelled_at", sa.DateTime(timezone=True)),
    )
    op.add_column(
        "managed_project_limit_requests", sa.Column("cancelled_by_issuer", sa.String())
    )
    op.add_column(
        "managed_project_limit_requests", sa.Column("cancelled_by_subject", sa.String())
    )
    op.add_column(
        "managed_project_limit_requests", sa.Column("cancellation_reason", sa.String())
    )
    op.create_check_constraint(
        "limit_request_cancellation",
        "managed_project_limit_requests",
        "(cancelled_at IS NULL AND cancelled_by_issuer IS NULL AND cancelled_by_subject IS NULL AND cancellation_reason IS NULL) OR "
        "(kind = 'temporary' AND status = 'approved' AND cancelled_at IS NOT NULL AND "
        "cancelled_by_issuer IS NOT NULL AND cancelled_by_subject IS NOT NULL AND cancellation_reason IS NOT NULL)",
    )
    op.create_foreign_key(
        "fk_limit_request_cancelled_by_human",
        "managed_project_limit_requests",
        "managed_humans",
        ["cancelled_by_issuer", "cancelled_by_subject"],
        ["issuer", "subject"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_limit_request_cancelled_by_human",
        "managed_project_limit_requests",
        type_="foreignkey",
    )
    op.drop_constraint(
        "limit_request_cancellation",
        "managed_project_limit_requests",
        type_="check",
    )
    op.drop_column("managed_project_limit_requests", "cancellation_reason")
    op.drop_column("managed_project_limit_requests", "cancelled_by_subject")
    op.drop_column("managed_project_limit_requests", "cancelled_by_issuer")
    op.drop_column("managed_project_limit_requests", "cancelled_at")
    op.drop_column("managed_audit", "detail")
