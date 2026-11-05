"""Carry the main key's requested budget and its reason on key requests."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "f2b3c4d5e6a7"
down_revision: Union[str, Sequence[str], None] = "e1a7b8c9d012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("managed_key_requests", sa.Column("requested_budget", sa.Numeric()))
    op.add_column("managed_key_requests", sa.Column("budget_reason", sa.String()))
    op.create_check_constraint(
        "requested_budget_reason_pair",
        "managed_key_requests",
        "(requested_budget IS NULL AND budget_reason IS NULL) OR "
        "(requested_budget IS NOT NULL AND requested_budget > 0 AND budget_reason IS NOT NULL)",
    )


def downgrade() -> None:
    op.drop_constraint(
        "requested_budget_reason_pair", "managed_key_requests", type_="check"
    )
    op.drop_column("managed_key_requests", "budget_reason")
    op.drop_column("managed_key_requests", "requested_budget")
