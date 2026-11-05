"""allow service keys without managed budgets

Revision ID: 7e40c64e19a2
Revises: 22a96aa8ec6e

"""

from typing import Sequence, Union

from alembic import op


revision: str = "7e40c64e19a2"
down_revision: Union[str, Sequence[str], None] = "22a96aa8ec6e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_constraint("service_key_budget", "managed_keys", type_="check")
    op.drop_constraint(
        "service_requested_budget", "managed_key_requests", type_="check"
    )
    op.drop_constraint("service_approved_budget", "managed_key_requests", type_="check")


def downgrade() -> None:
    op.create_check_constraint(
        "service_key_budget",
        "managed_keys",
        "owner_kind = 'personal' OR budget_id IS NOT NULL",
    )
    op.create_check_constraint(
        "service_requested_budget",
        "managed_key_requests",
        "owner_kind = 'personal' OR requested_budget_id IS NOT NULL",
    )
    op.create_check_constraint(
        "service_approved_budget",
        "managed_key_requests",
        "owner_kind = 'personal' OR status != 'approved' OR approved_budget_id IS NOT NULL",
    )
