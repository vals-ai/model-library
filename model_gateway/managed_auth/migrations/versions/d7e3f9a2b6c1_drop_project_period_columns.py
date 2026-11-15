"""Drop each project's recurring period and base and the period windows on temporary increases, which the Gateway no longer reads."""

from typing import Sequence, Union

from alembic import op

revision: str = "d7e3f9a2b6c1"
down_revision: Union[str, Sequence[str], None] = "c5d2e8f1a9b4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_constraint(
        "project_policy_cap_pair", "managed_project_policies", type_="check"
    )
    op.drop_column("managed_project_policies", "base_limit_usd")
    op.drop_column("managed_project_policies", "period")
    op.drop_column("managed_project_limit_requests", "period_end")
    op.drop_column("managed_project_limit_requests", "period_start")


def downgrade() -> None:
    raise RuntimeError(
        "Downgrade requires a period and base for every project and a window for every approved temporary increase"
    )
