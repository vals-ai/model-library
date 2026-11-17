"""Store the project service's version of each budget it last applied, so an older push cannot overwrite a newer one."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "a9c4e1f7b203"
down_revision: Union[str, Sequence[str], None] = "d7e3f9a2b6c1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "managed_project_policies",
        sa.Column("source_version", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("managed_project_policies", "source_version")
