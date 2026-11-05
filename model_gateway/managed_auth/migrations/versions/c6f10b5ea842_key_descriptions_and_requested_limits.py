"""Persist key descriptions and requested per-key limits."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "c6f10b5ea842"
down_revision: Union[str, Sequence[str], None] = "bd12a83694e1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("managed_keys", sa.Column("description", sa.String(), nullable=True))
    op.execute(
        """
        DO $$
        DECLARE unmatched_key uuid;
        BEGIN
            SELECT k.id INTO unmatched_key
            FROM managed_keys k
            LEFT JOIN managed_key_requests r
                ON r.key_id = k.id AND r.kind = 'create' AND r.status = 'approved'
            GROUP BY k.id
            HAVING count(r.id) <> 1
                OR count(r.id) FILTER (WHERE length(r.reason) > 0) <> 1
            LIMIT 1;
            IF unmatched_key IS NOT NULL THEN
                RAISE EXCEPTION 'Key % does not have exactly one approved create request with a nonempty reason', unmatched_key;
            END IF;
        END $$;
        """
    )
    op.execute(
        """
        UPDATE managed_keys k
        SET description = r.reason
        FROM managed_key_requests r
        WHERE r.key_id = k.id AND r.kind = 'create' AND r.status = 'approved'
        """
    )
    op.create_check_constraint(
        "key_description_nonempty", "managed_keys", "length(description) > 0"
    )
    op.add_column(
        "managed_key_requests", sa.Column("requested_key_limit_period", sa.String())
    )
    op.add_column(
        "managed_key_requests", sa.Column("requested_key_limit_usd", sa.Numeric())
    )
    op.create_check_constraint(
        "request_key_limit_cap_pair",
        "managed_key_requests",
        "(requested_key_limit_period IS NULL AND requested_key_limit_usd IS NULL) OR "
        "(creator_kind = 'human' AND requested_key_limit_period IS NOT NULL AND "
        "requested_key_limit_usd IS NOT NULL AND requested_key_limit_period IN "
        "('daily', 'weekly', 'monthly') AND requested_key_limit_usd >= 0)",
    )


def downgrade() -> None:
    op.drop_constraint(
        "request_key_limit_cap_pair", "managed_key_requests", type_="check"
    )
    op.drop_column("managed_key_requests", "requested_key_limit_usd")
    op.drop_column("managed_key_requests", "requested_key_limit_period")
    op.drop_constraint("key_description_nonempty", "managed_keys", type_="check")
    op.drop_column("managed_keys", "description")
