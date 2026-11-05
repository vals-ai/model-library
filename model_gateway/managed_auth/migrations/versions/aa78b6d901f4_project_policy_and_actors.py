"""Add project-bound keys, approved policies, and typed actors.

Backfill the verified DEV key population; refuse any other populated inventory.
"""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Sequence, Union
from uuid import UUID

from alembic import op
import sqlalchemy as sa

revision: str = "aa78b6d901f4"
down_revision: Union[str, Sequence[str], None] = "7e40c64e19a2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_PROJECT_ORGANIZATION = "431bc4f8-c66b-47eb-8d53-2c4622be04e6"
_PROJECT_ENVIRONMENT = "dev"
_APPROVED_KEY_REQUEST_PROJECTS = (
    (
        "ca6ae698-e5fd-4859-882b-b54ef10250e3",
        "483276d1-b136-41ac-ad4b-c68307c79621",
        "cb5760f5-c76e-44e6-bf42-1bc02c13310a",
    ),
    (
        "fc078ef7-e36d-4b80-b06e-9de39d074a68",
        "08d071a1-8fa6-4133-a862-cd024fe2e63e",
        "372ca605-aee0-4e05-862a-f96594dd24ca",
    ),
    (
        "b8ef4474-41d9-4359-9d7d-6013909c54c2",
        "d8076991-03a0-4c43-aa3d-22e4409690ec",
        "bb9a8186-f2f3-4cb2-874c-a461d42d1bf1",
    ),
    (
        "f20de50f-2be4-44b4-bc25-306db827fd76",
        "e664cd37-b6d1-4d73-9f57-e8cfde8616f4",
        "f749d1d2-9119-4732-9ca7-21df58da6c5e",
    ),
    (
        "7d9ee5db-a30e-4507-ad6a-7801bf1059c6",
        "b2b20528-7dfc-46b3-9adb-c77c9a420beb",
        "c9970734-457b-4bec-806b-212a5431cbbf",
    ),
    (
        "8e91ab65-dec6-42d5-9214-8dee700d81b6",
        "e8b2a299-31e6-4db5-85a3-fe94b24b2b72",
        "89c9cc80-fe36-4a7c-b110-c8a3c1d0680a",
    ),
    (
        "231d1d24-9104-4496-9ce1-5352af2b51f6",
        "c3f1791a-a1fd-4ecd-a86b-a9e71b042d00",
        "f749d1d2-9119-4732-9ca7-21df58da6c5e",
    ),
    (
        "9008f09f-44d1-4fc1-97ce-f2c026c97bd3",
        "ea09bb27-f113-44cf-9521-05ffca6e19a7",
        "e70b0f22-fe64-40fc-84be-12555064545c",
    ),
    (
        "b49a5484-4c08-4194-92b6-6a8c83a49a8a",
        "11d6053b-b1b2-4670-9587-d5c09126dbe7",
        "29011c39-60be-42cd-95fa-49e575ecf926",
    ),
    (
        "aef93c7c-4b2c-4dda-a1c8-33aaa2e0f120",
        "6fd85856-1fa8-4d1a-a078-5fb7ea595016",
        "77bb3ea8-55de-4d9d-8b6d-f7910285cd13",
    ),
    (
        "76ce5d10-4295-4683-b5b9-98b028ac211f",
        "c91a3358-3ad3-4fc3-b7ac-ce714b67af0e",
        "929fc456-80a7-4d49-9cc4-7b1f47785b43",
    ),
    (
        "73178426-7667-4a52-98ec-41b34198f4fa",
        "655dc53e-f008-43de-bfab-d694ea9d764c",
        "12257d67-d453-426d-9298-4387c37741ee",
    ),
    (
        "ce1c0bc1-9f94-4d86-adec-d11701033f7d",
        "ba8e447f-5a33-4f86-b856-e6589ded298a",
        "4fd784ba-0d38-4db2-81bd-251ed2aa5593",
    ),
    (
        "ae1e074b-8a9c-430a-a2e9-bf61ac5684de",
        "208ac89c-187c-479a-b2b1-671fd7f2889e",
        "f05afb28-95c7-41fe-9f74-b899874a6689",
    ),
)


def upgrade() -> None:
    connection = op.get_bind()
    key_projects = {
        UUID(key_id): UUID(project_id)
        for key_id, _, project_id in _APPROVED_KEY_REQUEST_PROJECTS
    }
    request_keys = {
        UUID(request_id): UUID(key_id)
        for key_id, request_id, _ in _APPROVED_KEY_REQUEST_PROJECTS
    }
    key_ids = set(
        connection.execute(
            sa.text("SELECT id FROM managed_keys").columns(id=sa.Uuid())
        ).scalars()
    )
    requests = connection.execute(
        sa.text("SELECT id, key_id, kind, status FROM managed_key_requests").columns(
            id=sa.Uuid(), key_id=sa.Uuid()
        )
    ).all()
    populated = bool(key_ids or requests)
    if populated and (
        key_ids != key_projects.keys()
        or {request.id: request.key_id for request in requests} != request_keys
        or any(
            request.kind != "create" or request.status != "approved"
            for request in requests
        )
    ):
        raise RuntimeError(
            "Managed keys and requests differ from the approved DEV migration inventory"
        )

    op.create_table(
        "managed_actors",
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("issuer", sa.String(), nullable=False),
        sa.Column("subject", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("kind", "issuer", "subject"),
        sa.CheckConstraint("kind IN ('human', 'machine', 'system')", name="actor_kind"),
    )
    op.execute(
        "INSERT INTO managed_actors (kind, issuer, subject) "
        "SELECT 'human', issuer, subject FROM managed_humans "
        "WHERE (issuer, subject) != ('managed-auth-system', 'eligibility-sync')"
    )
    op.execute(
        "INSERT INTO managed_actors (kind, issuer, subject) "
        "SELECT DISTINCT actor_kind, actor_issuer, actor_subject FROM managed_audit WHERE true "
        "ON CONFLICT (kind, issuer, subject) DO NOTHING"
    )
    op.create_table(
        "managed_machine_model_grants",
        sa.Column("actor_kind", sa.String(), nullable=False),
        sa.Column("issuer", sa.String(), nullable=False),
        sa.Column("subject", sa.String(), nullable=False),
        sa.Column(
            "group_id",
            sa.Uuid(),
            sa.ForeignKey("managed_model_groups.id"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("actor_kind", "issuer", "subject", "group_id"),
        sa.CheckConstraint("actor_kind = 'machine'", name="machine_model_grant_actor"),
        sa.ForeignKeyConstraint(
            ["actor_kind", "issuer", "subject"],
            ["managed_actors.kind", "managed_actors.issuer", "managed_actors.subject"],
        ),
    )
    op.create_table(
        "managed_project_policies",
        sa.Column("project_id", sa.Uuid(), primary_key=True),
        sa.Column("project_organization", sa.String(), nullable=False),
        sa.Column("project_environment", sa.String(), nullable=False),
        sa.Column("period", sa.String()),
        sa.Column("base_limit_usd", sa.Numeric()),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "(period IS NULL AND base_limit_usd IS NULL) OR "
            "(period IS NOT NULL AND base_limit_usd IS NOT NULL AND "
            "period IN ('daily', 'weekly', 'monthly') AND base_limit_usd >= 0)",
            name="project_policy_cap_pair",
        ),
    )
    op.create_table(
        "managed_project_limit_requests",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "project_id",
            sa.Uuid(),
            sa.ForeignKey("managed_project_policies.project_id"),
            nullable=False,
        ),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("increase_usd", sa.Numeric(), nullable=False),
        sa.Column("period_start", sa.DateTime(timezone=True)),
        sa.Column("period_end", sa.DateTime(timezone=True)),
        sa.Column("reason", sa.String(), nullable=False),
        sa.Column("creator_issuer", sa.String(), nullable=False),
        sa.Column("creator_subject", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("decided_by_issuer", sa.String()),
        sa.Column("decided_by_subject", sa.String()),
        sa.Column("decided_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("increase_usd > 0", name="limit_request_amount"),
        sa.CheckConstraint(
            "(kind = 'permanent' AND period_start IS NULL AND period_end IS NULL) OR "
            "(kind = 'temporary' AND ("
            "(status = 'approved' AND period_start IS NOT NULL AND period_end IS NOT NULL AND period_start < period_end) OR "
            "(status IN ('pending', 'denied') AND period_start IS NULL AND period_end IS NULL)))",
            name="limit_request_period",
        ),
        sa.CheckConstraint(
            "(status = 'pending' AND decided_at IS NULL AND decided_by_issuer IS NULL AND decided_by_subject IS NULL) OR "
            "(status IN ('approved', 'denied') AND decided_at IS NOT NULL AND decided_by_issuer IS NOT NULL AND decided_by_subject IS NOT NULL)",
            name="limit_request_state",
        ),
        sa.ForeignKeyConstraint(
            ["creator_issuer", "creator_subject"],
            ["managed_humans.issuer", "managed_humans.subject"],
        ),
        sa.ForeignKeyConstraint(
            ["decided_by_issuer", "decided_by_subject"],
            ["managed_humans.issuer", "managed_humans.subject"],
        ),
    )
    op.add_column("managed_keys", sa.Column("creator_kind", sa.String()))
    op.add_column("managed_keys", sa.Column("project_id", sa.Uuid()))
    op.add_column("managed_keys", sa.Column("key_limit_period", sa.String()))
    op.add_column("managed_keys", sa.Column("key_limit_usd", sa.Numeric()))
    op.add_column("managed_key_requests", sa.Column("creator_kind", sa.String()))
    op.add_column("managed_key_requests", sa.Column("project_id", sa.Uuid()))
    op.add_column(
        "managed_key_requests", sa.Column("project_organization", sa.String())
    )
    op.add_column("managed_key_requests", sa.Column("project_environment", sa.String()))

    if populated:
        policy = sa.table(
            "managed_project_policies",
            sa.column("project_id", sa.Uuid()),
            sa.column("project_organization", sa.String()),
            sa.column("project_environment", sa.String()),
            sa.column("period", sa.String()),
            sa.column("base_limit_usd", sa.Numeric()),
            sa.column("revision", sa.Integer()),
            sa.column("created_at", sa.DateTime(timezone=True)),
            sa.column("updated_at", sa.DateTime(timezone=True)),
        )
        now = datetime.now(UTC)
        connection.execute(
            sa.insert(policy),
            [
                {
                    "project_id": project_id,
                    "project_organization": _PROJECT_ORGANIZATION,
                    "project_environment": _PROJECT_ENVIRONMENT,
                    "period": "monthly",
                    "base_limit_usd": Decimal("10000"),
                    "revision": 1,
                    "created_at": now,
                    "updated_at": now,
                }
                for project_id in dict.fromkeys(key_projects.values())
            ],
        )
        key = sa.table(
            "managed_keys",
            sa.column("id", sa.Uuid()),
            sa.column("creator_kind", sa.String()),
            sa.column("project_id", sa.Uuid()),
        )
        connection.execute(
            sa.update(key)
            .where(key.c.id == sa.bindparam("approved_key_id", type_=sa.Uuid()))
            .values(
                creator_kind="human",
                project_id=sa.bindparam("approved_project_id", type_=sa.Uuid()),
            ),
            [
                {"approved_key_id": key_id, "approved_project_id": project_id}
                for key_id, project_id in key_projects.items()
            ],
        )
        connection.execute(
            sa.text(
                "UPDATE managed_key_requests AS request "
                "SET creator_kind = 'human', project_id = key.project_id, "
                "project_organization = policy.project_organization, "
                "project_environment = policy.project_environment "
                "FROM managed_keys AS key "
                "JOIN managed_project_policies AS policy ON policy.project_id = key.project_id "
                "WHERE request.key_id = key.id"
            )
        )

    op.alter_column("managed_keys", "creator_kind", nullable=False)
    op.alter_column("managed_keys", "project_id", nullable=False)
    op.alter_column("managed_key_requests", "creator_kind", nullable=False)
    op.alter_column("managed_key_requests", "project_id", nullable=False)
    op.alter_column("managed_key_requests", "project_organization", nullable=False)
    op.alter_column("managed_key_requests", "project_environment", nullable=False)
    op.create_check_constraint(
        "key_limit_cap_pair",
        "managed_keys",
        "(key_limit_period IS NULL AND key_limit_usd IS NULL) OR "
        "(key_limit_period IS NOT NULL AND key_limit_usd IS NOT NULL AND "
        "key_limit_period IN ('daily', 'weekly', 'monthly') AND key_limit_usd >= 0)",
    )
    op.create_check_constraint(
        "key_creator_kind", "managed_keys", "creator_kind IN ('human', 'machine')"
    )
    op.create_check_constraint(
        "request_creator_kind",
        "managed_key_requests",
        "creator_kind IN ('human', 'machine')",
    )
    op.drop_constraint(
        "managed_keys_creator_issuer_creator_subject_fkey",
        "managed_keys",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "managed_keys_creator_actor_fkey",
        "managed_keys",
        "managed_actors",
        ["creator_kind", "creator_issuer", "creator_subject"],
        ["kind", "issuer", "subject"],
    )
    op.create_foreign_key(
        "managed_keys_project_fkey",
        "managed_keys",
        "managed_project_policies",
        ["project_id"],
        ["project_id"],
    )
    op.drop_constraint(
        "managed_key_requests_creator_issuer_creator_subject_fkey",
        "managed_key_requests",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "managed_key_requests_creator_actor_fkey",
        "managed_key_requests",
        "managed_actors",
        ["creator_kind", "creator_issuer", "creator_subject"],
        ["kind", "issuer", "subject"],
    )
    op.drop_constraint(
        "managed_audit_actor_issuer_actor_subject_fkey",
        "managed_audit",
        type_="foreignkey",
    )
    op.drop_constraint("audit_actor_kind", "managed_audit", type_="check")
    op.create_check_constraint(
        "audit_actor_kind",
        "managed_audit",
        "actor_kind IN ('human', 'machine', 'system')",
    )
    op.create_foreign_key(
        "managed_audit_typed_actor_fkey",
        "managed_audit",
        "managed_actors",
        ["actor_kind", "actor_issuer", "actor_subject"],
        ["kind", "issuer", "subject"],
    )
    op.execute(
        """CREATE FUNCTION managed_key_project_immutable() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            IF NEW.project_id IS DISTINCT FROM OLD.project_id THEN
                RAISE EXCEPTION 'issued key project cannot change';
            END IF;
            RETURN NEW;
        END $$"""
    )
    op.execute(
        """CREATE TRIGGER managed_key_project_immutable_trigger
        BEFORE UPDATE ON managed_keys
        FOR EACH ROW EXECUTE FUNCTION managed_key_project_immutable()"""
    )


def downgrade() -> None:
    raise RuntimeError(
        "Downgrade requires an explicit project and machine actor data transition"
    )
