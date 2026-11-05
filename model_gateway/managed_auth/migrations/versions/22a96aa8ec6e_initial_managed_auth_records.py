"""initial managed auth records

Revision ID: 22a96aa8ec6e
Revises:
Create Date: 2026-10-26 02:27:57.926047

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "22a96aa8ec6e"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "managed_humans",
        sa.Column("issuer", sa.String(), nullable=False),
        sa.Column("subject", sa.String(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint("issuer", "subject"),
    )
    op.create_table(
        "managed_model_groups",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("permissions", sa.JSON(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_table(
        "managed_services",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "managed_audit",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("actor_issuer", sa.String(), nullable=False),
        sa.Column("actor_subject", sa.String(), nullable=False),
        sa.Column("actor_kind", sa.String(), server_default="human", nullable=False),
        sa.Column("target_kind", sa.String(), nullable=False),
        sa.Column("target_id", sa.Uuid(), nullable=False),
        sa.Column("change", sa.String(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "actor_kind IN ('human', 'system')", name="audit_actor_kind"
        ),
        sa.ForeignKeyConstraint(
            ["actor_issuer", "actor_subject"],
            ["managed_humans.issuer", "managed_humans.subject"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_managed_audit_target",
        "managed_audit",
        ["target_kind", "target_id"],
        unique=False,
    )
    op.create_table(
        "managed_budget_definitions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("period", sa.String(), nullable=False),
        sa.Column("limit_usd", sa.Numeric(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=True),
        sa.Column("project_organization", sa.String(), nullable=True),
        sa.Column("project_environment", sa.String(), nullable=True),
        sa.Column("key_id", sa.Uuid(), nullable=True),
        sa.CheckConstraint(
            "(kind = 'project' AND project_id IS NOT NULL AND project_organization IS NOT NULL AND project_environment IS NOT NULL AND key_id IS NULL) OR (kind = 'key' AND key_id IS NOT NULL AND project_id IS NULL AND project_organization IS NULL AND project_environment IS NULL)",
            name="budget_scope",
        ),
        sa.CheckConstraint(
            "period IN ('daily', 'weekly', 'monthly') AND limit_usd >= 0",
            name="budget_policy",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("id", "kind"),
    )
    op.create_index(
        "uq_managed_project_budget_target",
        "managed_budget_definitions",
        ["project_id"],
        unique=True,
        postgresql_where=sa.text("kind = 'project'"),
    )
    op.create_table(
        "managed_human_model_grants",
        sa.Column("issuer", sa.String(), nullable=False),
        sa.Column("subject", sa.String(), nullable=False),
        sa.Column("group_id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(
            ["group_id"],
            ["managed_model_groups.id"],
        ),
        sa.ForeignKeyConstraint(
            ["issuer", "subject"],
            ["managed_humans.issuer", "managed_humans.subject"],
        ),
        sa.PrimaryKeyConstraint("issuer", "subject", "group_id"),
    )
    op.create_table(
        "managed_service_managers",
        sa.Column("service_id", sa.Uuid(), nullable=False),
        sa.Column("issuer", sa.String(), nullable=False),
        sa.Column("subject", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(
            ["issuer", "subject"],
            ["managed_humans.issuer", "managed_humans.subject"],
        ),
        sa.ForeignKeyConstraint(
            ["service_id"],
            ["managed_services.id"],
        ),
        sa.PrimaryKeyConstraint("service_id", "issuer", "subject"),
    )
    op.create_table(
        "managed_human_funding_grants",
        sa.Column("issuer", sa.String(), nullable=False),
        sa.Column("subject", sa.String(), nullable=False),
        sa.Column("budget_id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(
            ["budget_id"],
            ["managed_budget_definitions.id"],
        ),
        sa.ForeignKeyConstraint(
            ["issuer", "subject"],
            ["managed_humans.issuer", "managed_humans.subject"],
        ),
        sa.PrimaryKeyConstraint("issuer", "subject", "budget_id"),
    )
    op.create_table(
        "managed_keys",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("display_name", sa.String(), nullable=False),
        sa.Column("creator_issuer", sa.String(), nullable=False),
        sa.Column("creator_subject", sa.String(), nullable=False),
        sa.Column("owner_kind", sa.String(), nullable=False),
        sa.Column("owner_issuer", sa.String(), nullable=True),
        sa.Column("owner_subject", sa.String(), nullable=True),
        sa.Column("service_id", sa.Uuid(), nullable=True),
        sa.Column("budget_id", sa.Uuid(), nullable=True),
        sa.Column("budget_kind", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("verifier_hash", sa.String(), nullable=True),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "(owner_kind = 'personal' AND owner_issuer IS NOT NULL AND owner_subject IS NOT NULL AND service_id IS NULL) OR (owner_kind = 'service' AND service_id IS NOT NULL AND owner_issuer IS NULL AND owner_subject IS NULL)",
            name="key_owner",
        ),
        sa.CheckConstraint(
            "(status = 'pending_collection' AND verifier_hash IS NULL) OR (status = 'active' AND verifier_hash IS NOT NULL) OR status = 'revoked'",
            name="key_state",
        ),
        sa.CheckConstraint(
            "(budget_id IS NULL) = (budget_kind IS NULL)", name="key_budget_pair"
        ),
        sa.CheckConstraint(
            "owner_kind = 'personal' OR budget_id IS NOT NULL",
            name="service_key_budget",
        ),
        sa.ForeignKeyConstraint(
            ["budget_id", "budget_kind"],
            ["managed_budget_definitions.id", "managed_budget_definitions.kind"],
        ),
        sa.ForeignKeyConstraint(
            ["creator_issuer", "creator_subject"],
            ["managed_humans.issuer", "managed_humans.subject"],
        ),
        sa.ForeignKeyConstraint(
            ["owner_issuer", "owner_subject"],
            ["managed_humans.issuer", "managed_humans.subject"],
        ),
        sa.ForeignKeyConstraint(
            ["service_id"],
            ["managed_services.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("verifier_hash"),
    )
    op.create_table(
        "managed_key_groups",
        sa.Column("key_id", sa.Uuid(), nullable=False),
        sa.Column("group_id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(
            ["group_id"],
            ["managed_model_groups.id"],
        ),
        sa.ForeignKeyConstraint(
            ["key_id"],
            ["managed_keys.id"],
        ),
        sa.PrimaryKeyConstraint("key_id", "group_id"),
    )
    op.create_table(
        "managed_key_requests",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("creator_issuer", sa.String(), nullable=False),
        sa.Column("creator_subject", sa.String(), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("display_name", sa.String(), nullable=False),
        sa.Column("reason", sa.String(), nullable=False),
        sa.Column("target_key_id", sa.Uuid(), nullable=True),
        sa.Column("target_revision", sa.Integer(), nullable=True),
        sa.Column("owner_kind", sa.String(), nullable=False),
        sa.Column("owner_issuer", sa.String(), nullable=True),
        sa.Column("owner_subject", sa.String(), nullable=True),
        sa.Column("service_id", sa.Uuid(), nullable=True),
        sa.Column("requested_budget_id", sa.Uuid(), nullable=True),
        sa.Column("requested_budget_kind", sa.String(), nullable=True),
        sa.Column("approved_budget_id", sa.Uuid(), nullable=True),
        sa.Column("approved_budget_kind", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("decided_by_issuer", sa.String(), nullable=True),
        sa.Column("decided_by_subject", sa.String(), nullable=True),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("key_id", sa.Uuid(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "(kind = 'create' AND target_key_id IS NULL AND target_revision IS NULL) OR (kind = 'change' AND target_key_id IS NOT NULL AND target_revision IS NOT NULL)",
            name="request_kind",
        ),
        sa.CheckConstraint(
            "(owner_kind = 'personal' AND owner_issuer IS NOT NULL AND owner_subject IS NOT NULL AND service_id IS NULL) OR (owner_kind = 'service' AND service_id IS NOT NULL AND owner_issuer IS NULL AND owner_subject IS NULL)",
            name="request_owner",
        ),
        sa.CheckConstraint(
            "(status = 'pending' AND decided_at IS NULL AND key_id IS NULL AND approved_budget_id IS NULL AND approved_budget_kind IS NULL) OR (status = 'denied' AND decided_at IS NOT NULL AND key_id IS NULL AND approved_budget_id IS NULL AND approved_budget_kind IS NULL) OR (status = 'approved' AND decided_at IS NOT NULL AND key_id IS NOT NULL)",
            name="request_state",
        ),
        sa.CheckConstraint(
            "(requested_budget_id IS NULL) = (requested_budget_kind IS NULL)",
            name="requested_budget_pair",
        ),
        sa.CheckConstraint(
            "(approved_budget_id IS NULL) = (approved_budget_kind IS NULL)",
            name="approved_budget_pair",
        ),
        sa.CheckConstraint(
            "owner_kind = 'personal' OR requested_budget_id IS NOT NULL",
            name="service_requested_budget",
        ),
        sa.CheckConstraint(
            "owner_kind = 'personal' OR status != 'approved' OR approved_budget_id IS NOT NULL",
            name="service_approved_budget",
        ),
        sa.ForeignKeyConstraint(
            ["approved_budget_id", "approved_budget_kind"],
            ["managed_budget_definitions.id", "managed_budget_definitions.kind"],
        ),
        sa.ForeignKeyConstraint(
            ["creator_issuer", "creator_subject"],
            ["managed_humans.issuer", "managed_humans.subject"],
        ),
        sa.ForeignKeyConstraint(
            ["decided_by_issuer", "decided_by_subject"],
            ["managed_humans.issuer", "managed_humans.subject"],
        ),
        sa.ForeignKeyConstraint(
            ["key_id"],
            ["managed_keys.id"],
        ),
        sa.ForeignKeyConstraint(
            ["owner_issuer", "owner_subject"],
            ["managed_humans.issuer", "managed_humans.subject"],
        ),
        sa.ForeignKeyConstraint(
            ["requested_budget_id", "requested_budget_kind"],
            ["managed_budget_definitions.id", "managed_budget_definitions.kind"],
        ),
        sa.ForeignKeyConstraint(
            ["service_id"],
            ["managed_services.id"],
        ),
        sa.ForeignKeyConstraint(
            ["target_key_id"],
            ["managed_keys.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "managed_request_groups",
        sa.Column("request_id", sa.Uuid(), nullable=False),
        sa.Column("group_id", sa.Uuid(), nullable=False),
        sa.Column("approved", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(
            ["group_id"],
            ["managed_model_groups.id"],
        ),
        sa.ForeignKeyConstraint(
            ["request_id"],
            ["managed_key_requests.id"],
        ),
        sa.PrimaryKeyConstraint("request_id", "group_id", "approved"),
    )
    op.execute("""CREATE FUNCTION managed_audit_immutable() RETURNS trigger
    LANGUAGE plpgsql AS $$ BEGIN
        RAISE EXCEPTION 'managed audit events are immutable';
    END $$""")
    op.execute("""CREATE TRIGGER managed_audit_immutable_trigger
    BEFORE UPDATE OR DELETE ON managed_audit
    FOR EACH ROW EXECUTE FUNCTION managed_audit_immutable()""")
    op.execute(
        "INSERT INTO managed_humans (issuer, subject, enabled) VALUES ('managed-auth-system', 'eligibility-sync', true) ON CONFLICT (issuer, subject) DO NOTHING"
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER managed_audit_immutable_trigger ON managed_audit")
    op.execute("DROP FUNCTION managed_audit_immutable()")
    op.drop_table("managed_request_groups")
    op.drop_table("managed_key_requests")
    op.drop_table("managed_key_groups")
    op.drop_table("managed_keys")
    op.drop_table("managed_human_funding_grants")
    op.drop_table("managed_service_managers")
    op.drop_table("managed_human_model_grants")
    op.drop_index(
        "uq_managed_project_budget_target", table_name="managed_budget_definitions"
    )
    op.drop_table("managed_budget_definitions")
    op.drop_index("ix_managed_audit_target", table_name="managed_audit")
    op.drop_table("managed_audit")
    op.drop_table("managed_services")
    op.drop_table("managed_model_groups")
    op.drop_table("managed_humans")
