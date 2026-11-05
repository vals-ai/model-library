from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    Numeric,
    PrimaryKeyConstraint,
    String,
    Table,
    UniqueConstraint,
    Uuid,
    text,
)

from model_gateway.managed_auth.database import metadata

humans = Table(
    "managed_humans",
    metadata,
    Column("issuer", String, nullable=False),
    Column("subject", String, nullable=False),
    Column("enabled", Boolean, nullable=False),
    PrimaryKeyConstraint("issuer", "subject"),
)

actors = Table(
    "managed_actors",
    metadata,
    Column("kind", String, nullable=False),
    Column("issuer", String, nullable=False),
    Column("subject", String, nullable=False),
    PrimaryKeyConstraint("kind", "issuer", "subject"),
    CheckConstraint("kind IN ('human', 'machine', 'system')", name="actor_kind"),
)

project_policies = Table(
    "managed_project_policies",
    metadata,
    Column("project_id", Uuid(), primary_key=True),
    Column("project_organization", String, nullable=False),
    Column("project_environment", String, nullable=False),
    Column("period", String, nullable=False),
    Column("base_limit_usd", Numeric(), nullable=False),
    Column("total_limit_usd", Numeric()),
    Column("total_started_at", DateTime(timezone=True)),
    Column("revision", Integer, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("budget_reason", String),
    Column("archived_at", DateTime(timezone=True)),
    CheckConstraint(
        "(period IS NULL AND base_limit_usd IS NULL) OR "
        "(period IS NOT NULL AND base_limit_usd IS NOT NULL AND "
        "period IN ('daily', 'weekly', 'monthly') AND base_limit_usd >= 0)",
        name="project_policy_cap_pair",
    ),
    CheckConstraint(
        "period IN ('daily', 'weekly', 'monthly') AND base_limit_usd > 0",
        name="project_policy_positive_base",
    ),
    CheckConstraint(
        "total_limit_usd IS NULL OR total_limit_usd > 0",
        name="project_policy_positive_total",
    ),
)

project_limit_requests = Table(
    "managed_project_limit_requests",
    metadata,
    Column("id", Uuid(), primary_key=True),
    Column(
        "project_id",
        Uuid(),
        ForeignKey("managed_project_policies.project_id"),
        nullable=False,
    ),
    Column("kind", String, nullable=False),
    Column("increase_usd", Numeric(), nullable=False),
    Column("target_total_usd", Numeric()),
    Column("source_revision", Integer),
    Column("period_start", DateTime(timezone=True)),
    Column("period_end", DateTime(timezone=True)),
    Column("reason", String, nullable=False),
    Column("creator_issuer", String, nullable=False),
    Column("creator_subject", String, nullable=False),
    Column("status", String, nullable=False),
    Column("decided_by_issuer", String),
    Column("decided_by_subject", String),
    Column("decided_at", DateTime(timezone=True)),
    Column("cancelled_at", DateTime(timezone=True)),
    Column("cancelled_by_issuer", String),
    Column("cancelled_by_subject", String),
    Column("cancellation_reason", String),
    Column("created_at", DateTime(timezone=True), nullable=False),
    CheckConstraint("increase_usd > 0", name="limit_request_amount"),
    CheckConstraint(
        "(kind = 'permanent' AND period_start IS NULL AND period_end IS NULL) OR "
        "(kind = 'temporary' AND ("
        "(status = 'approved' AND period_start IS NOT NULL AND period_end IS NOT NULL AND period_start < period_end) OR "
        "(status IN ('pending', 'denied') AND period_start IS NULL AND period_end IS NULL)))",
        name="limit_request_period",
    ),
    CheckConstraint(
        "(status = 'pending' AND decided_at IS NULL AND decided_by_issuer IS NULL AND decided_by_subject IS NULL) OR "
        "(status IN ('approved', 'denied') AND decided_at IS NOT NULL AND decided_by_issuer IS NOT NULL AND decided_by_subject IS NOT NULL)",
        name="limit_request_state",
    ),
    CheckConstraint(
        "(cancelled_at IS NULL AND cancelled_by_issuer IS NULL AND cancelled_by_subject IS NULL AND cancellation_reason IS NULL) OR "
        "(kind = 'temporary' AND status = 'approved' AND cancelled_at IS NOT NULL AND "
        "cancelled_by_issuer IS NOT NULL AND cancelled_by_subject IS NOT NULL AND cancellation_reason IS NOT NULL)",
        name="limit_request_cancellation",
    ),
    ForeignKeyConstraint(
        ["creator_issuer", "creator_subject"],
        ["managed_humans.issuer", "managed_humans.subject"],
    ),
    ForeignKeyConstraint(
        ["decided_by_issuer", "decided_by_subject"],
        ["managed_humans.issuer", "managed_humans.subject"],
    ),
    ForeignKeyConstraint(
        ["cancelled_by_issuer", "cancelled_by_subject"],
        ["managed_humans.issuer", "managed_humans.subject"],
    ),
)

services = Table(
    "managed_services",
    metadata,
    Column("id", Uuid(), primary_key=True),
    Column("name", String, nullable=False),
)

service_managers = Table(
    "managed_service_managers",
    metadata,
    Column("service_id", Uuid(), ForeignKey("managed_services.id"), nullable=False),
    Column("issuer", String, nullable=False),
    Column("subject", String, nullable=False),
    PrimaryKeyConstraint("service_id", "issuer", "subject"),
    ForeignKeyConstraint(
        ["issuer", "subject"], ["managed_humans.issuer", "managed_humans.subject"]
    ),
)

groups = Table(
    "managed_model_groups",
    metadata,
    Column("id", Uuid(), primary_key=True),
    Column("name", String, nullable=False, unique=True),
    Column("permissions", JSON(), nullable=False),
    Column("revision", Integer, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

budgets = Table(
    "managed_budget_definitions",
    metadata,
    Column("id", Uuid(), primary_key=True),
    Column("kind", String, nullable=False),
    Column("period", String, nullable=False),
    Column("limit_usd", Numeric(), nullable=False),
    Column("project_id", Uuid()),
    Column("project_organization", String),
    Column("project_environment", String),
    Column("key_id", Uuid()),
    CheckConstraint(
        "(kind = 'project' AND project_id IS NOT NULL AND project_organization IS NOT NULL AND project_environment IS NOT NULL AND key_id IS NULL) OR (kind = 'key' AND key_id IS NOT NULL AND project_id IS NULL AND project_organization IS NULL AND project_environment IS NULL)",
        name="budget_scope",
    ),
    CheckConstraint(
        "period IN ('daily', 'weekly', 'monthly') AND limit_usd >= 0",
        name="budget_policy",
    ),
    UniqueConstraint("id", "kind"),
)
Index(
    "uq_managed_project_budget_target",
    budgets.c.project_id,
    unique=True,
    postgresql_where=budgets.c.kind == "project",
)


human_model_grants = Table(
    "managed_human_model_grants",
    metadata,
    Column("issuer", String, nullable=False),
    Column("subject", String, nullable=False),
    Column("group_id", Uuid(), ForeignKey("managed_model_groups.id"), nullable=False),
    PrimaryKeyConstraint("issuer", "subject", "group_id"),
    ForeignKeyConstraint(
        ["issuer", "subject"], ["managed_humans.issuer", "managed_humans.subject"]
    ),
)

machine_model_grants = Table(
    "managed_machine_model_grants",
    metadata,
    Column("actor_kind", String, nullable=False),
    Column("issuer", String, nullable=False),
    Column("subject", String, nullable=False),
    Column("group_id", Uuid(), ForeignKey("managed_model_groups.id"), nullable=False),
    PrimaryKeyConstraint("actor_kind", "issuer", "subject", "group_id"),
    CheckConstraint("actor_kind = 'machine'", name="machine_model_grant_actor"),
    ForeignKeyConstraint(
        ["actor_kind", "issuer", "subject"],
        ["managed_actors.kind", "managed_actors.issuer", "managed_actors.subject"],
    ),
)

human_funding_grants = Table(
    "managed_human_funding_grants",
    metadata,
    Column("issuer", String, nullable=False),
    Column("subject", String, nullable=False),
    Column(
        "budget_id", Uuid(), ForeignKey("managed_budget_definitions.id"), nullable=False
    ),
    PrimaryKeyConstraint("issuer", "subject", "budget_id"),
    ForeignKeyConstraint(
        ["issuer", "subject"], ["managed_humans.issuer", "managed_humans.subject"]
    ),
)

keys = Table(
    "managed_keys",
    metadata,
    Column("id", Uuid(), primary_key=True),
    Column("display_name", String, nullable=False),
    Column("description", String),
    Column("creator_issuer", String, nullable=False),
    Column("creator_subject", String, nullable=False),
    Column("creator_kind", String, nullable=False),
    Column(
        "project_id",
        Uuid(),
        ForeignKey("managed_project_policies.project_id"),
        nullable=False,
    ),
    Column("owner_kind", String, nullable=False),
    Column("owner_issuer", String),
    Column("owner_subject", String),
    Column("service_id", Uuid(), ForeignKey("managed_services.id")),
    Column("budget_id", Uuid()),
    Column("budget_kind", String),
    Column("status", String, nullable=False),
    Column("verifier_hash", String, unique=True),
    Column("key_limit_period", String),
    Column("key_limit_usd", Numeric()),
    Column("revision", Integer, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    CheckConstraint("creator_kind IN ('human', 'machine')", name="key_creator_kind"),
    CheckConstraint("length(description) > 0", name="key_description_nonempty"),
    CheckConstraint(
        "(owner_kind = 'personal' AND owner_issuer IS NOT NULL AND owner_subject IS NOT NULL AND service_id IS NULL) OR (owner_kind = 'service' AND service_id IS NOT NULL AND owner_issuer IS NULL AND owner_subject IS NULL)",
        name="key_owner",
    ),
    CheckConstraint(
        "(status = 'pending_collection' AND verifier_hash IS NULL) OR (status = 'active' AND verifier_hash IS NOT NULL) OR status = 'revoked'",
        name="key_state",
    ),
    CheckConstraint(
        "(budget_id IS NULL) = (budget_kind IS NULL)", name="key_budget_pair"
    ),
    CheckConstraint(
        "(key_limit_period IS NULL AND key_limit_usd IS NULL) OR "
        "(key_limit_period IS NOT NULL AND key_limit_usd IS NOT NULL AND "
        "key_limit_period IN ('daily', 'weekly', 'monthly') AND key_limit_usd >= 0)",
        name="key_limit_cap_pair",
    ),
    ForeignKeyConstraint(
        ["creator_kind", "creator_issuer", "creator_subject"],
        ["managed_actors.kind", "managed_actors.issuer", "managed_actors.subject"],
    ),
    ForeignKeyConstraint(
        ["owner_issuer", "owner_subject"],
        ["managed_humans.issuer", "managed_humans.subject"],
    ),
    ForeignKeyConstraint(
        ["budget_id", "budget_kind"],
        ["managed_budget_definitions.id", "managed_budget_definitions.kind"],
    ),
)

key_groups = Table(
    "managed_key_groups",
    metadata,
    Column("key_id", Uuid(), ForeignKey("managed_keys.id"), nullable=False),
    Column("group_id", Uuid(), ForeignKey("managed_model_groups.id"), nullable=False),
    PrimaryKeyConstraint("key_id", "group_id"),
)

requests = Table(
    "managed_key_requests",
    metadata,
    Column("id", Uuid(), primary_key=True),
    Column("creator_issuer", String, nullable=False),
    Column("creator_subject", String, nullable=False),
    Column("creator_kind", String, nullable=False),
    Column("project_id", Uuid(), nullable=False),
    Column("project_organization", String, nullable=False),
    Column("project_environment", String, nullable=False),
    Column("kind", String, nullable=False),
    Column("display_name", String, nullable=False),
    Column("reason", String, nullable=False),
    Column("requested_key_limit_period", String),
    Column("requested_key_limit_usd", Numeric()),
    Column("requested_budget", Numeric()),
    Column("budget_reason", String),
    Column("target_key_id", Uuid(), ForeignKey("managed_keys.id")),
    Column("target_revision", Integer),
    Column("owner_kind", String, nullable=False),
    Column("owner_issuer", String),
    Column("owner_subject", String),
    Column("service_id", Uuid(), ForeignKey("managed_services.id")),
    Column("requested_budget_id", Uuid()),
    Column("requested_budget_kind", String),
    Column("approved_budget_id", Uuid()),
    Column("approved_budget_kind", String),
    Column("status", String, nullable=False),
    Column("decided_by_issuer", String),
    Column("decided_by_subject", String),
    Column("decided_at", DateTime(timezone=True)),
    Column("key_id", Uuid(), ForeignKey("managed_keys.id")),
    Column("created_at", DateTime(timezone=True), nullable=False),
    CheckConstraint(
        "creator_kind IN ('human', 'machine')", name="request_creator_kind"
    ),
    CheckConstraint(
        "(owner_kind = 'personal' AND owner_issuer IS NOT NULL AND owner_subject IS NOT NULL AND service_id IS NULL) OR (owner_kind = 'service' AND service_id IS NOT NULL AND owner_issuer IS NULL AND owner_subject IS NULL)",
        name="request_owner",
    ),
    CheckConstraint(
        "(kind = 'create' AND target_key_id IS NULL AND target_revision IS NULL) OR (kind = 'change' AND target_key_id IS NOT NULL AND target_revision IS NOT NULL)",
        name="request_kind",
    ),
    CheckConstraint(
        "(status = 'pending' AND decided_at IS NULL AND key_id IS NULL AND approved_budget_id IS NULL AND approved_budget_kind IS NULL) OR (status = 'denied' AND decided_at IS NOT NULL AND key_id IS NULL AND approved_budget_id IS NULL AND approved_budget_kind IS NULL) OR (status = 'approved' AND decided_at IS NOT NULL AND key_id IS NOT NULL)",
        name="request_state",
    ),
    CheckConstraint(
        "(requested_key_limit_period IS NULL AND requested_key_limit_usd IS NULL) OR "
        "(creator_kind = 'human' AND requested_key_limit_period IS NOT NULL AND "
        "requested_key_limit_usd IS NOT NULL AND requested_key_limit_period IN "
        "('daily', 'weekly', 'monthly') AND requested_key_limit_usd >= 0)",
        name="request_key_limit_cap_pair",
    ),
    CheckConstraint(
        "(requested_budget IS NULL AND budget_reason IS NULL) OR "
        "(requested_budget IS NOT NULL AND requested_budget > 0 AND budget_reason IS NOT NULL)",
        name="requested_budget_reason_pair",
    ),
    CheckConstraint(
        "(requested_budget_id IS NULL) = (requested_budget_kind IS NULL)",
        name="requested_budget_pair",
    ),
    CheckConstraint(
        "(approved_budget_id IS NULL) = (approved_budget_kind IS NULL)",
        name="approved_budget_pair",
    ),
    ForeignKeyConstraint(
        ["creator_kind", "creator_issuer", "creator_subject"],
        ["managed_actors.kind", "managed_actors.issuer", "managed_actors.subject"],
    ),
    ForeignKeyConstraint(
        ["owner_issuer", "owner_subject"],
        ["managed_humans.issuer", "managed_humans.subject"],
    ),
    ForeignKeyConstraint(
        ["decided_by_issuer", "decided_by_subject"],
        ["managed_humans.issuer", "managed_humans.subject"],
    ),
    ForeignKeyConstraint(
        ["requested_budget_id", "requested_budget_kind"],
        ["managed_budget_definitions.id", "managed_budget_definitions.kind"],
    ),
    ForeignKeyConstraint(
        ["approved_budget_id", "approved_budget_kind"],
        ["managed_budget_definitions.id", "managed_budget_definitions.kind"],
    ),
)

request_groups = Table(
    "managed_request_groups",
    metadata,
    Column("request_id", Uuid(), ForeignKey("managed_key_requests.id"), nullable=False),
    Column("group_id", Uuid(), ForeignKey("managed_model_groups.id"), nullable=False),
    Column("approved", Boolean, nullable=False),
    PrimaryKeyConstraint("request_id", "group_id", "approved"),
)

audit = Table(
    "managed_audit",
    metadata,
    Column("id", Uuid(), primary_key=True),
    Column("actor_issuer", String, nullable=False),
    Column("actor_subject", String, nullable=False),
    Column("actor_kind", String, nullable=False, server_default=text("'human'")),
    Column("target_kind", String, nullable=False),
    Column("target_id", Uuid(), nullable=False),
    Column("change", String, nullable=False),
    Column("detail", JSON()),
    Column("occurred_at", DateTime(timezone=True), nullable=False),
    CheckConstraint(
        "actor_kind IN ('human', 'machine', 'system')", name="audit_actor_kind"
    ),
    ForeignKeyConstraint(
        ["actor_kind", "actor_issuer", "actor_subject"],
        ["managed_actors.kind", "managed_actors.issuer", "managed_actors.subject"],
    ),
)
Index("ix_managed_audit_target", audit.c.target_kind, audit.c.target_id)
