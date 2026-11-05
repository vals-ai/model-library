from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Final, Literal
from uuid import UUID

from model_gateway.budgets.schema import BudgetPeriod


@dataclass(frozen=True)
class HumanIdentity:
    issuer: str
    subject: str


@dataclass(frozen=True)
class MachineIdentity:
    issuer: str
    subject: str


@dataclass(frozen=True)
class SystemIdentity:
    issuer: str
    subject: str


ActorIdentity = HumanIdentity | MachineIdentity | SystemIdentity


@dataclass(frozen=True)
class ProjectReference:
    project_id: UUID
    organization: str
    environment: str


@dataclass(frozen=True)
class PersonalOwner:
    identity: HumanIdentity


@dataclass(frozen=True)
class ServiceOwner:
    service_id: UUID


KeyOwner = PersonalOwner | ServiceOwner


@dataclass(frozen=True)
class KeyLimit:
    period: BudgetPeriod
    limit_usd: Decimal


class KeyStatus(StrEnum):
    PENDING_COLLECTION = "pending_collection"
    ACTIVE = "active"
    REVOKED = "revoked"


@dataclass(frozen=True)
class ManagedKey:
    id: UUID
    creator: HumanIdentity | MachineIdentity
    owner: KeyOwner
    description: str | None
    status: KeyStatus
    verifier_hash: str | None
    revision: int
    display_name: str
    created_at: datetime
    updated_at: datetime
    project: ProjectReference
    key_limit: KeyLimit | None


@dataclass(frozen=True)
class KeyRequest:
    id: UUID
    kind: Literal["create", "change"]
    display_name: str
    reason: str
    requested_key_limit: KeyLimit | None
    target_key_id: UUID | None
    target_revision: int | None
    creator: HumanIdentity | MachineIdentity
    owner: KeyOwner
    status: Literal["pending", "approved", "denied"]
    decided_by: HumanIdentity | None
    decided_at: datetime | None
    key_id: UUID | None
    created_at: datetime
    project: ProjectReference
    requested_budget: Decimal | None
    budget_reason: str | None


@dataclass(frozen=True)
class AuditEvent:
    id: UUID
    actor: ActorIdentity
    target_kind: str
    target_id: UUID
    change: str
    occurred_at: datetime
    actor_kind: Literal["human", "machine", "system"]


@dataclass(frozen=True)
class ProjectPolicy:
    project: ProjectReference
    period: BudgetPeriod
    base_limit_usd: Decimal
    revision: int
    created_at: datetime
    updated_at: datetime
    total_limit_usd: Decimal | None = None
    total_started_at: datetime | None = None
    budget_reason: str | None = None
    archived_at: datetime | None = None


@dataclass(frozen=True)
class ProjectAlertIdentity:
    name: str
    handle: str | None


LimitRequestKind = Literal["temporary", "permanent"]


class LimitRequestStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"


TEMPORARY_INCREASE: LimitRequestKind = "temporary"
PERMANENT_INCREASE: LimitRequestKind = "permanent"
PROJECT_POLICY_AUDIT: Final = "project_policy"


@dataclass(frozen=True)
class ProjectLimitRequest:
    id: UUID
    project: ProjectReference
    kind: LimitRequestKind
    increase_usd: Decimal
    period_start: datetime | None
    period_end: datetime | None
    reason: str
    creator: HumanIdentity
    status: LimitRequestStatus
    decided_by: HumanIdentity | None
    decided_at: datetime | None
    created_at: datetime
    cancelled_at: datetime | None
    cancelled_by: HumanIdentity | None
    cancellation_reason: str | None
