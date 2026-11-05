from datetime import datetime
from decimal import Decimal
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from model_gateway.budgets.schema import BudgetPeriod
from model_gateway.managed_auth.types import (
    LimitRequestStatus,
    HumanIdentity,
    KeyLimit,
    KeyOwner,
    KeyRequest,
    MachineIdentity,
    ManagedKey,
    PersonalOwner,
    ProjectLimitRequest,
    ProjectPolicy,
    ProjectReference,
    ServiceOwner,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IdentityDTO(StrictModel):
    issuer: str
    subject: str


class GrantIdentityDTO(IdentityDTO):
    kind: Literal["human", "machine"]


class PersonalOwnerDTO(StrictModel):
    kind: Literal["personal"]
    identity: IdentityDTO

    def owner(self) -> PersonalOwner:
        return PersonalOwner(HumanIdentity(**self.identity.model_dump()))


class ServiceOwnerDTO(StrictModel):
    kind: Literal["service"]
    service_id: UUID

    def owner(self) -> ServiceOwner:
        return ServiceOwner(self.service_id)


OwnerDTO = Annotated[PersonalOwnerDTO | ServiceOwnerDTO, Field(discriminator="kind")]


class ProjectDTO(StrictModel):
    project_id: UUID
    organization: str
    environment: str


class KeyLimitInput(StrictModel):
    period: BudgetPeriod
    limit_usd: Decimal = Field(ge=0)

    def limit(self) -> KeyLimit:
        return KeyLimit(self.period, self.limit_usd)


class RequestInput(StrictModel):
    project_id: UUID
    display_name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    owner: OwnerDTO
    key_limit: KeyLimitInput | None


class MachineRequestInput(StrictModel):
    project_id: UUID
    project_organization_id: UUID
    project_environment: Literal["dev", "bench", "prod"]
    display_name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    requested_budget: Decimal = Field(gt=0)
    budget_reason: str = Field(pattern=r"\S")


class InitialPolicyInput(StrictModel):
    period: BudgetPeriod
    base_limit_usd: Decimal = Field(gt=0)


class ApprovalInput(StrictModel):
    initial_policy: InitialPolicyInput | None = None

    @model_validator(mode="after")
    def explicit_policy(self) -> "ApprovalInput":
        if "initial_policy" in self.model_fields_set and self.initial_policy is None:
            raise ValueError("initial_policy must be an object")
        return self


class RevisionInput(StrictModel):
    expected_revision: int = Field(ge=1)


class KeyLimitChangeInput(StrictModel):
    expected_revision: int = Field(ge=1)
    key_limit: KeyLimitInput | None


class KeyDTO(StrictModel):
    id: UUID
    creator: GrantIdentityDTO
    owner: OwnerDTO
    project: ProjectDTO
    key_limit: KeyLimitInput | None
    status: str
    revision: int
    display_name: str
    description: str | None
    created_at: datetime
    updated_at: datetime


class SecretDTO(StrictModel):
    key: KeyDTO
    secret: str


class RequestDTO(StrictModel):
    id: UUID
    kind: str
    display_name: str
    reason: str
    requested_key_limit: KeyLimitInput | None
    requested_budget: Decimal | None
    budget_reason: str | None
    target_key_id: UUID | None
    target_revision: int | None
    creator: GrantIdentityDTO
    owner: OwnerDTO
    project: ProjectDTO
    status: str
    decided_by: IdentityDTO | None
    decided_at: datetime | None
    key_id: UUID | None
    created_at: datetime


class CatalogDTO(StrictModel):
    models: list[str]
    operations: list[str]


class KeyActionsDTO(StrictModel):
    collect: bool
    rotate: bool
    revoke: bool
    set_limit: bool


class CapabilitiesDTO(StrictModel):
    identity: IdentityDTO
    platform_admin: bool
    can_request: bool
    key_actions: dict[str, KeyActionsDTO]


class ProjectPolicyInput(StrictModel):
    expected_revision: int = Field(ge=1)
    period: BudgetPeriod
    base_limit_usd: Decimal = Field(gt=0)
    reason: str | None = None


class ProjectPolicyDTO(StrictModel):
    project: ProjectDTO
    period: BudgetPeriod
    base_limit_usd: Decimal
    total_limit_usd: Decimal | None
    budget_reason: str | None
    archived_at: datetime | None
    revision: int
    created_at: datetime
    updated_at: datetime


class TemporaryIncreaseInput(StrictModel):
    expected_revision: int = Field(ge=1)
    reason: str = Field(min_length=1)
    increase_usd: Decimal = Field(gt=0)
    expected_period_start: datetime
    expected_period_end: datetime


class TemporarySetInput(StrictModel):
    expected_revision: int = Field(ge=1)
    reason: str = Field(min_length=1)
    limit_usd: Decimal = Field(ge=0)
    expected_period_start: datetime
    expected_period_end: datetime


class TemporaryRemoveInput(StrictModel):
    expected_revision: int = Field(ge=1)
    reason: str = Field(min_length=1)


class LimitRequestDTO(StrictModel):
    id: UUID
    project: ProjectDTO
    kind: Literal["temporary", "permanent"]
    increase_usd: Decimal
    period_start: datetime | None
    period_end: datetime | None
    reason: str
    creator: IdentityDTO
    status: LimitRequestStatus
    decided_by: IdentityDTO | None
    decided_at: datetime | None
    cancelled_at: datetime | None
    cancelled_by: IdentityDTO | None
    cancellation_reason: str | None
    created_at: datetime


class CurrentProjectPeriodDTO(StrictModel):
    start: datetime
    end: datetime
    effective_limit_usd: Decimal
    spend_usd: Decimal | None
    remaining_usd: Decimal | None
    data_through: datetime | None


class ProjectAllowanceDTO(StrictModel):
    policy: ProjectPolicyDTO
    active_temporary_increases: list[LimitRequestDTO]
    current_period: CurrentProjectPeriodDTO


class HistoryExtraDTO(StrictModel):
    id: UUID
    increase_usd: Decimal
    period_start: datetime
    period_end: datetime


class ProjectAllowanceStateDTO(StrictModel):
    period: BudgetPeriod | None
    period_start: datetime | None
    period_end: datetime | None
    base_limit_usd: Decimal | None
    effective_limit_usd: Decimal | None
    active_temporary_increases: list[HistoryExtraDTO]


class ProjectHistoryEventDTO(StrictModel):
    id: UUID
    event: str
    occurred_at: datetime
    actor: IdentityDTO
    reason: str | None
    request: LimitRequestDTO | None
    before: ProjectAllowanceStateDTO | None
    after: ProjectAllowanceStateDTO | None
    affected_request_ids: list[UUID]


class LimitRequestInput(StrictModel):
    kind: Literal["temporary", "permanent"]
    increase_usd: Decimal = Field(gt=0)
    reason: str


class ProjectBudgetInput(StrictModel):
    expected_revision: int = Field(ge=1)
    limit_usd: Decimal = Field(gt=0)
    reason: str


class ProjectBudgetChangeDTO(StrictModel):
    policy: ProjectPolicyDTO
    pending_request: LimitRequestDTO | None


class LimitDecisionInput(StrictModel):
    expected_current_policy_revision: int = Field(ge=1)


class ServiceInput(StrictModel):
    name: str


class ServiceDTO(ServiceInput):
    id: UUID


def identity_dto(identity: HumanIdentity) -> IdentityDTO:
    return IdentityDTO(issuer=identity.issuer, subject=identity.subject)


def owner_dto(owner: KeyOwner) -> PersonalOwnerDTO | ServiceOwnerDTO:
    if isinstance(owner, PersonalOwner):
        return PersonalOwnerDTO(kind="personal", identity=identity_dto(owner.identity))
    return ServiceOwnerDTO(kind="service", service_id=owner.service_id)


def project_dto(project: ProjectReference) -> ProjectDTO:
    return ProjectDTO(
        project_id=project.project_id,
        organization=project.organization,
        environment=project.environment,
    )


def grant_identity_dto(identity: HumanIdentity | MachineIdentity) -> GrantIdentityDTO:
    return GrantIdentityDTO(
        kind="human" if isinstance(identity, HumanIdentity) else "machine",
        issuer=identity.issuer,
        subject=identity.subject,
    )


def key_dto(key: ManagedKey) -> KeyDTO:
    return KeyDTO(
        id=key.id,
        creator=grant_identity_dto(key.creator),
        owner=owner_dto(key.owner),
        project=project_dto(key.project),
        key_limit=KeyLimitInput(
            period=key.key_limit.period, limit_usd=key.key_limit.limit_usd
        )
        if key.key_limit is not None
        else None,
        status=key.status,
        revision=key.revision,
        display_name=key.display_name,
        description=key.description,
        created_at=key.created_at,
        updated_at=key.updated_at,
    )


def request_dto(request: KeyRequest) -> RequestDTO:
    return RequestDTO(
        id=request.id,
        kind=request.kind,
        display_name=request.display_name,
        reason=request.reason,
        requested_key_limit=KeyLimitInput(
            period=request.requested_key_limit.period,
            limit_usd=request.requested_key_limit.limit_usd,
        )
        if request.requested_key_limit is not None
        else None,
        requested_budget=request.requested_budget,
        budget_reason=request.budget_reason,
        target_key_id=request.target_key_id,
        target_revision=request.target_revision,
        creator=grant_identity_dto(request.creator),
        owner=owner_dto(request.owner),
        project=project_dto(request.project),
        status=request.status,
        decided_by=identity_dto(request.decided_by) if request.decided_by else None,
        decided_at=request.decided_at,
        key_id=request.key_id,
        created_at=request.created_at,
    )


def policy_dto(policy: ProjectPolicy) -> ProjectPolicyDTO:
    return ProjectPolicyDTO(
        project=project_dto(policy.project),
        period=policy.period,
        base_limit_usd=policy.base_limit_usd,
        total_limit_usd=policy.total_limit_usd,
        budget_reason=policy.budget_reason,
        archived_at=policy.archived_at,
        revision=policy.revision,
        created_at=policy.created_at,
        updated_at=policy.updated_at,
    )


def limit_request_dto(item: ProjectLimitRequest) -> LimitRequestDTO:
    return LimitRequestDTO(
        id=item.id,
        project=project_dto(item.project),
        kind=item.kind,
        increase_usd=item.increase_usd,
        period_start=item.period_start,
        period_end=item.period_end,
        reason=item.reason,
        creator=identity_dto(item.creator),
        status=item.status,
        decided_by=identity_dto(item.decided_by) if item.decided_by else None,
        decided_at=item.decided_at,
        created_at=item.created_at,
        cancelled_at=item.cancelled_at,
        cancelled_by=identity_dto(item.cancelled_by) if item.cancelled_by else None,
        cancellation_reason=item.cancellation_reason,
    )
