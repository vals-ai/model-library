from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal, cast
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from fastapi import APIRouter, Depends, Header, HTTPException, Request

from model_gateway.budgets.schema import period_bounds
from model_gateway.budgets.service import BudgetService
from model_gateway.managed_auth.identity import VerifiedHuman, VerifiedMachine
from model_gateway.managed_auth.schemas import (
    ApprovalInput,
    CapabilitiesDTO,
    CatalogDTO,
    CurrentProjectPeriodDTO,
    IdentityDTO,
    KeyActionsDTO,
    KeyDTO,
    KeyLimitChangeInput,
    LimitDecisionInput,
    LimitRequestDTO,
    LimitRequestInput,
    MachineRequestInput,
    ProjectAllowanceDTO,
    ProjectAllowanceStateDTO,
    ProjectHistoryEventDTO,
    ProjectBudgetChangeDTO,
    ProjectBudgetInput,
    ProjectPolicyDTO,
    ProjectPolicyInput,
    RequestDTO,
    RequestInput,
    RevisionInput,
    SecretDTO,
    ServiceDTO,
    ServiceInput,
    TemporaryIncreaseInput,
    TemporaryRemoveInput,
    TemporarySetInput,
    identity_dto,
    key_dto,
    limit_request_dto,
    policy_dto,
    request_dto,
)
from model_gateway.managed_auth.service import ManagedAuthService
from model_gateway.managed_auth.store import ManagedAuthStore
from model_gateway.managed_auth.types import (
    HumanIdentity,
    PersonalOwner,
    ProjectReference,
    ServiceOwner,
)
from model_gateway.run_token_types import RUN_TOKEN_OPERATION_PATHS
from model_library.registry_utils import get_model_names

router = APIRouter(prefix="/admin/managed-auth", tags=["managed-auth"])


def store(request: Request) -> ManagedAuthStore:
    return request.app.state.managed_auth_store


def service(request: Request) -> ManagedAuthService:
    return request.app.state.managed_auth_service


async def human(
    request: Request, authorization: str | None = Header(default=None)
) -> VerifiedHuman:
    scheme, _, token = authorization.partition(" ") if authorization else ("", "", "")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(401, "Human bearer token required")
    person = await request.app.state.managed_auth_verifier.verify(token)
    await service(request).register(person)
    return person


async def machine(
    request: Request, authorization: str | None = Header(default=None)
) -> VerifiedMachine:
    scheme, _, token = authorization.partition(" ") if authorization else ("", "", "")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(401, "Machine bearer token required")
    return await request.app.state.managed_machine_verifier.verify(token)


def _manages(person: VerifiedHuman, owner: PersonalOwner) -> bool:
    return person.identity == owner.identity


def _service_dto(service_id: UUID, name: str) -> ServiceDTO:
    return ServiceDTO(id=service_id, name=name)


@router.get("/catalog", response_model=CatalogDTO)
async def catalog(_: VerifiedHuman = Depends(human)):
    return {"models": get_model_names(), "operations": list(RUN_TOKEN_OPERATION_PATHS)}


@router.get("/capabilities", response_model=CapabilitiesDTO)
async def capabilities(
    person: VerifiedHuman = Depends(human),
    db: ManagedAuthStore = Depends(store),
) -> CapabilitiesDTO:
    admin = ManagedAuthService.is_admin(person)
    actions: dict[str, KeyActionsDTO] = {}
    for key in await db.list_keys():
        personal = isinstance(key.owner, PersonalOwner)
        manages = isinstance(key.owner, PersonalOwner) and _manages(person, key.owner)
        if manages or admin:
            actions[str(key.id)] = KeyActionsDTO(
                collect=(manages or (admin and not personal))
                and key.status == "pending_collection",
                rotate=(manages or (admin and not personal)) and key.status == "active",
                revoke=key.status != "revoked",
                set_limit=admin,
            )
    return CapabilitiesDTO(
        identity=identity_dto(person.identity),
        platform_admin=admin,
        can_request=True,
        key_actions=actions,
    )


@router.get("/keys", response_model=list[KeyDTO])
async def keys(
    _: VerifiedHuman = Depends(human), db: ManagedAuthStore = Depends(store)
):
    return [key_dto(key) for key in await db.list_keys()]


@router.get("/keys/{key_id}", response_model=KeyDTO)
async def key_detail(
    key_id: UUID,
    _: VerifiedHuman = Depends(human),
    svc: ManagedAuthService = Depends(service),
):
    return key_dto(await svc.key(key_id))


@router.post("/keys/{key_id}/collect", response_model=SecretDTO)
async def collect(
    key_id: UUID,
    body: RevisionInput,
    person: VerifiedHuman = Depends(human),
    svc: ManagedAuthService = Depends(service),
):
    key, secret = await svc.secret_action(
        person, key_id, body.expected_revision, "collect"
    )
    assert secret is not None
    return SecretDTO(key=key_dto(key), secret=secret)


@router.post("/keys/{key_id}/rotate", response_model=SecretDTO)
async def rotate(
    key_id: UUID,
    body: RevisionInput,
    person: VerifiedHuman = Depends(human),
    svc: ManagedAuthService = Depends(service),
):
    key, secret = await svc.secret_action(
        person, key_id, body.expected_revision, "rotate"
    )
    assert secret is not None
    return SecretDTO(key=key_dto(key), secret=secret)


@router.post("/keys/{key_id}/revoke", response_model=KeyDTO)
async def revoke(
    key_id: UUID,
    body: RevisionInput,
    person: VerifiedHuman = Depends(human),
    svc: ManagedAuthService = Depends(service),
):
    key, _ = await svc.secret_action(person, key_id, body.expected_revision, "revoke")
    return key_dto(key)


@router.put("/keys/{key_id}/limit", response_model=KeyDTO)
async def update_key_limit(
    key_id: UUID,
    body: KeyLimitChangeInput,
    person: VerifiedHuman = Depends(human),
    svc: ManagedAuthService = Depends(service),
):
    key_limit = body.key_limit.limit() if body.key_limit is not None else None
    key = await svc.update_key_limit(person, key_id, body.expected_revision, key_limit)
    return key_dto(key)


@router.get("/requests", response_model=list[RequestDTO])
async def requests(
    _: VerifiedHuman = Depends(human), db: ManagedAuthStore = Depends(store)
):
    return [request_dto(item) for item in await db.list_requests()]


@router.post("/requests", response_model=RequestDTO)
async def submit(
    body: RequestInput,
    person: VerifiedHuman = Depends(human),
    svc: ManagedAuthService = Depends(service),
):
    return request_dto(
        await svc.request(
            person,
            body.display_name,
            body.description,
            body.owner.owner(),
            body.key_limit.limit() if body.key_limit is not None else None,
            body.project_id,
        )
    )


@router.get("/requests/{request_id}", response_model=RequestDTO)
async def request_detail(
    request_id: UUID,
    _: VerifiedHuman = Depends(human),
    svc: ManagedAuthService = Depends(service),
):
    return request_dto(await svc.request_detail(request_id))


@router.post("/requests/{request_id}/approve", response_model=KeyDTO)
async def approve(
    request_id: UUID,
    body: ApprovalInput,
    person: VerifiedHuman = Depends(human),
    svc: ManagedAuthService = Depends(service),
    db: ManagedAuthStore = Depends(store),
):
    svc.admin(person)
    item = await svc.request_detail(request_id)
    existing_policy = await db.get_project_policy(item.project.project_id)
    if existing_policy is None and body.initial_policy is None:
        raise HTTPException(422, "First project approval requires initial_policy")
    if existing_policy is not None and body.initial_policy is not None:
        raise HTTPException(422, "Already-approved project cannot set initial_policy")
    initial_policy = (
        (body.initial_policy.period, body.initial_policy.base_limit_usd)
        if body.initial_policy is not None
        else None
    )
    key = await svc.decide(person, request_id, initial_policy, True)
    assert key is not None
    return key_dto(key)


@router.post("/requests/{request_id}/deny", status_code=204)
async def deny(
    request_id: UUID,
    person: VerifiedHuman = Depends(human),
    svc: ManagedAuthService = Depends(service),
):
    await svc.decide(person, request_id, None, False)


@router.post("/machine/requests", response_model=RequestDTO)
async def submit_machine_request(
    body: MachineRequestInput,
    principal: VerifiedMachine = Depends(machine),
    svc: ManagedAuthService = Depends(service),
):
    try:
        request = await svc.machine_request(
            principal,
            body.display_name,
            body.description,
            ProjectReference(
                body.project_id,
                str(body.project_organization_id),
                body.project_environment,
            ),
            body.requested_budget,
            body.budget_reason,
        )
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    return request_dto(request)


@router.get("/machine/requests/{request_id}", response_model=RequestDTO)
async def machine_request_detail(
    request_id: UUID,
    principal: VerifiedMachine = Depends(machine),
    svc: ManagedAuthService = Depends(service),
):
    return request_dto(await svc.machine_request_detail(principal, request_id))


@router.get("/machine/keys/{key_id}", response_model=KeyDTO)
async def machine_key_detail(
    key_id: UUID,
    principal: VerifiedMachine = Depends(machine),
    svc: ManagedAuthService = Depends(service),
):
    return key_dto(await svc.machine_key(principal, key_id))


@router.post("/machine/keys/{key_id}/collect", response_model=SecretDTO)
async def machine_collect(
    key_id: UUID,
    body: RevisionInput,
    principal: VerifiedMachine = Depends(machine),
    svc: ManagedAuthService = Depends(service),
):
    key, secret = await svc.machine_collect(principal, key_id, body.expected_revision)
    return SecretDTO(key=key_dto(key), secret=secret)


async def _set_archived(
    project_id: UUID, archived: bool, principal: VerifiedMachine, db: ManagedAuthStore
) -> ProjectPolicyDTO:
    if not await db.owns_project_key(ServiceOwner(principal.service_id), project_id):
        raise HTTPException(403, "Project belongs to another service")
    policy = await db.set_project_archived(project_id, archived, principal.identity)
    if policy is None:
        raise HTTPException(404, "Project policy not found")
    return policy_dto(policy)


@router.post("/machine/projects/{project_id}/archive", response_model=ProjectPolicyDTO)
async def machine_archive_project(
    project_id: UUID,
    principal: VerifiedMachine = Depends(machine),
    db: ManagedAuthStore = Depends(store),
):
    return await _set_archived(project_id, True, principal, db)


@router.post(
    "/machine/projects/{project_id}/unarchive", response_model=ProjectPolicyDTO
)
async def machine_unarchive_project(
    project_id: UUID,
    principal: VerifiedMachine = Depends(machine),
    db: ManagedAuthStore = Depends(store),
):
    return await _set_archived(project_id, False, principal, db)


@router.get("/projects/policies", response_model=list[ProjectPolicyDTO])
async def project_policies(
    _: VerifiedHuman = Depends(human),
    db: ManagedAuthStore = Depends(store),
):
    return [policy_dto(policy) for policy in await db.list_project_policies()]


@router.get("/projects/allowances", response_model=list[ProjectAllowanceDTO])
async def project_allowances(
    request: Request,
    _: VerifiedHuman = Depends(human),
    db: ManagedAuthStore = Depends(store),
) -> list[ProjectAllowanceDTO]:
    now = datetime.now(UTC)
    policies, extras = await db.list_project_allowance_sources(now)
    budget_service = request.app.state.budget_service
    snapshot = None
    if budget_service.redis is not None:
        payload = await budget_service.redis.get(budget_service.redis_key)
        if payload is not None:
            snapshot = budget_service._parse_snapshot(payload)
    result: list[ProjectAllowanceDTO] = []
    for policy in policies:
        active = extras.get(policy.project.project_id, ())
        start, end = period_bounds(policy.period, now)
        effective = policy.base_limit_usd + sum(
            (item.increase_usd for item in active), Decimal(0)
        )
        spend = None
        data_through = None
        if snapshot is not None:
            recorded = snapshot.periods[policy.period]
            project_id = str(policy.project.project_id)
            if (
                recorded.start == start
                and recorded.end == end
                and project_id in recorded.spend_usd
            ):
                spend = recorded.spend_usd[project_id]
                data_through = snapshot.data_through
        current = CurrentProjectPeriodDTO(
            start=start,
            end=end,
            effective_limit_usd=effective,
            spend_usd=spend,
            remaining_usd=effective - spend if spend is not None else None,
            data_through=data_through,
        )
        result.append(
            ProjectAllowanceDTO(
                policy=policy_dto(policy),
                active_temporary_increases=[limit_request_dto(item) for item in active],
                current_period=current,
            )
        )
    return result


@router.get("/projects/{project_id}/policy", response_model=ProjectPolicyDTO)
async def project_policy(
    project_id: UUID,
    _: VerifiedHuman = Depends(human),
    db: ManagedAuthStore = Depends(store),
):
    policy = await db.get_project_policy(project_id)
    if policy is None:
        raise HTTPException(404, "Project policy not found")
    return policy_dto(policy)


@router.put("/projects/{project_id}/policy", response_model=ProjectPolicyDTO)
async def update_project_policy(
    project_id: UUID,
    body: ProjectPolicyInput,
    person: VerifiedHuman = Depends(human),
    db: ManagedAuthStore = Depends(store),
    svc: ManagedAuthService = Depends(service),
):
    svc.admin(person)
    try:
        policy = await db.update_project_policy(
            project_id,
            body.expected_revision,
            body.period,
            body.base_limit_usd,
            person.identity,
            body.reason,
        )
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    if policy is None:
        raise HTTPException(409, "Project policy revision changed or not found")
    return policy_dto(policy)


@router.get(
    "/projects/{project_id}/history", response_model=list[ProjectHistoryEventDTO]
)
async def project_history(
    project_id: UUID,
    _: VerifiedHuman = Depends(human),
    db: ManagedAuthStore = Depends(store),
) -> list[ProjectHistoryEventDTO]:
    policy, audits, requests = await db.project_history_sources(project_id)
    if policy is None:
        raise HTTPException(404, "Project policy not found")
    approvals = {
        UUID(row["detail"]["request_id"]): row["detail"]
        for row in audits
        if row["change"] == "increase_approved"
        and row["detail"] is not None
        and "request_id" in row["detail"]
    }
    direct_ids = {
        UUID(request_id)
        for row in audits
        if row["detail"] is not None
        for request_id in row["detail"].get("direct_request_ids", [])
    }
    events: list[ProjectHistoryEventDTO] = []
    for row in audits:
        if row["change"] == "increase_approved":
            continue
        detail = row["detail"]
        events.append(
            ProjectHistoryEventDTO(
                id=row["id"],
                event=row["change"],
                occurred_at=row["occurred_at"],
                actor=IdentityDTO(
                    issuer=row["actor_issuer"], subject=row["actor_subject"]
                ),
                reason=detail["reason"] if detail is not None else None,
                request=None,
                before=ProjectAllowanceStateDTO.model_validate(detail["before"])
                if detail is not None and detail["before"] is not None
                else None,
                after=ProjectAllowanceStateDTO.model_validate(detail["after"])
                if detail is not None and detail["after"] is not None
                else None,
                affected_request_ids=[
                    UUID(item) for item in detail["affected_request_ids"]
                ]
                if detail is not None
                else [],
            )
        )
    for item in requests:
        if item.id in direct_ids:
            continue
        dto = limit_request_dto(item)
        events.append(
            ProjectHistoryEventDTO(
                id=uuid5(NAMESPACE_URL, f"{item.id}:submitted"),
                event="submitted",
                occurred_at=item.created_at,
                actor=identity_dto(item.creator),
                reason=item.reason,
                request=dto,
                before=None,
                after=None,
                affected_request_ids=[],
            )
        )
        if item.decided_at is not None:
            detail = approvals.get(item.id)
            events.append(
                ProjectHistoryEventDTO(
                    id=uuid5(NAMESPACE_URL, f"{item.id}:{item.status}"),
                    event=item.status,
                    occurred_at=item.decided_at,
                    actor=identity_dto(cast(HumanIdentity, item.decided_by)),
                    reason=item.reason,
                    request=dto,
                    before=ProjectAllowanceStateDTO.model_validate(detail["before"])
                    if detail is not None
                    else None,
                    after=ProjectAllowanceStateDTO.model_validate(detail["after"])
                    if detail is not None
                    else None,
                    affected_request_ids=[item.id] if detail is not None else [],
                )
            )
    return sorted(
        events, key=lambda item: (item.occurred_at, str(item.id)), reverse=True
    )


async def _change_temporary(
    db: ManagedAuthStore,
    project_id: UUID,
    person: VerifiedHuman,
    svc: ManagedAuthService,
    expected_revision: int,
    reason: str,
    action: Literal["add", "set", "remove"],
    amount: Decimal | None,
    expected_period: tuple[datetime, datetime] | None,
) -> ProjectPolicyDTO:
    svc.admin(person)
    try:
        policy = await db.change_temporary_increases(
            project_id,
            expected_revision,
            person.identity,
            reason,
            action,
            amount,
            expected_period,
        )
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    if policy is None:
        raise HTTPException(409, "Project policy revision changed or not found")
    return policy_dto(policy)


@router.post(
    "/projects/{project_id}/temporary-increases", response_model=ProjectPolicyDTO
)
async def add_temporary_increase(
    project_id: UUID,
    body: TemporaryIncreaseInput,
    person: VerifiedHuman = Depends(human),
    db: ManagedAuthStore = Depends(store),
    svc: ManagedAuthService = Depends(service),
):
    return await _change_temporary(
        db,
        project_id,
        person,
        svc,
        body.expected_revision,
        body.reason,
        "add",
        body.increase_usd,
        (body.expected_period_start, body.expected_period_end),
    )


@router.put(
    "/projects/{project_id}/temporary-increases", response_model=ProjectPolicyDTO
)
async def set_temporary_increase(
    project_id: UUID,
    body: TemporarySetInput,
    person: VerifiedHuman = Depends(human),
    db: ManagedAuthStore = Depends(store),
    svc: ManagedAuthService = Depends(service),
):
    return await _change_temporary(
        db,
        project_id,
        person,
        svc,
        body.expected_revision,
        body.reason,
        "set",
        body.limit_usd,
        (body.expected_period_start, body.expected_period_end),
    )


@router.delete(
    "/projects/{project_id}/temporary-increases", response_model=ProjectPolicyDTO
)
async def remove_temporary_increases(
    project_id: UUID,
    body: TemporaryRemoveInput,
    person: VerifiedHuman = Depends(human),
    db: ManagedAuthStore = Depends(store),
    svc: ManagedAuthService = Depends(service),
):
    return await _change_temporary(
        db,
        project_id,
        person,
        svc,
        body.expected_revision,
        body.reason,
        "remove",
        None,
        None,
    )


@router.post("/projects/{project_id}/limit-requests", response_model=LimitRequestDTO)
async def submit_limit_request(
    project_id: UUID,
    body: LimitRequestInput,
    person: VerifiedHuman = Depends(human),
    db: ManagedAuthStore = Depends(store),
    svc: ManagedAuthService = Depends(service),
):
    policy = await svc.project_member(person, project_id)
    try:
        item = await db.create_limit_request(
            uuid4(),
            policy.project,
            body.kind,
            body.increase_usd,
            body.reason,
            person.identity,
        )
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    return limit_request_dto(item)


@router.put("/projects/{project_id}/budget", response_model=ProjectBudgetChangeDTO)
async def change_project_budget(
    project_id: UUID,
    body: ProjectBudgetInput,
    request: Request,
    person: VerifiedHuman = Depends(human),
    db: ManagedAuthStore = Depends(store),
    svc: ManagedAuthService = Depends(service),
):
    policy = await svc.project_member(person, project_id)
    if policy.revision != body.expected_revision:
        raise HTTPException(409, "Project policy revision changed or not found")
    total = policy.total_limit_usd
    if total is None:
        raise HTTPException(409, "Project budget is not an overall total yet")
    if body.limit_usd > total:
        raised = await db.request_project_raise(
            uuid4(),
            project_id,
            body.expected_revision,
            body.limit_usd,
            body.reason,
            person.identity,
        )
        if raised is None:
            raise HTTPException(409, "Project policy revision changed or not found")
        reworded, item = raised
        return ProjectBudgetChangeDTO(
            policy=policy_dto(reworded), pending_request=limit_request_dto(item)
        )
    if body.limit_usd < total:
        budgets = cast(BudgetService, request.app.state.budget_service)
        spent = await budgets.project_total_spend(project_id)
        if spent is None:
            raise HTTPException(503, "Project spend is unavailable; try again shortly")
        if body.limit_usd < spent:
            raise HTTPException(
                409,
                f"The new budget ${body.limit_usd} is below the ${spent} "
                "this project has already spent",
            )
    changed = await db.change_project_total(
        project_id, body.expected_revision, body.limit_usd, body.reason, person.identity
    )
    if changed is None:
        raise HTTPException(409, "Project policy revision changed or not found")
    return ProjectBudgetChangeDTO(policy=policy_dto(changed), pending_request=None)


@router.get("/limit-requests", response_model=list[LimitRequestDTO])
async def limit_requests(
    _: VerifiedHuman = Depends(human),
    db: ManagedAuthStore = Depends(store),
):
    return [limit_request_dto(item) for item in await db.list_limit_requests()]


@router.get(
    "/projects/{project_id}/limit-requests", response_model=list[LimitRequestDTO]
)
async def project_limit_requests(
    project_id: UUID,
    _: VerifiedHuman = Depends(human),
    db: ManagedAuthStore = Depends(store),
):
    return [
        limit_request_dto(item)
        for item in await db.list_project_limit_requests(project_id)
    ]


@router.get("/limit-requests/{request_id}", response_model=LimitRequestDTO)
async def limit_request_detail(
    request_id: UUID,
    _: VerifiedHuman = Depends(human),
    db: ManagedAuthStore = Depends(store),
):
    item = await db.get_limit_request(request_id)
    if item is None:
        raise HTTPException(404, "Limit request not found")
    return limit_request_dto(item)


@router.post("/limit-requests/{request_id}/approve", response_model=LimitRequestDTO)
async def approve_limit_request(
    request_id: UUID,
    body: LimitDecisionInput,
    person: VerifiedHuman = Depends(human),
    db: ManagedAuthStore = Depends(store),
    svc: ManagedAuthService = Depends(service),
):
    svc.admin(person)
    try:
        item = await db.approve_limit_request(
            request_id, person.identity, body.expected_current_policy_revision
        )
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    return limit_request_dto(item)


@router.post("/limit-requests/{request_id}/deny", response_model=LimitRequestDTO)
async def deny_limit_request(
    request_id: UUID,
    person: VerifiedHuman = Depends(human),
    db: ManagedAuthStore = Depends(store),
    svc: ManagedAuthService = Depends(service),
):
    svc.admin(person)
    try:
        item = await db.deny_limit_request(request_id, person.identity)
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    return limit_request_dto(item)


@router.get("/services", response_model=list[ServiceDTO])
async def services(
    person: VerifiedHuman = Depends(human),
    db: ManagedAuthStore = Depends(store),
):
    return [
        _service_dto(service_id, name) for service_id, name in await db.list_services()
    ]


@router.get("/services/{service_id}", response_model=ServiceDTO)
async def service_detail(
    service_id: UUID,
    person: VerifiedHuman = Depends(human),
    db: ManagedAuthStore = Depends(store),
):
    name = await db.get_service(service_id)
    if name is None:
        raise HTTPException(404, "Service not found")
    return _service_dto(service_id, name)


@router.post("/services", response_model=ServiceDTO)
async def create_service(
    body: ServiceInput,
    person: VerifiedHuman = Depends(human),
    db: ManagedAuthStore = Depends(store),
    svc: ManagedAuthService = Depends(service),
):
    svc.admin(person)
    service_id = uuid4()
    await db.create_service(service_id, body.name, person.identity)
    return _service_dto(service_id, body.name)
