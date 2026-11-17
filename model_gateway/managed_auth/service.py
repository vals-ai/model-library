import hmac
from decimal import Decimal
from uuid import UUID, uuid4

from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError

from model_gateway.managed_auth.credentials import create_managed_key_secret
from model_gateway.managed_auth.identity import VerifiedHuman, VerifiedMachine
from model_gateway.managed_auth.store import ManagedAuthStore
from model_gateway.managed_auth.types import (
    KeyLimit,
    KeyOwner,
    KeyRequest,
    ManagedKey,
    PersonalOwner,
    ProjectPolicy,
    ProjectReference,
    ServiceOwner,
    SystemIdentity,
)


PLATFORM_ADMIN = "platform-admin"
ANOTHER_VERIFIER = "Key is active with another verifier"


class ManagedAuthService:
    def __init__(self, store: ManagedAuthStore, auto_approval_threshold_usd: Decimal):
        self.store = store
        self.auto_approval_threshold_usd = auto_approval_threshold_usd

    @staticmethod
    def is_admin(human: VerifiedHuman) -> bool:
        return PLATFORM_ADMIN in human.permissions

    @staticmethod
    def admin(human: VerifiedHuman) -> None:
        if not ManagedAuthService.is_admin(human):
            raise HTTPException(403, "Platform administrator required")

    async def project_policy(self, project_id: UUID) -> ProjectPolicy:
        policy = await self.store.get_project_policy(project_id)
        if policy is None:
            raise HTTPException(404, "Project policy not found")
        return policy

    async def project_member(
        self, human: VerifiedHuman, project_id: UUID
    ) -> ProjectPolicy:
        policy = await self.project_policy(project_id)
        if self.is_admin(human) or await self.store.owns_project_key(
            PersonalOwner(human.identity), project_id
        ):
            return policy
        raise HTTPException(403, "Project member or platform administrator required")

    async def register(self, human: VerifiedHuman) -> None:
        await self.store.record_human(human.identity, True, human.identity)

    @staticmethod
    def require_owner(human: VerifiedHuman, owner: KeyOwner) -> None:
        if isinstance(owner, PersonalOwner):
            if human.identity == owner.identity:
                return
        elif ManagedAuthService.is_admin(human):
            return
        raise HTTPException(403, "Key owner or platform administrator required")

    async def request(
        self,
        human: VerifiedHuman,
        display_name: str,
        description: str,
        owner: KeyOwner,
        key_limit: KeyLimit | None,
        project_id: UUID,
    ) -> KeyRequest:
        self.require_owner(human, owner)
        existing_policy = await self.store.get_project_policy(project_id)
        if existing_policy is None or not await self.store.has_approved_key_request(
            project_id
        ):
            raise HTTPException(409, "Project is not approved")
        project = existing_policy.project
        try:
            request = await self.store.create_request(
                uuid4(),
                display_name,
                description,
                human.identity,
                owner,
                key_limit,
                project,
                None,
                None,
            )
            await self.store.issue_additional_key(request.id, uuid4(), human.identity)
            updated = await self.store.get_request(request.id)
            assert updated is not None
            return updated
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    async def machine_request(
        self,
        machine: VerifiedMachine,
        display_name: str,
        description: str,
        project: ProjectReference,
        requested_budget: Decimal,
        budget_reason: str,
    ) -> KeyRequest:
        existing_policy = await self.store.get_project_policy(project.project_id)
        if existing_policy is not None and existing_policy.project != project:
            raise HTTPException(409, "Project reference differs from approved policy")
        request = await self.store.create_request(
            uuid4(),
            display_name,
            description,
            machine.identity,
            ServiceOwner(machine.service_id),
            None,
            project,
            requested_budget,
            budget_reason,
        )
        if existing_policy and await self.store.has_approved_key_request(
            project.project_id
        ):
            await self.store.issue_additional_key(request.id, uuid4(), machine.identity)
        elif self._within_auto_approval(requested_budget, existing_policy):
            await self._auto_approve(
                request.id, None if existing_policy else requested_budget
            )
        else:
            return request
        updated = await self.store.get_request(request.id)
        assert updated is not None
        return updated

    def _within_auto_approval(
        self, requested_budget: Decimal, policy: ProjectPolicy | None
    ) -> bool:
        threshold = self.auto_approval_threshold_usd
        return requested_budget <= threshold and (
            policy is None or policy.total_limit_usd <= threshold
        )

    async def _auto_approve(
        self, request_id: UUID, requested_budget: Decimal | None
    ) -> None:
        await self.store.decide_request(
            request_id,
            SystemIdentity("managed-auth-system", "requested-budget-auto-approval"),
            uuid4(),
            requested_budget,
            self.auto_approval_threshold_usd,
        )

    async def machine_request_detail(
        self, machine: VerifiedMachine, request_id: UUID
    ) -> KeyRequest:
        request = await self.store.get_request(request_id)
        if request is None:
            raise HTTPException(404, "Request not found")
        if request.owner != ServiceOwner(machine.service_id):
            raise HTTPException(403, "Machine request belongs to another service")
        return request

    async def machine_key(self, machine: VerifiedMachine, key_id: UUID) -> ManagedKey:
        key = await self.store.get_key(key_id)
        if key is None:
            raise HTTPException(404, "Key not found")
        if key.owner != ServiceOwner(machine.service_id):
            raise HTTPException(403, "Machine key belongs to another service")
        return key

    async def machine_collect(
        self, machine: VerifiedMachine, key_id: UUID, revision: int
    ) -> tuple[ManagedKey, str]:
        await self.machine_key(machine, key_id)
        secret, digest = create_managed_key_secret()
        result = await self.store.collect_key(
            key_id, revision, digest, machine.identity
        )
        if result is None:
            raise HTTPException(409, "Key state or revision changed")
        return result, secret

    async def machine_set_verifier(
        self, machine: VerifiedMachine, key_id: UUID, revision: int, verifier_hash: str
    ) -> ManagedKey:
        await self.machine_key(machine, key_id)
        try:
            result = await self.store.collect_key(
                key_id, revision, verifier_hash, machine.identity
            )
        except IntegrityError:
            raise HTTPException(409, "Verifier hash is already in use") from None
        if result is not None:
            return result
        key = await self.machine_key(machine, key_id)
        if key.status != "active" or key.verifier_hash is None:
            raise HTTPException(409, "Key state or revision changed")
        if not hmac.compare_digest(key.verifier_hash, verifier_hash):
            raise HTTPException(409, ANOTHER_VERIFIER)
        return key

    async def key(self, key_id: UUID) -> ManagedKey:
        key = await self.store.get_key(key_id)
        if key is None:
            raise HTTPException(404, "Key not found")
        return key

    async def request_detail(self, request_id: UUID) -> KeyRequest:
        request = await self.store.get_request(request_id)
        if request is None:
            raise HTTPException(404, "Request not found")
        return request

    async def decide(
        self,
        human: VerifiedHuman,
        request_id: UUID,
        first_total_usd: Decimal | None,
        approved: bool,
    ) -> ManagedKey | None:
        self.admin(human)
        request = await self.request_detail(request_id)
        if approved and request.kind != "create":
            raise HTTPException(
                409, "Historical change request requires cutover review"
            )
        key_id = uuid4() if approved else None
        try:
            return await self.store.decide_request(
                request_id, human.identity, key_id, first_total_usd, None
            )
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    async def secret_action(
        self, human: VerifiedHuman, key_id: UUID, revision: int, action: str
    ) -> tuple[ManagedKey, str | None]:
        key = await self.key(key_id)
        if isinstance(key.owner, PersonalOwner):
            if action != "revoke" or not self.is_admin(human):
                self.require_owner(human, key.owner)
        else:
            self.admin(human)
        if action == "revoke":
            result = await self.store.revoke_key(key_id, revision, human.identity)
            secret = None
        else:
            secret, digest = create_managed_key_secret()
            operation = (
                self.store.collect_key if action == "collect" else self.store.rotate_key
            )
            result = await operation(key_id, revision, digest, human.identity)
        if result is None:
            raise HTTPException(409, "Key state or revision changed")
        return result, secret

    async def update_key_limit(
        self,
        human: VerifiedHuman,
        key_id: UUID,
        expected_revision: int,
        key_limit: KeyLimit | None,
    ) -> ManagedKey:
        self.admin(human)
        key = await self.store.update_key_limit(
            key_id, expected_revision, key_limit, human.identity
        )
        if key is None:
            raise HTTPException(409, "Key revision changed or not found")
        return key
