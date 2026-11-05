import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from sqlalchemy import insert, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import RowMapping
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine
from sqlalchemy.sql.elements import ColumnElement

from model_gateway.budgets.schema import BudgetPeriod, period_bounds
from model_gateway.managed_auth import tables as t
from model_gateway.managed_auth.types import (
    ActorIdentity,
    AuditEvent,
    HumanIdentity,
    KeyLimit,
    KeyOwner,
    LimitRequestKind,
    TEMPORARY_INCREASE,
    LimitRequestStatus,
    PERMANENT_INCREASE,
    PROJECT_POLICY_AUDIT,
    KeyStatus,
    KeyRequest,
    MachineIdentity,
    ManagedKey,
    ProjectAlertIdentity,
    PersonalOwner,
    ProjectLimitRequest,
    ProjectPolicy,
    ProjectReference,
    ServiceOwner,
    SystemIdentity,
)

logger = logging.getLogger(__name__)


def _identity(identity: ActorIdentity, prefix: str = "") -> dict[str, str]:
    return {f"{prefix}issuer": identity.issuer, f"{prefix}subject": identity.subject}


def _actor_kind(identity: ActorIdentity) -> Literal["human", "machine", "system"]:
    if isinstance(identity, HumanIdentity):
        return "human"
    if isinstance(identity, MachineIdentity):
        return "machine"
    return "system"


def _read_actor(kind: str, issuer: str, subject: str) -> ActorIdentity:
    if kind == "human":
        return HumanIdentity(issuer, subject)
    if kind == "machine":
        return MachineIdentity(issuer, subject)
    return SystemIdentity(issuer, subject)


def _read_creator(
    kind: str, issuer: str, subject: str
) -> HumanIdentity | MachineIdentity:
    if kind == "human":
        return HumanIdentity(issuer, subject)
    return MachineIdentity(issuer, subject)


def _project(row: RowMapping) -> ProjectReference:
    return ProjectReference(
        row["project_id"], row["project_organization"], row["project_environment"]
    )


def _owner(owner: KeyOwner) -> dict[str, str | UUID]:
    if isinstance(owner, PersonalOwner):
        return {"owner_kind": "personal", **_identity(owner.identity, "owner_")}
    return {"owner_kind": "service", "service_id": owner.service_id}


def _read_owner(row: RowMapping) -> KeyOwner:
    if row["owner_kind"] == "personal":
        return PersonalOwner(HumanIdentity(row["owner_issuer"], row["owner_subject"]))
    return ServiceOwner(row["service_id"])


def _human_id(identity: HumanIdentity) -> UUID:
    return uuid5(NAMESPACE_URL, f"{identity.issuer}\0{identity.subject}")


class ManagedAuthStore:
    def __init__(self, engine: AsyncEngine):
        self.engine = engine

    async def _register_actor(
        self, connection: AsyncConnection, actor: ActorIdentity
    ) -> None:
        await connection.execute(
            pg_insert(t.actors)
            .values(kind=_actor_kind(actor), **_identity(actor))
            .on_conflict_do_nothing(
                index_elements=[t.actors.c.kind, t.actors.c.issuer, t.actors.c.subject]
            )
        )

    async def _audit(
        self,
        connection: AsyncConnection,
        actor: ActorIdentity,
        kind: str,
        target: UUID,
        change: str,
        detail: dict[str, object] | None = None,
        occurred_at: datetime | None = None,
    ) -> None:
        await self._register_actor(connection, actor)
        await connection.execute(
            insert(t.audit).values(
                id=uuid4(),
                **_identity(actor, "actor_"),
                actor_kind=_actor_kind(actor),
                target_kind=kind,
                target_id=target,
                change=change,
                detail=detail,
                occurred_at=occurred_at or datetime.now(UTC),
            )
        )

    async def record_human(
        self,
        identity: HumanIdentity,
        enabled: bool,
        actor: HumanIdentity | SystemIdentity,
    ) -> None:
        async with self.engine.begin() as connection:
            changed = (
                await connection.execute(
                    pg_insert(t.humans)
                    .values(
                        **_identity(identity),
                        enabled=enabled,
                    )
                    .on_conflict_do_update(
                        index_elements=[t.humans.c.issuer, t.humans.c.subject],
                        set_={"enabled": enabled},
                        where=t.humans.c.enabled != enabled,
                    )
                    .returning(t.humans.c.enabled)
                )
            ).scalar_one_or_none()
            if changed is not None:
                await self._audit(
                    connection,
                    actor,
                    "human",
                    _human_id(identity),
                    "enabled" if enabled else "disabled",
                )

    async def list_known_humans(self, expected_issuer: str) -> list[HumanIdentity]:
        async with self.engine.connect() as connection:
            rows = (
                (
                    await connection.execute(
                        select(t.humans.c.issuer, t.humans.c.subject)
                        .where(t.humans.c.issuer == expected_issuer)
                        .order_by(t.humans.c.subject)
                    )
                )
                .mappings()
                .all()
            )
            return [HumanIdentity(row["issuer"], row["subject"]) for row in rows]

    async def record_system_human_eligibility(
        self, identity: HumanIdentity, enabled: bool
    ) -> None:
        await self.record_human(
            identity, enabled, SystemIdentity("managed-auth-system", "eligibility-sync")
        )

    async def human_enabled(self, identity: HumanIdentity) -> bool | None:
        async with self.engine.connect() as connection:
            result = await connection.execute(
                select(t.humans.c.enabled).where(
                    t.humans.c.issuer == identity.issuer,
                    t.humans.c.subject == identity.subject,
                )
            )
            return result.scalar_one_or_none()

    async def create_service(
        self, service_id: UUID, name: str, actor: HumanIdentity
    ) -> None:
        async with self.engine.begin() as connection:
            await connection.execute(
                insert(t.services).values(id=service_id, name=name)
            )
            await self._audit(connection, actor, "service", service_id, "created")

    async def create_request(
        self,
        request_id: UUID,
        display_name: str,
        description: str,
        creator: HumanIdentity | MachineIdentity,
        owner: KeyOwner,
        key_limit: KeyLimit | None,
        project: ProjectReference,
        requested_budget: Decimal | None,
        budget_reason: str | None,
    ) -> KeyRequest:
        if isinstance(creator, MachineIdentity) and not isinstance(owner, ServiceOwner):
            raise ValueError("machine requests require a service owner")
        now = datetime.now(UTC)
        async with self.engine.begin() as connection:
            await self._register_actor(connection, creator)
            await connection.execute(
                insert(t.requests).values(
                    id=request_id,
                    kind="create",
                    display_name=display_name,
                    reason=description,
                    requested_key_limit_period=key_limit.period if key_limit else None,
                    requested_key_limit_usd=key_limit.limit_usd if key_limit else None,
                    requested_budget=requested_budget,
                    budget_reason=budget_reason,
                    **_identity(creator, "creator_"),
                    creator_kind=_actor_kind(creator),
                    project_id=project.project_id,
                    project_organization=project.organization,
                    project_environment=project.environment,
                    **_owner(owner),
                    status="pending",
                    created_at=now,
                )
            )
            await self._audit(connection, creator, "request", request_id, "submitted")
        return KeyRequest(
            request_id,
            "create",
            display_name,
            description,
            key_limit,
            None,
            None,
            creator,
            owner,
            "pending",
            None,
            None,
            None,
            now,
            project,
            requested_budget,
            budget_reason,
        )

    async def get_request(self, request_id: UUID) -> KeyRequest | None:
        async with self.engine.connect() as connection:
            row = (
                (
                    await connection.execute(
                        select(t.requests).where(t.requests.c.id == request_id)
                    )
                )
                .mappings()
                .first()
            )
            return self._request(row) if row is not None else None

    @staticmethod
    def _request(row: RowMapping) -> KeyRequest:
        requested_limit = (
            KeyLimit(row["requested_key_limit_period"], row["requested_key_limit_usd"])
            if row["requested_key_limit_period"] is not None
            else None
        )
        decider = (
            HumanIdentity(row["decided_by_issuer"], row["decided_by_subject"])
            if row["decided_by_issuer"] is not None
            else None
        )
        return KeyRequest(
            row["id"],
            row["kind"],
            row["display_name"],
            row["reason"],
            requested_limit,
            row["target_key_id"],
            row["target_revision"],
            _read_creator(
                row["creator_kind"], row["creator_issuer"], row["creator_subject"]
            ),
            _read_owner(row),
            row["status"],
            decider,
            row["decided_at"],
            row["key_id"],
            row["created_at"],
            _project(row),
            row["requested_budget"],
            row["budget_reason"],
        )

    async def decide_request(
        self,
        request_id: UUID,
        actor: HumanIdentity | SystemIdentity,
        key_id: UUID | None,
        first_policy: tuple[BudgetPeriod, Decimal] | None,
        auto_approval_threshold_usd: Decimal | None,
    ) -> ManagedKey | None:
        now = datetime.now(UTC)
        async with self.engine.begin() as connection:
            row = (
                (
                    await connection.execute(
                        select(t.requests)
                        .where(t.requests.c.id == request_id)
                        .with_for_update()
                    )
                )
                .mappings()
                .first()
            )
            if row is None or row["status"] != "pending":
                raise ValueError("request is not pending")
            if key_id is None:
                await connection.execute(
                    update(t.requests)
                    .where(t.requests.c.id == request_id)
                    .values(
                        status="denied",
                        **_identity(actor, "decided_by_"),
                        decided_at=now,
                    )
                )
                await self._audit(connection, actor, "request", request_id, "denied")
                return None
            if row["kind"] != "create":
                raise ValueError("historical change requests require cutover review")
            policy = (
                (
                    await connection.execute(
                        select(t.project_policies)
                        .where(t.project_policies.c.project_id == row["project_id"])
                        .with_for_update()
                    )
                )
                .mappings()
                .first()
            )
            if policy is None:
                if first_policy is None:
                    raise ValueError("first key approval requires a project policy")
                period, base_limit_usd = first_policy
                created_policy = (
                    await connection.execute(
                        pg_insert(t.project_policies)
                        .values(
                            project_id=row["project_id"],
                            project_organization=row["project_organization"],
                            project_environment=row["project_environment"],
                            period=period,
                            base_limit_usd=base_limit_usd,
                            total_limit_usd=base_limit_usd,
                            total_started_at=period_bounds(period, now)[0],
                            budget_reason=row["budget_reason"],
                            revision=1,
                            created_at=now,
                            updated_at=now,
                        )
                        .on_conflict_do_nothing(
                            index_elements=[t.project_policies.c.project_id]
                        )
                        .returning(t.project_policies.c.project_id)
                    )
                ).scalar_one_or_none()
                if created_policy is None:
                    raise ValueError("project was approved concurrently")
                await self._audit(
                    connection,
                    actor,
                    "project_policy",
                    row["project_id"],
                    "approved",
                    detail={
                        "reason": row["budget_reason"],
                        "before": None,
                        "after": self._allowance_state(period, base_limit_usd, now, ()),
                        "affected_request_ids": [],
                        "auto_approval_threshold_usd": str(auto_approval_threshold_usd)
                        if auto_approval_threshold_usd is not None
                        else None,
                    },
                    occurred_at=now,
                )
            elif (
                policy["project_organization"] != row["project_organization"]
                or policy["project_environment"] != row["project_environment"]
            ):
                raise ValueError("project reference changed since request")
            elif first_policy is not None:
                raise ValueError("project is already approved")
            created = (
                await connection.execute(
                    pg_insert(t.keys)
                    .values(
                        id=key_id,
                        display_name=row["display_name"],
                        description=row["reason"] or None,
                        creator_issuer=row["creator_issuer"],
                        creator_subject=row["creator_subject"],
                        creator_kind=row["creator_kind"],
                        project_id=row["project_id"],
                        **_owner(_read_owner(row)),
                        key_limit_period=row["requested_key_limit_period"],
                        key_limit_usd=row["requested_key_limit_usd"],
                        status="pending_collection",
                        revision=1,
                        created_at=now,
                        updated_at=now,
                    )
                    .on_conflict_do_nothing(index_elements=[t.keys.c.id])
                    .returning(t.keys.c.id)
                )
            ).first()
            if created is None:
                raise ValueError("key already exists")
            await connection.execute(
                update(t.requests)
                .where(t.requests.c.id == request_id)
                .values(
                    status="approved",
                    **(
                        _identity(actor, "decided_by_")
                        if isinstance(actor, HumanIdentity)
                        else {}
                    ),
                    decided_at=now,
                    key_id=key_id,
                )
            )
            await self._audit(connection, actor, "request", request_id, "approved")
            await self._audit(connection, actor, "key", key_id, "created")
            key = await self._load_key(connection, key_id)
            assert key is not None
            return key

    async def issue_additional_key(
        self,
        request_id: UUID,
        key_id: UUID,
        creator: HumanIdentity | MachineIdentity,
    ) -> ManagedKey:
        now = datetime.now(UTC)
        system = SystemIdentity("managed-auth-system", "additional-key-issuance")
        async with self.engine.begin() as connection:
            request = (
                (
                    await connection.execute(
                        select(t.requests)
                        .where(t.requests.c.id == request_id)
                        .with_for_update()
                    )
                )
                .mappings()
                .first()
            )
            if (
                request is None
                or request["status"] != "pending"
                or request["kind"] != "create"
                or request["creator_kind"] != _actor_kind(creator)
                or request["creator_issuer"] != creator.issuer
                or request["creator_subject"] != creator.subject
            ):
                raise ValueError(
                    "additional-key request is not pending for this creator"
                )
            policy = (
                (
                    await connection.execute(
                        select(t.project_policies)
                        .where(t.project_policies.c.project_id == request["project_id"])
                        .with_for_update()
                    )
                )
                .mappings()
                .first()
            )
            if policy is None or (
                policy["project_organization"] != request["project_organization"]
                or policy["project_environment"] != request["project_environment"]
            ):
                raise ValueError("project is not approved")
            if (
                isinstance(creator, MachineIdentity)
                and request["owner_kind"] != "service"
            ):
                raise ValueError("machine can only issue its service key")
            await connection.execute(
                insert(t.keys).values(
                    id=key_id,
                    display_name=request["display_name"],
                    description=request["reason"],
                    creator_kind=request["creator_kind"],
                    creator_issuer=request["creator_issuer"],
                    creator_subject=request["creator_subject"],
                    project_id=request["project_id"],
                    **_owner(_read_owner(request)),
                    key_limit_period=request["requested_key_limit_period"],
                    key_limit_usd=request["requested_key_limit_usd"],
                    status="pending_collection",
                    revision=1,
                    created_at=now,
                    updated_at=now,
                )
            )
            await connection.execute(
                update(t.requests)
                .where(t.requests.c.id == request_id)
                .values(status="approved", decided_at=now, key_id=key_id)
            )
            await self._audit(
                connection, system, "request", request_id, "conditionally_issued"
            )
            await self._audit(connection, system, "key", key_id, "created")
            key = await self._load_key(connection, key_id)
            assert key is not None
            return key

    async def get_key(self, key_id: UUID) -> ManagedKey | None:
        async with self.engine.connect() as connection:
            return await self._load_key(connection, key_id)

    async def _load_key(
        self, connection: AsyncConnection, key_id: UUID
    ) -> ManagedKey | None:
        return await self._select_key(connection, t.keys.c.id == key_id)

    @staticmethod
    def _key_query():
        return select(
            t.keys,
            t.project_policies.c.project_organization,
            t.project_policies.c.project_environment,
        ).join(
            t.project_policies, t.keys.c.project_id == t.project_policies.c.project_id
        )

    @staticmethod
    def _key(row: RowMapping) -> ManagedKey:
        return ManagedKey(
            row["id"],
            _read_creator(
                row["creator_kind"], row["creator_issuer"], row["creator_subject"]
            ),
            _read_owner(row),
            row["description"],
            row["status"],
            row["verifier_hash"],
            row["revision"],
            row["display_name"],
            row["created_at"],
            row["updated_at"],
            _project(row),
            KeyLimit(row["key_limit_period"], row["key_limit_usd"])
            if row["key_limit_period"] is not None
            else None,
        )

    async def _select_key(
        self, connection: AsyncConnection, *conditions: ColumnElement[bool]
    ) -> ManagedKey | None:
        row = (
            (await connection.execute(self._key_query().where(*conditions)))
            .mappings()
            .first()
        )
        return self._key(row) if row is not None else None

    async def get_key_by_verifier(self, verifier_hash: str) -> ManagedKey | None:
        async with self.engine.connect() as connection:
            return await self._select_key(
                connection,
                t.keys.c.verifier_hash == verifier_hash,
                t.keys.c.status == "active",
            )

    async def collect_key(
        self,
        key_id: UUID,
        expected_revision: int,
        verifier_hash: str,
        actor: HumanIdentity | MachineIdentity,
    ) -> ManagedKey | None:
        return await self._change_secret(
            key_id,
            expected_revision,
            verifier_hash,
            actor,
            "pending_collection",
            "collected",
        )

    async def rotate_key(
        self,
        key_id: UUID,
        expected_revision: int,
        verifier_hash: str,
        actor: HumanIdentity,
    ) -> ManagedKey | None:
        return await self._change_secret(
            key_id, expected_revision, verifier_hash, actor, "active", "rotated"
        )

    async def _change_secret(
        self,
        key_id: UUID,
        expected_revision: int,
        verifier_hash: str,
        actor: HumanIdentity | MachineIdentity,
        current: str,
        change: str,
    ) -> ManagedKey | None:
        now = datetime.now(UTC)
        async with self.engine.begin() as connection:
            row = (
                await connection.execute(
                    update(t.keys)
                    .where(
                        t.keys.c.id == key_id,
                        t.keys.c.status == current,
                        t.keys.c.revision == expected_revision,
                    )
                    .values(
                        status="active",
                        verifier_hash=verifier_hash,
                        revision=expected_revision + 1,
                        updated_at=now,
                    )
                    .returning(t.keys.c.id)
                )
            ).first()
            if row is None:
                return None
            await self._audit(connection, actor, "key", key_id, change)
            return await self._load_key(connection, key_id)

    async def revoke_key(
        self, key_id: UUID, expected_revision: int, actor: HumanIdentity
    ) -> ManagedKey | None:
        now = datetime.now(UTC)
        async with self.engine.begin() as connection:
            row = (
                await connection.execute(
                    update(t.keys)
                    .where(
                        t.keys.c.id == key_id,
                        t.keys.c.status != "revoked",
                        t.keys.c.revision == expected_revision,
                    )
                    .values(
                        status="revoked",
                        verifier_hash=None,
                        revision=expected_revision + 1,
                        updated_at=now,
                    )
                    .returning(t.keys.c.id)
                )
            ).first()
            if row is None:
                return None
            await self._audit(connection, actor, "key", key_id, "revoked")
            return await self._load_key(connection, key_id)

    async def update_key_limit(
        self,
        key_id: UUID,
        expected_revision: int,
        key_limit: KeyLimit | None,
        actor: HumanIdentity,
    ) -> ManagedKey | None:
        now = datetime.now(UTC)
        async with self.engine.begin() as connection:
            changed = (
                await connection.execute(
                    update(t.keys)
                    .where(
                        t.keys.c.id == key_id,
                        t.keys.c.revision == expected_revision,
                    )
                    .values(
                        key_limit_period=key_limit.period if key_limit else None,
                        key_limit_usd=key_limit.limit_usd if key_limit else None,
                        revision=expected_revision + 1,
                        updated_at=now,
                    )
                    .returning(t.keys.c.id)
                )
            ).first()
            if changed is None:
                return None
            await self._audit(connection, actor, "key", key_id, "limit_changed")
            return await self._load_key(connection, key_id)

    async def audit_for(
        self, target_kind: str, target_id: UUID
    ) -> tuple[AuditEvent, ...]:
        async with self.engine.connect() as connection:
            rows = (
                (
                    await connection.execute(
                        select(t.audit)
                        .where(
                            t.audit.c.target_kind == target_kind,
                            t.audit.c.target_id == target_id,
                        )
                        .order_by(t.audit.c.occurred_at, t.audit.c.id)
                    )
                )
                .mappings()
                .all()
            )
            return tuple(
                AuditEvent(
                    row["id"],
                    _read_actor(
                        row["actor_kind"], row["actor_issuer"], row["actor_subject"]
                    ),
                    row["target_kind"],
                    row["target_id"],
                    row["change"],
                    row["occurred_at"],
                    row["actor_kind"],
                )
                for row in rows
            )

    async def list_keys(self) -> tuple[ManagedKey, ...]:
        async with self.engine.connect() as connection:
            rows = (
                (
                    await connection.execute(
                        self._key_query().order_by(t.keys.c.created_at.desc())
                    )
                )
                .mappings()
                .all()
            )
            return tuple(self._key(row) for row in rows)

    async def list_requests(self) -> tuple[KeyRequest, ...]:
        async with self.engine.connect() as connection:
            rows = (
                (
                    await connection.execute(
                        select(t.requests).order_by(t.requests.c.created_at.desc())
                    )
                )
                .mappings()
                .all()
            )
            return tuple(self._request(row) for row in rows)

    @staticmethod
    def _policy(row: RowMapping) -> ProjectPolicy:
        return ProjectPolicy(
            _project(row),
            row["period"],
            row["base_limit_usd"],
            row["revision"],
            row["created_at"],
            row["updated_at"],
            row["total_limit_usd"],
            row["total_started_at"],
            row["budget_reason"],
            row["archived_at"],
        )

    @staticmethod
    def _allowance_state(
        period: BudgetPeriod,
        base: Decimal,
        now: datetime,
        extras: tuple[RowMapping | dict[str, Any], ...],
    ) -> dict[str, object]:
        start, end = period_bounds(period, now)
        return {
            "period": period,
            "period_start": start.isoformat(),
            "period_end": end.isoformat(),
            "base_limit_usd": str(base),
            "effective_limit_usd": str(
                base + sum((row["increase_usd"] for row in extras), Decimal(0))
            ),
            "active_temporary_increases": [
                {
                    "id": str(row["id"]),
                    "increase_usd": str(row["increase_usd"]),
                    "period_start": row["period_start"].isoformat(),
                    "period_end": row["period_end"].isoformat(),
                }
                for row in extras
            ],
        }

    async def _active_extras(
        self, connection: AsyncConnection, project_id: UUID, now: datetime
    ) -> tuple[RowMapping, ...]:
        return tuple(
            (
                await connection.execute(
                    select(t.project_limit_requests)
                    .where(
                        t.project_limit_requests.c.project_id == project_id,
                        t.project_limit_requests.c.kind == "temporary",
                        t.project_limit_requests.c.status == "approved",
                        t.project_limit_requests.c.cancelled_at.is_(None),
                        t.project_limit_requests.c.period_start <= now,
                        t.project_limit_requests.c.period_end > now,
                    )
                    .order_by(t.project_limit_requests.c.id)
                )
            )
            .mappings()
            .all()
        )

    async def list_project_allowance_sources(
        self, now: datetime
    ) -> tuple[tuple[ProjectPolicy, ...], dict[UUID, tuple[ProjectLimitRequest, ...]]]:
        async with self.engine.connect() as connection:
            await connection.execution_options(isolation_level="REPEATABLE READ")
            async with connection.begin():
                policies = (
                    (
                        await connection.execute(
                            select(t.project_policies).order_by(
                                t.project_policies.c.project_id
                            )
                        )
                    )
                    .mappings()
                    .all()
                )
                rows = (
                    (
                        await connection.execute(
                            self._limit_request_query()
                            .where(
                                t.project_limit_requests.c.kind == "temporary",
                                t.project_limit_requests.c.status == "approved",
                                t.project_limit_requests.c.cancelled_at.is_(None),
                                t.project_limit_requests.c.period_start <= now,
                                t.project_limit_requests.c.period_end > now,
                            )
                            .order_by(
                                t.project_limit_requests.c.project_id,
                                t.project_limit_requests.c.id,
                            )
                        )
                    )
                    .mappings()
                    .all()
                )
        extras: dict[UUID, list[ProjectLimitRequest]] = {}
        for row in rows:
            extras.setdefault(row["project_id"], []).append(self._limit_request(row))
        return (
            tuple(self._policy(row) for row in policies),
            {project_id: tuple(items) for project_id, items in extras.items()},
        )

    async def list_project_alert_identities(self) -> dict[UUID, ProjectAlertIdentity]:
        async with self.engine.connect() as connection:
            rows = (
                (
                    await connection.execute(
                        select(
                            t.project_policies.c.project_id,
                            t.keys.c.display_name.label("alert_name"),
                            t.keys.c.description.label("alert_description"),
                            t.keys.c.creator_kind.label("alert_creator_kind"),
                        ).select_from(
                            t.project_policies.outerjoin(
                                t.keys,
                                (t.keys.c.project_id == t.project_policies.c.project_id)
                                & (t.keys.c.owner_kind == "service")
                                & (t.keys.c.status != "revoked"),
                            )
                        )
                    )
                )
                .mappings()
                .all()
            )
        candidates: dict[
            UUID, tuple[set[ProjectAlertIdentity], set[ProjectAlertIdentity]]
        ] = {}
        for row in rows:
            machine, human = candidates.setdefault(row["project_id"], (set(), set()))
            if row["alert_name"] is None:
                continue
            identity = ProjectAlertIdentity(
                name=row["alert_name"],
                handle=(
                    row["alert_description"].removeprefix("Valkyrie project ")
                    if row["alert_creator_kind"] == "machine"
                    and row["alert_description"] is not None
                    and row["alert_description"].startswith("Valkyrie project ")
                    else None
                ),
            )
            (machine if row["alert_creator_kind"] == "machine" else human).add(identity)

        resolved: dict[UUID, ProjectAlertIdentity] = {}
        for project_id, (machine, human) in candidates.items():
            preferred = machine or human
            if len(preferred) == 1:
                resolved[project_id] = next(iter(preferred))
            else:
                logger.warning(
                    "Skipping project alert: project_id=%s reason=%s",
                    project_id,
                    "conflicting" if preferred else "missing",
                )
        return resolved

    async def get_project_policy(self, project_id: UUID) -> ProjectPolicy | None:
        async with self.engine.connect() as connection:
            row = (
                (
                    await connection.execute(
                        select(t.project_policies).where(
                            t.project_policies.c.project_id == project_id
                        )
                    )
                )
                .mappings()
                .first()
            )
            return self._policy(row) if row is not None else None

    async def owns_project_key(self, owner: KeyOwner, project_id: UUID) -> bool:
        async with self.engine.connect() as connection:
            row = (
                await connection.execute(
                    select(t.keys.c.id)
                    .where(
                        t.keys.c.project_id == project_id,
                        *(
                            t.keys.c[column] == value
                            for column, value in _owner(owner).items()
                        ),
                        t.keys.c.status != KeyStatus.REVOKED,
                    )
                    .limit(1)
                )
            ).first()
            return row is not None

    async def change_project_total(
        self,
        project_id: UUID,
        expected_revision: int,
        total_limit_usd: Decimal,
        reason: str,
        actor: HumanIdentity,
    ) -> ProjectPolicy | None:
        now = datetime.now(UTC)
        async with self.engine.begin() as connection:
            before = (
                await connection.execute(
                    select(t.project_policies.c.total_limit_usd).where(
                        t.project_policies.c.project_id == project_id
                    )
                )
            ).scalar_one_or_none()
            row = (
                (
                    await connection.execute(
                        update(t.project_policies)
                        .where(
                            t.project_policies.c.project_id == project_id,
                            t.project_policies.c.revision == expected_revision,
                        )
                        .values(
                            total_limit_usd=total_limit_usd,
                            budget_reason=reason,
                            revision=expected_revision + 1,
                            updated_at=now,
                        )
                        .returning(t.project_policies)
                    )
                )
                .mappings()
                .first()
            )
            if row is None:
                return None
            await self._audit(
                connection,
                actor,
                PROJECT_POLICY_AUDIT,
                project_id,
                "total_changed",
                detail={
                    "reason": reason,
                    "before": None,
                    "after": None,
                    "affected_request_ids": [],
                    "total_before_usd": str(before),
                    "total_after_usd": str(total_limit_usd),
                },
                occurred_at=now,
            )
            return self._policy(row)

    async def set_project_archived(
        self, project_id: UUID, archived: bool, actor: ActorIdentity
    ) -> ProjectPolicy | None:
        now = datetime.now(UTC)
        async with self.engine.begin() as connection:
            row = (
                (
                    await connection.execute(
                        update(t.project_policies)
                        .where(t.project_policies.c.project_id == project_id)
                        .values(
                            archived_at=now if archived else None,
                            updated_at=now,
                        )
                        .returning(t.project_policies)
                    )
                )
                .mappings()
                .first()
            )
            if row is None:
                return None
            await self._audit(
                connection,
                actor,
                PROJECT_POLICY_AUDIT,
                project_id,
                "archived" if archived else "unarchived",
                occurred_at=now,
            )
            return self._policy(row)

    async def list_project_policies(self) -> tuple[ProjectPolicy, ...]:
        async with self.engine.connect() as connection:
            rows = (
                (
                    await connection.execute(
                        select(t.project_policies).order_by(
                            t.project_policies.c.project_id
                        )
                    )
                )
                .mappings()
                .all()
            )
            return tuple(self._policy(row) for row in rows)

    async def update_project_policy(
        self,
        project_id: UUID,
        expected_revision: int,
        period: BudgetPeriod,
        base_limit_usd: Decimal,
        actor: HumanIdentity,
        reason: str | None = None,
    ) -> ProjectPolicy | None:
        async with self.engine.begin() as connection:
            current = (
                (
                    await connection.execute(
                        select(t.project_policies)
                        .where(t.project_policies.c.project_id == project_id)
                        .with_for_update()
                    )
                )
                .mappings()
                .first()
            )
            if current is None or current["revision"] != expected_revision:
                return None
            now = datetime.now(UTC)
            extras = await self._active_extras(connection, project_id, now)
            before = self._allowance_state(
                current["period"], current["base_limit_usd"], now, extras
            )
            total: Decimal | None = None
            if current["total_limit_usd"] is not None:
                moved: Decimal = (
                    current["total_limit_usd"]
                    + base_limit_usd
                    - current["base_limit_usd"]
                )
                if moved <= 0:
                    raise ValueError("project total must stay above zero")
                total = moved
            row = (
                (
                    await connection.execute(
                        update(t.project_policies)
                        .where(
                            t.project_policies.c.project_id == project_id,
                            t.project_policies.c.revision == expected_revision,
                        )
                        .values(
                            period=period,
                            base_limit_usd=base_limit_usd,
                            total_limit_usd=total,
                            revision=expected_revision + 1,
                            updated_at=now,
                        )
                        .returning(t.project_policies)
                    )
                )
                .mappings()
                .first()
            )
            if row is None:
                return None
            await self._audit(
                connection,
                actor,
                "project_policy",
                project_id,
                "policy_changed",
                detail={
                    "reason": reason,
                    "before": before,
                    "after": self._allowance_state(period, base_limit_usd, now, extras),
                    "affected_request_ids": [],
                },
                occurred_at=now,
            )
            return self._policy(row)

    @staticmethod
    def _limit_request_query():
        return select(
            t.project_limit_requests,
            t.project_policies.c.project_organization,
            t.project_policies.c.project_environment,
        ).join(t.project_policies)

    @staticmethod
    def _limit_request(row: RowMapping) -> ProjectLimitRequest:
        return ProjectLimitRequest(
            row["id"],
            ProjectReference(
                row["project_id"],
                row["project_organization"],
                row["project_environment"],
            ),
            row["kind"],
            row["increase_usd"],
            row["period_start"],
            row["period_end"],
            row["reason"],
            HumanIdentity(row["creator_issuer"], row["creator_subject"]),
            row["status"],
            HumanIdentity(row["decided_by_issuer"], row["decided_by_subject"])
            if row["decided_by_issuer"] is not None
            else None,
            row["decided_at"],
            row["created_at"],
            row["cancelled_at"],
            HumanIdentity(row["cancelled_by_issuer"], row["cancelled_by_subject"])
            if row["cancelled_by_issuer"] is not None
            else None,
            row["cancellation_reason"],
        )

    async def get_limit_request(self, request_id: UUID) -> ProjectLimitRequest | None:
        async with self.engine.connect() as connection:
            row = (
                (
                    await connection.execute(
                        self._limit_request_query().where(
                            t.project_limit_requests.c.id == request_id
                        )
                    )
                )
                .mappings()
                .first()
            )
            return self._limit_request(row) if row is not None else None

    async def list_limit_requests(self) -> tuple[ProjectLimitRequest, ...]:
        async with self.engine.connect() as connection:
            rows = (
                (
                    await connection.execute(
                        self._limit_request_query()
                        .where(
                            or_(
                                t.project_policies.c.archived_at.is_(None),
                                t.project_limit_requests.c.status
                                != LimitRequestStatus.PENDING,
                            )
                        )
                        .order_by(t.project_limit_requests.c.created_at.desc())
                    )
                )
                .mappings()
                .all()
            )
            return tuple(self._limit_request(row) for row in rows)

    async def list_project_limit_requests(
        self, project_id: UUID
    ) -> tuple[ProjectLimitRequest, ...]:
        async with self.engine.connect() as connection:
            rows = (
                (
                    await connection.execute(
                        self._limit_request_query()
                        .where(t.project_limit_requests.c.project_id == project_id)
                        .order_by(t.project_limit_requests.c.created_at.desc())
                    )
                )
                .mappings()
                .all()
            )
            return tuple(self._limit_request(row) for row in rows)

    async def project_history_sources(
        self, project_id: UUID
    ) -> tuple[
        ProjectPolicy | None, tuple[RowMapping, ...], tuple[ProjectLimitRequest, ...]
    ]:
        async with self.engine.connect() as connection:
            await connection.execution_options(isolation_level="REPEATABLE READ")
            async with connection.begin():
                policy_row = (
                    (
                        await connection.execute(
                            select(t.project_policies).where(
                                t.project_policies.c.project_id == project_id
                            )
                        )
                    )
                    .mappings()
                    .first()
                )
                audit_rows = (
                    (
                        await connection.execute(
                            select(t.audit)
                            .where(
                                t.audit.c.target_kind == "project_policy",
                                t.audit.c.target_id == project_id,
                            )
                            .order_by(t.audit.c.occurred_at, t.audit.c.id)
                        )
                    )
                    .mappings()
                    .all()
                )
                request_rows = (
                    (
                        await connection.execute(
                            self._limit_request_query()
                            .where(t.project_limit_requests.c.project_id == project_id)
                            .order_by(t.project_limit_requests.c.created_at.desc())
                        )
                    )
                    .mappings()
                    .all()
                )
        return (
            self._policy(policy_row) if policy_row is not None else None,
            tuple(audit_rows),
            tuple(self._limit_request(row) for row in request_rows),
        )

    async def change_temporary_increases(
        self,
        project_id: UUID,
        expected_revision: int,
        actor: HumanIdentity,
        reason: str,
        action: Literal["add", "set", "remove"],
        amount: Decimal | None,
        expected_period: tuple[datetime, datetime] | None,
    ) -> ProjectPolicy | None:
        async with self.engine.begin() as connection:
            policy = (
                (
                    await connection.execute(
                        select(t.project_policies)
                        .where(t.project_policies.c.project_id == project_id)
                        .with_for_update()
                    )
                )
                .mappings()
                .first()
            )
            if policy is None or policy["revision"] != expected_revision:
                return None
            now = datetime.now(UTC)
            if action != "remove":
                if period_bounds(policy["period"], now) != expected_period:
                    raise ValueError("project budget period changed since preview")
            if action == "set" and amount < policy["base_limit_usd"]:
                raise ValueError("temporary limit must be at least the recurring base")
            extras = await self._active_extras(connection, project_id, now)
            before = self._allowance_state(
                policy["period"], policy["base_limit_usd"], now, extras
            )
            cancelled_ids: list[UUID] = []
            if action != "add":
                cancelled_ids = [item["id"] for item in extras]
                if cancelled_ids:
                    await connection.execute(
                        update(t.project_limit_requests)
                        .where(t.project_limit_requests.c.id.in_(cancelled_ids))
                        .values(
                            cancelled_at=now,
                            **_identity(actor, "cancelled_by_"),
                            cancellation_reason=reason,
                        )
                    )
            if action == "add":
                increase = amount
            elif action == "set":
                increase = amount - policy["base_limit_usd"]
            else:
                increase = None
            added: tuple[RowMapping, ...] = ()
            if increase is not None and increase > 0:
                start, end = period_bounds(policy["period"], now)
                added_id = uuid4()
                added_row = (
                    (
                        await connection.execute(
                            insert(t.project_limit_requests)
                            .values(
                                id=added_id,
                                project_id=project_id,
                                kind=TEMPORARY_INCREASE,
                                increase_usd=increase,
                                period_start=start,
                                period_end=end,
                                reason=reason,
                                **_identity(actor, "creator_"),
                                status=LimitRequestStatus.APPROVED,
                                **_identity(actor, "decided_by_"),
                                decided_at=now,
                                created_at=now,
                            )
                            .returning(t.project_limit_requests)
                        )
                    )
                    .mappings()
                    .one()
                )
                added = (added_row,)
            values: dict[str, Decimal | int | datetime] = {
                "revision": expected_revision + 1,
                "updated_at": now,
            }
            if policy["total_limit_usd"] is not None:
                total: Decimal = (
                    policy["total_limit_usd"]
                    + sum((item["increase_usd"] for item in added), Decimal(0))
                    - sum(
                        (item["increase_usd"] for item in extras if action != "add"),
                        Decimal(0),
                    )
                )
                if total <= 0:
                    raise ValueError("project total must stay above zero")
                values["total_limit_usd"] = total
            row = (
                (
                    await connection.execute(
                        update(t.project_policies)
                        .where(t.project_policies.c.project_id == project_id)
                        .values(**values)
                        .returning(t.project_policies)
                    )
                )
                .mappings()
                .one()
            )
            after_extras = (extras if action == "add" else ()) + added
            await self._audit(
                connection,
                actor,
                "project_policy",
                project_id,
                {
                    "add": "temporary_added",
                    "set": "temporary_set",
                    "remove": "temporary_removed",
                }[action],
                detail={
                    "reason": reason,
                    "before": before,
                    "after": self._allowance_state(
                        policy["period"], policy["base_limit_usd"], now, after_extras
                    ),
                    "affected_request_ids": [
                        *(str(item) for item in cancelled_ids),
                        *(str(item["id"]) for item in added),
                    ],
                    "direct_request_ids": [str(item["id"]) for item in added],
                },
                occurred_at=now,
            )
            return self._policy(row)

    async def create_limit_request(
        self,
        request_id: UUID,
        project: ProjectReference,
        kind: LimitRequestKind,
        increase_usd: Decimal,
        reason: str,
        creator: HumanIdentity,
    ) -> ProjectLimitRequest:
        now = datetime.now(UTC)
        async with self.engine.begin() as connection:
            policy = (
                (
                    await connection.execute(
                        select(t.project_policies).where(
                            t.project_policies.c.project_id == project.project_id
                        )
                    )
                )
                .mappings()
                .first()
            )
            if policy is None or _project(policy) != project:
                raise ValueError("a capped approved project is required")
            await connection.execute(
                insert(t.project_limit_requests).values(
                    id=request_id,
                    project_id=project.project_id,
                    kind=kind,
                    increase_usd=increase_usd,
                    reason=reason,
                    **_identity(creator, "creator_"),
                    status=LimitRequestStatus.PENDING,
                    created_at=now,
                )
            )
            await self._audit(
                connection, creator, "project_limit_request", request_id, "submitted"
            )
        request = await self.get_limit_request(request_id)
        assert request is not None
        return request

    async def request_project_raise(
        self,
        request_id: UUID,
        project_id: UUID,
        expected_revision: int,
        target_total_usd: Decimal,
        reason: str,
        creator: HumanIdentity,
    ) -> tuple[ProjectPolicy, ProjectLimitRequest] | None:
        now = datetime.now(UTC)
        async with self.engine.begin() as connection:
            policy = (
                (
                    await connection.execute(
                        select(t.project_policies)
                        .where(t.project_policies.c.project_id == project_id)
                        .with_for_update()
                    )
                )
                .mappings()
                .first()
            )
            if policy is None or policy["revision"] != expected_revision:
                return None
            row = (
                (
                    await connection.execute(
                        update(t.project_policies)
                        .where(t.project_policies.c.project_id == project_id)
                        .values(
                            budget_reason=reason,
                            revision=expected_revision + 1,
                            updated_at=now,
                        )
                        .returning(t.project_policies)
                    )
                )
                .mappings()
                .one()
            )
            await connection.execute(
                insert(t.project_limit_requests).values(
                    id=request_id,
                    project_id=project_id,
                    kind=PERMANENT_INCREASE,
                    increase_usd=target_total_usd - policy["total_limit_usd"],
                    target_total_usd=target_total_usd,
                    source_revision=expected_revision + 1,
                    reason=reason,
                    **_identity(creator, "creator_"),
                    status=LimitRequestStatus.PENDING,
                    created_at=now,
                )
            )
            await self._audit(
                connection, creator, "project_limit_request", request_id, "submitted"
            )
        request = await self.get_limit_request(request_id)
        assert request is not None
        return self._policy(row), request

    async def _pending_limit_request(
        self, connection: AsyncConnection, request_id: UUID
    ) -> RowMapping:
        request = (
            (
                await connection.execute(
                    select(t.project_limit_requests)
                    .where(t.project_limit_requests.c.id == request_id)
                    .with_for_update()
                )
            )
            .mappings()
            .first()
        )
        if request is None or request["status"] != LimitRequestStatus.PENDING:
            raise ValueError("limit request is not pending")
        return request

    async def approve_limit_request(
        self, request_id: UUID, actor: HumanIdentity, expected_revision: int
    ) -> ProjectLimitRequest:
        async with self.engine.begin() as connection:
            request = await self._pending_limit_request(connection, request_id)
            policy = (
                (
                    await connection.execute(
                        select(t.project_policies)
                        .where(t.project_policies.c.project_id == request["project_id"])
                        .with_for_update()
                    )
                )
                .mappings()
                .one()
            )
            if policy["archived_at"] is not None:
                raise ValueError("project is archived; unarchive it to approve")
            if policy["revision"] != expected_revision:
                raise ValueError("project policy changed since approval was prepared")
            if (
                request["source_revision"] is not None
                and request["source_revision"] != policy["revision"]
            ):
                raise ValueError(
                    "project budget changed after this raise was filed; file it again"
                )
            now = datetime.now(UTC)
            extras = await self._active_extras(connection, request["project_id"], now)
            before = self._allowance_state(
                policy["period"], policy["base_limit_usd"], now, extras
            )
            decision = {
                "status": LimitRequestStatus.APPROVED,
                **_identity(actor, "decided_by_"),
                "decided_at": now,
            }
            values: dict[str, Decimal | int | datetime]
            base_after: Decimal = policy["base_limit_usd"]
            if request["kind"] == TEMPORARY_INCREASE:
                start, end = period_bounds(policy["period"], now)
                decision.update(period_start=start, period_end=end)
            if policy["total_limit_usd"] is not None:
                values = {
                    "total_limit_usd": request["target_total_usd"]
                    or policy["total_limit_usd"] + request["increase_usd"],
                    "revision": expected_revision + 1,
                    "updated_at": now,
                }
            elif request["kind"] == TEMPORARY_INCREASE:
                values = {"revision": expected_revision + 1, "updated_at": now}
            else:
                new_base: Decimal = policy["base_limit_usd"] + request["increase_usd"]
                base_after = new_base
                values = {
                    "base_limit_usd": new_base,
                    "revision": expected_revision + 1,
                    "updated_at": now,
                }
            await connection.execute(
                update(t.project_policies)
                .where(
                    t.project_policies.c.project_id == request["project_id"],
                    t.project_policies.c.revision == expected_revision,
                )
                .values(**values)
            )
            await connection.execute(
                update(t.project_limit_requests)
                .where(t.project_limit_requests.c.id == request_id)
                .values(**decision)
            )
            await self._audit(
                connection,
                actor,
                "project_policy",
                request["project_id"],
                "increase_approved",
                detail={
                    "reason": request["reason"],
                    "before": before,
                    "after": self._allowance_state(
                        policy["period"],
                        base_after,
                        now,
                        extras
                        + (
                            (dict(request) | decision,)
                            if request["kind"] == TEMPORARY_INCREASE
                            else ()
                        ),
                    ),
                    "affected_request_ids": [str(request_id)],
                    "request_id": str(request_id),
                },
                occurred_at=now,
            )
            await self._audit(
                connection, actor, "project_limit_request", request_id, "approved"
            )
        result = await self.get_limit_request(request_id)
        assert result is not None
        return result

    async def deny_limit_request(
        self, request_id: UUID, actor: HumanIdentity
    ) -> ProjectLimitRequest:
        async with self.engine.begin() as connection:
            await self._pending_limit_request(connection, request_id)
            await connection.execute(
                update(t.project_limit_requests)
                .where(t.project_limit_requests.c.id == request_id)
                .values(
                    status="denied",
                    **_identity(actor, "decided_by_"),
                    decided_at=datetime.now(UTC),
                )
            )
            await self._audit(
                connection, actor, "project_limit_request", request_id, "denied"
            )
        result = await self.get_limit_request(request_id)
        assert result is not None
        return result

    async def list_services(self) -> tuple[tuple[UUID, str], ...]:
        async with self.engine.connect() as connection:
            rows = (
                (
                    await connection.execute(
                        select(t.services).order_by(t.services.c.name)
                    )
                )
                .mappings()
                .all()
            )
            return tuple((row["id"], row["name"]) for row in rows)

    async def get_service(self, service_id: UUID) -> str | None:
        async with self.engine.connect() as connection:
            return (
                await connection.execute(
                    select(t.services.c.name).where(t.services.c.id == service_id)
                )
            ).scalar_one_or_none()
