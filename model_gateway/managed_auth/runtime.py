import json
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Protocol, cast
from uuid import UUID

from redis.asyncio import Redis

from model_gateway.budgets.schema import BudgetPeriod
from model_gateway.managed_auth.types import (
    HumanIdentity,
    KeyLimit,
    PersonalOwner,
    ProjectReference,
    ServiceOwner,
)
from model_gateway.run_token_types import ModelPermissions


@dataclass(frozen=True)
class TemporaryProjectIncrease:
    increase_usd: Decimal
    period_start: datetime
    period_end: datetime


@dataclass(frozen=True)
class RuntimeProjectLimit:
    period: BudgetPeriod
    base_limit_usd: Decimal
    temporary_increases: tuple[TemporaryProjectIncrease, ...]
    total_limit_usd: Decimal | None = None


@dataclass(frozen=True)
class RuntimeAuthority:
    key_id: UUID
    owner: PersonalOwner | ServiceOwner
    permissions: ModelPermissions
    project: ProjectReference
    project_limit: RuntimeProjectLimit | None
    key_limit: KeyLimit | None


def authority_hash_key(namespace: str) -> str:
    return f"gateway:managed-auth:{namespace}:active"


class AuthorityReaderRedis(Protocol):
    async def hget(self, name: str, key: str) -> str | None: ...

    async def hmget(self, name: str, keys: list[str]) -> list[str | None]: ...


class ManagedAuthorityResolver:
    def __init__(self, redis_client: Redis, *, namespace: str):
        self.redis: AuthorityReaderRedis = cast(AuthorityReaderRedis, redis_client)
        self._active_key = authority_hash_key(namespace)

    async def has_published_generation(self) -> bool:
        marker = await self.redis.hget(self._active_key, "__published__")
        return marker is not None and marker.startswith("v1:")

    async def by_key_id(self, key_id: UUID) -> RuntimeAuthority | None:
        return await self._resolve(f"k:{key_id}")

    async def by_verifier(self, verifier_hash: str) -> RuntimeAuthority | None:
        return await self._resolve(f"v:{verifier_hash}")

    async def _resolve(self, field: str) -> RuntimeAuthority | None:
        marker, record = await self.redis.hmget(
            self._active_key, ["__published__", field]
        )
        if marker is None or not marker.startswith("v1:"):
            raise RuntimeError("Managed authority generation is not published")
        if record is None:
            return None
        return deserialize_authority(record)


def serialize_authority(authority: RuntimeAuthority) -> str:
    owner = authority.owner
    owner_data = (
        {
            "kind": "personal",
            "issuer": owner.identity.issuer,
            "subject": owner.identity.subject,
        }
        if isinstance(owner, PersonalOwner)
        else {"kind": "service", "service_id": str(owner.service_id)}
    )
    project = authority.project
    project_limit = authority.project_limit
    key_limit = authority.key_limit
    return json.dumps(
        {
            "key_id": str(authority.key_id),
            "owner": owner_data,
            "permissions": authority.permissions.model_dump(),
            "project": {
                "project_id": str(project.project_id),
                "organization": project.organization,
                "environment": project.environment,
            },
            "project_limit": (
                {
                    "period": project_limit.period,
                    "base_limit_usd": str(project_limit.base_limit_usd),
                    "temporary_increases": [
                        {
                            "increase_usd": str(increase.increase_usd),
                            "period_start": increase.period_start.isoformat(),
                            "period_end": increase.period_end.isoformat(),
                        }
                        for increase in project_limit.temporary_increases
                    ],
                    "total_limit_usd": (
                        str(project_limit.total_limit_usd)
                        if project_limit.total_limit_usd is not None
                        else None
                    ),
                }
                if project_limit is not None
                else None
            ),
            "key_limit": (
                {"period": key_limit.period, "limit_usd": str(key_limit.limit_usd)}
                if key_limit is not None
                else None
            ),
        },
        separators=(",", ":"),
    )


def deserialize_authority(record: str) -> RuntimeAuthority:
    data = json.loads(record)
    owner_data = data["owner"]
    owner = (
        PersonalOwner(HumanIdentity(owner_data["issuer"], owner_data["subject"]))
        if owner_data["kind"] == "personal"
        else ServiceOwner(UUID(owner_data["service_id"]))
    )
    project_data = data["project"]
    project_limit_data = data["project_limit"]
    key_limit_data = data["key_limit"]
    project_limit = (
        RuntimeProjectLimit(
            period=project_limit_data["period"],
            base_limit_usd=Decimal(project_limit_data["base_limit_usd"]),
            temporary_increases=tuple(
                TemporaryProjectIncrease(
                    increase_usd=Decimal(increase["increase_usd"]),
                    period_start=datetime.fromisoformat(increase["period_start"]),
                    period_end=datetime.fromisoformat(increase["period_end"]),
                )
                for increase in project_limit_data["temporary_increases"]
            ),
            total_limit_usd=(
                Decimal(project_limit_data["total_limit_usd"])
                if project_limit_data.get("total_limit_usd") is not None
                else None
            ),
        )
        if project_limit_data is not None
        else None
    )
    key_limit = (
        KeyLimit(
            period=key_limit_data["period"],
            limit_usd=Decimal(key_limit_data["limit_usd"]),
        )
        if key_limit_data is not None
        else None
    )
    return RuntimeAuthority(
        key_id=UUID(data["key_id"]),
        owner=owner,
        # Old publishers may still send group limits during a rolling release.
        # TODO: Drop the serialized permissions field after old readers retire.
        permissions=ModelPermissions(
            allowed_models=None,
            allowed_embedding_models=[],
            allowed_operations=None,
        ),
        project=ProjectReference(
            UUID(project_data["project_id"]),
            project_data["organization"],
            project_data["environment"],
        ),
        project_limit=project_limit,
        key_limit=key_limit,
    )
