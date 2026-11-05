import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from decimal import Decimal
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from sqlalchemy.exc import IntegrityError
from fastapi import FastAPI
from jwt.exceptions import PyJWTError

from model_gateway.managed_auth.api import router
from model_gateway.managed_auth.database import create_engine
from model_gateway.managed_auth.identity import HumanVerifier, MachineVerifier
from model_gateway.managed_auth.service import PLATFORM_ADMIN, ManagedAuthService
from model_gateway.managed_auth.store import ManagedAuthStore
from model_gateway.managed_auth.types import (
    LimitRequestStatus,
    HumanIdentity,
    PersonalOwner,
)

ISSUER = "https://identity.example"
PREFIX = "/admin/managed-auth"


@dataclass
class Project:
    client: httpx.AsyncClient
    path: str
    spend: dict[str, Decimal]
    store: ManagedAuthStore
    member: HumanIdentity

    async def call(self, token: str, method: str, path: str, **kwargs: Any):
        return await self.client.request(
            method,
            PREFIX + path,
            headers={"Authorization": f"Bearer {token}"},
            **kwargs,
        )

    async def policy(self) -> dict[str, Any]:
        return (await self.call("admin", "GET", f"{self.path}/policy")).json()

    async def edit(self, token: str, limit_usd: str, reason: str):
        return await self.call(
            token,
            "PUT",
            f"{self.path}/budget",
            json={
                "expected_revision": (await self.policy())["revision"],
                "limit_usd": limit_usd,
                "reason": reason,
            },
        )


@asynccontextmanager
async def project_with_member() -> AsyncIterator[Project]:
    engine = create_engine(os.environ["MODEL_GATEWAY_AUTH_DATABASE_URL"], 5)
    db = ManagedAuthStore(engine)
    tenant_id = f"descope-tenant-{uuid4()}"
    machine_key_id = f"registered-machine-{uuid4()}"
    service_id = uuid4()
    project_id = uuid4()
    subjects = {name: f"{name}-{uuid4()}" for name in ("member", "admin", "outsider")}

    async def validate_session(token: str):
        if token == "machine-access-key":
            raise PyJWTError("Access key is not a human session")
        return {"sessionToken": {"iss": ISSUER, "sub": subjects[token]}}

    async def load_user(user_id: str):
        admin = user_id == subjects["admin"]
        return {
            "user": {
                "userId": user_id,
                "status": "enabled",
                "userTenants": [{"tenantId": tenant_id}],
                "permissions": [PLATFORM_ADMIN] if admin else [],
            }
        }

    async def exchange_access_key(access_key: str):
        return {
            "sessionToken": {"iss": ISSUER},
            "keyId": machine_key_id,
            "tenants": {tenant_id: {}},
        }

    human_verifier = HumanVerifier(
        "descope-project", ISSUER, tenant_id, "management-key", machine_key_id
    )
    human_verifier._client = SimpleNamespace(
        validate_session=validate_session,
        mgmt=SimpleNamespace(user=SimpleNamespace(load_by_user_id=load_user)),
    )
    machine_verifier = MachineVerifier(
        "descope-project", ISSUER, tenant_id, machine_key_id, service_id
    )
    machine_verifier._client = SimpleNamespace(exchange_access_key=exchange_access_key)
    spend = {str(project_id): Decimal("0")}

    async def project_total_spend(spent_project_id):
        return spend.get(str(spent_project_id))

    app = FastAPI()
    app.include_router(router)
    app.state.managed_auth_store = db
    app.state.managed_auth_service = ManagedAuthService(db, Decimal("500"))
    app.state.managed_auth_verifier = human_verifier
    app.state.managed_machine_verifier = machine_verifier
    app.state.budget_service = SimpleNamespace(project_total_spend=project_total_spend)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            project = Project(
                client,
                f"/projects/{project_id}",
                spend,
                db,
                HumanIdentity(ISSUER, subjects["member"]),
            )
            await db.create_service(
                service_id,
                f"registered-{uuid4()}",
                HumanIdentity(ISSUER, subjects["admin"]),
            )
            requested = await project.call(
                "machine-access-key",
                "POST",
                "/machine/requests",
                json={
                    "project_id": str(project_id),
                    "project_organization_id": str(uuid4()),
                    "project_environment": "dev",
                    "display_name": "project key",
                    "description": "worker",
                    "requested_budget": "5000",
                    "budget_reason": "full rerun",
                },
            )
            assert requested.status_code == 200, requested.text
            approved = await project.call(
                "admin",
                "POST",
                f"/requests/{requested.json()['id']}/approve",
                json={
                    "initial_policy": {"period": "monthly", "base_limit_usd": "100.00"}
                },
            )
            assert approved.status_code == 200, approved.text
            member_key = await project.call(
                "member",
                "POST",
                "/requests",
                json={
                    "project_id": str(project_id),
                    "display_name": "member key",
                    "description": "research",
                    "key_limit": None,
                    "owner": {
                        "kind": "personal",
                        "identity": {"issuer": ISSUER, "subject": subjects["member"]},
                    },
                },
            )
            assert member_key.json()["status"] == "approved", member_key.text
            yield project
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_raising_the_budget_files_a_pending_request():
    async with project_with_member() as project:
        raised = await project.edit("member", "150.00", "longer eval sweep")
        assert raised.status_code == 200, raised.text
        assert Decimal(raised.json()["policy"]["total_limit_usd"]) == Decimal("100")
        pending = raised.json()["pending_request"]
        assert pending["status"] == "pending" and Decimal(
            pending["increase_usd"]
        ) == Decimal("50")
        assert pending["reason"] == "longer eval sweep"
        assert raised.json()["policy"]["budget_reason"] == "longer eval sweep"
        assert (await project.policy())["budget_reason"] == "longer eval sweep"
        assert Decimal((await project.policy())["total_limit_usd"]) == Decimal("100")


@pytest.mark.asyncio
async def test_lowering_the_budget_applies_now():
    async with project_with_member() as project:
        project.spend[project.path.removeprefix("/projects/")] = Decimal("30")
        lowered = await project.edit("member", "60.00", "sweep finished early")
        assert lowered.status_code == 200, lowered.text
        assert lowered.json()["pending_request"] is None
        policy = await project.policy()
        assert Decimal(policy["total_limit_usd"]) == Decimal("60")
        assert policy["budget_reason"] == "sweep finished early"


@pytest.mark.asyncio
async def test_lowering_below_spend_is_refused_with_the_spent_amount():
    async with project_with_member() as project:
        project.spend[project.path.removeprefix("/projects/")] = Decimal("80")
        refused = await project.edit("member", "50.00", "trim")
        assert refused.status_code == 409
        assert refused.json()["detail"] == (
            "The new budget $50.00 is below the $80 this project has already spent"
        )
        assert Decimal((await project.policy())["total_limit_usd"]) == Decimal("100")


@pytest.mark.asyncio
async def test_editing_the_reason_applies_without_approval():
    async with project_with_member() as project:
        edited = await project.edit("member", "100.00", "now covers the legal sweep")
        assert edited.status_code == 200, edited.text
        assert edited.json()["pending_request"] is None
        policy = await project.policy()
        assert policy["budget_reason"] == "now covers the legal sweep"
        assert Decimal(policy["total_limit_usd"]) == Decimal("100")


@pytest.mark.asyncio
async def test_members_file_raises_and_only_admins_approve():
    async with project_with_member() as project:
        assert (await project.edit("outsider", "150.00", "more")).status_code == 403
        filed = await project.call(
            "member",
            "POST",
            f"{project.path}/limit-requests",
            json={
                "kind": "permanent",
                "increase_usd": "25.00",
                "reason": "more tasks",
            },
        )
        assert filed.status_code == 200, filed.text
        approve_path = f"/limit-requests/{filed.json()['id']}/approve"
        revision = {
            "expected_current_policy_revision": (await project.policy())["revision"]
        }
        assert (
            await project.call("member", "POST", approve_path, json=revision)
        ).status_code == 403
        approved = await project.call("admin", "POST", approve_path, json=revision)
        assert approved.status_code == 200 and approved.json()["status"] == "approved"
        assert Decimal((await project.policy())["total_limit_usd"]) == Decimal("125")


@pytest.mark.asyncio
async def test_archived_raise_is_hidden_and_unapprovable_until_unarchived():
    async with project_with_member() as project:
        raised = await project.edit("member", "150.00", "more")
        request_id = raised.json()["pending_request"]["id"]

        async def inbox_ids():
            return {
                item["id"]
                for item in (
                    await project.call("admin", "GET", "/limit-requests")
                ).json()
            }

        assert request_id in await inbox_ids()
        archived = await project.call(
            "machine-access-key", "POST", f"/machine{project.path}/archive"
        )
        assert (
            archived.status_code == 200 and archived.json()["archived_at"] is not None
        )
        assert request_id not in await inbox_ids()
        project_list = (
            await project.call("admin", "GET", f"{project.path}/limit-requests")
        ).json()
        assert request_id in {item["id"] for item in project_list}
        approve_path = f"/limit-requests/{request_id}/approve"
        revision = {
            "expected_current_policy_revision": (await project.policy())["revision"]
        }
        refused = await project.call("admin", "POST", approve_path, json=revision)
        assert refused.status_code == 409
        assert (
            await project.call("admin", "GET", f"/limit-requests/{request_id}")
        ).json()["status"] == "pending"
        unarchived = await project.call(
            "machine-access-key", "POST", f"/machine{project.path}/unarchive"
        )
        assert (
            unarchived.status_code == 200 and unarchived.json()["archived_at"] is None
        )
        assert request_id in await inbox_ids()
        assert (
            await project.call("admin", "POST", approve_path, json=revision)
        ).status_code == 200


@pytest.mark.asyncio
async def test_archive_hides_only_pending_raises_from_the_admin_list():
    async with project_with_member() as project:
        decided = (await project.edit("member", "150.00", "more")).json()[
            "pending_request"
        ]["id"]
        revision = {
            "expected_current_policy_revision": (await project.policy())["revision"]
        }
        approved = await project.call(
            "admin", "POST", f"/limit-requests/{decided}/approve", json=revision
        )
        assert approved.status_code == 200, approved.text
        pending = (await project.edit("member", "200.00", "again")).json()[
            "pending_request"
        ]["id"]
        await project.call(
            "machine-access-key", "POST", f"/machine{project.path}/archive"
        )
        listed = {
            item["id"]
            for item in (await project.call("admin", "GET", "/limit-requests")).json()
        }
        assert decided in listed and pending not in listed


@pytest.mark.asyncio
async def test_lowering_answers_503_while_spend_is_unavailable():
    async with project_with_member() as project:
        project.spend.clear()
        refused = await project.edit("member", "50.00", "trim")
        assert refused.status_code == 503, refused.text
        assert Decimal((await project.policy())["total_limit_usd"]) == Decimal("100")


@pytest.mark.asyncio
async def test_a_raise_is_filed_only_at_the_revision_it_was_priced_on():
    async with project_with_member() as project:
        policy = await project.policy()
        project_id = UUID(policy["project"]["project_id"])
        stale = await project.store.request_project_raise(
            uuid4(),
            project_id,
            policy["revision"] + 1,
            Decimal("150"),
            "more",
            project.member,
        )
        assert stale is None
        filed = await project.store.request_project_raise(
            uuid4(),
            project_id,
            policy["revision"],
            Decimal("150"),
            "more",
            project.member,
        )
        assert filed is not None and filed[1].status == LimitRequestStatus.PENDING


@pytest.mark.asyncio
async def test_a_raise_that_cannot_be_filed_leaves_the_reason_unchanged():
    async with project_with_member() as project:
        policy = await project.policy()
        project_id = UUID(policy["project"]["project_id"])
        request_id = uuid4()
        first = await project.store.request_project_raise(
            request_id,
            project_id,
            policy["revision"],
            Decimal("150"),
            "first",
            project.member,
        )
        assert first is not None
        with pytest.raises(IntegrityError):
            await project.store.request_project_raise(
                request_id,
                project_id,
                first[0].revision,
                Decimal("200"),
                "second",
                project.member,
            )
        after = await project.policy()
        assert (after["budget_reason"], after["revision"]) == (
            "first",
            first[0].revision,
        )


@pytest.mark.asyncio
async def test_a_raise_lands_on_its_target_or_must_be_filed_again():
    async with project_with_member() as project:
        stale = (await project.edit("member", "150.00", "more")).json()[
            "pending_request"
        ]["id"]
        assert (await project.edit("member", "60.00", "trim")).status_code == 200
        revision = {
            "expected_current_policy_revision": (await project.policy())["revision"]
        }
        refused = await project.call(
            "admin", "POST", f"/limit-requests/{stale}/approve", json=revision
        )
        assert refused.status_code == 409, refused.text
        fresh = (await project.edit("member", "150.00", "more")).json()[
            "pending_request"
        ]["id"]
        revision = {
            "expected_current_policy_revision": (await project.policy())["revision"]
        }
        approved = await project.call(
            "admin", "POST", f"/limit-requests/{fresh}/approve", json=revision
        )
        assert approved.status_code == 200, approved.text
        assert Decimal((await project.policy())["total_limit_usd"]) == Decimal("150")


@pytest.mark.asyncio
async def test_a_machine_archives_only_projects_its_service_holds_keys_in():
    async with project_with_member() as project:
        other = await project.call(
            "machine-access-key", "POST", f"/machine/projects/{uuid4()}/archive"
        )
        assert other.status_code == 403, other.text
        own = await project.call(
            "machine-access-key", "POST", f"/machine{project.path}/archive"
        )
        assert own.status_code == 200, own.text


@pytest.mark.asyncio
async def test_a_member_whose_key_is_revoked_cannot_edit():
    async with project_with_member() as project:
        key = next(
            key
            for key in await project.store.list_keys()
            if key.owner == PersonalOwner(project.member)
        )
        assert (
            await project.store.revoke_key(key.id, key.revision, project.member)
            is not None
        )
        assert (await project.edit("member", "150.00", "more")).status_code == 403


@pytest.mark.asyncio
async def test_a_new_project_keeps_the_reason_it_was_requested_with():
    async with project_with_member() as project:
        assert (await project.policy())["budget_reason"] == "full rerun"
