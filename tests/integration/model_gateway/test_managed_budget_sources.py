import os
from decimal import Decimal
from uuid import NAMESPACE_URL, uuid4, uuid5

import pytest

from model_gateway.managed_auth.budget_sources import list_key_spend_sources
from model_gateway.managed_auth.database import create_engine
from model_gateway.managed_auth.store import ManagedAuthStore
from model_gateway.managed_auth.types import (
    HumanIdentity, KeyLimit, PersonalOwner, ProjectReference, ServiceOwner,
)


@pytest.mark.asyncio
async def test_approved_project_sources_and_system_eligibility():
    engine = create_engine(os.environ["MODEL_GATEWAY_AUTH_DATABASE_URL"], 3)
    store = ManagedAuthStore(engine)
    issuer = f"https://issuer-{uuid4()}.example"
    actor = HumanIdentity(issuer, "admin")
    owner = HumanIdentity(issuer, "owner")
    project = ProjectReference(uuid4(), "organization", "environment")
    request_id = uuid4()
    try:
        await store.record_human(actor, True, actor)
        await store.record_system_human_eligibility(owner, False)
        assert await store.list_known_humans(issuer) == [actor, owner]
        assert await store.human_enabled(owner) is False
        audit_id = uuid5(NAMESPACE_URL, f"{owner.issuer}\0{owner.subject}")
        await store.record_system_human_eligibility(owner, False)
        assert [(event.change, event.actor_kind, event.actor.subject)
                for event in await store.audit_for("human", audit_id)] == [
                    ("disabled", "system", "eligibility-sync")]
        await store.record_system_human_eligibility(owner, True)
        assert await store.human_enabled(owner) is True
        assert [(event.change, event.actor_kind)
                for event in await store.audit_for("human", audit_id)] == [
                    ("disabled", "system"), ("enabled", "system")]
        assert project.project_id not in {
            policy.project.project_id for policy in await store.list_project_policies()
        }
        await store.create_request(
            request_id, "project key", "first approval", actor,
            PersonalOwner(actor), None, project, None, None,
        )
        assert project.project_id not in {
            policy.project.project_id for policy in await store.list_project_policies()
        }
        key = await store.decide_request(
            request_id, actor, uuid4(), ("weekly", Decimal("100")), None,
        )
        assert key is not None and key.project == project
        policy = await store.get_project_policy(project.project_id)
        assert policy is not None
        assert policy.project == project
        assert policy.period == "weekly" and policy.base_limit_usd == Decimal("100")
        sources = [policy.project.project_id for policy in await store.list_project_policies()]
        assert sources.count(project.project_id) == 1
        assert key.id not in await list_key_spend_sources(engine)
        changed_reference = ProjectReference(project.project_id, "other-organization", "environment")
        changed_request = uuid4()
        await store.create_request(
            changed_request, "other project key", "wrong binding", actor,
            PersonalOwner(actor), None, changed_reference, None, None,
        )
        with pytest.raises(ValueError, match="project reference changed"):
            await store.decide_request(changed_request, actor, uuid4(), None, None)
        assert (await store.get_project_policy(project.project_id)) == policy
        assert [policy.project.project_id for policy in await store.list_project_policies()].count(
            project.project_id
        ) == 1
        project_with_optional_key_cap = ProjectReference(uuid4(), "organization", "environment")
        optional_key_request = uuid4()
        await store.create_request(
            optional_key_request, "independent key cap", "project with optional key cap", actor,
            PersonalOwner(actor), None, project_with_optional_key_cap, None, None,
        )
        optional_key = await store.decide_request(
            optional_key_request, actor, uuid4(), ("monthly", Decimal("25")), None,
        )
        assert optional_key is not None and optional_key.key_limit is None
        assert project_with_optional_key_cap.project_id in {
            policy.project.project_id for policy in await store.list_project_policies()
        }
        capped = await store.update_key_limit(
            optional_key.id, optional_key.revision, KeyLimit("daily", Decimal("5")), actor,
        )
        assert capped is not None
        assert (await list_key_spend_sources(engine)).count(capped.id) == 1
        assert project_with_optional_key_cap.project_id in {
            policy.project.project_id for policy in await store.list_project_policies()
        }
        cleared = await store.update_key_limit(capped.id, capped.revision, None, actor)
        assert cleared is not None and cleared.key_limit is None
        assert capped.id not in await list_key_spend_sources(engine)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_first_service_key_approval_creates_shared_project_policy():
    engine = create_engine(os.environ["MODEL_GATEWAY_AUTH_DATABASE_URL"], 3)
    store = ManagedAuthStore(engine)
    actor = HumanIdentity(f"https://issuer-{uuid4()}.example", "admin")
    service_id, project_id = uuid4(), uuid4()
    try:
        await store.record_human(actor, True, actor)
        await store.create_service(service_id, "worker", actor)
        project = ProjectReference(project_id, "organization", "environment")
        requested = await store.create_request(
            uuid4(), "worker key", "deployment", actor,
            ServiceOwner(service_id), None, project, None, None,
        )
        key = await store.decide_request(
            requested.id, actor, uuid4(), ("monthly", Decimal("50")), None,
        )
        assert key is not None
        policy = await store.get_project_policy(project_id)
        assert policy is not None
        assert policy.project == project
        assert policy.period == "monthly" and policy.base_limit_usd == Decimal("50")
        assert project_id in {
            policy.project.project_id for policy in await store.list_project_policies()
        }
        assert key.project == project
        assert key.status == "pending_collection"
        requests = await store.list_requests()
        request = next(request for request in requests if request.key_id == key.id)
        assert request.status == "approved"
    finally:
        await engine.dispose()


