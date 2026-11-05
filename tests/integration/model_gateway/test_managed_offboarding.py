import os
from decimal import Decimal
from uuid import uuid4

import httpx
import pytest
from redis.asyncio import Redis

from model_gateway.managed_auth.authority_publisher import ManagedAuthorityPublisher
from model_gateway.managed_auth.database import create_engine
from model_gateway.managed_auth.identity import HumanVerifier
from model_gateway.managed_auth.reconciliation import HumanReconciler
from model_gateway.managed_auth.runtime import ManagedAuthorityResolver, authority_hash_key
from model_gateway.managed_auth.store import ManagedAuthStore
from model_gateway.managed_auth.types import (
    HumanIdentity, PersonalOwner, ProjectReference, ServiceOwner,
)


ISSUER = "https://api.descope.com/project-test"
ORGANIZATION = "organization-test"


@pytest.mark.integration
async def test_background_offboarding_changes_personal_authority_without_service_disruption():
    engine = create_engine(os.environ["MODEL_GATEWAY_AUTH_DATABASE_URL"], 4)
    store = ManagedAuthStore(engine)
    redis = Redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
    namespace = f"offboarding-{uuid4().hex}"
    owner_key = f"gateway:managed-auth:{namespace}:owner"
    token = uuid4().hex
    resolver = ManagedAuthorityResolver(redis, namespace=namespace)
    owner = HumanIdentity(ISSUER, f"owner-{uuid4()}")
    service_id = uuid4()
    personal_id, service_key_id = uuid4(), uuid4()
    project = ProjectReference(uuid4(), ORGANIZATION, "test")
    service_project = ProjectReference(uuid4(), ORGANIZATION, "test")
    verifier = HumanVerifier("project-test", ISSUER, ORGANIZATION, "management-test", f"machine-{uuid4()}")
    state = {"status": "enabled", "tenants": [{"tenantId": ORGANIZATION}], "http_status": 200}

    def respond(request):
        assert request.url.path == "/v1/mgmt/user"
        assert request.url.params["userId"] == owner.subject
        if state["http_status"] != 200:
            return httpx.Response(state["http_status"], json={"errorMessage": "unavailable"})
        return httpx.Response(200, json={"user": {
            "userId": owner.subject, "status": state["status"],
            "userTenants": state["tenants"], "permissions": [],
        }})

    verifier._client._mgmt_http._async_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    verifier._client._license_attempted = True
    reconciler = HumanReconciler(store, verifier, ISSUER)
    publisher = ManagedAuthorityPublisher(engine, redis, reconciler, namespace=namespace)
    try:
        await store.record_human(owner, True, owner)
        await store.create_service(service_id, "background", owner)
        personal_request = await store.create_request(
            uuid4(), "personal", "reason", owner, PersonalOwner(owner), None, project, None, None,
        )
        service_request = await store.create_request(
            uuid4(), "service", "reason", owner, ServiceOwner(service_id), None,
            service_project, None, None,
        )
        personal_key = await store.decide_request(
            personal_request.id, owner, personal_id, ("monthly", Decimal("15")), None,
        )
        service_key = await store.decide_request(
            service_request.id, owner, service_key_id, ("monthly", Decimal("20")), None,
        )
        await store.collect_key(personal_id, personal_key.revision, f"personal-{personal_id}", owner)
        await store.collect_key(service_key_id, service_key.revision, f"service-{service_key_id}", owner)
        assert await redis.set(owner_key, token, nx=True, ex=60)
        assert await publisher.publish_once(token)
        assert (await resolver.by_key_id(personal_id)).project == project
        assert (await resolver.by_key_id(personal_id)).project_limit.base_limit_usd == Decimal("15")
        assert (await resolver.by_key_id(service_key_id)).project == service_project
        assert (await resolver.by_key_id(service_key_id)).project_limit.base_limit_usd == Decimal("20")

        state["tenants"] = []
        await reconciler.reconcile_once()
        assert await publisher.publish_once(token)
        assert await store.human_enabled(owner) is False
        assert await resolver.by_key_id(personal_id) is None
        assert await resolver.by_key_id(service_key_id) is not None

        state["http_status"] = 503
        state["tenants"] = [{"tenantId": ORGANIZATION}]
        await reconciler.reconcile_once()
        assert await store.human_enabled(owner) is False
        state["http_status"] = 200
        await reconciler.reconcile_once()
        assert await publisher.publish_once(token)
        assert await store.human_enabled(owner) is True
        assert await resolver.by_key_id(personal_id) is not None
        state["http_status"] = 404
        await reconciler.reconcile_once()
        assert await publisher.publish_once(token)
        assert await store.human_enabled(owner) is False
        assert await resolver.by_key_id(service_key_id) is not None
    finally:
        await redis.delete(authority_hash_key(namespace), owner_key)
        await redis.aclose()
        await verifier.aclose()
        await engine.dispose()
