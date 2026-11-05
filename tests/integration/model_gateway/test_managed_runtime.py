import os
from decimal import Decimal
from uuid import uuid4

import pytest
from redis.asyncio import Redis

from model_gateway.managed_auth.authority_publisher import ManagedAuthorityPublisher
from model_gateway.managed_auth.database import create_engine
from model_gateway.managed_auth.identity import HumanVerifier
from model_gateway.managed_auth.reconciliation import HumanReconciler
from model_gateway.managed_auth.runtime import ManagedAuthorityResolver, authority_hash_key
from model_gateway.managed_auth.store import ManagedAuthStore
from model_gateway.managed_auth.types import (
    HumanIdentity, KeyLimit, PersonalOwner, ProjectReference, ServiceOwner,
)


@pytest.mark.asyncio
async def test_current_personal_and_service_authority():
    engine = create_engine(os.environ["MODEL_GATEWAY_AUTH_DATABASE_URL"], 4)
    store = ManagedAuthStore(engine)
    redis = Redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
    namespace = f"runtime-{uuid4().hex}"
    owner_key = f"gateway:managed-auth:{namespace}:owner"
    token = uuid4().hex
    verifier = HumanVerifier(
        "project-test", "https://issuer.example", "organization-test", "management-test",
        f"machine-{uuid4()}",
    )
    publisher = ManagedAuthorityPublisher(
        engine, redis, HumanReconciler(store, verifier, "https://issuer.example"),
        namespace=namespace,
    )
    resolver = ManagedAuthorityResolver(redis, namespace=namespace)
    person = HumanIdentity("https://issuer.example", f"owner-{uuid4()}")
    creator = HumanIdentity("https://issuer.example", f"creator-{uuid4()}")
    project = ProjectReference(uuid4(), "organization", "dev")
    service_project = ProjectReference(uuid4(), "organization", "dev")
    service_id, personal_key_id, service_key_id = uuid4(), uuid4(), uuid4()
    personal_verifier, service_verifier = f"personal-{uuid4()}", f"service-{uuid4()}"
    try:
        await store.record_human(person, True, person)
        await store.record_human(creator, True, creator)
        await store.create_service(service_id, "batch", creator)
        personal_request_id = uuid4()
        service_request_id = uuid4()
        await store.create_request(personal_request_id, "personal", "work", person,
                                   PersonalOwner(person), None, project, None, None)
        await store.create_request(service_request_id, "service", "work", creator,
                                   ServiceOwner(service_id), None, service_project, None, None)
        personal_pending = await store.decide_request(
            personal_request_id, creator, personal_key_id, ("weekly", Decimal("70")), None,
        )
        service_pending = await store.decide_request(
            service_request_id, creator, service_key_id, ("monthly", Decimal("30")), None,
        )
        assert await redis.set(owner_key, token, nx=True, ex=60)
        assert await publisher.publish_once(token)
        assert await resolver.by_key_id(personal_key_id) is None
        assert await resolver.by_key_id(service_key_id) is None
        personal = await store.collect_key(personal_key_id, personal_pending.revision, personal_verifier, person)
        service = await store.collect_key(service_key_id, service_pending.revision, service_verifier, creator)
        personal_limit = KeyLimit("daily", Decimal("12.50"))
        limited = await store.update_key_limit(personal_key_id, personal.revision, personal_limit, creator)
        assert await publisher.publish_once(token)
        assert await resolver.by_verifier("missing-verifier") is None
        authority = await resolver.by_verifier(personal_verifier)
        assert authority.key_id == personal.id
        assert authority.owner == personal.owner
        assert authority.permissions.allowed_models is None
        assert authority.permissions.allowed_embedding_models == []
        assert authority.permissions.allowed_operations is None
        assert authority.project == project
        assert authority.project_limit.period == "weekly"
        assert authority.project_limit.base_limit_usd == Decimal("70")
        assert authority.key_limit == personal_limit
        service_authority = await resolver.by_key_id(service_key_id)
        assert service_authority.key_id == service.id
        assert service_authority.owner == service.owner
        assert service_authority.project == service_project
        assert service_authority.project_limit.base_limit_usd == Decimal("30")
        assert service_authority.key_limit is None
        assert service_authority.permissions.allowed_models is None

        await store.record_human(person, False, creator)
        await store.record_human(creator, False, creator)
        assert await publisher.publish_once(token)
        assert await resolver.by_key_id(personal_key_id) is None
        assert await resolver.by_verifier(personal_verifier) is None
        assert (await resolver.by_verifier(service_verifier)).key_id == service_key_id

        await store.record_human(person, True, creator)
        assert await publisher.publish_once(token)
        updated = await resolver.by_key_id(personal_key_id)
        assert updated.key_id == personal_key_id
        assert updated.permissions.allowed_models is None
        assert updated.project == project
        assert updated.project_limit.base_limit_usd == Decimal("70")
        assert updated.key_limit == personal_limit

        await store.revoke_key(personal_key_id, limited.revision, creator)
        await store.revoke_key(service_key_id, service.revision, creator)
        assert await publisher.publish_once(token)
        assert await resolver.by_key_id(personal_key_id) is None
        assert await resolver.by_verifier(personal_verifier) is None
        assert await resolver.by_key_id(service_key_id) is None
        assert await resolver.by_verifier(service_verifier) is None
    finally:
        await redis.delete(authority_hash_key(namespace), owner_key)
        await redis.aclose()
        await verifier.aclose()
        await engine.dispose()
