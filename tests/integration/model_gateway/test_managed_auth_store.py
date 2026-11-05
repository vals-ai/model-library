import asyncio
import hashlib
import os
from contextlib import asynccontextmanager
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import delete, select, update
from sqlalchemy.exc import DBAPIError

from model_gateway.managed_auth import tables
from model_gateway.managed_auth.database import create_engine
from model_gateway.managed_auth.store import ManagedAuthStore
from model_gateway.managed_auth.types import (
    HumanIdentity, KeyLimit, PersonalOwner, ProjectReference, ServiceOwner,
)


async def _interleave_read(store, engine, read, mutate):
    selected = asyncio.Event()
    resume = asyncio.Event()

    @asynccontextmanager
    async def connect():
        async with engine.connect() as connection:
            class PausedConnection:
                async def execute(self, *args, **kwargs):
                    result = await connection.execute(*args, **kwargs)
                    selected.set()
                    await resume.wait()
                    return result

            yield PausedConnection()

    store.engine = SimpleNamespace(connect=connect)
    task = asyncio.create_task(read())
    try:
        await asyncio.wait_for(selected.wait(), 5)
        await mutate()
    finally:
        resume.set()
        store.engine = engine
    return await task


@pytest.mark.asyncio
async def test_approval_collection_rotation_limit_and_immutable_audit():
    engine = create_engine(os.environ["MODEL_GATEWAY_AUTH_DATABASE_URL"], 4)
    store = ManagedAuthStore(engine)
    person = HumanIdentity("https://issuer.example", f"person-{uuid4()}")
    project = ProjectReference(uuid4(), "organization-a", "dev")
    try:
        await store.record_human(person, True, person)
        writer = ManagedAuthStore(engine)
        request_id, key_id = uuid4(), uuid4()
        await store.create_request(
            request_id, "analysis", "team need", person, PersonalOwner(person), None, project, None, None,
        )
        assert await store.get_project_policy(project.project_id) is None
        results = await asyncio.gather(
            store.decide_request(request_id, person, key_id, ("monthly", Decimal("35.00")), None),
            store.decide_request(request_id, person, key_id, ("monthly", Decimal("35.00")), None),
            return_exceptions=True,
        )
        assert len([result for result in results if not isinstance(result, Exception)]) == 1
        assert len([result for result in results if isinstance(result, ValueError)]) == 1
        policy = await store.get_project_policy(project.project_id)
        assert policy.project == project and policy.period == "monthly"
        assert policy.base_limit_usd == Decimal("35.00") and policy.revision == 1
        key = await store.get_key(key_id)
        assert key.status == "pending_collection" and key.verifier_hash is None
        assert key.project == project
        assert key.key_limit is None
        assert (await store.get_request(request_id)).status == "approved"

        secret = "r8-test-secret-" + str(uuid4())
        verifier = hashlib.sha256(secret.encode()).hexdigest()
        collected = await store.collect_key(key_id, key.revision, verifier, person)
        assert collected.status == "active"
        assert await store.collect_key(key_id, key.revision, "another-hash", person) is None
        assert (await store.get_key_by_verifier(verifier)).id == key_id
        async with engine.connect() as connection:
            row = (await connection.execute(select(tables.keys).where(tables.keys.c.id == key_id))).one()
            assert secret not in str(row)
            assert row.verifier_hash == verifier and row.project_id == project.project_id
        key_limit = KeyLimit("weekly", Decimal("8.00"))
        limited = await store.update_key_limit(key_id, collected.revision, key_limit, person)
        assert limited.key_limit == key_limit and limited.project == project
        assert limited.verifier_hash == verifier
        assert (await store.get_key_by_verifier(verifier)).key_limit == key_limit
        prior_key = await _interleave_read(
            store, engine, lambda: store.get_key_by_verifier(verifier),
            lambda: writer.rotate_key(key_id, limited.revision, "replacement-verifier", person),
        )
        assert prior_key.status == "active" and prior_key.verifier_hash == verifier
        rotated = await store.get_key(key_id)
        assert rotated.id == key_id and rotated.verifier_hash == "replacement-verifier"
        assert rotated.key_limit == key_limit
        assert await store.get_key_by_verifier(verifier) is None
        assert await store.rotate_key(key_id, limited.revision, "losing-verifier", person) is None
        events = await store.audit_for("key", key_id)
        assert [event.change for event in events] == [
            "created", "collected", "limit_changed", "rotated",
        ]
        with pytest.raises(DBAPIError):
            async with engine.begin() as connection:
                await connection.execute(update(tables.audit).where(tables.audit.c.id == events[0].id).values(change="rewritten"))
        with pytest.raises(DBAPIError):
            async with engine.begin() as connection:
                await connection.execute(delete(tables.audit).where(tables.audit.c.id == events[0].id))
        assert (await store.audit_for("key", key_id))[0] == events[0]
        cleared = await store.update_key_limit(key_id, rotated.revision, None, person)
        assert cleared.key_limit is None and cleared.verifier_hash == "replacement-verifier"
        assert cleared.project == project
        assert (await store.get_key_by_verifier("replacement-verifier")).key_limit is None
        assert [event.change for event in await store.audit_for("key", key_id)] == [
            *[event.change for event in events], "limit_changed",
        ]
        revoked = await store.revoke_key(key_id, cleared.revision, person)
        assert revoked.status == "revoked"
        assert await store.get_key_by_verifier("replacement-verifier") is None
        assert await store.rotate_key(key_id, revoked.revision, "invalid", person) is None
        assert await store.human_enabled(person) is True
        await store.record_human(person, False, person)
        assert await store.human_enabled(person) is False
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_service_owner_project_policy():
    engine = create_engine(os.environ["MODEL_GATEWAY_AUTH_DATABASE_URL"], 3)
    store = ManagedAuthStore(engine)
    creator = HumanIdentity("https://issuer.example", f"creator-{uuid4()}")
    service_id, key_id = uuid4(), uuid4()
    owner = ServiceOwner(service_id)
    project = ProjectReference(uuid4(), "organization-a", "dev")
    try:
        await store.record_human(creator, True, creator)
        await store.create_service(service_id, "worker", creator)
        request_id = uuid4()
        await store.create_request(
            request_id, "worker key", "deployment", creator, owner, None, project, None, None,
        )
        assert await store.get_project_policy(project.project_id) is None
        await store.decide_request(request_id, creator, key_id, ("monthly", Decimal("10")), None)
        policy = await store.get_project_policy(project.project_id)
        assert policy.period == "monthly" and policy.base_limit_usd == Decimal("10")
        key = await store.get_key(key_id)
        assert key.creator == creator and key.owner == owner and key.project == project
        await store.collect_key(key_id, key.revision, str(uuid4()), creator)
        assert (await store.get_key(key_id)).project == project
    finally:
        await engine.dispose()
